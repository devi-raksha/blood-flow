/* --------------------------------------------------------------------------
 * Blood flow simulation in 1D (dim=1, spacedim=3), run on a
 * parallel::fullydistributed::Triangulation.
 *
 * Unknown layout:
 *   FESystem( FE_DGQ(fe_degree), 2,   -> comps 0,1 : cell  A, U
 *             FE_DGQ(1),        2 )   -> comps 2,3 : trace A_hat, U_hat
 *   followed by n_rcr_dofs capacitor pressures appended after the FE range.
 *
 * Ownership follows the mesh partition: the rank owning a cell owns that
 * cell's cell-DoFs and its trace-DoFs, because every one of them is a DoF of
 * `dof_handler_`.  The only bookkeeping this file adds on top of deal.II is
 * face_dof_map, which records *which* of the four trace DoFs at a face is the
 * canonical one -- it does not decide who owns them.
 *
 * Assembly rule, obeyed by every routine below: read from the ghosted
 * vectors, write only rows this rank owns, and close with one compress().
 * --------------------------------------------------------------------------
 */
#include <deal.II/base/function_parser.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/types.h>

#include <deal.II/distributed/tria_base.h>

#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_description.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparsity_tools.h>

#include <deal.II/meshworker/mesh_loop.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/vector_tools.h>

#include <idas/idas.h>
#include <metric_flow_x/blood_flow_system.h>
#include <metric_flow_x/io/vtk_utils.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>

namespace MetricFlowX
{
  using namespace dealii;

  // ===========================================================================
  // IDA statistics access shim.
  //
  // SUNDIALS::IDA keeps its `ida_mem` handle private, but IDAGetNumSteps and
  // friends need it.  An explicit instantiation may name a private member, so
  // Planter<..., &IDA::ida_mem> captures the pointer-to-member and exposes it
  // through the ADL-found friend steal().  Same trick as the pre-MetricFlowX
  // driver; it lives in exactly one translation unit.
  // ===========================================================================
  namespace internal_ida_access
  {
    template <typename VecType>
    struct IdaMemAccessor
    {
      using type = void *dealii::SUNDIALS::IDA<VecType>::*;
      friend type        steal(IdaMemAccessor);
    };

    template <typename VecType, typename IdaMemAccessor<VecType>::type M>
    struct Planter
    {
      friend typename IdaMemAccessor<VecType>::type
      steal(IdaMemAccessor<VecType>)
      {
        return M;
      }
    };

    template struct Planter<VectorType,
                            &dealii::SUNDIALS::IDA<VectorType>::ida_mem>;

    inline void *
    get_ida_mem(const dealii::SUNDIALS::IDA<VectorType> &ida)
    {
      return ida.*steal(IdaMemAccessor<VectorType>{});
    }
  } // namespace internal_ida_access

  // ==========================================================================
  // Constructor
  // ==========================================================================
  template <int dim, int spacedim>
  BloodFlowSystem<dim, spacedim>::BloodFlowSystem(const MPI_Comm comm)
    : ParameterAcceptor("MetricFlowSystem<" + std::to_string(dim) + ", " +
                        std::to_string(spacedim) + ">")
    , mpi_communicator_(comm)
    , n_mpi_processes(Utilities::MPI::n_mpi_processes(comm))
    , this_mpi_process(Utilities::MPI::this_mpi_process(comm))
    , pcout(std::cout, this_mpi_process == 0)
    , computing_timer(mpi_communicator_,
                      pcout,
                      TimerOutput::summary,
                      TimerOutput::wall_times)
    , direct_solver_control(1000, 1e-10)
    , par("Metric Flow Parameters",
          {"rho", "mu", "xi", "m", "Rt"},
          {1060, 0.004, 2.0, 0.5, 0.5},
          {"Density",
           "Viscosity coefficient",
           "Profile constant for friction term",
           "Tube law exponent",
           "Reflection coefficient at outflow boundary"})
    , triangulation_(mpi_communicator_)
    , dof_handler_(triangulation_)
    , fe_(nullptr)
    , rhs_function("Functions",
                   "0.0; 0.0",
                   "RHS expression",
                   par,
                   dealii::FunctionParser<spacedim>::default_variable_names() +
                     ",t")
    , exact_solution(
        "Functions",
        "1e-4; 0.0; 0.0; 0.0",
        "Exact solution",
        par,
        dealii::FunctionParser<spacedim>::default_variable_names() + ",t")
    , inflow_function("Functions", "0.0", "Inflow function", par, "x,t")
  {
    add_parameter("Finite element degree", fe_degree);
    add_parameter("Problem constants", constants);
    add_parameter("Output filename", output_filename);
    add_parameter("Use direct solver", use_direct_solver);
    add_parameter("Number of refinement cycles", n_refinement_cycles);
    add_parameter("Number of global refinement", n_global_refinements);
    add_parameter("Theta (penalty parameter)", theta);
    add_parameter("Theta Boundary (stability parameter)", theta_bd);
    add_parameter("Gamma (Total pressure factor)", gamma);
    add_parameter("Verbosity (console depth)", verbosity);
    add_parameter("Output directory", output_directory);
    add_parameter(
      "Output time interval",
      time_interval,
      "Positive value sets the interval between written outputs; zero writes "
      "at every IDA callback.");
    add_parameter("Numerical flux type", numerical_flux_type_str);
    add_parameter("Use Riemann Invariants", use_riemann_invariants);
    add_parameter("Use junction mesh", use_junction_mesh);
    add_parameter("Outlet boundary condition type", outlet_type);
    add_parameter("Vtk file path for mesh input", vtk_file_path);

    this->enter_subsection("IDA parameters");
    this->enter_my_subsection(this->prm);
    ida_parameters.add_parameters(this->prm);
    this->leave_my_subsection(this->prm);
    this->leave_subsection();
  }

  // ============================================================================
  // initialize_params
  // ============================================================================
  template <int dim, int spacedim>
  void
  BloodFlowSystem<dim, spacedim>::initialize_params(const std::string &filename)
  {
    TimerOutput::Scope t(computing_timer, "initialize_params");

    ParameterAcceptor::initialize(filename,
                                  "last_used_parameters.prm",
                                  ParameterHandler::Short,
                                  this->prm,
                                  ParameterHandler::Short);

    AssertThrow(std::isfinite(time_interval) && time_interval >= 0.0,
                ExcMessage("Output time interval must be finite and "
                           "non-negative."));
    if (time_interval > 0.0)
      AssertThrow(ida_parameters.initial_time + time_interval >
                      ida_parameters.initial_time &&
                    ida_parameters.final_time - time_interval <
                      ida_parameters.final_time,
                  ExcMessage("Output time interval is too small to be "
                             "represented at the simulation time scale."));

    // deallog is written by rank 0 only; every other rank stays silent so that
    // the console is readable and the log file is not written K times over.
    deallog.depth_console(this_mpi_process == 0 ? verbosity : 0);
    deallog.depth_file(this_mpi_process == 0 ? verbosity : 0);

    exact_solution.update_constants(par);
    rhs_function.update_constants(par);
    inflow_function.update_constants(par);

    if (numerical_flux_type_str == "HLL")
      numerical_flux_type = NumericalFluxType::HLL;
    else if (numerical_flux_type_str == "HLL_HDG")
      numerical_flux_type = NumericalFluxType::HLL_HDG;
    else if (numerical_flux_type_str == "LAX_FRIEDRICHS")
      numerical_flux_type = NumericalFluxType::LAX_FRIEDRICHS;
    else
      AssertThrow(false,
                  ExcMessage("Unknown numerical flux type: " +
                             numerical_flux_type_str));
  }

  template <int dim, int spacedim>
  void
  BloodFlowSystem<dim, spacedim>::setup()
  {
    create_triangulation();
    setup_system();
    initialize_terminal_capacitors();
    build_per_cell_mass_inv();
  }

  template <int dim, int spacedim>
  VectorType
  BloodFlowSystem<dim, spacedim>::make_state() const
  {
    return VectorType(locally_owned_dofs_, mpi_communicator_);
  }

  template <int dim, int spacedim>
  void
  BloodFlowSystem<dim, spacedim>::reinit_state(VectorType &state) const
  {
    state.reinit(locally_owned_dofs_, mpi_communicator_);
  }

  template <int dim, int spacedim>
  void
  BloodFlowSystem<dim, spacedim>::initialize_state(VectorType  &state,
                                                   const double t0)
  {
    reinit_state(state);
    compute_initial_solution(state, t0);
    initialize_trace_unknowns(state, t0);
  }

  template <int dim, int spacedim>
  void
  BloodFlowSystem<dim, spacedim>::initialize_state_derivative(
    VectorType &state_dot,
    const double /*t0*/) const
  {
    reinit_state(state_dot);
    state_dot = 0.0;
  }

  template <int dim, int spacedim>
  const parallel::fullydistributed::Triangulation<dim, spacedim> &
  BloodFlowSystem<dim, spacedim>::triangulation() const
  {
    return triangulation_;
  }

  template <int dim, int spacedim>
  const DoFHandler<dim, spacedim> &
  BloodFlowSystem<dim, spacedim>::dof_handler() const
  {
    return dof_handler_;
  }

  template <int dim, int spacedim>
  const FiniteElement<dim, spacedim> &
  BloodFlowSystem<dim, spacedim>::finite_element() const
  {
    AssertThrow(fe_ != nullptr, ExcMessage("BloodFlowSystem is not set up."));
    return *fe_;
  }

  template <int dim, int spacedim>
  const AffineConstraints<double> &
  BloodFlowSystem<dim, spacedim>::constraints() const
  {
    return constraints_;
  }

  template <int dim, int spacedim>
  MPI_Comm
  BloodFlowSystem<dim, spacedim>::mpi_communicator() const
  {
    return mpi_communicator_;
  }

  template <int dim, int spacedim>
  const IndexSet &
  BloodFlowSystem<dim, spacedim>::locally_owned_dofs() const
  {
    return locally_owned_dofs_;
  }

  template <int dim, int spacedim>
  const IndexSet &
  BloodFlowSystem<dim, spacedim>::locally_relevant_dofs() const
  {
    return locally_relevant_dofs_;
  }

  template <int dim, int spacedim>
  const IndexSet &
  BloodFlowSystem<dim, spacedim>::differential_dofs() const
  {
    return differential_dofs_;
  }

  template <int dim, int spacedim>
  const IndexSet &
  BloodFlowSystem<dim, spacedim>::algebraic_dofs() const
  {
    return algebraic_dofs_;
  }

  template <int dim, int spacedim>
  const IndexSet &
  BloodFlowSystem<dim, spacedim>::component_dofs(
    const Component component) const
  {
    return component_dofs_[static_cast<unsigned int>(component)];
  }

  template <int dim, int spacedim>
  FEValuesExtractors::Scalar
  BloodFlowSystem<dim, spacedim>::area_extractor() const
  {
    return FEValuesExtractors::Scalar(0);
  }

  template <int dim, int spacedim>
  FEValuesExtractors::Scalar
  BloodFlowSystem<dim, spacedim>::velocity_extractor() const
  {
    return FEValuesExtractors::Scalar(1);
  }

  // ============================================================================
  // create_triangulation
  //
  // Every rank reads the VTK file and builds the coarse mesh serially, then
  // partitions it and keeps only its own part plus a ghost layer.  For a 1-D
  // network the coarse mesh and the per-vessel data are small; what has to
  // scale is the refined mesh, and that is what the fully distributed
  // Triangulation stores.  If the coarse mesh ever stops fitting in one rank's
  // memory, this is the function to replace with
  // create_description_from_triangulation_in_groups().
  // ============================================================================
  template <int dim, int spacedim>
  void
  BloodFlowSystem<dim, spacedim>::create_triangulation()
  {
    TimerOutput::Scope timer(computing_timer, "create_triangulation");

    Triangulation<dim, spacedim> serial_triangulation;

    {
      GridIn<dim, spacedim> grid_in;
      grid_in.attach_triangulation(serial_triangulation);
      if (verbosity > 0)
        pcout << "Reading VTK file: " << vtk_file_path << std::endl;
      std::ifstream mesh_file(vtk_file_path);
      AssertThrow(mesh_file, ExcMessage("Cannot open " + vtk_file_path));
      grid_in.read_vtk(mesh_file);
    }
    if (verbosity > 0)
      pcout << "Coarse cells = " << serial_triangulation.n_active_cells()
            << std::endl;

    VTKUtils::read_cell_data(vtk_file_path, "vessel_id", cell_vessel_ids);
    VTKUtils::read_cell_data(vtk_file_path, "a0", cell_a0);
    VTKUtils::read_cell_data(vtk_file_path, "a_d", cell_a_d);
    VTKUtils::read_cell_data(vtk_file_path, "E", cell_E);
    VTKUtils::read_cell_data(vtk_file_path, "h_wall", cell_h_wall);
    VTKUtils::read_cell_data(vtk_file_path, "p_d", cell_p_d);
    VTKUtils::read_cell_data(vtk_file_path, "p0", cell_p0);
    VTKUtils::read_cell_data(vtk_file_path, "L", cell_L);
    VTKUtils::read_cell_data(vtk_file_path, "r_d", cell_r_d);

    // Tapered radii are optional.
    try
      {
        VTKUtils::read_cell_data(vtk_file_path, "r_in", cell_r_in);
      }
    catch (...)
      {
        cell_r_in.reinit(0);
      }
    try
      {
        VTKUtils::read_cell_data(vtk_file_path, "r_out", cell_r_out);
      }
    catch (...)
      {
        cell_r_out.reinit(0);
      }

    VTKUtils::read_vertex_data(vtk_file_path, "R1", point_R1);
    VTKUtils::read_vertex_data(vtk_file_path, "R2", point_R2);
    VTKUtils::read_vertex_data(vtk_file_path, "C", point_C);
    VTKUtils::read_vertex_data(vtk_file_path, "P_out", point_P_out);
    VTKUtils::read_vertex_data(vtk_file_path, "boundary_id", point_boundary_id);

    // Material IDs + boundary IDs from VTK.  These are attributes of the coarse
    // cells and are inherited by their children through refinement and carried
    // into the distributed description, so they are available on every rank
    // afterwards without any communication.
    {
      unsigned int cell_idx = 0;
      for (auto &cell : serial_triangulation.active_cell_iterators())
        {
          cell->set_material_id(
            static_cast<unsigned int>(cell_vessel_ids[cell_idx]));
          for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
            if (cell->face(f)->at_boundary())
              {
                const unsigned int v = cell->face(f)->vertex_index(0);
                cell->face(f)->set_boundary_id(
                  static_cast<types::boundary_id>(point_boundary_id[v]));
              }
          ++cell_idx;
        }
    }

    // RCR map, keyed by boundary id: global data, identical on every rank
    // because every rank reads the same file.
    rcr_map.clear();
    for (const auto &cell : serial_triangulation.active_cell_iterators())
      for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
        if (cell->face(f)->at_boundary())
          {
            const types::boundary_id bid = cell->face(f)->boundary_id();
            if (bid == 0)
              continue; // inflow
            const unsigned int v = cell->face(f)->vertex_index(0);
            RCRPhysics         rcr;
            rcr.R1    = point_R1[v];
            rcr.R2    = point_R2[v];
            rcr.C     = point_C[v];
            rcr.P_out = point_P_out[v];
            if (rcr.R2 > 0.0)
              rcr_map[bid] = rcr;
          }

    serial_triangulation.refine_global(n_global_refinements);

    // Partition and hand the description to the fully distributed mesh.  The
    // z-order partitioner keeps a vessel's cells together, which keeps the
    // number of junctions straddling a subdomain boundary small.
    GridTools::partition_triangulation_zorder(n_mpi_processes,
                                              serial_triangulation);

    const auto description = TriangulationDescription::Utilities::
      create_description_from_triangulation(serial_triangulation,
                                            mpi_communicator_);

    triangulation_.create_triangulation(description);
    if (verbosity > 0)
      pcout << "Global active cells = "
            << triangulation_.n_global_active_cells() << std::endl;
    if (verbosity > 0)
      {
        const std::vector<unsigned int> owned =
          Utilities::MPI::gather(mpi_communicator_,
                                 triangulation_.n_locally_owned_active_cells());
        if (this_mpi_process == 0)
          for (unsigned int r = 0; r < owned.size(); ++r)
            pcout << "  rank " << r << " owns " << owned[r] << " cells"
                  << std::endl;
      }
  }

  // ============================================================================
  // build_global_vessel_data
  //
  // vessel_map and rcr_map describe the network, not the local part of it: a
  // rank needs the pressure law of every vessel whose cells it can see, and
  // compute_a_d_local() interpolates between a vessel's *global* end radii.  So
  // vessel_map comes from the replicated VTK arrays and the arc-length bounds
  // are reduced over the communicator.  Without the reduction a tapered vessel
  // split across two ranks would get a different a_d on each side of the cut.
  // ============================================================================
  template <int dim, int spacedim>
  void
  BloodFlowSystem<dim, spacedim>::build_global_vessel_data()
  {
    TimerOutput::Scope timer(computing_timer, "build_global_vessel_data");

    vessel_map.clear();

    unsigned int max_vid = 0;
    for (unsigned int i = 0; i < cell_vessel_ids.size(); ++i)
      max_vid =
        std::max(max_vid, static_cast<unsigned int>(cell_vessel_ids[i]));
    const unsigned int n_vid = max_vid + 1;

    for (unsigned int i = 0; i < cell_vessel_ids.size(); ++i)
      {
        const unsigned int vid = static_cast<unsigned int>(cell_vessel_ids[i]);
        if (vessel_map.count(vid))
          continue;

        VesselPhysicalProperties vp;
        vp.a0     = cell_a0[vid];
        vp.E      = cell_E[vid];
        vp.h_wall = cell_h_wall[vid];
        vp.p_d    = cell_p_d[vid];
        vp.p0     = cell_p0[vid];
        vp.a_d    = cell_a_d[vid];
        vp.L      = cell_L[vid];
        vp.r_d    = cell_r_d[vid];
        vp.r_in   = (cell_r_in.size() > vid) ? cell_r_in[vid] : 0.0;
        vp.r_out  = (cell_r_out.size() > vid) ? cell_r_out[vid] : 0.0;

        vessel_map[vid] = vp;
      }

    // Arc-length bounds: local extrema first, then a global min/max so that
    // every rank ends up with the same [s_min, s_max] per vessel.
    std::vector<double> s_min(n_vid, std::numeric_limits<double>::max());
    std::vector<double> s_max(n_vid, std::numeric_limits<double>::lowest());

    for (const auto &cell : triangulation_.active_cell_iterators())
      {
        if (!cell->is_locally_owned())
          continue;

        const unsigned int vid = cell->material_id();

        const Tensor<1, spacedim> d_hat =
          (cell->vertex(1) - cell->vertex(0)) /
          cell->vertex(1).distance(cell->vertex(0));

        const double s0 = cell->vertex(0) * d_hat;
        const double s1 = cell->vertex(1) * d_hat;

        s_min[vid] = std::min(s_min[vid], std::min(s0, s1));
        s_max[vid] = std::max(s_max[vid], std::max(s0, s1));
      }

    std::vector<double> global_s_min(n_vid);
    std::vector<double> global_s_max(n_vid);

    MPI_Allreduce(s_min.data(),
                  global_s_min.data(),
                  n_vid,
                  MPI_DOUBLE,
                  MPI_MIN,
                  mpi_communicator_);

    MPI_Allreduce(s_max.data(),
                  global_s_max.data(),
                  n_vid,
                  MPI_DOUBLE,
                  MPI_MAX,
                  mpi_communicator_);

    s_min.swap(global_s_min);
    s_max.swap(global_s_max);
    vessel_s_bounds.clear();
    for (const auto &[vid, vp] : vessel_map)
      {
        (void)vp;
        if (s_min[vid] <= s_max[vid])
          vessel_s_bounds[vid] = std::make_pair(s_min[vid], s_max[vid]);
      }

    // Terminal boundary ids: derived from rcr_map, which is replicated, so this
    // set is identical on every rank and needs no reduction.
    terminal_boundary_ids.clear();
    for (const auto &[bid, rcr] : rcr_map)
      {
        (void)rcr;
        terminal_boundary_ids.insert(bid);
      }

    if (verbosity > 0)
      {
        pcout << "\n=== vessel arc lengths ===\n  vid   L[mm]\n";
        for (const auto &[vid, b] : vessel_s_bounds)
          pcout << "  " << std::setw(3) << vid << "   " << std::setw(10)
                << (b.second - b.first) * 1000.0 << "\n";
      }
  }

  // ============================================================================
  // detect_junctions
  //
  // Runs over the locally relevant cells (owned + ghost).  The ghost layer is
  // vertex-adjacent, so a rank owning any cell incident to a junction vertex
  // also stores every other cell incident to it: the JunctionInfo it builds is
  // complete, and no rank ever sees a partial junction.
  //
  // The half-faces are sorted by (CellId, face_no) so that every rank that sees
  // a junction orders its vessels identically.  That matters because the row of
  // each junction equation is chosen by position in this list -- without the
  // sort, two ranks could disagree about which row carries mass conservation.
  // ============================================================================
  template <int dim, int spacedim>
  void
  BloodFlowSystem<dim, spacedim>::detect_junctions()
  {
    TimerOutput::Scope timer(computing_timer, "detect_junctions");

    junctions.clear();
    all_junction_faces.clear();

    // vertex -> list of (cell, local_face_no) pairs
    std::map<unsigned int,
             std::vector<std::pair<
               typename DoFHandler<dim, spacedim>::active_cell_iterator,
               unsigned int>>>
      vertex_to_half_faces;

    for (const auto &cell : dof_handler_.active_cell_iterators())
      {
        if (cell->is_artificial())
          continue;

        for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v)
          vertex_to_half_faces[cell->vertex_index(v)].emplace_back(cell, v);
      }

    unsigned int n_local_junctions = 0;

    for (auto &[v_idx, half_faces] : vertex_to_half_faces)
      {
        std::sort(half_faces.begin(),
                  half_faces.end(),
                  [](const auto &a, const auto &b) {
                    if (a.first->id() != b.first->id())
                      return a.first->id() < b.first->id();
                    return a.second < b.second;
                  });

        const std::size_t n_inc = half_faces.size();

        // Classify by how many cells meet at the vertex:
        //   n_inc == 1                    -> boundary (inlet / terminal).
        //   n_inc == 2, same vessel id    -> ordinary interior face.
        //   n_inc == 2, different ids     -> two vessels joined end to end, an
        //                                    in-line connection, treated as a
        //                                    2-way junction.
        //   n_inc >= 3                    -> junction.
        if (n_inc == 1)
          continue;

        if (n_inc == 2)
          {
            const unsigned int vid_a = half_faces[0].first->material_id();
            const unsigned int vid_b = half_faces[1].first->material_id();
            if (vid_a == vid_b)
              continue;

            if (unify_two_way_junctions)
              {
                if (two_way_pressure_laws_match(vid_a, vid_b))
                  {
                    // Same p(A) on both sides: the K=2 junction conditions
                    // reduce to a single unique trace pair.  Demote to an
                    // interior face, handled by
                    // assemble_trace_interior_equations() plus the continuity
                    // rows like every other valence-<=2 face.
                    continue;
                  }
                pcout
                  << "WARNING: 2-way node between vessels " << vid_a << " and "
                  << vid_b
                  << " has differing pressure laws; keeping the per-vessel\n"
                     "         junction equations (a unique trace there "
                     "would violate total-pressure continuity).\n";
              }
          }

        JunctionInfo J;
        J.location = triangulation_.get_vertices()[v_idx];

        for (const auto &[cell, local_face] : half_faces)
          {
            JunctionHalfFace jhf;
            jhf.cell        = cell;
            jhf.face_no     = local_face;
            jhf.orientation = (local_face == 1) ? 1 : -1;

            J.half_faces.push_back(jhf);
            all_junction_faces.emplace(cell->id(), local_face);
          }

        junctions.push_back(std::move(J));
        ++n_local_junctions;
      }

    const unsigned int n_global_junctions =
      Utilities::MPI::sum(n_local_junctions, mpi_communicator_);
    if (verbosity > 0)
      pcout
        << "Detected " << n_global_junctions
        << " junction half-face groups (counting each junction once per rank "
           "that touches it)."
        << std::endl;
  }

  // ============================================================================
  // canonical_face_key
  //
  // Interior face: key = the half-face whose cell has the smaller CellId.
  // Boundary / junction face: key = the unique owning cell.
  //
  // CellId ordering does not depend on the partition, so all ranks agree on the
  // canonical side of a face that straddles a subdomain boundary.
  // ============================================================================
  template <int dim, int spacedim>
  std::pair<CellId, unsigned int>
  BloodFlowSystem<dim, spacedim>::canonical_face_key(
    const typename DoFHandler<dim, spacedim>::active_cell_iterator &cell,
    const unsigned int face_no) const
  {
    if (cell->face(face_no)->at_boundary() ||
        is_junction_face(cell->id(), face_no))
      return {cell->id(), face_no};

    const auto        &nb         = cell->neighbor(face_no);
    const unsigned int nb_face_no = cell->neighbor_of_neighbor(face_no);

    if (cell->id() < nb->id())
      return {cell->id(), face_no};

    return {nb->id(), nb_face_no};
  }

  // ============================================================================
  // build_face_dof_map
  //
  // Records, for every face this rank can reach from one of its locally owned
  // cells, which (A_hat, U_hat) pair is the canonical one.  The indices come
  // straight out of the FESystem, so this map answers "which DoF" and never
  // "whose DoF" -- ownership is read off locally_owned_dofs_ at the point of
  // use.
  //
  // Both sides of an interior face are locally relevant (the ghost layer is
  // vertex-adjacent), so the canonical indices can always be looked up, even
  // when the canonical cell belongs to another rank.
  //
  // The non-canonical side of an ordinary interior face is a duplicate: this
  // rank owns its two rows whenever it owns that cell, regardless of who owns
  // the canonical side, and ties them down with the continuity rows.
  // ============================================================================
  template <int dim, int spacedim>
  void
  BloodFlowSystem<dim, spacedim>::build_face_dof_map()
  {
    TimerOutput::Scope timer(computing_timer, "build_face_dof_map");

    face_dof_map.clear();
    trace_continuity_pairs.clear();

    std::vector<types::global_dof_index> ldofs(fe_->n_dofs_per_cell());
    std::vector<types::global_dof_index> nb_dofs(fe_->n_dofs_per_cell());

    for (const auto &cell : dof_handler_.active_cell_iterators())
      {
        if (!(cell->is_locally_owned()))
          continue;

        cell->get_dof_indices(ldofs);

        for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
          {
            const auto key = canonical_face_key(cell, f);
            // if (cell->face(f)->at_boundary() || is_junction_face(cell->id(),
            // f))
            //   {
            //     std::cout << "Rank " << this_mpi_process
            //               << " inserting junction key " << key.first << " "
            //               << key.second << " owned=" <<
            //               cell->is_locally_owned()
            //               << " ghost=" << cell->is_ghost() << std::endl;
            //   }
            const auto [a_here, u_here] = face_trace_dofs(ldofs, f);

            const bool this_side_is_canonical = (key.first == cell->id());

            if (this_side_is_canonical)
              {
                face_dof_map[key] = FaceTraceDof{a_here, u_here};
              }
            else
              {
                // The canonical side is the neighbour, owned by this rank or a
                // ghost.  Either way its DoF indices are readable.
                const auto        &nb   = cell->neighbor(f);
                const unsigned int nb_f = cell->neighbor_of_neighbor(f);
                nb->get_dof_indices(nb_dofs);
                const auto [a_canon, u_canon] = face_trace_dofs(nb_dofs, nb_f);

                face_dof_map[key] = FaceTraceDof{a_canon, u_canon};

                // This cell's own pair is the duplicate side.  Junction and
                // boundary half-faces are never duplicated: each is its own
                // canonical owner, so they never reach this branch.
                if (!cell->face(f)->at_boundary() &&
                    !is_junction_face(cell->id(), f))
                  trace_continuity_pairs.push_back(
                    TraceContinuityPair{a_here, u_here, a_canon, u_canon});
              }
          }
      }

    const types::global_dof_index n_pairs =
      Utilities::MPI::sum<types::global_dof_index>(
        trace_continuity_pairs.size(), mpi_communicator_);
    if (verbosity > 0)
      pcout << "Face DOF map: FE dofs = " << dof_handler_.n_dofs()
            << "  continuity pairs = " << n_pairs << std::endl;
  }

  // ==========================================================================
  // build_rcr_dof_map
  //
  // One capacitor pressure per RCR terminal with C > 0 or we write R_2 > 0,
  // appended after the FE range.  The terminal boundary ids come from rcr_map,
  // which is replicated, so the enumeration is identical on every rank without
  // communication -- a necessary property, since a global index must mean the
  // same thing everywhere.
  //
  // A Pc index is *owned* by the rank owning the cell that carries the terminal
  // face.  Each terminal face has exactly one incident cell, so this assigns
  // every Pc to exactly one rank and locally_owned_dofs_ stays a partition.
  // ==========================================================================
  template <int dim, int spacedim>
  void
  BloodFlowSystem<dim, spacedim>::build_rcr_dof_map()
  {
    TimerOutput::Scope timer(computing_timer, "build_rcr_dof_map");

    rcr_pc_dof.clear();
    n_rcr_dofs = 0;

    const types::global_dof_index first_pc = dof_handler_.n_dofs();

    // Numbering: identical on every rank, no communication needed -- rcr_map
    // is replicated, so every rank enumerates it the same way.
    if (outlet_type != "Reflection")
      for (const auto &[bid, rcr] : rcr_map)
        {
          if (bid == 0)
            continue; // inflow
          if (rcr.C <= 0.0)
            continue; // single-R terminal carries no capacitor unknown

          rcr_pc_dof[bid] = first_pc + n_rcr_dofs;
          ++n_rcr_dofs;
        }

    n_total_dofs = first_pc + n_rcr_dofs;

    // Ownership: dof_handler_'s own FE numbering is already split into
    // contiguous, rank-ordered blocks with no gaps between them (that's
    // intrinsic to how a parallel DoFHandler numbers its own dofs -- rank r's
    // block ends exactly where rank r+1's begins).  The Pc range sits right
    // after all of that, starting at first_pc.  For a rank's combined FE+Pc
    // range to be one contiguous interval (which is what PETSc requires),
    // every Pc must go to the *one* rank whose FE block ends at first_pc -- no
    // other rank's FE block is adjacent to the Pc range, so splitting the Pc's
    // up by terminal ownership (or by any other rule) always leaves a gap for
    // every rank except that one.  Assembly is unaffected: it still happens on
    // whichever rank owns each terminal's cell, and off-owner contributions
    // are already routed correctly by the compress(VectorOperation::add)
    // calls used throughout this file.
    unsigned int local_candidate = n_mpi_processes; // sentinel: "not me"
    if (first_pc > 0 &&
        dof_handler_.locally_owned_dofs().is_element(first_pc - 1))
      local_candidate = this_mpi_process;

    const unsigned int pc_owner =
      Utilities::MPI::min(local_candidate, mpi_communicator_);

    Assert(pc_owner < n_mpi_processes,
           ExcMessage("Could not find the rank owning the last FE DoF."));

    rcr_dofs_owned.clear();
    rcr_dofs_owned.set_size(n_total_dofs);
    if (this_mpi_process == pc_owner)
      for (types::global_dof_index i = first_pc; i < n_total_dofs; ++i)
        rcr_dofs_owned.add_index(i);
    rcr_dofs_owned.compress();

    Assert(Utilities::MPI::sum<types::global_dof_index>(
             rcr_dofs_owned.n_elements(), mpi_communicator_) == n_rcr_dofs,
           ExcMessage(
             "Every capacitor DOF must be owned by exactly one rank."));
    if (verbosity > 0)
      pcout << "RCR capacitor DOFs: n_rcr = " << n_rcr_dofs
            << "  n_total = " << n_total_dofs << std::endl;
  }

  // ============================================================================
  // update_ghosted_vectors
  //
  // The single point where the solver communicates during assembly: copy the
  // locally owned `y` into the two ghosted read vectors.  Everything that
  // follows reads from these and writes only owned rows, so no assembly routine
  // needs a reduction of its own.
  // ============================================================================
  template <int dim, int spacedim>
  void
  BloodFlowSystem<dim, spacedim>::update_ghosted_vectors(
    const VectorType &y) const
  {
    TimerOutput::Scope timer(computing_timer, "update_ghosted_vectors");


    y_relevant = y;
    // std::cout << "Rank " << this_mpi_process << '\n'
    //           << "  y size        = " << y.size() << '\n'
    //           << "  owned dofs    = " << locally_owned_dofs_.n_elements() <<
    //           '\n'
    //           << "  owned FE dofs = " << locally_owned_fe_dofs.n_elements()
    //           << '\n'
    //           << "  range = [" << y.local_range().first << ", "
    //           << y.local_range().second << ")\n";
    const auto range = y.local_range();

    for (const auto i : locally_owned_fe_dofs)
      {
        if (i < range.first || i >= range.second)
          {
            std::cout << "Rank " << this_mpi_process << " invalid FE index "
                      << i << " local range [" << range.first << ","
                      << range.second << ")\n";
            std::abort();
          }

        y_fe_owned(i) = y(i);
      }
    y_fe_owned.compress(VectorOperation::insert);
    y_fe_relevant = y_fe_owned;
  }

  // ============================================================================
  // get_face_trace
  // ============================================================================
  template <int dim, int spacedim>
  void
  BloodFlowSystem<dim, spacedim>::get_face_trace(
    const VectorType                                               &y,
    const typename DoFHandler<dim, spacedim>::active_cell_iterator &cell,
    const unsigned int                                              face_no,
    double                                                         &A_hat,
    double                                                         &U_hat) const
  {
    const auto key = canonical_face_key(cell, face_no);

    const auto it = face_dof_map.find(key);
    Assert(it != face_dof_map.end(),
           ExcMessage("Face not found in face_dof_map."));

    A_hat = y(it->second.a_hat_dof);
    U_hat = y(it->second.u_hat_dof);
  }

  // ============================================================================
  // Sparsity
  //
  // Every builder below mirrors, one for one, the ownership guards used by the
  // matching assembly routine: a rank inserts a row into the pattern exactly
  // when it will later write that row.  Keeping the two in step is what allows
  // jacobian_matrix.reinit() to be given locally_owned_dofs_ as its row map.
  // ============================================================================

  // Cell rows: volume terms and the face flux against this cell's own trace.
  // The neighbour columns are kept so the pattern also covers the trace rows'
  // dependence on the two incident cells, assembled in build_trace_sparsity().
  template <int dim, int spacedim>
  void
  BloodFlowSystem<dim, spacedim>::build_cell_sparsity(
    DynamicSparsityPattern &dsp)
  {
    std::vector<types::global_dof_index> ldofs(fe_->n_dofs_per_cell());
    std::vector<types::global_dof_index> nb_dofs(fe_->n_dofs_per_cell());

    for (const auto &cell : dof_handler_.active_cell_iterators())
      {
        if (!cell->is_locally_owned())
          continue;

        cell->get_dof_indices(ldofs);

        for (const auto i : ldofs)
          for (const auto j : ldofs)
            dsp.add(i, j);

        for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
          {
            if (cell->face(f)->at_boundary())
              continue;

            cell->neighbor(f)->get_dof_indices(nb_dofs);
            for (const auto i : ldofs)
              for (const auto j : nb_dofs)
                dsp.add(i, j);
          }
      }
  }

  // Trace rows of interior and boundary faces: coupled to their own pair and to
  // the cell DoFs on both sides of the face.
  template <int dim, int spacedim>
  void
  BloodFlowSystem<dim, spacedim>::build_trace_sparsity(
    DynamicSparsityPattern &dsp)
  {
    std::vector<types::global_dof_index> cell_dofs(fe_->n_dofs_per_cell());
    std::vector<types::global_dof_index> nb_dofs(fe_->n_dofs_per_cell());

    for (const auto &cell : dof_handler_.active_cell_iterators())
      {
        if (!cell->is_locally_owned())
          continue;

        cell->get_dof_indices(cell_dofs);

        for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
          {
            const auto it = face_dof_map.find(canonical_face_key(cell, f));
            if (it == face_dof_map.end())
              continue;

            const auto a_hat = it->second.a_hat_dof;
            const auto u_hat = it->second.u_hat_dof;

            // Rows belonging to a canonical side on another rank are that
            // rank's business.
            if (!locally_owned_dofs_.is_element(a_hat))
              continue;

            dsp.add(a_hat, a_hat);
            dsp.add(a_hat, u_hat);
            dsp.add(u_hat, a_hat);
            dsp.add(u_hat, u_hat);

            for (const auto ci : cell_dofs)
              {
                dsp.add(a_hat, ci);
                dsp.add(u_hat, ci);
              }

            if (!cell->face(f)->at_boundary())
              {
                cell->neighbor(f)->get_dof_indices(nb_dofs);
                for (const auto ni : nb_dofs)
                  {
                    dsp.add(a_hat, ni);
                    dsp.add(u_hat, ni);
                  }
              }
          }
      }
  }

  // Junction rows: fully coupled among the 2K trace unknowns, plus the cell
  // DoFs feeding each vessel's Riemann invariant.
  template <int dim, int spacedim>
  void
  BloodFlowSystem<dim, spacedim>::build_junction_sparsity(
    DynamicSparsityPattern &dsp)
  {
    std::vector<types::global_dof_index> cell_dofs(fe_->n_dofs_per_cell());

    for (const auto &J : junctions)
      {
        const unsigned int K = J.n_vessels();

        std::vector<types::global_dof_index> a_row(K), u_row(K);
        std::vector<types::global_dof_index> ldofs(fe_->n_dofs_per_cell());

        for (unsigned int i = 0; i < K; ++i)
          {
            J.half_faces[i].cell->get_dof_indices(ldofs);

            const auto [a_hat_dof, u_hat_dof] =
              face_trace_dofs(ldofs, J.half_faces[i].face_no);

            a_row[i] = a_hat_dof;
            u_row[i] = u_hat_dof;
          }
        for (unsigned int i = 0; i < K; ++i)
          for (const auto row : {a_row[i], u_row[i]})
            {
              if (!locally_owned_dofs_.is_element(row))
                continue;

              for (unsigned int j = 0; j < K; ++j)
                {
                  dsp.add(row, a_row[j]);
                  dsp.add(row, u_row[j]);

                  J.half_faces[j].cell->get_dof_indices(cell_dofs);
                  for (const auto cj : cell_dofs)
                    dsp.add(row, cj);
                }
            }
      }
  }

  template <int dim, int spacedim>
  void
  BloodFlowSystem<dim, spacedim>::build_rcr_sparsity(
    DynamicSparsityPattern &dsp)
  {
    for (const auto &cell : dof_handler_.active_cell_iterators())
      {
        if (!cell->is_locally_owned())
          continue;

        for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
          {
            if (!cell->face(f)->at_boundary())
              continue;
            if (is_junction_face(cell->id(), f))
              continue;

            const auto pit = rcr_pc_dof.find(cell->face(f)->boundary_id());
            if (pit == rcr_pc_dof.end())
              continue;

            const auto it = face_dof_map.find(canonical_face_key(cell, f));
            if (it == face_dof_map.end())
              continue;

            const auto pc = pit->second;

            // Both the Pc row and the boundary trace rows are owned here: the
            // terminal face has this cell as its only neighbour.
            dsp.add(pc, pc);
            dsp.add(pc, it->second.a_hat_dof);
            dsp.add(pc, it->second.u_hat_dof);
            dsp.add(it->second.a_hat_dof, pc);
          }
      }
  }

  template <int dim, int spacedim>
  void
  BloodFlowSystem<dim, spacedim>::build_trace_continuity_sparsity(
    DynamicSparsityPattern &dsp)
  {
    // The duplicate side is always a locally owned cell's own pair, so these
    // rows are owned by construction.
    for (const auto &p : trace_continuity_pairs)
      {
        dsp.add(p.a_dup, p.a_dup);
        dsp.add(p.a_dup, p.a_canon);
        dsp.add(p.u_dup, p.u_dup);
        dsp.add(p.u_dup, p.u_canon);
      }
  }

  // ============================================================================
  // build_extended_sparsity_pattern
  //
  // The pattern is assembled locally over the relevant rows, exchanged with
  // SparsityTools::distribute_sparsity_pattern() so that each rank learns the
  // off-rank entries written into rows it owns, and then handed to the
  // matrices with locally_owned_dofs_ as the row map.
  // ============================================================================
  template <int dim, int spacedim>
  void
  BloodFlowSystem<dim, spacedim>::build_extended_sparsity_pattern()
  {
    TimerOutput::Scope timer(computing_timer,
                             "build_extended_sparsity_pattern");

    DynamicSparsityPattern dsp(n_total_dofs,
                               n_total_dofs,
                               locally_relevant_dofs_);

    build_cell_sparsity(dsp);
    build_trace_sparsity(dsp);
    build_junction_sparsity(dsp);
    build_rcr_sparsity(dsp);
    build_trace_continuity_sparsity(dsp);

    SparsityTools::distribute_sparsity_pattern(dsp,
                                               locally_owned_dofs_,
                                               mpi_communicator_,
                                               locally_relevant_dofs_);

    jacobian_matrix.reinit(locally_owned_dofs_,
                           locally_owned_dofs_,
                           dsp,
                           mpi_communicator_);
    state_jacobian_matrix_.reinit(locally_owned_dofs_,
                                  locally_owned_dofs_,
                                  dsp,
                                  mpi_communicator_);
    derivative_jacobian_matrix_.reinit(locally_owned_dofs_,
                                       locally_owned_dofs_,
                                       dsp,
                                       mpi_communicator_);
    linear_system_matrix.reinit(locally_owned_dofs_,
                                locally_owned_dofs_,
                                dsp,
                                mpi_communicator_);
  }

  // ============================================================================
  // setup_system
  // ============================================================================
  template <int dim, int spacedim>
  void
  BloodFlowSystem<dim, spacedim>::setup_system()
  {
    TimerOutput::Scope timer(computing_timer, "setup_system");

    if (!fe_)
      fe_ = std::make_unique<FESystem<dim, spacedim>>(
        FE_DGQ<dim, spacedim>(fe_degree),
        2, // comps 0,1 : cell A, U (DG)
        FE_DGQ<dim, spacedim>(1),
        2); // comps 2,3 : trace A_hat, U_hat

    dof_handler_.distribute_dofs(*fe_);

    // DoFRenumbering::component_wise() is deliberately not applied.  It would
    // not produce a global [cell | trace] block layout in parallel -- it
    // renumbers within each rank's owned range -- and nothing needs one: the
    // component index sets below take over every role the old integer ranges
    // played.

    // Global network data first: everything downstream (junction detection via
    // the pressure laws, compute_a_d_local() via the arc-length bounds) reads
    // it.
    build_global_vessel_data();

    if (outlet_type == "RCR")
      AssertThrow(!terminal_boundary_ids.empty(),
                  ExcMessage("No terminal boundaries found anywhere in the "
                             "network."));

    detect_junctions();
    build_face_dof_map();
    build_rcr_dof_map(); // sets n_total_dofs and rcr_dofs_owned

    // ---- index sets ---------------------------------------------------------
    locally_owned_fe_dofs = dof_handler_.locally_owned_dofs();

    locally_relevant_fe_dofs =
      DoFTools::extract_locally_relevant_dofs(dof_handler_);

    locally_owned_dofs_.clear();
    locally_owned_dofs_.set_size(n_total_dofs);
    locally_owned_dofs_.add_indices(locally_owned_fe_dofs);
    locally_owned_dofs_.add_indices(rcr_dofs_owned);
    locally_owned_dofs_.compress();
    locally_relevant_dofs_.clear();
    locally_relevant_dofs_.set_size(n_total_dofs);
    locally_relevant_dofs_.add_indices(locally_relevant_fe_dofs);
    locally_relevant_dofs_.add_indices(rcr_dofs_owned);

    // A rank reads Pc only for terminals whose cell it owns, but make every Pc
    // it can reach through a ghost relevant as well so that diagnostics and
    // output can read them without a special case.
    for (const auto &cell : dof_handler_.active_cell_iterators())
      {
        if (cell->is_artificial())
          continue;

        for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
          {
            if (!cell->face(f)->at_boundary())
              continue;
            const auto it = rcr_pc_dof.find(cell->face(f)->boundary_id());
            if (it != rcr_pc_dof.end())
              locally_relevant_dofs_.add_index(it->second);
          }
      }
    locally_relevant_dofs_.compress();

    // Component index sets: the differential rows of the DAE (cell A, U and the
    // capacitor pressures) versus the algebraic ones (the traces).  Taken from
    // a ComponentMask, so they are right with or without renumbering.
    {
      const ComponentMask cell_mask({true, true, false, false});
      const ComponentMask trace_mask({false, false, true, true});

      const IndexSet cell_fe = DoFTools::extract_dofs(dof_handler_, cell_mask);
      const IndexSet trace_fe =
        DoFTools::extract_dofs(dof_handler_, trace_mask);

      cell_dofs_owned.clear();
      cell_dofs_owned.set_size(n_total_dofs);
      cell_dofs_owned.add_indices(cell_fe);
      cell_dofs_owned.compress();

      trace_dofs_owned.clear();
      trace_dofs_owned.set_size(n_total_dofs);
      trace_dofs_owned.add_indices(trace_fe);
      trace_dofs_owned.compress();

      const std::array<ComponentMask, 4> component_masks = {
        ComponentMask({true, false, false, false}),
        ComponentMask({false, true, false, false}),
        ComponentMask({false, false, true, false}),
        ComponentMask({false, false, false, true})};
      for (unsigned int c = 0; c < component_masks.size(); ++c)
        {
          component_dofs_[c].clear();
          component_dofs_[c].set_size(n_total_dofs);
          component_dofs_[c].add_indices(
            DoFTools::extract_dofs(dof_handler_, component_masks[c]));
          component_dofs_[c].compress();
        }

      differential_dofs_.clear();
      differential_dofs_.set_size(n_total_dofs);
      differential_dofs_.add_indices(cell_dofs_owned);
      differential_dofs_.add_indices(rcr_dofs_owned);
      differential_dofs_.compress();

      algebraic_dofs_ = trace_dofs_owned;
    }

    for (const auto i : locally_owned_fe_dofs)
      {
        if (!locally_owned_dofs_.is_element(i))
          {
            std::cout << "Rank " << this_mpi_process
                      << " FE DoF missing from system IndexSet: " << i
                      << std::endl;
          }
      }
    // ---- sparsity + matrices ------------------------------------------------
    build_extended_sparsity_pattern();

    // ---- vectors ------------------------------------------------------------
    solution.reinit(locally_owned_dofs_, mpi_communicator_);
    solution_dot.reinit(locally_owned_dofs_, mpi_communicator_);
    pressure_.reinit(locally_owned_dofs_, mpi_communicator_);
    residual_F.reinit(locally_owned_dofs_, mpi_communicator_);

    y_relevant.reinit(locally_owned_dofs_,
                      locally_relevant_dofs_,
                      mpi_communicator_);
    y_fe_owned.reinit(locally_owned_fe_dofs, mpi_communicator_);
    y_fe_relevant.reinit(locally_owned_fe_dofs,
                         locally_relevant_fe_dofs,
                         mpi_communicator_);
    AssertThrow(y_fe_relevant.size() == dof_handler_.n_dofs(),
                ExcInternalError());
    if (verbosity > 0)
      pcout << "  cell DoFs (differential): "
            << Utilities::MPI::sum<types::global_dof_index>(
                 cell_dofs_owned.n_elements(), mpi_communicator_)
            << "\n  trace DoFs (algebraic)  : "
            << Utilities::MPI::sum<types::global_dof_index>(
                 trace_dofs_owned.n_elements(), mpi_communicator_)
            << "\n  total DoFs              : " << n_total_dofs << std::endl;
  }

  // ============================================================================
  // initialize_terminal_capacitors
  // ============================================================================
  template <int dim, int spacedim>
  void
  BloodFlowSystem<dim, spacedim>::initialize_terminal_capacitors()
  {
    terminal_Pc_storage.clear();

    for (const auto &cell : dof_handler_.active_cell_iterators())
      {
        if (!cell->is_locally_owned())
          continue;

        const unsigned int vid = cell->material_id();
        const auto        &vpp = vessel_map.at(vid);

        for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
          {
            if (!cell->face(f)->at_boundary())
              continue;
            if (is_junction_face(cell->id(), f))
              continue;

            const types::boundary_id bid = cell->face(f)->boundary_id();
            if (bid != 0 && is_terminal_boundary(bid))
              terminal_Pc_storage.try_emplace(bid, vpp.p_d);
          }
      }
  }

  // ============================================================================
  // compute_initial_solution
  // ============================================================================
  template <int dim, int spacedim>
  void
  BloodFlowSystem<dim, spacedim>::compute_initial_solution(VectorType &dst,
                                                           const double /*t*/)
  {
    TimerOutput::Scope timer(computing_timer, "compute_initial_solution");

    dst = 0.0;

    // ---- cell block and this cell's own trace pair --------------------------
    // Both are DoFs of a locally owned cell, hence owned here, so the whole
    // loop writes with insert semantics and no row is touched twice.
    std::vector<types::global_dof_index> ldofs(fe_->n_dofs_per_cell());

    for (const auto &cell : dof_handler_.active_cell_iterators())
      {
        if (!cell->is_locally_owned())
          continue;

        cell->get_dof_indices(ldofs);
        const double a_d_local = compute_a_d_local(cell);

        for (unsigned int i = 0; i < fe_->n_dofs_per_cell(); ++i)
          {
            const unsigned int comp = fe_->system_to_component_index(i).first;
            dst(ldofs[i])           = (comp == 0) ? a_d_local : 0.0;
          }

        // Trace: seed A_hat with the diastolic area at that face and U_hat with
        // zero, which is consistent for every face type (inflow, terminal,
        // interior and junction all start from a flux A_hat*U_hat*bn = 0).
        // Seeding both sides of an interior face from the same rule also starts
        // the continuity residual at zero.
        for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
          {
            const auto [a_here, u_here] = face_trace_dofs(ldofs, f);
            dst(a_here)                 = compute_a_d_at_face(cell, f);
            dst(u_here)                 = 0.0;
          }
      }

    // ---- capacitor block ----------------------------------------------------
    for (const auto &cell : dof_handler_.active_cell_iterators())
      {
        if (!cell->is_locally_owned())
          continue;

        for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
          {
            if (!cell->face(f)->at_boundary())
              continue;
            if (is_junction_face(cell->id(), f))
              continue;

            const auto pit = rcr_pc_dof.find(cell->face(f)->boundary_id());
            if (pit == rcr_pc_dof.end())
              continue;

            dst(pit->second) = vessel_map.at(cell->material_id()).p_d;
          }
      }

    dst.compress(VectorOperation::insert);
  }

  // ============================================================================
  // initialize_trace_unknowns
  //
  // Newton on the algebraic rows only, with the cell and capacitor rows pinned
  // by an identity so the same distributed matrix and solver can be reused.
  // ============================================================================
  template <int dim, int spacedim>
  void
  BloodFlowSystem<dim, spacedim>::initialize_trace_unknowns(VectorType  &sol,
                                                            const double t)
  {
    TimerOutput::Scope timer(computing_timer, "initialize_trace_unknowns");

    const double       tol      = 1.0e-8;
    const unsigned int max_iter = 50;
    if (verbosity > 0)
      pcout << "\n=== initialize_trace_unknowns (Newton) ===\n";

    VectorType G(locally_owned_dofs_, mpi_communicator_);
    VectorType rhs(locally_owned_dofs_, mpi_communicator_);
    VectorType delta(locally_owned_dofs_, mpi_communicator_);

    for (unsigned int iter = 0; iter < max_iter; ++iter)
      {
        update_ghosted_vectors(sol);

        // ---- G(yhat) = F_trace(y_cell^0, yhat) ------------------------------
        G = 0.0;
        assemble_trace_interior_equations(y_relevant, G);
        assemble_trace_boundary_equations(t, y_relevant, G);
        assemble_trace_junction_equations(y_relevant, G);
        assemble_trace_continuity_equations(y_relevant, G);
        G.compress(VectorOperation::add);

        // ---- convergence, over the algebraic rows only ----------------------
        double gnorm_inf_local = 0.0;
        for (const auto i : trace_dofs_owned)
          gnorm_inf_local = std::max(gnorm_inf_local, std::abs(G(i)));

        const double gnorm_inf =
          Utilities::MPI::max(gnorm_inf_local, mpi_communicator_);
        if (verbosity > 0)
          pcout << "  iter " << std::setw(3) << iter
                << "  ||G||_inf = " << std::scientific << std::setprecision(4)
                << gnorm_inf << "\n";

        if (gnorm_inf < tol)
          {
            if (verbosity > 0)
              pcout << "  Converged in " << iter << " Newton iteration(s).\n";
            break;
          }

        if (iter == max_iter - 1)
          {
            pcout << "WARNING: initialize_trace_unknowns did not converge.\n"
                  << "         tol=" << tol << "  ||G||_inf=" << gnorm_inf
                  << "  after " << max_iter << " iterations.\n";
            break;
          }

        // ---- J_tt, with an identity on every non-trace row
        // -------------------
        jacobian_matrix = 0.0;
        assemble_jacobian_trace_interior_block(y_relevant);
        assemble_jacobian_trace_boundary_block(t, y_relevant);
        assemble_jacobian_trace_junction_block(y_relevant);
        assemble_jacobian_trace_continuity_block();
        jacobian_matrix.compress(VectorOperation::add);

        linear_system_matrix = 0.0;
        for (const auto i : trace_dofs_owned)
          for (auto it = jacobian_matrix.begin(i); it != jacobian_matrix.end(i);
               ++it)
            linear_system_matrix.set(i, it->column(), it->value());

        for (const auto i : cell_dofs_owned)
          linear_system_matrix.set(i, i, 1.0);
        for (const auto i : rcr_dofs_owned)
          linear_system_matrix.set(i, i, 1.0);
        linear_system_matrix.compress(VectorOperation::insert);

        // ---- b = -G on the trace rows, zero elsewhere
        // ------------------------ The identity rows then return delta = 0
        // there, so the cell values and the seeded capacitor pressures are left
        // untouched.
        rhs = 0.0;
        for (const auto i : trace_dofs_owned)
          rhs(i) = -G(i);
        rhs.compress(VectorOperation::insert);

        // ---- solve and update
        // ------------------------------------------------
        {
          SolverControl solver_control(1, 0.0);
#ifdef USE_PETSC_LA
          PETScWrappers::SparseDirectMUMPS newton_solver(solver_control);
          newton_solver.solve(linear_system_matrix, delta, rhs);
#else
          TrilinosWrappers::SolverDirect newton_solver(solver_control);
          newton_solver.solve(linear_system_matrix, delta, rhs);
#endif
        }

        for (const auto i : trace_dofs_owned)
          sol(i) += delta(i);
        sol.compress(VectorOperation::add);
      }

    jacobian_matrix      = 0.0;
    linear_system_matrix = 0.0;
    jacobian_matrix.compress(VectorOperation::insert);
    linear_system_matrix.compress(VectorOperation::insert);
  }

  // ============================================================================
  // build_per_cell_mass_inv
  //
  // For every locally owned cell K:
  //   per_cell_mass     : M_K, used for the M*ydot term of the residual and the
  //                       alpha*M term of the Jacobian
  //   per_cell_mass_inv : M_K^{-1}, diagnostic only
  //
  // Indexed by cell->active_cell_index(): deal.II numbers the locally stored
  // active cells contiguously from zero, so the vectors are sized to the local
  // cell count and no CellId comparison happens in the residual inner loop.
  // ============================================================================
  template <int dim, int spacedim>
  void
  BloodFlowSystem<dim, spacedim>::build_per_cell_mass_inv()
  {
    TimerOutput::Scope timer(computing_timer, "build_per_cell_mass_inv");

    const unsigned int n_local_cells = triangulation_.n_active_cells();
    per_cell_mass.assign(n_local_cells, FullMatrix<double>());
    per_cell_mass_inv.assign(n_local_cells, FullMatrix<double>());

    const QGauss<dim>       quad(fe_degree + 1);
    FEValues<dim, spacedim> fev(*fe_, quad, update_values | update_JxW_values);

    const unsigned int n_dofs = fe_->n_dofs_per_cell();

    for (const auto &cell : dof_handler_.active_cell_iterators())
      {
        if (!cell->is_locally_owned())
          continue;

        fev.reinit(cell);
        FullMatrix<double> M(n_dofs, n_dofs);

        // FESystem(FE_DGQ, 2) shape functions are component specific: phi_i is
        // nonzero only for its own component, so M(i,j) = 0 whenever
        // component(i) != component(j).  Cross-component pairs are skipped;
        // otherwise M would have zero rows and gauss_jordan() would abort.
        // Only the cell components (0,1) carry a mass term, so the trace rows
        // and columns of M stay exactly zero -- which is what makes alpha*M
        // leave the trace rows algebraic.
        for (unsigned int q = 0; q < fev.n_quadrature_points; ++q)
          for (unsigned int i = 0; i < n_dofs; ++i)
            {
              const unsigned int ci = fe_->system_to_component_index(i).first;
              if (ci >= 2)
                continue;
              for (unsigned int j = 0; j < n_dofs; ++j)
                {
                  const unsigned int cj =
                    fe_->system_to_component_index(j).first;
                  if (ci != cj)
                    continue;
                  M(i, j) +=
                    fev.shape_value(i, q) * fev.shape_value(j, q) * fev.JxW(q);
                }
            }

        per_cell_mass[cell->active_cell_index()] = M;

        // For the diagnostic inverse, put 1 on the trace diagonal so the dense
        // inverse exists; that block is never applied.
        FullMatrix<double> M_inv(M);
        for (unsigned int i = 0; i < n_dofs; ++i)
          if (fe_->system_to_component_index(i).first >= 2)
            M_inv(i, i) = 1.0;
        M_inv.gauss_jordan();
        per_cell_mass_inv[cell->active_cell_index()] = std::move(M_inv);
      }
  }

  // ============================================================================
  // open_csv_files
  //
  // One CSV per vessel, written at the vessel's arc-length midpoint.  The probe
  // cell is chosen by a global argmin over the distance to the midpoint: each
  // rank offers its best locally owned candidate and the rank holding the
  // overall winner opens the file.  Every vessel is therefore probed exactly
  // once, by whichever rank owns the middle of it, and no file is written by
  // two ranks.
  // ============================================================================
  template <int dim, int spacedim>
  void
  BloodFlowSystem<dim, spacedim>::open_csv_files()
  {
    const std::string dir =
      output_directory + (output_directory.empty() ? "" : "/");

    // Every rank may open a vessel file, so every rank makes sure the
    // directory exists (create_directories is a no-op when it already does).
    if (!output_directory.empty())
      std::filesystem::create_directories(output_directory);

    close_csv_files();
    csv_vessel.clear();
    probe_targets.clear();

    const std::string hdr = "time_s,P_dynpcm2,Q_cm3ps,A_cm2,U_cmps\n";

    for (const auto &[vid, vpp] : vessel_map)
      {
        (void)vpp;

        double     s_mid = 0.0;
        const auto it    = vessel_s_bounds.find(vid);
        if (it != vessel_s_bounds.end())
          s_mid = 0.5 * (it->second.first + it->second.second);

        double best_dist = std::numeric_limits<double>::max();
        typename DoFHandler<dim, spacedim>::active_cell_iterator best_cell;
        bool                                                     found = false;

        for (const auto &cell : dof_handler_.active_cell_iterators())
          {
            if (!cell->is_locally_owned())
              continue;
            if (cell->material_id() != vid)
              continue;

            const Tensor<1, spacedim> d_hat = compute_directional_vector(cell);
            const double              s     = cell->center() * d_hat;
            const double              d     = std::abs(s - s_mid);
            if (d < best_dist)
              {
                best_dist = d;
                best_cell = cell;
                found     = true;
              }
          }

        // Ties are broken by rank, so exactly one rank claims the vessel even
        // if two of them report the same distance.
        const auto winner =
          Utilities::MPI::min_max_avg(best_dist, mpi_communicator_);
        const unsigned int winning_rank =
          Utilities::MPI::min(found && best_dist <= winner.min ?
                                this_mpi_process :
                                numbers::invalid_unsigned_int,
                              mpi_communicator_);

        if (winning_rank != this_mpi_process || !found)
          continue;

        probe_targets.emplace_back(vid, best_cell);

        std::ofstream &os = csv_vessel[vid];
        os.open(dir + "HDG_IDA_Vessel_" + std::to_string(vid) + ".csv");
        os << hdr;
      }
  }

  // ============================================================================
  // write_csv_row
  // ============================================================================
  template <int dim, int spacedim>
  void
  BloodFlowSystem<dim, spacedim>::write_csv_row(const double      t,
                                                const VectorType &sol)
  {
    const unsigned int dofs_per_cell = fe_->n_dofs_per_cell();

    // Reference-cell midpoint (DGQ, dim == 1 -> xi = 0.5).
    const Point<dim> xi_mid = (dim == 1) ? Point<dim>(0.5) : Point<dim>();

    for (const auto &[vid, cell] : probe_targets)
      {
        std::ofstream &os = csv_vessel.at(vid);

        // The probe cell is locally owned, so every one of its DoFs is owned
        // here and can be read straight out of the non-ghosted vector.
        std::vector<types::global_dof_index> ldofs(dofs_per_cell);
        cell->get_dof_indices(ldofs);

        double A_val = 0.0, U_val = 0.0;
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          {
            const double       phi  = fe_->shape_value(i, xi_mid);
            const unsigned int comp = fe_->system_to_component_index(i).first;
            if (comp == 0)
              A_val += sol(ldofs[i]) * phi;
            else if (comp == 1)
              U_val += sol(ldofs[i]) * phi;
          }

        const double A_cm2  = A_val * 1.0e4;
        const double U_cmps = U_val * 1.0e2;

        const double P_Pa =
          compute_pressure_value(A_val, vid, compute_a_d_local(cell));
        const double P_dynpcm2 = P_Pa * 10.0;

        const double Q_cm3ps = A_val * U_val * 1.0e6;

        os << std::scientific << std::setprecision(8) << t << "," << P_dynpcm2
           << "," << Q_cm3ps << "," << A_cm2 << "," << U_cmps << "\n";
        os.flush();
      }
  }

  // ============================================================================
  // close_csv_files
  // ============================================================================
  template <int dim, int spacedim>
  void
  BloodFlowSystem<dim, spacedim>::close_csv_files()
  {
    for (auto &kv : csv_vessel)
      if (kv.second.is_open())
        kv.second.close();
  }

  // ============================================================================
  // HLL flux (residual)
  // ============================================================================
  template <int dim, int spacedim>
  std::array<double, 2>
  BloodFlowSystem<dim, spacedim>::hll_flux(const double       bn_L,
                                           const double       bn_R,
                                           const double       A_L,
                                           const double       U_L,
                                           const double       A_R,
                                           const double       U_R,
                                           const unsigned int vid_L,
                                           const unsigned int vid_R,
                                           const double       ad_L,
                                           const double       ad_R) const
  {
    const double c_L   = compute_wave_speed(A_L, vid_L, ad_L);
    const double c_R   = compute_wave_speed(A_R, vid_R, ad_R);
    const double U_bar = 0.5 * (U_L + U_R);
    const double c_bar = 0.5 * (c_L + c_R);
    const double s_L   = U_bar - c_bar;
    const double s_R   = U_bar + c_bar;
    // const double s_L = std::min(U_L - c_L, U_R - c_R);

    // const double s_R = std::max(U_L + c_L, U_R + c_R);

    const double FAL = scalar_area_flux(bn_L, A_L, U_L);
    const double FUL = scalar_momentum_flux(
      bn_L, U_L, compute_pressure_value(A_L, vid_L, ad_L), par["rho"]);
    const double FAR = scalar_area_flux(bn_R, A_R, U_R);
    const double FUR = scalar_momentum_flux(
      bn_R, U_R, compute_pressure_value(A_R, vid_R, ad_R), par["rho"]);

    if (s_L >= 0.0)
      return {{FAL, FUL}};
    if (s_R <= 0.0)
      return {{FAR, FUR}};

    const double inv = 1.0 / (s_R - s_L);
    return {{(s_R * FAL - s_L * FAR + s_R * s_L * (A_R - A_L)) * inv,
             (s_R * FUL - s_L * FUR + s_R * s_L * (U_R - U_L)) * inv}};
  }

  // ============================================================================
  // HLL flux Jacobian (linearised w.r.t. trial perturbations dA_L,
  // dU_L, …)
  // ============================================================================
  template <int dim, int spacedim>
  std::array<double, 2>
  BloodFlowSystem<dim, spacedim>::hll_flux_jac(const double       bn_L,
                                               const double       bn_R,
                                               const double       A_L,
                                               const double       U_L,
                                               const double       A_R,
                                               const double       U_R,
                                               const double       dA_L,
                                               const double       dU_L,
                                               const double       dA_R,
                                               const double       dU_R,
                                               const unsigned int vid_L,
                                               const unsigned int vid_R,
                                               const double       ad_L,
                                               const double       ad_R) const
  {
    const double c_L   = compute_wave_speed(A_L, vid_L, ad_L);
    const double c_R   = compute_wave_speed(A_R, vid_R, ad_R);
    const double U_bar = 0.5 * (U_L + U_R);
    const double c_bar = 0.5 * (c_L + c_R);
    const double s_L   = U_bar - c_bar;
    const double s_R   = U_bar + c_bar;
    // const double s_L = std::min(U_L - c_L, U_R - c_R);

    // const double s_R = std::max(U_L + c_L, U_R + c_R);

    const double c2L_over_AL = c_L * c_L / A_L;
    const double c2R_over_AR = c_R * c_R / A_R;

    const double FAL_j = scalar_area_flux_jac(bn_L, A_L, U_L, dA_L, dU_L);
    const double FUL_j =
      scalar_momentum_flux_jac(bn_L, c2L_over_AL, U_L, dA_L, dU_L);
    const double FAR_j = scalar_area_flux_jac(bn_R, A_R, U_R, dA_R, dU_R);
    const double FUR_j =
      scalar_momentum_flux_jac(bn_R, c2R_over_AR, U_R, dA_R, dU_R);

    if (s_L >= 0.0)
      return {{FAL_j, FUL_j}};
    if (s_R <= 0.0)
      return {{FAR_j, FUR_j}};

    const double inv = 1.0 / (s_R - s_L);
    return {{(s_R * FAL_j - s_L * FAR_j + s_R * s_L * (dA_R - dA_L)) * inv,
             (s_R * FUL_j - s_L * FUR_j + s_R * s_L * (dU_R - dU_L)) * inv}};
  }

  // HLL-HDG flux based on Paper "Hybridisable discontinuous
  // Galerkin formulation of compressible flows "

  template <int dim, int spacedim>
  std::array<double, 2>
  BloodFlowSystem<dim, spacedim>::hll_hdg_flux(const double bn_L,
                                               const double /*bn_R*/,
                                               const double A_L,
                                               const double U_L, // interior U_e
                                               const double A_R,
                                               const double U_R, // trace U_b
                                               const unsigned int /*vid_L*/,
                                               const unsigned int vid_R,
                                               const double /*ad_L*/,
                                               const double ad_R) const
  {
    // Stabilization from TRACE state only (eq. 37 of Vila-Perez et
    // al.)
    const double c_b    = compute_wave_speed(A_R, vid_R, ad_R);
    const double s_plus = std::max(0.0, U_R * bn_L + c_b); // s⁺ at trace

    // Physical flux at TRACE (F(U_b)·n)
    const double FA_b = scalar_area_flux(bn_L, A_R, U_R);
    const double FU_b = scalar_momentum_flux(
      bn_L, U_R, compute_pressure_value(A_R, vid_R, ad_R), par["rho"]);

    // Stabilization term: s⁺(U_e - U_b)
    const double FA = FA_b + s_plus * (A_L - A_R);
    const double FU = FU_b + s_plus * (U_L - U_R);

    return {{FA, FU}};
  }

  template <int dim, int spacedim>
  std::array<double, 2>
  BloodFlowSystem<dim, spacedim>::hll_hdg_flux_jac(
    const double bn_L,
    const double /*bn_R*/,
    const double A_L,
    const double U_L,
    const double A_R,
    const double U_R,
    const double dA_L,
    const double dU_L, // perturbation direction
    const double dA_R,
    const double dU_R,
    const unsigned int /*vid_L*/,
    const unsigned int vid_R,
    const double /*ad_L*/,
    const double ad_R) const
  {
    const double c_b    = compute_wave_speed(A_R, vid_R, ad_R);
    const double dc_b   = compute_wave_speed_derivative(A_R, vid_R, ad_R);
    const double s_plus = std::max(0.0, U_R * bn_L + c_b);

    // dF(U_b)·n / dU_b  (trace perturbation dA_R, dU_R)
    const double c2_over_A_b = c_b * c_b / A_R;
    const double dFA_b       = scalar_area_flux_jac(bn_L, A_R, U_R, dA_R, dU_R);
    const double dFU_b =
      scalar_momentum_flux_jac(bn_L, c2_over_A_b, U_R, dA_R, dU_R);

    if (s_plus <= 0.0)
      {
        // Pure upwind from trace — only trace terms survive
        return {{dFA_b, dFU_b}};
      }

    // d(s⁺)/d(A_R) = dc_b/dA_R * dA_R (if U_R*bn + c_b > 0)
    const double ds_plus_dAR = dc_b * dA_R;
    const double ds_plus_dUR = bn_L * dU_R;
    const double ds_plus     = ds_plus_dAR + ds_plus_dUR;

    const double dFA = dFA_b                     // dF(U_b)/d(U_b) * dU_b
                       + ds_plus * (A_L - A_R)   // d(s⁺)/d(U_b) * (U_e - U_b)
                       + s_plus * (dA_L - dA_R); // s⁺ * d(U_e - U_b)
    const double dFU = dFU_b + ds_plus * (U_L - U_R) + s_plus * (dU_L - dU_R);

    return {{dFA, dFU}};
  }

  // ============================================================================
  // Lax–Friedrichs flux (residual)
  // ============================================================================
  template <int dim, int spacedim>
  std::array<double, 2>
  BloodFlowSystem<dim, spacedim>::lf_flux(const double       bn_L,
                                          const double       bn_R,
                                          const double       A_L,
                                          const double       U_L,
                                          const double       A_R,
                                          const double       U_R,
                                          const unsigned int vid_L,
                                          const unsigned int vid_R,
                                          const double       ad_L,
                                          const double       ad_R) const
  {
    const double FAL = scalar_area_flux(bn_L, A_L, U_L);
    const double FUL = scalar_momentum_flux(
      bn_L, U_L, compute_pressure_value(A_L, vid_L, ad_L), par["rho"]);
    const double FAR = scalar_area_flux(bn_R, A_R, U_R);
    const double FUR = scalar_momentum_flux(
      bn_R, U_R, compute_pressure_value(A_R, vid_R, ad_R), par["rho"]);
    const double alpha =
      theta * compute_LF_penalty(
                A_L, A_R, U_L, U_R, bn_L, bn_R, vid_L, vid_R, ad_L, ad_R);

    return {{0.5 * (FAL + FAR) - 0.5 * alpha * (A_R - A_L),
             0.5 * (FUL + FUR) - 0.5 * alpha * (U_R - U_L)}};
  }

  // ============================================================================
  // Lax–Friedrichs flux Jacobian
  // ============================================================================
  template <int dim, int spacedim>
  std::array<double, 2>
  BloodFlowSystem<dim, spacedim>::lf_flux_jac(const double       bn_L,
                                              const double       bn_R,
                                              const double       A_L,
                                              const double       U_L,
                                              const double       A_R,
                                              const double       U_R,
                                              const double       dA_L,
                                              const double       dU_L,
                                              const double       dA_R,
                                              const double       dU_R,
                                              const unsigned int vid_L,
                                              const unsigned int vid_R,
                                              const double       ad_L,
                                              const double       ad_R) const
  {
    const double c2L = compute_wave_speed(A_L, vid_L, ad_L);
    const double c2R = compute_wave_speed(A_R, vid_R, ad_R);

    const double FAL_j = scalar_area_flux_jac(bn_L, A_L, U_L, dA_L, dU_L);
    const double FUL_j =
      scalar_momentum_flux_jac(bn_L, c2L * c2L / A_L, U_L, dA_L, dU_L);
    const double FAR_j = scalar_area_flux_jac(bn_R, A_R, U_R, dA_R, dU_R);
    const double FUR_j =
      scalar_momentum_flux_jac(bn_R, c2R * c2R / A_R, U_R, dA_R, dU_R);
    const double alpha =
      theta * compute_LF_penalty(
                A_L, A_R, U_L, U_R, bn_L, bn_R, vid_L, vid_R, ad_L, ad_R);

    return {{0.5 * (FAL_j + FAR_j) - 0.5 * alpha * (dA_R - dA_L),
             0.5 * (FUL_j + FUR_j) - 0.5 * alpha * (dU_R - dU_L)}};
  }

  template <int dim, int spacedim>
  void
  BloodFlowSystem<dim, spacedim>::assemble_cell_residuals(
    const double t,
    const VectorType & /*y*/,
    VectorType &F)
  {
    TimerOutput::Scope timer(computing_timer, "assemble_cell_residuals");

    // FEValues and FEFaceValues read through the ghosted FE-range vector.  It
    // is sized dof_handler_.n_dofs(), which get_function_values() asserts, and
    // it holds every entry a locally owned cell can ask for, including the ones
    // on its ghost neighbours.
    const VectorType &y_cell = y_fe_relevant;

    const FEValuesExtractors::Scalar area_extractor(0);
    const FEValuesExtractors::Scalar velocity_extractor(1);
    const FEValuesExtractors::Scalar a_hat_extractor(2); // trace area
    const FEValuesExtractors::Scalar u_hat_extractor(3); // trace velocity

    const QGauss<dim>     quad_cell(fe_->tensor_degree() + 1);
    const QGauss<dim - 1> quad_face(fe_->tensor_degree() + 1);

    FEValues<dim, spacedim> fev(*fe_,
                                quad_cell,
                                update_values | update_gradients |
                                  update_quadrature_points | update_JxW_values);
    FEFaceValues<dim, spacedim> fef(*fe_,
                                    quad_face,
                                    update_values | update_JxW_values |
                                      update_normal_vectors);

    rhs_function.set_time(t);

    const double rho = par["rho"];
    const double eta = 2.0 * (par["xi"] + 2.0) * numbers::PI * par["mu"] / rho;

    for (const auto &cell : dof_handler_.active_cell_iterators())
      {
        if (!cell->is_locally_owned())
          continue;

        const unsigned int vid    = cell->material_id();
        const unsigned int n_dofs = fe_->n_dofs_per_cell();
        fev.reinit(cell);

        std::vector<types::global_dof_index> ldofs(n_dofs);
        cell->get_dof_indices(ldofs);

        const auto &JxW = fev.get_JxW_values();

        std::vector<double> A_h(fev.n_quadrature_points);
        std::vector<double> U_h(fev.n_quadrature_points);
        fev[area_extractor].get_function_values(y_cell, A_h);
        fev[velocity_extractor].get_function_values(y_cell, U_h);

        Vector<double> cell_rhs(n_dofs);

        // ---- Volume integral
        // --------------------------------------------------
        for (unsigned int q = 0; q < fev.n_quadrature_points; ++q)
          {
            const double A = std::max(A_h[q], 1e-10);
            const double U = U_h[q];
            const double P =
              compute_pressure_value(A, vid, compute_a_d_local(cell));
            // const double              P =
            // compute_pressure_value(A, vid);
            const Tensor<1, spacedim> b = compute_directional_vector(cell);

            const double rhs_A =
              rhs_function.value(fev.get_quadrature_points()[q], 0);
            const double rhs_U =
              rhs_function.value(fev.get_quadrature_points()[q], 1);

            for (unsigned int i = 0; i < n_dofs; ++i)
              {
                const unsigned int comp =
                  fe_->system_to_component_index(i).first;

                if (comp == 0)
                  {
                    cell_rhs(i) +=
                      (rhs_A * fev[area_extractor].value(i, q) +
                       A * U * (b * fev[area_extractor].gradient(i, q))) *
                      JxW[q];
                  }
                else
                  {
                    const double phi_u = fev[velocity_extractor].value(i, q);
                    cell_rhs(i) +=
                      (rhs_U * phi_u +
                       (0.5 * U * U + P / rho) *
                         (b * fev[velocity_extractor].gradient(i, q)) -
                       eta * U / A * phi_u) *
                      JxW[q];
                  }
              }
          }

        const double ad_local = compute_a_d_local(cell);
        // ---- Face flux: interior cell value vs. global face trace

        for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
          {
            fef.reinit(cell, f);
            const double ad_face = compute_a_d_at_face(cell, f);
            const auto  &normals = fef.get_normal_vectors();
            const auto  &JxW     = fef.get_JxW_values();

            // Interior cell values at the face quadrature point
            std::vector<double> Ah_q(fef.n_quadrature_points);
            std::vector<double> Uh_q(fef.n_quadrature_points);
            fef[area_extractor].get_function_values(y_cell, Ah_q);
            fef[velocity_extractor].get_function_values(y_cell, Uh_q);

            // Face-trace values: this cell's OWN (A_hat, U_hat) on
            // face f, read with the same extractor idiom as the
            // cell values.  For boundary faces and K>=3 junction
            // half-faces this DOF *is* the unique/per-vessel trace;
            // on an ordinary interior (or 2-way) face it is tied to
            // the canonical side by the continuity rows, so the
            // converged value is identical to the canonical one.
            std::vector<double> Ahat_q(fef.n_quadrature_points);
            std::vector<double> Uhat_q(fef.n_quadrature_points);
            fef[a_hat_extractor].get_function_values(y_cell, Ahat_q);
            fef[u_hat_extractor].get_function_values(y_cell, Uhat_q);

            for (unsigned int q = 0; q < fef.n_quadrature_points; ++q)
              {
                const double A_in  = std::max(Ah_q[q], 1e-10);
                const double U_in  = Uh_q[q];
                const double A_hat = std::max(Ahat_q[q], 1e-10);
                const double U_hat = Uhat_q[q];
                const double bn =
                  compute_tangent_normal_product(cell, normals[q]);

                // numerical_flux designed for left/right states.
                // In current implementation there is no physical
                // neighbour cell on the other side so we are ;

                const auto [FA, FU] = numerical_flux(bn,
                                                     bn,
                                                     A_in,
                                                     U_in,
                                                     A_hat,
                                                     U_hat,
                                                     vid,
                                                     vid,
                                                     ad_local,
                                                     ad_face);

                for (unsigned int i = 0; i < n_dofs; ++i)
                  {
                    const unsigned int comp =
                      fe_->system_to_component_index(i).first;
                    if (comp >= 2)
                      continue; // trace components carry no cell
                                // flux
                    cell_rhs(i) -=
                      (comp == 0 ? FA : FU) * fef.shape_value(i, q) * JxW[q];
                  }
              }
          }

        // ---- Scatter into global F
        // -------------------------------------------
        // Every DoF of this FESystem is discontinuous, so all of `ldofs`
        // belongs to this cell alone and is owned by this rank: no other rank
        // contributes to these rows.
        for (unsigned int i = 0; i < n_dofs; ++i)
          F(ldofs[i]) += cell_rhs(i);
      }
  }

  template <int dim, int spacedim>
  void
  BloodFlowSystem<dim, spacedim>::assemble_trace_interior_equations(
    const VectorType &y,
    VectorType       &F)
  {
    TimerOutput::Scope timer(computing_timer,
                             "assemble_trace_interior_equations");

    // FEValues and FEFaceValues read through the ghosted FE-range vector.  It
    // is sized dof_handler_.n_dofs(), which get_function_values() asserts, and
    // it holds every entry a locally owned cell can ask for, including the ones
    // on its ghost neighbours.
    const VectorType &y_cell = y_fe_relevant;

    const FEValuesExtractors::Scalar area_extractor(0);
    const FEValuesExtractors::Scalar velocity_extractor(1);
    // const FEValuesExtractors::Scalar a_hat_extractor(2); // trace area
    // const FEValuesExtractors::Scalar u_hat_extractor(3); // trace velocity

    const QGauss<dim - 1> quad_face(1); // each 1-D face is a single 0-D point
    FEFaceValues<dim, spacedim> fef(*fe_,
                                    quad_face,
                                    update_values | update_normal_vectors);

    // Avoid double assembly of the same face from the left and the right
    // cell.  The set is per rank, which is enough: the ownership guard below
    // lets exactly one rank write a given face's rows.
    std::set<std::pair<CellId, unsigned int>> processed;

    for (const auto &cell : dof_handler_.active_cell_iterators())
      {
        if (!cell->is_locally_owned())
          continue;

        for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
          {
            if (cell->face(f)->at_boundary())
              continue;
            if (is_junction_face(cell->id(), f))
              continue;
            const auto key = canonical_face_key(cell, f);
            if (processed.count(key))
              continue;
            processed.insert(key);

            // A face's rows live on its canonical side.  If that side sits on
            // another rank, that rank writes them and this one skips the face.
            const auto trace_it = face_dof_map.find(key);
            Assert(trace_it != face_dof_map.end(), ExcInternalError());
            if (!locally_owned_dofs_.is_element(trace_it->second.a_hat_dof))
              continue;

            const auto        &nb   = cell->neighbor(f);
            const unsigned int nb_f = cell->neighbor_of_neighbor(f);

            const unsigned int vid_L   = cell->material_id();
            const unsigned int vid_R   = nb->material_id();
            const double       ad_L    = compute_a_d_local(cell);
            const double       ad_R    = compute_a_d_local(nb);
            const double       ad_face = compute_a_d_at_face(cell, f);
            // Interior cell values at face — use y_cell, not y
            fef.reinit(cell, f);
            const double bn_L =
              compute_tangent_normal_product(cell, fef.get_normal_vectors()[0]);

            std::vector<double> A_L_v(1), U_L_v(1);
            fef[area_extractor].get_function_values(y_cell, A_L_v);
            fef[velocity_extractor].get_function_values(y_cell, U_L_v);

            fef.reinit(nb, nb_f);
            const double bn_R =
              compute_tangent_normal_product(nb, fef.get_normal_vectors()[0]);

            std::vector<double> A_R_v(1), U_R_v(1);
            fef[area_extractor].get_function_values(y_cell, A_R_v);
            fef[velocity_extractor].get_function_values(y_cell, U_R_v);

            const double A_L = std::max(A_L_v[0], 1e-10);
            const double U_L = U_L_v[0];
            const double A_R = std::max(A_R_v[0], 1e-10);
            const double U_R = U_R_v[0];

            // Trace state
            double A_hat = 0.0, U_hat = 0.0;
            get_face_trace(y, cell, f, A_hat, U_hat);
            A_hat = std::max(A_hat, 1e-10);

            // Left HDG type flux residuals (face integrals with
            // interior state from left cell) numerical_flux is
            // designed for left/right states. The trace acts as the
            // exterior state, therefore use opposite orientation on
            // the trace side.
            const auto [FA_L, FU_L] = numerical_flux(
              bn_L, bn_L, A_L, U_L, A_hat, U_hat, vid_L, vid_R, ad_L, ad_face);

            // Right HDG type flux residuals (face integrals with
            // interior state from right cell)
            const auto [FA_R, FU_R] = numerical_flux(
              bn_R, bn_R, A_R, U_R, A_hat, U_hat, vid_R, vid_L, ad_R, ad_face);

            // Trace dofs
            const FaceTraceDof &td = trace_it->second;

            // HDG type transmission equations
            // sum of left and right fluxes must be zero
            F(td.a_hat_dof) += FA_L + FA_R;
            F(td.u_hat_dof) += FU_L + FU_R;
          }
      }
  }

  template <int dim, int spacedim>
  void
  BloodFlowSystem<dim, spacedim>::assemble_trace_boundary_equations(
    const double      t,
    const VectorType &y,
    VectorType       &F)
  {
    TimerOutput::Scope timer(computing_timer,
                             "assemble_trace_boundary_equations");

    // FEValues and FEFaceValues read through the ghosted FE-range vector.  It
    // is sized dof_handler_.n_dofs(), which get_function_values() asserts, and
    // it holds every entry a locally owned cell can ask for, including the ones
    // on its ghost neighbours.
    const VectorType &y_cell = y_fe_relevant;

    const FEValuesExtractors::Scalar area_extractor(0);
    const FEValuesExtractors::Scalar velocity_extractor(1);
    const FEValuesExtractors::Scalar a_hat_extractor(2); // trace area
    const FEValuesExtractors::Scalar u_hat_extractor(3); // trace velocity

    const QGauss<dim - 1>       quad_face(1);
    FEFaceValues<dim, spacedim> fef(*fe_, quad_face, update_values);

    inflow_function.set_time(t);

    for (const auto &cell : dof_handler_.active_cell_iterators())
      {
        if (!cell->is_locally_owned())
          continue;

        for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
          {
            if (!cell->face(f)->at_boundary())
              continue;
            if (is_junction_face(cell->id(), f))
              continue;

            const types::boundary_id bid = cell->face(f)->boundary_id();
            const unsigned int       vid = cell->material_id();


            // Interior cell value at face — use y_cell, not y
            fef.reinit(cell, f);
            std::vector<double> A_int_v(1), U_int_v(1);
            fef[area_extractor].get_function_values(y_cell, A_int_v);
            fef[velocity_extractor].get_function_values(y_cell, U_int_v);

            const double A_int     = std::max(A_int_v[0], 1e-10);
            const double U_int     = U_int_v[0];
            const double a_d_local = compute_a_d_at_face(cell, f);
            const double c0    = compute_wave_speed(a_d_local, vid, a_d_local);
            const double c_int = compute_wave_speed(A_int, vid, a_d_local);

            // Current face trace (from trace block of y)
            double A_hat_cur = 0.0, U_hat_cur = 0.0;
            get_face_trace(y, cell, f, A_hat_cur, U_hat_cur);
            const double A_hat = std::max(A_hat_cur, 1e-10);
            const double c_hat = compute_wave_speed(A_hat, vid, a_d_local);

            double res_A = 0.0, res_U = 0.0;

            if (bid == 0) // inflow
              {
                inflow_function.set_time(t);
                const double Q_in   = inflow_function.value(Point<1>(0.0));
                const double W2_int = U_int - 4.0 * (c_int - c0);

                res_A = A_hat_cur * U_hat_cur - Q_in;
                res_U = (U_hat_cur - 4.0 * (c_hat - c0)) - W2_int;
              }
            else if (outlet_type == "RCR" && rcr_map.count(bid) &&
                     (rcr_map.at(bid).R1 > 0.0 || rcr_map.at(bid).R2 > 0.0))

              {
                const auto  &rcr = rcr_map.at(bid);
                const double Q   = A_hat_cur * U_hat_cur;
                if (rcr.C > 0.0)
                  {
                    // const double Pc =
                    // terminal_Pc_storage.at(bid); res_A =
                    //   compute_pressure_value(A_hat, vid,
                    //   a_d_local) - (rcr.R1 * Q + Pc);
                    const double Pc = y(rcr_pc_dof.at(bid));
                    res_A = compute_pressure_value(A_hat, vid, a_d_local) -
                            (rcr.R1 * Q + Pc);
                  }
                else
                  { // single R: P = R2*Q + P_out
                    res_A = compute_pressure_value(A_hat, vid, a_d_local) -
                            (rcr.R2 * Q + rcr.P_out);
                  }
                const double W1_int = U_int + 4.0 * (c_int - c0);
                res_U               = (U_hat_cur + 4.0 * (c_hat - c0)) - W1_int;
              }
            else // Reflection outlet
              {
                const double Rt     = par["Rt"];
                const double W1_int = U_int + 4.0 * (c_int - c0);
                const double W2_tgt = -Rt * W1_int;

                res_A = (U_hat_cur + 4.0 * (c_hat - c0)) - W1_int;
                res_U = (U_hat_cur - 4.0 * (c_hat - c0)) - W2_tgt;
              }

            // A boundary face has exactly one incident cell, so its rows are
            // owned wherever that cell is owned.  Writing with `+=` into a
            // zeroed vector gives the same value as `=` and keeps the whole
            // residual on add semantics for a single compress().
            const FaceTraceDof &td =
              face_dof_map.at(canonical_face_key(cell, f));
            Assert(locally_owned_dofs_.is_element(td.a_hat_dof),
                   ExcInternalError());
            F(td.a_hat_dof) += res_A;
            F(td.u_hat_dof) += res_U;
          }
      }
  }

  template <int dim, int spacedim>
  void
  BloodFlowSystem<dim, spacedim>::assemble_rcr_capacitor_equations(
    const VectorType &y,
    const VectorType &ydot,
    VectorType       &F)
  {
    if (rcr_pc_dof.empty())
      return;

    // ydot is IDA's raw vector: unlike y (ghosted into y_relevant by the
    // caller), it only holds entries this rank owns.  Pc row ownership is now
    // assigned to a single designated rank for PETSc's sake (see
    // build_rcr_dof_map), which in general differs from the rank that owns
    // the terminal's cell -- the rank that needs Pc_dot here to write the
    // residual is often not the one that owns it.  n_rcr_dofs is tiny, so a
    // manual all-reduce of that short array is far cheaper than standing up a
    // full ghosted copy of ydot just for this.
    const types::global_dof_index first_pc = n_total_dofs - n_rcr_dofs;
    std::vector<double>           pc_dot_local(n_rcr_dofs, 0.0);
    for (const auto &[bid, pc_dof] : rcr_pc_dof)
      if (locally_owned_dofs_.is_element(pc_dof))
        pc_dot_local[pc_dof - first_pc] = ydot(pc_dof);

    // Utilities::MPI::sum's std::vector<double> overload isn't explicitly
    // instantiated in every deal.II build (it's only compiled for a fixed set
    // of types), so call MPI_Allreduce directly instead -- it's always
    // available once <deal.II/base/mpi.h> is included.
    std::vector<double> pc_dot_global(n_rcr_dofs, 0.0);
    if (n_rcr_dofs > 0)
      MPI_Allreduce(pc_dot_local.data(),
                    pc_dot_global.data(),
                    static_cast<int>(n_rcr_dofs),
                    MPI_DOUBLE,
                    MPI_SUM,
                    mpi_communicator_);

    for (const auto &cell : dof_handler_.active_cell_iterators())
      {
        if (!cell->is_locally_owned())
          continue;

        for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
          {
            if (!cell->face(f)->at_boundary())
              continue;
            if (is_junction_face(cell->id(), f))
              continue;
            const auto pit = rcr_pc_dof.find(cell->face(f)->boundary_id());
            if (pit == rcr_pc_dof.end())
              continue;

            const types::global_dof_index pc_dof = pit->second;
            const auto &rcr = rcr_map.at(cell->face(f)->boundary_id());

            double A_hat = 0.0, U_hat = 0.0;
            get_face_trace(y, cell, f, A_hat, U_hat);
            const double Q      = A_hat * U_hat;
            const double Pc     = y(pc_dof);
            const double Pc_dot = pc_dot_global[pc_dof - first_pc];

            // C*Ṗc - Q + (Pc - P_out)/R2 = 0
            // Written from whichever rank owns the terminal's cell; the row
            // itself may be owned by a *different* rank for PETSc's
            // partitioning, and the off-owner add is routed there by the
            // caller's compress(VectorOperation::add).
            F(pc_dof) += rcr.C * Pc_dot - Q + (Pc - rcr.P_out) / rcr.R2;
          }
      }
  }

  template <int dim, int spacedim>
  void
  BloodFlowSystem<dim, spacedim>::assemble_trace_junction_equations(
    const VectorType &y,
    VectorType       &F)
  {
    TimerOutput::Scope timer(computing_timer,
                             "assemble_trace_junction_equations");

    if (junctions.empty())
      return;

    // FEValues and FEFaceValues read through the ghosted FE-range vector.  It
    // is sized dof_handler_.n_dofs(), which get_function_values() asserts, and
    // it holds every entry a locally owned cell can ask for, including the ones
    // on its ghost neighbours.
    const VectorType &y_cell = y_fe_relevant;

    const FEValuesExtractors::Scalar area_extractor(0);
    const FEValuesExtractors::Scalar velocity_extractor(1);
    const FEValuesExtractors::Scalar a_hat_extractor(2); // trace area
    const FEValuesExtractors::Scalar u_hat_extractor(3); // trace velocity

    const QGauss<dim - 1>       quad_face(1);
    FEFaceValues<dim, spacedim> fef(*fe_, quad_face, update_values);

    const double rho   = par["rho"];
    const double A_min = 1e-10;

    for (const auto &J : junctions)
      {
        const unsigned int K = J.n_vessels();

        // Per-vessel quantities
        std::vector<double>                  A_int(K), U_int(K), c0(K), W(K);
        std::vector<double>                  A_hat(K), U_hat(K), c_hat(K);
        std::vector<int>                     s(K);
        std::vector<types::global_dof_index> a_idx(K), u_idx(K);

        for (unsigned int i = 0; i < K; ++i)
          {
            const auto        &hf  = J.half_faces[i];
            const unsigned int vid = hf.cell->material_id();

            // Interior cell value at the junction face — use y_cell
            fef.reinit(hf.cell, hf.face_no);
            std::vector<double> Av(1), Uv(1);
            fef[area_extractor].get_function_values(y_cell, Av);
            fef[velocity_extractor].get_function_values(y_cell, Uv);

            A_int[i] = std::max(Av[0], A_min);
            U_int[i] = Uv[0];
            s[i]     = hf.orientation; // +1 if junction is at face 1
                                       // (right end)
            const double a_d_face = compute_a_d_at_face(hf.cell, hf.face_no);
            c0[i]                 = compute_wave_speed(a_d_face, vid, a_d_face);

            // Outgoing Riemann invariant from cell i toward the
            // junction
            const double c_i = compute_wave_speed(A_int[i], vid, a_d_face);
            W[i] = U_int[i] + static_cast<double>(s[i]) * 4.0 * (c_i - c0[i]);

            // Trace DOF indices and current trace values (from
            // trace block of y)
            // const FaceTraceDof &td =
            //   face_dof_map.at(canonical_face_key(hf.cell, hf.face_no));
            // std::cout << "Rank " << this_mpi_process
            //           << " owned=" << hf.cell->is_locally_owned()
            //           << " ghost=" << hf.cell->is_ghost()
            //           << " artificial=" << hf.cell->is_artificial() <<
            //           std::endl;
            // const auto key = canonical_face_key(hf.cell, hf.face_no);

            // auto it = face_dof_map.find(key);

            // if (it == face_dof_map.end())
            //   {
            //     std::cout << "Rank " << this_mpi_process << " missing face
            //     key
            //     "
            //               << key.first << " " << key.second << std::endl;

            //     std::abort();
            //   }

            // const FaceTraceDof &td = it->second;

            // a_idx[i] = td.a_hat_dof;
            // u_idx[i] = td.u_hat_dof;
            std::vector<types::global_dof_index> ldofs(fe_->n_dofs_per_cell());
            hf.cell->get_dof_indices(ldofs);

            const auto [a_hat_dof, u_hat_dof] =
              face_trace_dofs(ldofs, hf.face_no);

            a_idx[i] = a_hat_dof;
            u_idx[i] = u_hat_dof;

            A_hat[i] = std::max(y(a_idx[i]), A_min);
            U_hat[i] = y(u_idx[i]);
            c_hat[i] = compute_wave_speed(A_hat[i], vid, a_d_face);
          }

        // (a) Mass conservation -> row a_idx[0].
        //
        // The 2K rows of a junction are spread over its K half-faces, so a
        // junction cut by a subdomain boundary has its rows split between the
        // two ranks.  Each rank writes only the rows it owns and reads the
        // other vessels' states from its ghost layer, which is complete here
        // because the layer is vertex-adjacent.  No reduction closes these
        // equations.
        if (locally_owned_dofs_.is_element(a_idx[0]))
          {
            double mass_res = 0.0;
            for (unsigned int i = 0; i < K; ++i)
              mass_res += static_cast<double>(s[i]) * A_hat[i] * U_hat[i];
            F(a_idx[0]) += mass_res;
          }

        // (b) Total-head continuity: H_0 − H_i = 0 -> rows
        // u_idx[0..K−2]
        {
          const double a_d0 =
            compute_a_d_at_face(J.half_faces[0].cell, J.half_faces[0].face_no);
          const double H0 =
            0.5 * gamma * U_hat[0] * U_hat[0] +
            compute_pressure_value(A_hat[0],
                                   J.half_faces[0].cell->material_id(),
                                   a_d0) /
              rho;

          for (unsigned int i = 1; i < K; ++i)
            {
              const double a_di = compute_a_d_at_face(J.half_faces[i].cell,
                                                      J.half_faces[i].face_no);
              const double Hi =
                0.5 * gamma * U_hat[i] * U_hat[i] +
                compute_pressure_value(A_hat[i],
                                       J.half_faces[i].cell->material_id(),
                                       a_di) /
                  rho;

              if (locally_owned_dofs_.is_element(u_idx[i - 1]))
                F(u_idx[i - 1]) += H0 - Hi;
            }
        }

        // (c) Riemann compatibility: U_hat_i + s_i*4(c_hat_i−c0_i)
        // − W_i = 0
        //     vessel 0 -> row u_idx[K−1]
        //     vessel i>=1 -> row a_idx[i]
        for (unsigned int i = 0; i < K; ++i)
          {
            const double compatibility_res =
              U_hat[i] + static_cast<double>(s[i]) * 4.0 * (c_hat[i] - c0[i]) -
              W[i];

            const types::global_dof_index row =
              (i == 0) ? u_idx[K - 1] : a_idx[i];

            if (locally_owned_dofs_.is_element(row))
              F(row) += compatibility_res;
          }
      }
  }

  template <int dim, int spacedim>
  void
  BloodFlowSystem<dim, spacedim>::assemble_trace_continuity_equations(
    const VectorType &y,
    VectorType       &F)
  {
    TimerOutput::Scope timer(computing_timer,
                             "assemble_trace_continuity_equations");
    for (const auto &p : trace_continuity_pairs)
      {
        // The duplicate side is always a locally owned cell's own pair, so
        // these rows are owned by construction; the canonical values are read
        // from the ghosted vector and may live on another rank.
        Assert(locally_owned_dofs_.is_element(p.a_dup), ExcInternalError());
        F(p.a_dup) += y(p.a_dup) - y(p.a_canon);
        F(p.u_dup) += y(p.u_dup) - y(p.u_canon);
      }
  }

  template <int dim, int spacedim>
  void
  BloodFlowSystem<dim, spacedim>::assemble_jacobian_trace_continuity_block()
  {
    // Owned by construction: the duplicate side is a locally owned cell's own
    // pair.  The canonical column may belong to another rank, which is fine --
    // only rows are partitioned.
    for (const auto &p : trace_continuity_pairs)
      {
        jacobian_matrix.add(p.a_dup, p.a_dup, 1.0);
        jacobian_matrix.add(p.a_dup, p.a_canon, -1.0);
        jacobian_matrix.add(p.u_dup, p.u_dup, 1.0);
        jacobian_matrix.add(p.u_dup, p.u_canon, -1.0);
      }
  }

  // ============================================================================
  // assemble_residual
  //
  // IDA's F(t, y, ydot) = 0.  Reads go through the ghosted vectors filled by
  // update_ghosted_vectors(); writes go to the locally owned `residual` and are
  // closed by a single compress(add).  Every routine called here uses add
  // semantics, so the two are never mixed in one compress.
  // ============================================================================
  template <int dim, int spacedim>
  void
  BloodFlowSystem<dim, spacedim>::assemble_residual(const double      t,
                                                    const VectorType &y,
                                                    const VectorType &ydot,
                                                    VectorType       &residual)
  {
    TimerOutput::Scope timer(computing_timer, "assemble_residual");
    if (verbosity > 1)
      {
        deallog.push("assemble_residual");
        deallog << "t=" << t << std::endl;
      }

    update_ghosted_vectors(y);

    // ---- raw residuals R(y) -------------------------------------------------
    residual_F = 0.0;
    assemble_cell_residuals(t, y_relevant, residual_F);
    assemble_trace_interior_equations(y_relevant, residual_F);
    assemble_trace_boundary_equations(t, y_relevant, residual_F);
    assemble_trace_junction_equations(y_relevant, residual_F);
    assemble_trace_continuity_equations(y_relevant, residual_F);
    residual_F.compress(VectorOperation::add);

    // ---- residual = M_K * ydot - R, uniformly over the FE range -------------
    // per_cell_mass has an exactly zero block on the trace components (2,3), so
    // this single formula produces the right thing for both kinds of row:
    //   cell rows  (differential):  M ydot - R_cell
    //   trace rows (algebraic)   :  0      - R_trace  =  -R_trace
    // Every FE DoF, trace ones included, belongs to exactly one cell, and that
    // cell is locally owned, so each row is written by exactly one rank once.
    residual = 0.0;

    const unsigned int n_dofs = fe_->n_dofs_per_cell();
    Vector<double>     local_F(n_dofs), local_Mydot(n_dofs), local_res(n_dofs);
    std::vector<types::global_dof_index> ldofs(n_dofs);

    for (const auto &cell : dof_handler_.active_cell_iterators())
      {
        if (!cell->is_locally_owned())
          continue;

        cell->get_dof_indices(ldofs);

        for (unsigned int i = 0; i < n_dofs; ++i)
          {
            local_F(i)     = residual_F(ldofs[i]);
            local_Mydot(i) = ydot(ldofs[i]);
          }

        per_cell_mass[cell->active_cell_index()].vmult(local_res, local_Mydot);

        for (unsigned int i = 0; i < n_dofs; ++i)
          residual(ldofs[i]) += local_res(i) - local_F(i);
      }

    // RCR capacitor rows: differential, written directly in M*ydot - R form.
    assemble_rcr_capacitor_equations(y_relevant, ydot, residual);

    residual.compress(VectorOperation::add);
    if (verbosity > 1)
      deallog.pop();
  }

  // ============================================================================
  // assemble_jacobian
  //
  // J_IDA = dF/dy + alpha * dF/dydot = -dR/dy + alpha * M.
  //
  // The blocks are added first and compressed once; only then is the matrix
  // scaled and the mass term added, because *= and the ghost exchange of add()
  // both need the matrix in a compressed state.
  // ============================================================================
  template <int dim, int spacedim>
  void
  BloodFlowSystem<dim, spacedim>::assemble_jacobian(const double      t,
                                                    const VectorType &y,
                                                    const VectorType &ydot,
                                                    const double      alpha)
  {
    TimerOutput::Scope timer(computing_timer, "assemble_jacobian");
    if (verbosity > 1)
      deallog.push("assemble_jacobian");
    deallog << "t=" << t << std::endl;

    assemble_state_jacobian(t, y, ydot);
    assemble_derivative_jacobian(t, y, ydot);

    jacobian_matrix.copy_from(state_jacobian_matrix_);
    jacobian_matrix.add(alpha, derivative_jacobian_matrix_);

    if (verbosity > 1)
      deallog.pop();
  }

  template <int dim, int spacedim>
  void
  BloodFlowSystem<dim, spacedim>::assemble_state_jacobian(
    const double      t,
    const VectorType &y,
    const VectorType &ydot)
  {
    (void)ydot;
    update_ghosted_vectors(y);
    jacobian_matrix = 0.0;
    assemble_jacobian_cell_block(t, y_relevant);
    assemble_jacobian_trace_interior_block(y_relevant);
    assemble_jacobian_trace_boundary_block(t, y_relevant);
    assemble_jacobian_trace_junction_block(y_relevant);
    assemble_jacobian_trace_continuity_block();
    assemble_jacobian_rcr_capacitor_block(y_relevant);
    jacobian_matrix.compress(VectorOperation::add);
    jacobian_matrix *= -1.0;
    jacobian_matrix.compress(VectorOperation::insert);
    state_jacobian_matrix_.copy_from(jacobian_matrix);
  }

  template <int dim, int spacedim>
  void
  BloodFlowSystem<dim, spacedim>::assemble_derivative_jacobian(
    const double      t,
    const VectorType &y,
    const VectorType &ydot)
  {
    (void)t;
    (void)y;
    (void)ydot;
    derivative_jacobian_matrix_ = 0.0;

    const unsigned int                   n_dofs = fe_->n_dofs_per_cell();
    std::vector<types::global_dof_index> ldofs(n_dofs);
    for (const auto &cell : dof_handler_.active_cell_iterators())
      {
        if (!cell->is_locally_owned())
          continue;
        cell->get_dof_indices(ldofs);
        const FullMatrix<double> &M_K =
          per_cell_mass[cell->active_cell_index()];
        for (unsigned int i = 0; i < n_dofs; ++i)
          for (unsigned int j = 0; j < n_dofs; ++j)
            derivative_jacobian_matrix_.add(ldofs[i], ldofs[j], M_K(i, j));
      }

    for (const auto &[bid, pc_dof] : rcr_pc_dof)
      if (locally_owned_dofs_.is_element(pc_dof))
        derivative_jacobian_matrix_.add(pc_dof, pc_dof, rcr_map.at(bid).C);

    derivative_jacobian_matrix_.compress(VectorOperation::add);
  }

  template <int dim, int spacedim>
  const MatrixType &
  BloodFlowSystem<dim, spacedim>::state_jacobian_matrix() const
  {
    return state_jacobian_matrix_;
  }

  template <int dim, int spacedim>
  const MatrixType &
  BloodFlowSystem<dim, spacedim>::derivative_jacobian_matrix() const
  {
    return derivative_jacobian_matrix_;
  }

  template <int dim, int spacedim>
  void
  BloodFlowSystem<dim, spacedim>::assemble_jacobian_cell_block(
    const double t,
    const VectorType & /*y*/)
  {
    TimerOutput::Scope timer(computing_timer, "assemble_jacobian_cell_block");

    // FEValues and FEFaceValues read through the ghosted FE-range vector.  It
    // is sized dof_handler_.n_dofs(), which get_function_values() asserts, and
    // it holds every entry a locally owned cell can ask for, including the ones
    // on its ghost neighbours.
    const VectorType &y_cell = y_fe_relevant;

    const FEValuesExtractors::Scalar area_extractor(0);
    const FEValuesExtractors::Scalar velocity_extractor(1);
    const FEValuesExtractors::Scalar a_hat_extractor(2); // trace area
    const FEValuesExtractors::Scalar u_hat_extractor(3); // trace velocity

    const QGauss<dim>     quad_cell(fe_->tensor_degree() + 1);
    const QGauss<dim - 1> quad_face(fe_->tensor_degree() + 1);

    FEValues<dim, spacedim> fev(
      *fe_, quad_cell, update_values | update_gradients | update_JxW_values);
    FEFaceValues<dim, spacedim> fef(*fe_,
                                    quad_face,
                                    update_values | update_JxW_values |
                                      update_normal_vectors);

    const double rho = par["rho"];
    const double eta = 2.0 * (par["xi"] + 2.0) * numbers::PI * par["mu"] / rho;
    (void)t;

    for (const auto &cell : dof_handler_.active_cell_iterators())
      {
        if (!cell->is_locally_owned())
          continue;

        const unsigned int vid    = cell->material_id();
        const unsigned int n_dofs = fe_->n_dofs_per_cell();

        fev.reinit(cell);

        const auto                          &JxW      = fev.get_JxW_values();
        const double                         ad_local = compute_a_d_local(cell);
        std::vector<types::global_dof_index> ldofs(n_dofs);
        cell->get_dof_indices(ldofs);

        std::vector<double> A_h(fev.n_quadrature_points);
        std::vector<double> U_h(fev.n_quadrature_points);
        fev[area_extractor].get_function_values(y_cell, A_h);
        fev[velocity_extractor].get_function_values(y_cell, U_h);

        FullMatrix<double> cell_matrix(n_dofs, n_dofs);

        // ---- Block (1,1) volume - cell_matrix
        // -------------------------------
        for (unsigned int q = 0; q < fev.n_quadrature_points; ++q)
          {
            const double A    = std::max(A_h[q], 1e-10);
            const double U    = U_h[q];
            const double dpdA = compute_pressure_derivative(A, vid, ad_local);
            const double c2_A = A / rho * dpdA;

            for (unsigned int i = 0; i < n_dofs; ++i)
              {
                const unsigned int ci = fe_->system_to_component_index(i).first;
                const Tensor<1, spacedim> grad_phiA =
                  fev[area_extractor].gradient(i, q);
                const Tensor<1, spacedim> grad_phiU =
                  fev[velocity_extractor].gradient(i, q);
                const double phi_U = fev[velocity_extractor].value(i, q);
                const Tensor<1, spacedim> b = compute_directional_vector(cell);

                for (unsigned int j = 0; j < n_dofs; ++j)
                  {
                    const double trial_A = fev[area_extractor].value(j, q);
                    const double trial_U = fev[velocity_extractor].value(j, q);

                    double contrib = 0.0;
                    if (ci == 0)
                      contrib = (U * trial_A + A * trial_U) * (b * grad_phiA);
                    else
                      contrib =
                        (c2_A / A * trial_A + U * trial_U) * (b * grad_phiU) -
                        eta * (trial_U / A - U * trial_A / (A * A)) * phi_U;

                    cell_matrix(i, j) += contrib * JxW[q];
                  }
              }
          }

        // ---- Block (1,1) face + Block (1,2)
        // --------------------------------- Face cell-cell ->
        // cell_matrix

        for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
          {
            fef.reinit(cell, f);
            const auto         &normals = fef.get_normal_vectors();
            const auto         &JxW_f   = fef.get_JxW_values();
            const double        ad_face = compute_a_d_at_face(cell, f);
            std::vector<double> Ah_q(fef.n_quadrature_points);
            std::vector<double> Uh_q(fef.n_quadrature_points);
            fef[area_extractor].get_function_values(y_cell, Ah_q);
            fef[velocity_extractor].get_function_values(y_cell, Uh_q);

            // This cell's OWN trace values / DOFs on face f — must
            // match the extractor read in assemble_cell_residuals,
            // otherwise the analytic Jacobian differentiates w.r.t.
            // the wrong column.
            std::vector<double> Ahat_q(fef.n_quadrature_points);
            std::vector<double> Uhat_q(fef.n_quadrature_points);
            fef[a_hat_extractor].get_function_values(y_cell, Ahat_q);
            fef[u_hat_extractor].get_function_values(y_cell, Uhat_q);
            const double A_hat = std::max(Ahat_q[0], 1e-10);
            const double U_hat = Uhat_q[0];

            const auto [loc_a_hat_dof, loc_u_hat_dof] =
              face_trace_dofs(ldofs, f);
            const FaceTraceDof td{loc_a_hat_dof, loc_u_hat_dof};

            for (unsigned int q = 0; q < fef.n_quadrature_points; ++q)
              {
                const double A_in = std::max(Ah_q[q], 1e-10);
                const double U_in = Uh_q[q];
                const double bn =
                  compute_tangent_normal_product(cell, normals[q]);

                // (1,1) face -> cell_matrix
                for (unsigned int j = 0; j < n_dofs; ++j)
                  {
                    const double dA = fef[area_extractor].value(j, q);
                    const double dU = fef[velocity_extractor].value(j, q);

                    const auto [dFA, dFU] = numerical_flux_jac(bn,
                                                               bn,
                                                               A_in,
                                                               U_in,
                                                               A_hat,
                                                               U_hat,
                                                               dA,
                                                               dU,
                                                               0.0,
                                                               0.0,
                                                               vid,
                                                               vid,
                                                               ad_local,
                                                               ad_face);

                    for (unsigned int i = 0; i < n_dofs; ++i)
                      {
                        const unsigned int comp =
                          fe_->system_to_component_index(i).first;
                        if (comp >= 2)
                          continue; // trace rows carry no cell flux
                        const double phi = fef.shape_value(i, q);
                        cell_matrix(i, j) -=
                          (comp == 0 ? dFA : dFU) * phi * JxW_f[q];
                      }
                  }

                // (1,2) -> jacobian_matrix (trace cols)
                for (const auto trace_col : {td.a_hat_dof, td.u_hat_dof})
                  {
                    const bool   is_a   = (trace_col == td.a_hat_dof);
                    const double dA_hat = is_a ? 1.0 : 0.0;
                    const double dU_hat = is_a ? 0.0 : 1.0;

                    const auto [dFA, dFU] = numerical_flux_jac(bn,
                                                               bn,
                                                               A_in,
                                                               U_in,
                                                               A_hat,
                                                               U_hat,
                                                               0.0,
                                                               0.0,
                                                               dA_hat,
                                                               dU_hat,
                                                               vid,
                                                               vid,
                                                               ad_local,
                                                               ad_face);

                    for (unsigned int i = 0; i < n_dofs; ++i)
                      {
                        const unsigned int comp =
                          fe_->system_to_component_index(i).first;
                        if (comp >= 2)
                          continue; // trace rows carry no cell flux
                        const double phi = fef.shape_value(i, q);
                        jacobian_matrix.add(ldofs[i],
                                            trace_col,
                                            -(comp == 0 ? dFA : dFU) * phi *
                                              JxW_f[q]);
                      }
                  }
              }
          }
        // Single scatter: volume + face cell-cell ->
        // jacobian_matrix

        for (unsigned int i = 0; i < n_dofs; ++i)
          for (unsigned int j = 0; j < n_dofs; ++j)
            jacobian_matrix.add(ldofs[i], ldofs[j], cell_matrix(i, j));
      }
  }

  template <int dim, int spacedim>
  void
  BloodFlowSystem<dim, spacedim>::assemble_jacobian_trace_interior_block(
    const VectorType &y)
  {
    TimerOutput::Scope timer(computing_timer,
                             "assemble_jacobian_trace_interior_block");

    // FEValues and FEFaceValues read through the ghosted FE-range vector.  It
    // is sized dof_handler_.n_dofs(), which get_function_values() asserts, and
    // it holds every entry a locally owned cell can ask for, including the ones
    // on its ghost neighbours.
    const VectorType &y_cell = y_fe_relevant;

    const FEValuesExtractors::Scalar area_extractor(0);
    const FEValuesExtractors::Scalar velocity_extractor(1);
    const FEValuesExtractors::Scalar a_hat_extractor(2); // trace area
    const FEValuesExtractors::Scalar u_hat_extractor(3); // trace velocity

    const QGauss<dim - 1>       quad_face(1);
    FEFaceValues<dim, spacedim> fef(*fe_,
                                    quad_face,
                                    update_values | update_normal_vectors);

    std::set<std::pair<CellId, unsigned int>> processed;

    for (const auto &cell : dof_handler_.active_cell_iterators())
      {
        if (!cell->is_locally_owned())
          continue;

        for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
          {
            if (cell->face(f)->at_boundary())
              continue;
            if (is_junction_face(cell->id(), f))
              continue;

            const auto key = canonical_face_key(cell, f);
            if (processed.count(key))
              continue;
            processed.insert(key);

            // Same rule as the residual: the canonical side owns the rows.
            {
              const auto tit = face_dof_map.find(key);
              Assert(tit != face_dof_map.end(), ExcInternalError());
              if (!locally_owned_dofs_.is_element(tit->second.a_hat_dof))
                continue;
            }

            const auto        &nb      = cell->neighbor(f);
            const unsigned int nb_f    = cell->neighbor_of_neighbor(f);
            const unsigned int vid_L   = cell->material_id();
            const unsigned int vid_R   = nb->material_id();
            const double       ad_L    = compute_a_d_local(cell);
            const double       ad_R    = compute_a_d_local(nb);
            const double       ad_face = compute_a_d_at_face(cell, f);
            // ---- Left cell: state and normal
            // --------------------------------
            fef.reinit(cell, f);
            const double bn_L =
              compute_tangent_normal_product(cell, fef.get_normal_vectors()[0]);

            std::vector<double> A_Lv(1), U_Lv(1);
            fef[area_extractor].get_function_values(y_cell, A_Lv);
            fef[velocity_extractor].get_function_values(y_cell, U_Lv);
            const double A_L = std::max(A_Lv[0], 1e-10);
            const double U_L = U_Lv[0];

            std::vector<types::global_dof_index> ldofs_L(
              fe_->n_dofs_per_cell());
            cell->get_dof_indices(ldofs_L);

            // ---- Right cell: state and normal
            // --------------------------------
            fef.reinit(nb, nb_f);
            const double bn_R =
              compute_tangent_normal_product(nb, fef.get_normal_vectors()[0]);

            std::vector<double> A_Rv(1), U_Rv(1);
            fef[area_extractor].get_function_values(y_cell, A_Rv);
            fef[velocity_extractor].get_function_values(y_cell, U_Rv);
            const double A_R = std::max(A_Rv[0], 1e-10);
            const double U_R = U_Rv[0];

            std::vector<types::global_dof_index> ldofs_R(
              fe_->n_dofs_per_cell());
            nb->get_dof_indices(ldofs_R);

            // ---- Trace state
            // -------------------------------------------------
            double A_hat_cur = 0.0, U_hat_cur = 0.0;
            get_face_trace(y, cell, f, A_hat_cur, U_hat_cur);
            const double A_hat = std::max(A_hat_cur, 1e-10);
            const double U_hat = U_hat_cur;

            const FaceTraceDof           &td    = face_dof_map.at(key);
            const types::global_dof_index a_row = td.a_hat_dof;
            const types::global_dof_index u_row = td.u_hat_dof;

            // ================================================================
            // Block (trace, cell_L): dR/dw_L
            // Linearise F_hat(bn_L,bn_L, A_L,U_L, A_hat,U_hat)
            // w.r.t. w_L. Interior = L (trial nonzero), exterior =
            // trace (trial zero).
            // ================================================================
            fef.reinit(cell, f);
            for (unsigned int j = 0; j < fe_->n_dofs_per_cell(); ++j)
              {
                const double phi_A = fef[area_extractor].value(j, 0);
                const double phi_U = fef[velocity_extractor].value(j, 0);

                const auto [dFA, dFU] =
                  numerical_flux_jac(bn_L,
                                     bn_L,
                                     A_L,
                                     U_L,
                                     A_hat,
                                     U_hat,
                                     phi_A,
                                     phi_U, // trial on interior (L)
                                     0.0,
                                     0.0, // trace DOF fixed here
                                     vid_L,
                                     vid_R,
                                     ad_L,
                                     ad_face);

                jacobian_matrix.add(a_row, ldofs_L[j], dFA);
                jacobian_matrix.add(u_row, ldofs_L[j], dFU);
              }

            // ================================================================
            // Block (trace, cell_R): dR/dw_R
            // Linearise F_hat(bn_R,bn_R, A_R,U_R, A_hat,U_hat)
            // w.r.t. w_R. Interior = R (trial nonzero), exterior =
            // trace (trial zero).
            // ================================================================
            fef.reinit(nb, nb_f);
            for (unsigned int j = 0; j < fe_->n_dofs_per_cell(); ++j)
              {
                const double phi_A = fef[area_extractor].value(j, 0);
                const double phi_U = fef[velocity_extractor].value(j, 0);

                const auto [dFA, dFU] =
                  numerical_flux_jac(bn_R,
                                     bn_R,
                                     A_R,
                                     U_R,
                                     A_hat,
                                     U_hat,
                                     phi_A,
                                     phi_U, // trial on interior (R)
                                     0.0,
                                     0.0, // trace DOF fixed here
                                     vid_R,
                                     vid_L,
                                     ad_R,
                                     ad_face);

                jacobian_matrix.add(a_row, ldofs_R[j], dFA);
                jacobian_matrix.add(u_row, ldofs_R[j], dFU);
              }

            // ================================================================
            // Block (trace, trace): dR/d(A_hat, U_hat)
            // Trace appears as the EXTERIOR state in both fluxes.
            // Contribution from left flux: trial on exterior =
            // (dA_hat, dU_hat). Contribution from right flux: same
            // trial. Sum both contributions.
            // ================================================================
            // Unit perturbations in A_hat and U_hat:
            for (const auto &[dA_hat_trial, dU_hat_trial, col] :
                 {std::tuple{1.0, 0.0, td.a_hat_dof},
                  std::tuple{0.0, 1.0, td.u_hat_dof}})
              {
                // Left flux: exterior trial = (dA_hat_trial,
                // dU_hat_trial)
                const auto [dFA_L, dFU_L] =
                  numerical_flux_jac(bn_L,
                                     bn_L,
                                     A_L,
                                     U_L,
                                     A_hat,
                                     U_hat,
                                     0.0,
                                     0.0, // interior trial = 0
                                     dA_hat_trial,
                                     dU_hat_trial,
                                     vid_L,
                                     vid_R,
                                     ad_L,
                                     ad_face);

                // Right flux: exterior trial = (dA_hat_trial,
                // dU_hat_trial)
                const auto [dFA_R, dFU_R] =
                  numerical_flux_jac(bn_R,
                                     bn_R,
                                     A_R,
                                     U_R,
                                     A_hat,
                                     U_hat,
                                     0.0,
                                     0.0, // interior trial = 0
                                     dA_hat_trial,
                                     dU_hat_trial,
                                     vid_R,
                                     vid_L,
                                     ad_R,
                                     ad_face);

                jacobian_matrix.add(a_row, col, dFA_L + dFA_R);
                jacobian_matrix.add(u_row, col, dFU_L + dFU_R);
              }
          }
      }
  }

  template <int dim, int spacedim>
  void
  BloodFlowSystem<dim, spacedim>::assemble_jacobian_trace_boundary_block(
    const double      t,
    const VectorType &y)
  {
    TimerOutput::Scope timer(computing_timer,
                             "assemble_jacobian_trace_boundary_block");

    // FEValues and FEFaceValues read through the ghosted FE-range vector.  It
    // is sized dof_handler_.n_dofs(), which get_function_values() asserts, and
    // it holds every entry a locally owned cell can ask for, including the ones
    // on its ghost neighbours.
    const VectorType &y_cell = y_fe_relevant;

    const FEValuesExtractors::Scalar area_extractor(0);
    const FEValuesExtractors::Scalar velocity_extractor(1);
    const FEValuesExtractors::Scalar a_hat_extractor(2); // trace area
    const FEValuesExtractors::Scalar u_hat_extractor(3); // trace velocity

    const QGauss<dim - 1>       quad_face(1);
    FEFaceValues<dim, spacedim> fef(*fe_, quad_face, update_values);

    (void)t;

    for (const auto &cell : dof_handler_.active_cell_iterators())
      {
        if (!cell->is_locally_owned())
          continue;

        for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
          {
            if (!cell->face(f)->at_boundary())
              continue;
            if (is_junction_face(cell->id(), f))
              continue;

            const types::boundary_id bid = cell->face(f)->boundary_id();
            const unsigned int       vid = cell->material_id();
            // const auto              &vpp = vessel_map.at(vid);

            fef.reinit(cell, f);
            std::vector<double> A_int_v(1), U_int_v(1);
            fef[area_extractor].get_function_values(y_cell, A_int_v);
            fef[velocity_extractor].get_function_values(y_cell, U_int_v);

            const double A_int     = std::max(A_int_v[0], 1e-10);
            const double a_d_local = compute_a_d_at_face(cell, f);
            // const double U_int  = U_int_v[0];
            // const double c0     = compute_wave_speed(vpp.a_d,
            // vid);
            const double dc_int =
              compute_wave_speed_derivative(A_int, vid, a_d_local);

            double A_hat_cur = 0.0, U_hat_cur = 0.0;
            get_face_trace(y, cell, f, A_hat_cur, U_hat_cur);
            const double A_hat = std::max(A_hat_cur, 1e-10);
            const double dc_hat =
              compute_wave_speed_derivative(A_hat, vid, a_d_local);
            const double dPdA_hat =
              compute_pressure_derivative(A_hat, vid, a_d_local);

            const auto                    key   = canonical_face_key(cell, f);
            const auto                   &td    = face_dof_map.at(key);
            const types::global_dof_index a_row = td.a_hat_dof;
            const types::global_dof_index u_row = td.u_hat_dof;

            std::vector<types::global_dof_index> ldofs(fe_->n_dofs_per_cell());
            cell->get_dof_indices(ldofs);

            if (bid == 0) // inflow
              {
                // R1 = A_hat * U_hat - Q_in(t)
                // R2 = U_hat - 4(c_hat - c0) - W2_int
                // where W2_int = U_int - 4(c_int - c0)

                // \partialR1/ \partialA_hat, \partialR1/
                // \partialU_hat
                jacobian_matrix.add(a_row, a_row, U_hat_cur);
                jacobian_matrix.add(a_row, u_row, A_hat);

                // \partialR2/\partialA_hat,
                // \partialR2/\partialU_hat
                jacobian_matrix.add(u_row, a_row, -4.0 * dc_hat);
                jacobian_matrix.add(u_row, u_row, 1.0);

                // \partialR2/\partial(A_int, U_int) — block (2,1)
                for (unsigned int j = 0; j < fe_->n_dofs_per_cell(); ++j)
                  {
                    const double phi_A = fef[area_extractor].value(j, 0);
                    const double phi_U = fef[velocity_extractor].value(j, 0);
                    // \partialW2_int/\partialw_int = phi_U -
                    // 4*dc_int*phi_A
                    jacobian_matrix.add(u_row,
                                        ldofs[j],
                                        -(phi_U - 4.0 * dc_int * phi_A));
                  }
              }
            else if (outlet_type == "RCR" && rcr_map.count(bid) &&
                     (rcr_map.at(bid).R1 > 0.0 || rcr_map.at(bid).R2 > 0.0))

              {
                const auto &rcr = rcr_map.at(bid);

                if (rcr.C > 0.0)
                  {
                    // ---- Full RCR
                    // -------------------------------------------------------
                    // res_A = P(A_hat) - R1*(A_hat*U_hat) - Pc
                    // res_U = U_hat + 4(c_hat - c0) - W1_int
                    // dres_A/dA_hat = dP/dA_hat - R1*U_hat
                    // dres_A/dU_hat = -R1*A_hat
                    jacobian_matrix.add(a_row,
                                        a_row,
                                        dPdA_hat - rcr.R1 * U_hat_cur);
                    jacobian_matrix.add(a_row, u_row, -rcr.R1 * A_hat);

                    // dres_U/dA_hat = 4*dc_hat
                    // dres_U/dU_hat = 1
                    jacobian_matrix.add(u_row, a_row, 4.0 * dc_hat);
                    jacobian_matrix.add(u_row, u_row, 1.0);

                    // dres_U/d(A_int, U_int)
                    for (unsigned int j = 0; j < fe_->n_dofs_per_cell(); ++j)
                      {
                        const double phi_A = fef[area_extractor].value(j, 0);
                        const double phi_U =
                          fef[velocity_extractor].value(j, 0);
                        jacobian_matrix.add(u_row,
                                            ldofs[j],
                                            -(phi_U + 4.0 * dc_int * phi_A));
                      }
                  }
                else
                  {
                    // Single R
                    // -------------------------------------------------------
                    // res_A = P(A_hat) - R2*(A_hat*U_hat) - P_out
                    // res_U = U_hat + 4(c_hat - c0) - W1_int
                    //
                    // dres_A/dA_hat = dP/dA_hat - R2*U_hat   (same
                    // form as RCR with R2) dres_A/dU_hat =
                    // -R2*A_hat               (R2 replaces R1) Pc
                    // term drops out (no capacitor, no time
                    // derivative)
                    jacobian_matrix.add(a_row,
                                        a_row,
                                        dPdA_hat - rcr.R2 * U_hat_cur);
                    jacobian_matrix.add(a_row, u_row, -rcr.R2 * A_hat);

                    // dres_U/dA_hat = 4*dc_hat
                    // dres_U/dU_hat = 1
                    // (identical to RCR — res_U has no R
                    // dependence)
                    jacobian_matrix.add(u_row, a_row, 4.0 * dc_hat);
                    jacobian_matrix.add(u_row, u_row, 1.0);

                    // dres_U/d(A_int, U_int)
                    for (unsigned int j = 0; j < fe_->n_dofs_per_cell(); ++j)
                      {
                        const double phi_A = fef[area_extractor].value(j, 0);
                        const double phi_U =
                          fef[velocity_extractor].value(j, 0);
                        jacobian_matrix.add(u_row,
                                            ldofs[j],
                                            -(phi_U + 4.0 * dc_int * phi_A));
                      }
                  }
              }
            else // Reflection
              {
                const double Rt = par["Rt"];

                // R1 = U_hat + 4(c_hat - c0) - W1_int
                // R2 = U_hat - 4(c_hat - c0) - (-Rt * W1_int)

                // \partialR1/\partialA_hat,
                // \partialR1/\partialU_hat
                jacobian_matrix.add(a_row, a_row, 4.0 * dc_hat);
                jacobian_matrix.add(a_row, u_row, 1.0);

                // \partialR2/\partialA_hat,
                // \partialR2/\partialU_hat
                jacobian_matrix.add(u_row, a_row, -4.0 * dc_hat);
                jacobian_matrix.add(u_row, u_row, 1.0);

                // \partialR1 and \partialR2 /\partial(A_int,U_int)
                // via W1_int
                for (unsigned int j = 0; j < fe_->n_dofs_per_cell(); ++j)
                  {
                    const double phi_A = fef[area_extractor].value(j, 0);
                    const double phi_U = fef[velocity_extractor].value(j, 0);
                    const double dW1   = phi_U + 4.0 * dc_int * phi_A;
                    jacobian_matrix.add(a_row, ldofs[j], -dW1);
                    jacobian_matrix.add(u_row, ldofs[j], Rt * dW1);
                  }
              }
          }
      }
  }

  template <int dim, int spacedim>
  void
  BloodFlowSystem<dim, spacedim>::assemble_jacobian_rcr_capacitor_block(
    const VectorType &y)
  {
    if (rcr_pc_dof.empty())
      return;

    for (const auto &cell : dof_handler_.active_cell_iterators())
      {
        if (!cell->is_locally_owned())
          continue;

        for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
          {
            if (!cell->face(f)->at_boundary())
              continue;
            if (is_junction_face(cell->id(), f))
              continue;
            const auto pit = rcr_pc_dof.find(cell->face(f)->boundary_id());
            if (pit == rcr_pc_dof.end())
              continue;

            const types::global_dof_index pc_dof = pit->second;
            const auto &rcr = rcr_map.at(cell->face(f)->boundary_id());
            const auto &td  = face_dof_map.at(canonical_face_key(cell, f));

            double A_hat = 0.0, U_hat = 0.0;
            get_face_trace(y, cell, f, A_hat, U_hat);

            // R_pc = A_hat*U_hat - (Pc - P_out)/R2
            jacobian_matrix.add(pc_dof, td.a_hat_dof, U_hat);
            jacobian_matrix.add(pc_dof, td.u_hat_dof, A_hat);
            jacobian_matrix.add(pc_dof, pc_dof, -1.0 / rcr.R2);

            // res_A = P(Â) - R1*Q - Pc  ->  d(res_A)/dPc = -1
            jacobian_matrix.add(td.a_hat_dof, pc_dof, -1.0);
          }
      }
  }

  template <int dim, int spacedim>
  void
  BloodFlowSystem<dim, spacedim>::assemble_jacobian_trace_junction_block(
    const VectorType &y)
  {
    TimerOutput::Scope timer(computing_timer,
                             "assemble_jacobian_trace_junction_block");

    // FEValues and FEFaceValues read through the ghosted FE-range vector.  It
    // is sized dof_handler_.n_dofs(), which get_function_values() asserts, and
    // it holds every entry a locally owned cell can ask for, including the ones
    // on its ghost neighbours.
    const VectorType &y_cell = y_fe_relevant;

    const FEValuesExtractors::Scalar area_extractor(0);
    const FEValuesExtractors::Scalar velocity_extractor(1);
    const FEValuesExtractors::Scalar a_hat_extractor(2); // trace area
    const FEValuesExtractors::Scalar u_hat_extractor(3); // trace velocity

    const QGauss<dim - 1>       quad_face(1);
    FEFaceValues<dim, spacedim> fef(*fe_, quad_face, update_values);

    const double rho   = par["rho"];
    const double A_min = 1e-10;

    for (const auto &J : junctions)
      {
        const unsigned int K = J.n_vessels();

        std::vector<double>                  A_int(K), U_int(K), c0(K);
        std::vector<double>                  A_hat(K), U_hat(K);
        std::vector<double>                  c_hat_v(K), dP_hat(K), dc_hat_v(K);
        std::vector<double>                  dc_int_v(K);
        std::vector<int>                     orient(K);
        std::vector<types::global_dof_index> a_row(K), u_row(K);
        std::vector<std::vector<types::global_dof_index>> cell_dofs(K);

        for (unsigned int i = 0; i < K; ++i)
          {
            const auto        &hf  = J.half_faces[i];
            const unsigned int vid = hf.cell->material_id();

            fef.reinit(hf.cell, hf.face_no);
            std::vector<double> Av(1), Uv(1);
            fef[area_extractor].get_function_values(y_cell, Av);
            fef[velocity_extractor].get_function_values(y_cell, Uv);

            A_int[i]  = std::max(Av[0], A_min);
            U_int[i]  = Uv[0];
            orient[i] = hf.orientation;

            const double a_d_face = compute_a_d_at_face(hf.cell, hf.face_no);
            c0[i]                 = compute_wave_speed(a_d_face, vid, a_d_face);
            dc_int_v[i] =
              compute_wave_speed_derivative(A_int[i], vid, a_d_face);


            // c0[i]       =
            // compute_wave_speed(vessel_map.at(vid).a_d, vid);
            // dc_int_v[i] = compute_wave_speed_derivative(A_int[i],
            // vid);

            // const auto  key = canonical_face_key(hf.cell, hf.face_no);
            // const auto &td  = face_dof_map.at(key);
            // a_row[i]        = td.a_hat_dof;
            // u_row[i]        = td.u_hat_dof;

            // A_hat[i] = std::max(y(a_row[i]), A_min);
            // U_hat[i] = y(u_row[i]);

            std::vector<types::global_dof_index> ldofs(fe_->n_dofs_per_cell());
            hf.cell->get_dof_indices(ldofs);

            const auto [a_hat_dof, u_hat_dof] =
              face_trace_dofs(ldofs, hf.face_no);

            a_row[i] = a_hat_dof;
            u_row[i] = u_hat_dof;

            A_hat[i] = std::max(y(a_row[i]), A_min);
            U_hat[i] = y(u_row[i]);

            c_hat_v[i] = compute_wave_speed(A_hat[i], vid, a_d_face);
            dc_hat_v[i] =
              compute_wave_speed_derivative(A_hat[i], vid, a_d_face);
            dP_hat[i] = compute_pressure_derivative(A_hat[i], vid, a_d_face);

            cell_dofs[i].resize(fe_->n_dofs_per_cell());
            hf.cell->get_dof_indices(cell_dofs[i]);
          }

        // Row a_row[0]: mass conservation \sum s_i A_hat_i U_hat_i = 0.
        // Each row group is guarded exactly as in the residual, so the two stay
        // in step and the sparsity built by build_junction_sparsity() matches.
        if (locally_owned_dofs_.is_element(a_row[0]))
          for (unsigned int i = 0; i < K; ++i)
            {
              const double s = static_cast<double>(orient[i]);
              jacobian_matrix.add(a_row[0], a_row[i], s * U_hat[i]);
              jacobian_matrix.add(a_row[0], u_row[i], s * A_hat[i]);
            }

        // Rows u_row[0..K-2]: H_0 − H_i = 0
        for (unsigned int i = 1; i < K; ++i)
          {
            if (!locally_owned_dofs_.is_element(u_row[i - 1]))
              continue;

            // \partialH_0/\partialA_hat_0 ,
            // \partialH_0/\partialU_hat_0
            jacobian_matrix.add(u_row[i - 1], a_row[0], dP_hat[0] / rho);
            jacobian_matrix.add(u_row[i - 1], u_row[0], gamma * U_hat[0]);
            //  \partial(-H_i)/\partialA_hat_i,
            //  \partial(-H_i)/\partialU_hat_i
            jacobian_matrix.add(u_row[i - 1], a_row[i], -dP_hat[i] / rho);
            jacobian_matrix.add(u_row[i - 1], u_row[i], -gamma * U_hat[i]);
          }

        // Compat row for vessel 0 -> u_row[K-1]
        // Compat row for vessel i>=1 -> a_row[i]
        for (unsigned int i = 0; i < K; ++i)
          {
            const double                  s = static_cast<double>(orient[i]);
            const types::global_dof_index rr =
              (i == 0) ? u_row[K - 1] : a_row[i];

            if (!locally_owned_dofs_.is_element(rr))
              continue;

            // \partial/\partialA_hat_i, \partial/\partialU_hat_i of
            // (U_hat_i + s*4(c_hat_i - c0_i) - W_i)
            jacobian_matrix.add(rr, a_row[i], s * 4.0 * dc_hat_v[i]);
            jacobian_matrix.add(rr, u_row[i], 1.0);

            // \partial(-W_i)/\partial(A_int_i, U_int_i)
            const unsigned int vid = J.half_faces[i].cell->material_id();
            fef.reinit(J.half_faces[i].cell, J.half_faces[i].face_no);

            for (unsigned int j = 0; j < fe_->n_dofs_per_cell(); ++j)
              {
                const double phi_A = fef[area_extractor].value(j, 0);
                const double phi_U = fef[velocity_extractor].value(j, 0);
                // \partialW_i/\partialw = phi_U +
                // s*4*dc_int_v[i]*phi_A
                const double dW = phi_U + s * 4.0 * dc_int_v[i] * phi_A;
                jacobian_matrix.add(rr, cell_dofs[i][j], -dW);
              }
            (void)vid;
          }
      }
  }

  // ============================================================================
  // compute_pressure
  // ============================================================================
  template <int dim, int spacedim>
  void
  BloodFlowSystem<dim, spacedim>::compute_pressure(const VectorType &y,
                                                   VectorType       &p) const
  {
    TimerOutput::Scope timer(computing_timer, "compute_pressure");

    p = 0.0;

    std::vector<types::global_dof_index> ldofs(fe_->n_dofs_per_cell());

    for (const auto &cell : dof_handler_.active_cell_iterators())
      {
        if (!cell->is_locally_owned())
          continue;

        cell->get_dof_indices(ldofs);

        for (unsigned int i = 0; i < fe_->n_dofs_per_cell(); ++i)
          if (fe_->system_to_component_index(i).first == 0)
            {
              const double A = y(ldofs[i]);
              p(ldofs[i])    = compute_pressure_value(A,
                                                      cell->material_id(),
                                                      compute_a_d_local(cell));
            }
      }

    p.compress(VectorOperation::insert);
  }

  template <int dim, int spacedim>
  double
  BloodFlowSystem<dim, spacedim>::pressure(const double       area,
                                           const unsigned int vessel_id) const
  {
    return compute_pressure_value(area, vessel_id);
  }

  template <int dim, int spacedim>
  double
  BloodFlowSystem<dim, spacedim>::pressure_derivative(
    const double       area,
    const unsigned int vessel_id) const
  {
    return compute_pressure_derivative(area, vessel_id);
  }

  template <int dim, int spacedim>
  double
  BloodFlowSystem<dim, spacedim>::wave_speed(const double       area,
                                             const unsigned int vessel_id) const
  {
    return compute_wave_speed(area, vessel_id);
  }

  template <int dim, int spacedim>
  const typename BloodFlowSystem<dim, spacedim>::VesselProperties &
  BloodFlowSystem<dim, spacedim>::vessel_properties(
    const unsigned int vessel_id) const
  {
    return vessel_map.at(vessel_id);
  }


  // ============================================================================
  // output_results
  //
  // DataOut needs a ghosted vector over the FE range: it evaluates on every
  // locally owned cell, whose DoFs are all owned, but the ghosted form is what
  // the interface expects and it costs one exchange per field.  Each rank
  // writes its own .vtu and rank 0 records the .pvtu/.pvd, which is the
  // standard deal.II parallel output path -- no gathering to one rank.
  // ============================================================================
  template <int dim, int spacedim>
  void
  BloodFlowSystem<dim, spacedim>::output_results(const VectorType &y,
                                                 const VectorType &pressure_vec,
                                                 const unsigned int cycle) const
  {
    TimerOutput::Scope timer(computing_timer, "output_results");

    const std::string dir =
      output_directory + (output_directory.empty() ? "" : "/");

    // Restrict each field to the FE range and give it ghost entries.
    auto to_fe_ghosted = [this](const VectorType &src) {
      VectorType owned(locally_owned_fe_dofs, mpi_communicator_);
      for (const auto i : locally_owned_fe_dofs)
        owned(i) = src(i);
      owned.compress(VectorOperation::insert);

      VectorType ghosted(locally_owned_fe_dofs,
                         locally_relevant_fe_dofs,
                         mpi_communicator_);
      ghosted = owned;
      return ghosted;
    };

    const VectorType sol_fe = to_fe_ghosted(y);
    const VectorType p_fe   = to_fe_ghosted(pressure_vec);

    DataOut<dim, spacedim> data_out;
    data_out.attach_dof_handler(dof_handler_);

    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      interp(4, DataComponentInterpretation::component_is_scalar);

    std::vector<std::string> names = {"area", "velocity", "A_hat", "U_hat"};
    data_out.add_data_vector(sol_fe,
                             names,
                             DataOut<dim, spacedim>::type_dof_data,
                             interp);

    names = {"pressure", "unused", "unused_p2", "unused_p3"};
    data_out.add_data_vector(p_fe,
                             names,
                             DataOut<dim, spacedim>::type_dof_data,
                             interp);

    // Which rank owns which cell, so the partition is visible in the output.
    Vector<double> subdomain(triangulation_.n_active_cells());
    for (unsigned int i = 0; i < subdomain.size(); ++i)
      subdomain(i) = triangulation_.locally_owned_subdomain();
    data_out.add_data_vector(subdomain, "subdomain");

    data_out.build_patches();

    // Writes <name>-<cycle>.<rank>.vtu on each rank plus the .pvtu record, and
    // appends the time step to the .pvd on rank 0.
    static std::vector<std::pair<double, std::string>> pvd_records;
    const std::string pvtu = data_out.write_vtu_with_pvtu_record(
      dir, output_filename, cycle, mpi_communicator_, 5);

    if (this_mpi_process == 0)
      {
        pvd_records.emplace_back(time, pvtu);
        std::ofstream pvd(dir + output_filename + ".pvd");
        DataOutBase::write_pvd_record(pvd, pvd_records);
        pcout << "  Wrote <" << pvtu << ">" << std::endl;
      }
  }

  // ============================================================================
  // compute_errors
  //
  // integrate_difference() fills one entry per locally owned cell and
  // compute_global_error() does the reduction across ranks, so the printed
  // numbers are the global ones.
  // ============================================================================
  template <int dim, int spacedim>
  void
  BloodFlowSystem<dim, spacedim>::compute_errors(const unsigned int k)
  {
    TimerOutput::Scope timer(computing_timer, "compute_errors");

    // Existing parameter files describe only the physical (A,U) fields.  The
    // HDG state now also contains trace fields, so such an exact solution is
    // not dimensionally compatible with the four-component finite element.
    // Keep those runs usable and reserve error integration for a matching
    // manufactured solution.
    if (exact_solution.n_components != fe_->n_components())
      {
        pcout << "Skipping error computation: exact solution has "
              << exact_solution.n_components << " components, while the finite"
              << " element has " << fe_->n_components() << "." << std::endl;
        return;
      }

    // The FE has 4 components (A, U, A_hat, U_hat), so the masks select out of
    // 4 and `exact_solution` must be a 4-component Function; components 2,3 are
    // masked out.  Meaningful only for manufactured-solution verification runs.
    const ComponentSelectFunction<spacedim> area_mask(0, 1.0, 4);
    const ComponentSelectFunction<spacedim> vel_mask(1, 1.0, 4);

    Vector<float> diff(triangulation_.n_active_cells());

    VectorType cell_sol(locally_owned_fe_dofs, mpi_communicator_);
    for (const auto i : locally_owned_fe_dofs)
      cell_sol(i) = solution(i);
    cell_sol.compress(VectorOperation::insert);

    VectorType cell_sol_ghosted(locally_owned_fe_dofs,
                                locally_relevant_fe_dofs,
                                mpi_communicator_);
    cell_sol_ghosted = cell_sol;

    exact_solution.set_time(time);

    auto l2_error = [&](const ComponentSelectFunction<spacedim> &mask) {
      VectorTools::integrate_difference(dof_handler_,
                                        cell_sol_ghosted,
                                        exact_solution,
                                        diff,
                                        QGauss<dim>(fe_degree + 3),
                                        VectorTools::L2_norm,
                                        &mask);
      return VectorTools::compute_global_error(triangulation_,
                                               diff,
                                               VectorTools::L2_norm);
    };
    auto h1_error = [&](const ComponentSelectFunction<spacedim> &mask) {
      VectorTools::integrate_difference(dof_handler_,
                                        cell_sol_ghosted,
                                        exact_solution,
                                        diff,
                                        QGauss<dim>(fe_degree + 3),
                                        VectorTools::H1_seminorm,
                                        &mask);
      return VectorTools::compute_global_error(triangulation_,
                                               diff,
                                               VectorTools::H1_seminorm);
    };

    const double AL2 = l2_error(area_mask);
    const double AH1 = h1_error(area_mask);
    const double UL2 = l2_error(vel_mask);
    const double UH1 = h1_error(vel_mask);

    static double prev_AL2 = 0, prev_AH1 = 0, prev_UL2 = 0, prev_UH1 = 0;

    auto rate = [&](double prev, double cur) -> double {
      return (k == 0 || prev == 0.0) ? 0.0 :
                                       std::log(prev / cur) / std::log(2.0);
    };

    pcout << std::scientific << std::setprecision(3) << "=== Errors t=" << time
          << " cycle " << k + 1 << " ===\n"
          << " A  L2=" << AL2 << " rate=" << rate(prev_AL2, AL2) << "\n"
          << " A  H1=" << AH1 << " rate=" << rate(prev_AH1, AH1) << "\n"
          << " U  L2=" << UL2 << " rate=" << rate(prev_UL2, UL2) << "\n"
          << " U  H1=" << UH1 << " rate=" << rate(prev_UH1, UH1) << "\n"
          << " DoFs cell="
          << Utilities::MPI::sum<types::global_dof_index>(
               cell_dofs_owned.n_elements(), mpi_communicator_)
          << " trace="
          << Utilities::MPI::sum<types::global_dof_index>(
               trace_dofs_owned.n_elements(), mpi_communicator_)
          << " total=" << n_total_dofs << "\n"
          << std::string(60, '=') << "\n";

    prev_AL2 = AL2;
    prev_AH1 = AH1;
    prev_UL2 = UL2;
    prev_UH1 = UH1;
  }

  // ============================================================================
  // check_mass_conservation  ->  mass_conservation.csv
  //
  // Compares dV/dt (finite difference between consecutive outputs of the
  // integrated cross-sectional area) with the net flux through the inlet and
  // the terminals.  The reductions are collective; only rank 0 writes.
  // ============================================================================
  template <int dim, int spacedim>
  void
  BloodFlowSystem<dim, spacedim>::check_mass_conservation(const VectorType &y,
                                                          const double t) const
  {
    TimerOutput::Scope timer(computing_timer, "check_mass_conservation");

    // Force y_fe_relevant to reflect *this* y, not whatever trial iterate IDA
    // last evaluated a residual/Jacobian at.  Without this, the volume
    // integral below can read a stale snapshot, producing exact-zero dV/dt on
    // some steps and an overshooting "catch-up" jump on the next.
    update_ghosted_vectors(y);

    const FEValuesExtractors::Scalar area_extr(0);
    const QGauss<dim>                quad(fe_degree + 2);
    FEValues<dim, spacedim> fev(*fe_, quad, update_values | update_JxW_values);

    std::vector<double> A_q(quad.size());
    double              local_volume = 0.0;
    for (const auto &cell : dof_handler_.active_cell_iterators())
      {
        if (!cell->is_locally_owned())
          continue;
        fev.reinit(cell);
        fev[area_extr].get_function_values(y_fe_relevant, A_q);
        for (unsigned int q = 0; q < quad.size(); ++q)
          local_volume += A_q[q] * fev.JxW(q);
      }
    const double total_volume =
      Utilities::MPI::sum(local_volume, mpi_communicator_);

    double local_net_flux = 0.0;
    for (const auto &cell : dof_handler_.active_cell_iterators())
      {
        if (!cell->is_locally_owned())
          continue;
        for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
          {
            if (!cell->face(f)->at_boundary())
              continue;
            if (is_junction_face(cell->id(), f))
              continue;

            double A_hat = 0.0, U_hat = 0.0;
            get_face_trace(y_relevant, cell, f, A_hat, U_hat);
            const double Q   = A_hat * U_hat;
            const auto   bid = cell->face(f)->boundary_id();
            local_net_flux += (bid == 0) ? Q : -Q;
          }
      }
    const double net_flux =
      Utilities::MPI::sum(local_net_flux, mpi_communicator_);

    if (!mass_initialized)
      {
        mass_prev_volume = total_volume;
        mass_prev_t      = t;
        mass_initialized = true;
      }

    const double dt = t - mass_prev_t;
    const double dVdt =
      (dt > 0.0) ? (total_volume - mass_prev_volume) / dt : 0.0;
    const double imbalance = dVdt - net_flux;

    // Normalise against a slowly varying flow scale (running peak |net_flux|),
    // not the instantaneous value: net_flux crosses zero every cardiac cycle,
    // and dividing by it there would blow the ratio up for a tiny imbalance.
    mass_flow_scale = std::max(mass_flow_scale, std::abs(net_flux));
    const double rel_imbalance =
      std::abs(imbalance) / std::max(1e-12, mass_flow_scale);

    if (this_mpi_process == 0 && mass_log.is_open())
      {
        mass_log << std::scientific << std::setprecision(8) << t << ","
                 << total_volume << "," << dVdt << "," << net_flux << ","
                 << imbalance << "," << rel_imbalance << "\n";
        mass_log.flush();
      }

    // Two independent gates: relative-to-peak-flow AND an absolute floor, so
    // a genuine problem cannot hide behind either one.
    constexpr double rel_imbalance_warn_tol = 0.20;
    constexpr double abs_imbalance_warn_tol = 1e-5;
    if (rel_imbalance > rel_imbalance_warn_tol &&
        std::abs(imbalance) > abs_imbalance_warn_tol)
      pcout << "  [mass check WARNING] t=" << t << " dV/dt=" << dVdt
            << " net_flux=" << net_flux << " imbalance=" << imbalance << " ("
            << rel_imbalance * 100.0 << "% of peak flow)" << std::endl;

    mass_prev_volume = total_volume;
    mass_prev_t      = t;
  }

  // ============================================================================
  // open_output_files
  //
  // Opens the per-vessel probes and mass_conservation.csv, and resets the
  // running state of the mass check and of the BDF-order histogram.  Call once
  // per simulation (per refinement cycle), before ida.solve_dae().
  // ============================================================================
  template <int dim, int spacedim>
  void
  BloodFlowSystem<dim, spacedim>::open_output_files()
  {
    open_csv_files();

    mass_initialized = false;
    mass_prev_volume = 0.0;
    mass_prev_t      = 0.0;
    mass_flow_scale  = 0.0;

    if (mass_log.is_open())
      mass_log.close();
    if (this_mpi_process == 0)
      {
        const std::string dir =
          output_directory.empty() ? "." : output_directory;
        std::filesystem::create_directories(dir);
        mass_log.open(dir + "/mass_conservation.csv", std::ios::trunc);
        AssertThrow(mass_log,
                    ExcMessage("Cannot open " + dir +
                               "/mass_conservation.csv"));
        mass_log << "t,total_volume,dVdt,net_flux,imbalance,rel_imbalance\n";
      }

    bdf_steps.fill(0);
    bdf_prev_nst = 0;
  }

  // ============================================================================
  // write_output_step
  //
  // One row in every HDG_IDA_Vessel_<vid>.csv owned by this rank and one row
  // in mass_conservation.csv.
  // ============================================================================
  template <int dim, int spacedim>
  void
  BloodFlowSystem<dim, spacedim>::write_output_step(const double      t,
                                                    const VectorType &sol)
  {
    write_csv_row(t, sol);
    check_mass_conservation(sol, t);
  }

  // ============================================================================
  // record_ida_step
  //
  // output_step fires once per output period, so every step accepted since
  // the previous call is attributed to the order of the last accepted step.
  // The per-order counts then sum exactly to the total number of steps.
  // ============================================================================
  template <int dim, int spacedim>
  void
  BloodFlowSystem<dim, spacedim>::record_ida_step(
    const SUNDIALS::IDA<VectorType> &ida)
  {
    void *mem = internal_ida_access::get_ida_mem(ida);
    if (mem == nullptr)
      return;

    long int nst_now = 0;
    int      klast   = 0;
    IDAGetNumSteps(mem, &nst_now);
    IDAGetLastOrder(mem, &klast);

    const long int delta = nst_now - bdf_prev_nst;
    if (klast >= 1 && klast <= 5 && delta > 0)
      bdf_steps[klast] += delta;
    bdf_prev_nst = nst_now;
  }

  // ============================================================================
  // write_ida_statistics  ->  ida_statistics.csv
  //
  // ida_mem is still alive after solve_dae() returns (it is freed in the IDA
  // destructor or the next reset), so the cumulative counters can be read
  // here.  One row per simulation, appended, so successive runs (e.g. a sweep
  // over the maximum BDF order) accumulate in the same file.
  // ============================================================================
  template <int dim, int spacedim>
  void
  BloodFlowSystem<dim, spacedim>::write_ida_statistics(
    const SUNDIALS::IDA<VectorType> &ida,
    const double                     solve_wall_time) const
  {
    void *mem = internal_ida_access::get_ida_mem(ida);

    long int nst = 0, netf = 0, nni = 0, ncfn = 0;
    if (mem != nullptr)
      {
        IDAGetNumSteps(mem, &nst);                // accepted steps
        IDAGetNumErrTestFails(mem, &netf);        // error-test failures
        IDAGetNumNonlinSolvIters(mem, &nni);      // nonlinear iterations
        IDAGetNumNonlinSolvConvFails(mem, &ncfn); // nonlinear conv. failures
      }

    // Collective: the slowest rank defines the wall time of the solve.
    const double wall_time =
      Utilities::MPI::max(solve_wall_time, mpi_communicator_);

    if (this_mpi_process != 0)
      return;

    const std::string dir = output_directory.empty() ? "." : output_directory;
    std::filesystem::create_directories(dir);

    const std::string filename    = dir + "/ida_statistics.csv";
    const bool        file_exists = std::ifstream(filename).good();

    std::ofstream csv(filename, std::ios::app);
    AssertThrow(csv, ExcMessage("Cannot open " + filename));

    if (!file_exists)
      csv << "max_order,wall_time,n_steps,n_err_test_fails,"
             "n_nonlin_iters,n_nonlin_conv_fails,"
             "bdf1_steps,bdf2_steps,bdf3_steps,bdf4_steps,bdf5_steps\n";

    csv << ida_parameters.maximum_order << ',' << wall_time << ',' << nst << ','
        << netf << ',' << nni << ',' << ncfn << ',' << bdf_steps[1] << ','
        << bdf_steps[2] << ',' << bdf_steps[3] << ',' << bdf_steps[4] << ','
        << bdf_steps[5] << '\n';
  }

  // ============================================================================
  // finalize_output
  // ============================================================================
  template <int dim, int spacedim>
  void
  BloodFlowSystem<dim, spacedim>::finalize_output(
    const SUNDIALS::IDA<VectorType> &ida,
    const double                     solve_wall_time)
  {
    close_csv_files();
    if (mass_log.is_open())
      mass_log.close();
    write_ida_statistics(ida, solve_wall_time);
  }

  // ============================================================================
  // write_scaling_timings  ->  scaling_timings.csv
  //
  // Accumulated wall-clock time per TimerOutput section, tagged with the
  // process count.  Run the executable with mpirun -np 1, 2, 4, ...; each
  // invocation appends one row per section to the same CSV.
  // ============================================================================
  template <int dim, int spacedim>
  void
  BloodFlowSystem<dim, spacedim>::write_scaling_timings() const
  {
    const auto summary = computing_timer.get_summary_data(
      TimerOutput::OutputData::total_wall_time);

    if (this_mpi_process != 0)
      return;

    const std::string dir = output_directory.empty() ? "." : output_directory;
    std::filesystem::create_directories(dir);

    const std::string filename    = dir + "/scaling_timings.csv";
    const bool        file_exists = std::ifstream(filename).good();

    std::ofstream csv(filename, std::ios::app);
    AssertThrow(csv, ExcMessage("Cannot open " + filename));

    if (!file_exists)
      csv << "n_mpi_processes,section,wall_time\n";

    for (const auto &entry : summary)
      csv << n_mpi_processes << "," << entry.first << "," << entry.second
          << "\n";
  }

  // Explicit instantiation
  template class BloodFlowSystem<1, 3>;

} // namespace MetricFlowX