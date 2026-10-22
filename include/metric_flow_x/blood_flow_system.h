#ifndef METRIC_FLOW_X_BLOOD_FLOW_SYSTEM_H
#define METRIC_FLOW_X_BLOOD_FLOW_SYSTEM_H

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/function_parser.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/parsed_function.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>

#include <deal.II/distributed/fully_distributed_tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_interface_values.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_values_extractors.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/sundials/ida.h>

#include <metric_flow_x/parsed_tools/constants.h>
#include <metric_flow_x/parsed_tools/function.h>

#include <algorithm>
#include <array>
#include <fstream>
#include <map>
#include <memory>
#include <set>
#include <tuple>
#include <utility>
#include <vector>

void
test();

namespace MetricFlowX
{
  using dealii::AffineConstraints;
  using dealii::CellId;
  using dealii::ComponentMask;
  using dealii::ConditionalOStream;
  using dealii::DoFHandler;
  using dealii::DynamicSparsityPattern;
  using dealii::FE_DGQ;
  using dealii::FEInterfaceValues;
  using dealii::FESystem;
  using dealii::FEValues;
  using dealii::FiniteElement;
  using dealii::FullMatrix;
  using dealii::IndexSet;
  using dealii::ParameterAcceptor;
  using dealii::ParameterHandler;
  using dealii::Point;
  using dealii::Quadrature;
  using dealii::SolverControl;
  using dealii::Tensor;
  using dealii::TimerOutput;
  using dealii::Triangulation;
  using dealii::update_gradients;
  using dealii::update_JxW_values;
  using dealii::update_normal_vectors;
  using dealii::update_quadrature_points;
  using dealii::update_values;
  using dealii::UpdateFlags;
  using dealii::Vector;
  using dealii::VectorOperation;
  namespace DoFTools              = dealii::DoFTools;
  namespace FEValuesExtractors    = dealii::FEValuesExtractors;
  namespace LinearAlgebraPETSc    = dealii::LinearAlgebraPETSc;
  //namespace LinearAlgebraTrilinos = dealii::LinearAlgebraTrilinos;
  namespace PETScWrappers         = dealii::PETScWrappers;
  namespace SUNDIALS              = dealii::SUNDIALS;
  namespace SparsityTools         = dealii::SparsityTools;
  //namespace TrilinosWrappers      = dealii::TrilinosWrappers;
  namespace numbers               = dealii::numbers;
  namespace parallel              = dealii::parallel;
  namespace types                 = dealii::types;
  namespace Utilities             = dealii::Utilities;

  // ---------------------------------------------------------------------------
  // Linear algebra backend.
  //
  // PETSc is picked when it is available and built with real scalars, unless
  // FORCE_USE_OF_TRILINOS asks for the other one.  Everything below is written
  // against the LA aliases, so the solver compiles unchanged either way; the
  // two places where the backends genuinely differ, the direct solver and the
  // AdditionalData of the preconditioners, are guarded by USE_PETSC_LA.
  // ---------------------------------------------------------------------------
  namespace LA
  {
// #define FORCE_USE_OF_TRILINOS
#if defined(DEAL_II_WITH_PETSC) && !defined(DEAL_II_PETSC_WITH_COMPLEX) && \
  !(defined(DEAL_II_WITH_TRILINOS) && defined(FORCE_USE_OF_TRILINOS))
    namespace MPI     = dealii::LinearAlgebraPETSc::MPI;
    using SolverGMRES = dealii::LinearAlgebraPETSc::SolverGMRES;
#  define USE_PETSC_LA
#elif defined(DEAL_II_WITH_TRILINOS)
    namespace MPI     = dealii::LinearAlgebraTrilinos::MPI;
    using SolverGMRES = dealii::LinearAlgebraTrilinos::SolverGMRES;
#else
#  error DEAL_II_WITH_PETSC or DEAL_II_WITH_TRILINOS required
#endif
  } // namespace LA

  using BloodFlowParameters = ParsedTools::Constants;

  template <int dim, int spacedim>
  class BloodFlowIDARunner;

  // ---------------------------------------------------------------------------
  // Distributed linear algebra types.
  //
  // Every unknown of the system is a DoF of `dof_handler_` (components 0,1
  // carry the cell values A, U; components 2,3 carry the face traces A_hat,
  // U_hat), with the sole exception of the RCR capacitor pressures, which
  // occupy the index range appended after dof_handler_.n_dofs().  Ownership of
  // an unknown is consequently the ownership of the cell that carries it, as
  // determined by the Triangulation partitioning, and this class defines no
  // second ownership rule.
  //
  // SUNDIALS::IDA is instantiated for both LA::MPI::Vector alternatives, so the
  // time integrator follows the backend choice without any further work.
  // ---------------------------------------------------------------------------
  using VectorType = LA::MPI::Vector;
  using MatrixType = LA::MPI::SparseMatrix;

  // ---------------------------------------------------------------------------
  // FaceTraceDof
  //
  // One (A_hat, U_hat) trace pair lives on every unique face of the mesh.  In
  // 1-D (dim=1, spacedim=3) a face is a single vertex, so there is exactly one
  // quadrature point on it and one pair of scalar unknowns per face.
  //
  // The pair is addressed by the canonical face key (CellId,
  // local_face_number), where the canonical side of an interior face is the one
  // whose CellId compares less among the two neighbours.  Both indices are DoFs
  // of the FESystem, owned by the rank owning the canonical cell.
  // ---------------------------------------------------------------------------
  struct FaceTraceDof
  {
    types::global_dof_index a_hat_dof = numbers::invalid_dof_index;
    types::global_dof_index u_hat_dof = numbers::invalid_dof_index;
  };

  // ---------------------------------------------------------------------------
  // Scratch / copy-data structures for cell and face integrals.
  // ---------------------------------------------------------------------------
  template <int dim, int spacedim>
  struct BloodFlowScratchData
  {
    BloodFlowScratchData(
      const FiniteElement<dim, spacedim> &fe_,
      const Quadrature<dim>              &quadrature,
      const Quadrature<dim - 1>          &quadrature_face,
      const UpdateFlags update_flags = update_values | update_gradients |
                                       update_quadrature_points |
                                       update_JxW_values,
      const UpdateFlags interface_update_flags = update_values |
                                                 update_gradients |
                                                 update_quadrature_points |
                                                 update_JxW_values |
                                                 update_normal_vectors)
      : fe_values(fe_, quadrature, update_flags)
      , fe_interface_values(fe_, quadrature_face, interface_update_flags)
    {}

    BloodFlowScratchData(const BloodFlowScratchData<dim, spacedim> &src)
      : fe_values(src.fe_values.get_fe(),
                  src.fe_values.get_quadrature(),
                  src.fe_values.get_update_flags())
      , fe_interface_values(src.fe_interface_values.get_fe(),
                            src.fe_interface_values.get_quadrature(),
                            src.fe_interface_values.get_update_flags())
    {}

    FEValues<dim, spacedim>          fe_values;
    FEInterfaceValues<dim, spacedim> fe_interface_values;
  };

  struct BloodFlowCopyDataFace
  {
    FullMatrix<double>                   cell_matrix;
    Vector<double>                       cell_rhs;
    std::vector<types::global_dof_index> joint_dof_indices;
  };

  struct BloodFlowCopyData
  {
    FullMatrix<double>                   cell_matrix;
    Vector<double>                       cell_rhs;
    std::vector<types::global_dof_index> local_dof_indices;
    std::vector<BloodFlowCopyDataFace>   face_data;

    template <class Iterator>
    void
    reinit(const Iterator &cell, const unsigned int dofs_per_cell)
    {
      cell_matrix.reinit(dofs_per_cell, dofs_per_cell);
      cell_rhs.reinit(dofs_per_cell);
      local_dof_indices.resize(dofs_per_cell);
      cell->get_dof_indices(local_dof_indices);
    }
  };

  // ===========================================================================
  // Main class
  //
  // PARALLEL MODEL
  // --------------
  // The mesh is a parallel::fullydistributed::Triangulation built from a
  // description that is generated on a small number of group masters, so no
  // rank ever stores the entire fine mesh.  The ghost layer is the standard
  // vertex-adjacent one, which in 1-D means that a rank owning any cell
  // incident to a vertex also stores every other cell incident to that vertex.
  // Junction coupling therefore never needs data beyond the locally relevant
  // range.
  //
  // Assembly obeys a single rule: a rank writes exactly those rows of the
  // residual and of the Jacobian whose index it owns, and reads everything else
  // from ghosted vectors.  The rows of a junction with K incident vessels are
  // distributed over the incident half-faces rather than assigned to one owner:
  //
  //   row u_hat_i  : Riemann compatibility along vessel i,      i = 0 ... K-1
  //   row a_hat_0  : mass conservation over the whole junction
  //   row a_hat_i  : total-head continuity vessel 0 <-> i,      i = 1 ... K-1
  //
  // Each of these rows belongs to the rank owning vessel i's incident cell, and
  // that rank can read all K states from its ghost layer.  Every contribution
  // is consequently produced by exactly one rank and no reduction is needed to
  // close the junction equations.
  // ===========================================================================
  template <int dim, int spacedim = dim>
  class BloodFlowSystem : public ParameterAcceptor
  {
  public:
    template <int, int>
    friend class BloodFlowIDARunner;

    BloodFlowSystem(const MPI_Comm comm = MPI_COMM_WORLD);

    enum class Component
    {
      area,
      velocity,
      area_trace,
      velocity_trace
    };

    struct VesselProperties
    {
      double a0     = 0.0;
      double r_d    = 0.0;
      double a_d    = 0.0;
      double E      = 0.0;
      double h_wall = 0.0;
      double p_d    = 0.0;
      double p0     = 0.0;
      double L      = 0.0;
      double r_in   = 0.0;
      double r_out  = 0.0;
    };

    void
    initialize_params(const std::string &filename = "");

    void
    setup();

    VectorType
    make_state() const;

    void
    reinit_state(VectorType &state) const;

    void
    initialize_state(VectorType &state, const double t0);

    void
    initialize_state_derivative(VectorType &state_dot, const double t0) const;

    const parallel::fullydistributed::Triangulation<dim, spacedim> &
    triangulation() const;

    const DoFHandler<dim, spacedim> &
    dof_handler() const;

    const FiniteElement<dim, spacedim> &
    finite_element() const;

    const AffineConstraints<double> &
    constraints() const;

    MPI_Comm
    mpi_communicator() const;

    const IndexSet &
    locally_owned_dofs() const;

    const IndexSet &
    locally_relevant_dofs() const;

    const IndexSet &
    differential_dofs() const;

    const IndexSet &
    algebraic_dofs() const;

    const IndexSet &
    component_dofs(const Component component) const;

    FEValuesExtractors::Scalar
    area_extractor() const;

    FEValuesExtractors::Scalar
    velocity_extractor() const;

    // Read the VTK file on the group masters, distribute the resulting
    // description, and broadcast the per-vessel and per-terminal data, which is
    // small enough to be replicated on every rank.
    void
    create_triangulation();

    void
    setup_system();

    // Register the canonical (A_hat, U_hat) pair of every face touched by a
    // locally relevant cell, and collect the duplicate/canonical pairs that the
    // trace continuity rows tie together.  Called after distribute_dofs().
    void
    build_face_dof_map();

    void
    detect_junctions();

    void
    initialize_terminal_capacitors();

    // Residual F(t,y) for IDA; assembles the cell, trace, junction, continuity
    // and capacitor rows.
    void
    assemble_residual(const double      t,
                      const VectorType &y,
                      const VectorType &ydot,
                      VectorType       &residual);

    // Jacobian dF/dy + alpha * dF/dydot.
    void
    assemble_jacobian(const double      t,
                      const VectorType &y,
                      const VectorType &ydot,
                      const double      alpha);
    void 
    export_jacobian_matrix_market(const std::string &filename,
                                     const double       t,
                                     const double       alpha) const;

   std::string jacobian_export_filename = "";
   double      jacobian_export_time = 0.0;
   bool        jacobian_exported        = false;                  

    void
    assemble_state_jacobian(const double      t,
                            const VectorType &y,
                            const VectorType &ydot);

    void
    assemble_derivative_jacobian(const double      t,
                                 const VectorType &y,
                                 const VectorType &ydot);

    const MatrixType &
    state_jacobian_matrix() const;

    const MatrixType &
    derivative_jacobian_matrix() const;

    // Local mass matrices M_K and their inverses for every locally owned cell.
    void
    build_per_cell_mass_inv();

    void
    compute_initial_solution(VectorType &dst, const double t);

    void
    initialize_trace_unknowns(VectorType &sol, const double t);

    void
    output_results(const VectorType  &y,
                   const VectorType  &pressure_vec,
                   const unsigned int cycle) const;

    void
    compute_pressure(const VectorType &y, VectorType &pressure_vec) const;

    double
    pressure(const double area, const unsigned int vessel_id) const;

    double
    pressure_derivative(const double area, const unsigned int vessel_id) const;

    double
    wave_speed(const double area, const unsigned int vessel_id) const;

    const VesselProperties &
    vessel_properties(const unsigned int vessel_id) const;

    void
    compute_theoretical_peak(VectorType &theoretical_peak) const;

    void
    compute_errors(const unsigned int k);

    // -----------------------------------------------------------------------
    // CSV diagnostics (ported from the pre-MetricFlowX run() loop)
    //
    //   HDG_IDA_Vessel_<vid>.csv : P, Q, A, U at each vessel midpoint
    //   mass_conservation.csv    : dV/dt vs. net boundary flux
    //   ida_statistics.csv       : IDA step / failure counts + BDF-order
    //                              histogram (one row per simulation)
    //   ida_step_history.csv     : one row per output time: internal IDA
    //                              steps and nonlinear iterations taken to
    //                              reach it (per interval and cumulative)
    //   scaling_timings.csv      : TimerOutput wall time per section,
    //                              tagged with the MPI process count
    //
    // Call sequence from the time-integration driver:
    //   open_output_files();                     // before ida.solve_dae()
    //   record_ida_step(ida, t);                 // first line of output_step
    //   write_output_step(t, sol);               // inside output_step
    //   finalize_output(ida, solve_wall_time);   // after ida.solve_dae()
    //   write_scaling_timings();                 // once, at program end
    //
    // Every one of these is collective: call it on all ranks.
    // -----------------------------------------------------------------------
    void
    open_output_files();

    void
    write_output_step(const double t, const VectorType &sol);

    // Per-output-time IDA bookkeeping.  Every field is identical on all ranks
    // because the time integration itself is global.
    struct IDAOutputRecord
    {
      double   output_time         = 0.0; // the output time we asked for
      double   ida_time            = 0.0; // IDA's internal time (>= output)
      long int steps_interval      = 0;   // internal steps since last output
      long int steps_cumulative    = 0;
      long int nonlin_iters_interval   = 0; // Newton iters since last output
      long int nonlin_iters_cumulative = 0;
      long int nonlin_fails_interval   = 0; // nonlinear conv. failures
      long int err_test_fails_interval = 0; // local error-test failures
      long int residual_evals_interval = 0;
      long int lin_setups_interval     = 0; // setup_jacobian calls
      double   last_step_size      = 0.0;
      int      last_order          = 0;
    };

    void
    record_ida_step(const SUNDIALS::IDA<VectorType> &ida, const double t);

    const std::vector<IDAOutputRecord> &
    ida_output_history() const
    {
      return ida_history;
    }

    void
    poll_ida_internal_step(const SUNDIALS::IDA<VectorType> &ida,
                           const bool resync_time = false);

    void
    finalize_output(const SUNDIALS::IDA<VectorType> &ida,
                    const double                     solve_wall_time);

    void
    write_ida_statistics(const SUNDIALS::IDA<VectorType> &ida,
                         const double                     solve_wall_time) const;

    void
    write_scaling_timings() const;

    void
    check_mass_conservation(const VectorType &y, const double t) const;

    void
    open_csv_files();
    void
    write_csv_row(const double t, const VectorType &sol);
    void
    close_csv_files();

    enum class NumericalFluxType
    {
      HLL,
      HLL_HDG,
      LAX_FRIEDRICHS
    };

    void
    set_numerical_flux(const NumericalFluxType flux_type)
    {
      numerical_flux_type = flux_type;
    }

    NumericalFluxType
    get_numerical_flux() const
    {
      return numerical_flux_type;
    }

  private:
    using VertexKey = unsigned int;
    using FaceKey   = std::pair<CellId, unsigned int>;

    // -----------------------------------------------------------------------
    // Parallel infrastructure
    // -----------------------------------------------------------------------
    MPI_Comm            mpi_communicator_;
    const unsigned int  n_mpi_processes;
    const unsigned int  this_mpi_process;
    ConditionalOStream  pcout;
    mutable TimerOutput computing_timer;
    SolverControl       direct_solver_control;

    // -----------------------------------------------------------------------
    // Physical parameters
    // -----------------------------------------------------------------------
    std::map<FaceKey, FaceTraceDof> face_dof_map;

    // VertexKey
    // face_key(const typename DoFHandler<dim, spacedim>::active_cell_iterator
    // &cell,
    //          const unsigned int face_no) const;

    ParsedTools::Constants    par;
    AffineConstraints<double> constraints_;
    double                    current_dt  = 0.0;
    double                    last_rcr_dt = 0.0;

    // -----------------------------------------------------------------------
    // Vessel / RCR physics (read from VTK)
    //
    // These maps are keyed by vessel id and boundary id, hold a handful of
    // doubles each, and are replicated on every rank: the pressure law of a
    // vessel is needed wherever one of its cells is locally relevant, and the
    // arc-length bounds entering compute_a_d_local() are a property of the
    // vessel as a whole rather than of the locally stored part of it.
    // -----------------------------------------------------------------------
    using VesselPhysicalProperties = VesselProperties;
    std::map<unsigned int, VesselPhysicalProperties> vessel_map;

    struct RCRPhysics
    {
      double R1, R2, C, P_out;
    };
    std::map<unsigned int, RCRPhysics>   rcr_map;
    std::map<unsigned int, unsigned int> vid_to_rcr_vertex; // key = vessel id
    std::map<types::boundary_id, double> terminal_Pc_storage;
    std::set<types::boundary_id>         terminal_boundary_ids;

    // Global arc-length extent [s_min, s_max] of every vessel, obtained by
    // reducing the locally stored cell centres over the communicator.
    std::map<unsigned int, std::pair<double, double>> vessel_s_bounds;

    // Raw VTK cell/point data arrays, indexed by the serial cell/vertex
    // numbering of the file and read on the group masters only.
    Vector<double> cell_vessel_ids, cell_a0, cell_r_d, cell_a_d, cell_E;
    Vector<double> cell_h_wall, cell_p_d, cell_p0, cell_L;
    Vector<double> cell_r_in, cell_r_out;
    Vector<double> point_boundary_id, point_R1, point_R2, point_C, point_P_out;

    // Fill vessel_map, rcr_map, terminal_boundary_ids and vessel_s_bounds so
    // that they describe the whole network on every rank.
    void
    build_global_vessel_data();

    // -----------------------------------------------------------------------
    // Mesh and FE space
    //
    //   FESystem( FE_DGQ(fe_degree), 2,   -> components 0,1 : cell  A, U
    //             FE_DGQ(1),        2 )   -> components 2,3 : trace A_hat,
    //             U_hat
    //
    // Solution vector layout: [ FE range | capacitor pressures ], the FE range
    // being numbered by dof_handler_ and the capacitor block occupying the
    // n_rcr_dofs indices starting at dof_handler_.n_dofs().
    // -----------------------------------------------------------------------
    parallel::fullydistributed::Triangulation<dim, spacedim> triangulation_;
    DoFHandler<dim, spacedim>                                dof_handler_;
    std::unique_ptr<FiniteElement<dim, spacedim>>            fe_;

    types::global_dof_index n_total_dofs = 0;

    // -----------------------------------------------------------------------
    // Index sets
    //
    // The *_fe_dofs sets span the FE range only, sized dof_handler_.n_dofs(),
    // as required by FEValues::get_function_values().  The unqualified sets
    // span the full system, sized n_total_dofs.
    //
    // cell_dofs_owned and trace_dofs_owned come from DoFTools::extract_dofs()
    // with a ComponentMask, and hence identify the differential and the
    // algebraic rows of the DAE irrespective of any renumbering.
    // -----------------------------------------------------------------------
    IndexSet locally_owned_dofs_;
    IndexSet locally_relevant_dofs_;
    IndexSet locally_owned_fe_dofs;
    IndexSet locally_relevant_fe_dofs;
    IndexSet cell_dofs_owned;  // components 0,1 : differential rows
    IndexSet trace_dofs_owned; // components 2,3 : algebraic rows
    IndexSet rcr_dofs_owned;   // capacitor rows owned by this rank
    IndexSet differential_dofs_;
    IndexSet algebraic_dofs_;
    std::array<IndexSet, 4> component_dofs_;

    // -----------------------------------------------------------------------
    // Trace continuity
    //
    // Each cell carries its own (A_hat, U_hat) at every face.  On an ordinary
    // interior face the two sides are tied together: the canonical side is
    // registered in face_dof_map and carries the Riemann equation, while the
    // opposite side is slaved by the algebraic rows
    //     A_hat_dup - A_hat_canon = 0 ,   U_hat_dup - U_hat_canon = 0 .
    // Junction and boundary half-faces are each their own canonical owner and
    // are untouched by this mechanism.
    //
    // A pair is stored on the rank owning the duplicate side, which is the rank
    // that owns the two rows; the canonical indices are read from the ghosted
    // solution.
    // -----------------------------------------------------------------------
    struct TraceContinuityPair
    {
      types::global_dof_index a_dup, u_dup;     // slaved (duplicate) side
      types::global_dof_index a_canon, u_canon; // master (canonical) side
    };
    std::vector<TraceContinuityPair> trace_continuity_pairs;

    // Return the two global trace DoFs (component 2 -> A_hat, component 3 ->
    // U_hat) with support on local face `f`, given a cell's local dof indices.
    // In 1-D, FE_DGQ(1) has exactly one such DoF per component per face.
    std::pair<types::global_dof_index, types::global_dof_index>
    face_trace_dofs(const std::vector<types::global_dof_index> &ldofs,
                    const unsigned int                          f) const
    {
      types::global_dof_index a = numbers::invalid_dof_index;
      types::global_dof_index u = numbers::invalid_dof_index;
      for (unsigned int i = 0; i < fe_->n_dofs_per_cell(); ++i)
        {
          const unsigned int c = fe_->system_to_component_index(i).first;
          if ((c == 2 || c == 3) && fe_->has_support_on_face(i, f))
            (c == 2 ? a : u) = ldofs[i];
        }
      return {a, u};
    }

    // -----------------------------------------------------------------------
    // Capacitor pressures as differential DAE unknowns, one per RCR terminal
    // with C > 0.
    //
    // The terminal boundary ids are global data, so every rank enumerates the
    // capacitor block identically.  A given Pc index is owned by the rank
    // owning the unique cell holding that terminal face, so its row is always
    // assembled locally.
    // -----------------------------------------------------------------------
    std::map<types::boundary_id, types::global_dof_index> rcr_pc_dof;
    types::global_dof_index                               n_rcr_dofs = 0;

    void
    build_rcr_dof_map();

    void
    assemble_rcr_capacitor_equations(const VectorType &y,
                                     const VectorType &ydot,
                                     VectorType       &F);

    void
    assemble_jacobian_rcr_capacitor_block(const VectorType &y);

    // -----------------------------------------------------------------------
    // Junction detection
    //
    // A junction is a mesh vertex touched by two or more cells having different
    // vessel ids.  all_junction_faces holds (CellId, local_face_no) for every
    // half-face ending at a junction vertex, and is used to skip those faces in
    // the ordinary boundary-condition assembly so that they are handled
    // exclusively by assemble_trace_junction_equations().
    //
    // Detection runs over the locally relevant cells.  Since the ghost layer is
    // vertex-adjacent, a rank incident to a junction sees all of its
    // half-faces, and the list of junctions it builds is complete for every row
    // it owns.
    // -----------------------------------------------------------------------
    struct JunctionHalfFace
    {
      typename DoFHandler<dim, spacedim>::active_cell_iterator cell;
      unsigned int                                             face_no;
      int orientation; // ±1
    };

    struct JunctionInfo
    {
      Point<spacedim>               location;
      std::vector<JunctionHalfFace> half_faces; // one entry per incident vessel

      unsigned int
      n_vessels() const
      {
        return static_cast<unsigned int>(half_faces.size());
      }
    };

    std::vector<JunctionInfo>                 junctions;
    std::set<std::pair<CellId, unsigned int>> all_junction_faces;

    // ----------------------------------------------------------------------
    // If true, a valence-2 node joining two different vessel ids is demoted to
    // an ordinary interior face: it then carries a single trace pair, exactly
    // like an inflow, outflow or interior face, with the far side slaved by the
    // continuity rows.
    //
    // PHYSICS GATE: this is equivalent to the K=2 junction equations only when
    // both vessels share the same pressure law p(A).  If they differ, forcing
    // A_hat_L = A_hat_R satisfies mass conservation but violates total-pressure
    // continuity  p_L(A) + rho/2 U^2 = p_R(A) + rho/2 U^2.  detect_junctions()
    // checks this and refuses to demote, with a warning, when the laws differ.
    // ----------------------------------------------------------------------
    bool unify_two_way_junctions = false;

    // Do the two vessels share an identical pressure law?
    bool
    two_way_pressure_laws_match(const unsigned int vid_a,
                                const unsigned int vid_b) const
    {
      if (!vessel_map.count(vid_a) || !vessel_map.count(vid_b))
        return false;
      const auto &A  = vessel_map.at(vid_a);
      const auto &B  = vessel_map.at(vid_b);
      const auto  eq = [](const double x, const double y) {
        return std::abs(x - y) <=
               1e-10 * std::max(1.0, std::max(std::abs(x), std::abs(y)));
      };
      return eq(A.a0, B.a0) && eq(A.a_d, B.a_d) && eq(A.E, B.E) &&
             eq(A.h_wall, B.h_wall) && eq(A.p0, B.p0) && eq(A.p_d, B.p_d) &&
             eq(A.r_in, B.r_in) && eq(A.r_out, B.r_out);
    }

    // -----------------------------------------------------------------------
    // Linear algebra
    // -----------------------------------------------------------------------
    MatrixType jacobian_matrix;
    MatrixType state_jacobian_matrix_;
    MatrixType derivative_jacobian_matrix_;
    MatrixType linear_system_matrix;

    // The direct solver is the one object with no common alias in the LA
    // namespace: PETSc reaches MUMPS through SparseDirectMUMPS, Trilinos
    // reaches Amesos through SolverDirect.  Both are constructed from a
    // SolverControl and solve through solve(A, x, b), so only the declaration
    // differs.

#ifdef USE_PETSC_LA
    std::unique_ptr<PETScWrappers::SparseDirectMUMPS> direct_solver;
#else
    std::unique_ptr<TrilinosWrappers::SolverDirect> direct_solver;
#endif

    // Preconditioner for the iterative path.  Note that PETSc's PCILU is a
    // single-rank preconditioner: with more than one process, initialize() it
    // through a block-Jacobi outer level or fall back to PreconditionAMG.
    std::unique_ptr<LA::MPI::PreconditionILU> ilu_preconditioner;

    // Per-cell mass matrices, indexed by cell->active_cell_index(): deal.II
    // numbers the locally stored active cells contiguously, so no CellId lookup
    // is needed in the residual inner loop.  Only locally owned entries are
    // filled.
    std::vector<FullMatrix<double>> per_cell_mass_inv;
    std::vector<FullMatrix<double>> per_cell_mass;

    VectorType solution;
    VectorType solution_dot;
    VectorType pressure_;
    VectorType theoretical_peak;

    // -----------------------------------------------------------------------
    // Ghosted read vectors.
    //
    // All assembly reads go through these and all assembly writes go to a
    // non-ghosted vector followed by compress(), which is the only
    // communication in the whole assembly.
    //
    //   y_relevant    : full system, for direct y[dof] reads of trace and
    //                   capacitor unknowns.
    //   y_fe_relevant : FE range only, as required by
    //                   FEValues::get_function_values(), which asserts that the
    //                   vector has size dof_handler_.n_dofs().
    //   y_fe_owned    : scratch used to build y_fe_relevant.
    // -----------------------------------------------------------------------
    mutable VectorType y_relevant;
    mutable VectorType y_fe_relevant;
    mutable VectorType y_fe_owned;
    VectorType         residual_F;

    void
    update_ghosted_vectors(const VectorType &y) const;

    // -----------------------------------------------------------------------
    // User parameters
    // -----------------------------------------------------------------------
    unsigned int fe_degree              = 1;
    std::string  constants              = "1.0";
    std::string  output_filename        = "solution";
    bool         use_direct_solver      = true;
    bool         use_junction_mesh      = true;
    bool         use_riemann_invariants = true;
    unsigned int n_refinement_cycles    = 1;
    unsigned int n_global_refinements   = 5;
    std::string  vtk_file_path          = "mesh.vtk";
    std::string  output_directory       = "";
    unsigned int verbosity              = 0;
    std::string  outlet_type;
    double       theta    = 0.5;
    double       theta_bd = 0.5;
    double gamma = 1; // 0.5*gamma*(U^2) term in total pressure continuity at
                      // junctions chnage only for 56 arteries <= 0.8
    double time = 0.0;
    // Positive values throttle file output; zero writes every IDA callback.
    double time_interval = 0.0;
    
    NumericalFluxType numerical_flux_type     = NumericalFluxType::HLL;
    std::string       numerical_flux_type_str = "HLL";

    SUNDIALS::IDA<VectorType>::AdditionalData ida_parameters;

    ParsedTools::Function<spacedim> rhs_function;
    ParsedTools::Function<spacedim> exact_solution;
    ParsedTools::Function<1>        inflow_function;

    // -----------------------------------------------------------------------
    // CSV time-series output, one file per vessel.
    //
    // A probe is opened only on the rank owning the probed cell, so each file
    // is written by exactly one rank and no output is duplicated.
    // -----------------------------------------------------------------------
    std::map<unsigned int, std::ofstream> csv_vessel;
    // (vessel id, cell containing the vessel's arc-length midpoint, reference
    // coordinate of the midpoint inside that cell).
    std::vector<
      std::tuple<unsigned int,
                 typename DoFHandler<dim, spacedim>::active_cell_iterator,
                 double>>
      probe_targets;

    // mass_conservation.csv state (rank 0 writes the file; every rank keeps
    // the running values because the reductions are collective).
    mutable std::ofstream mass_log;
    mutable double        mass_prev_volume = 0.0;
    mutable double        mass_prev_t      = 0.0;
    mutable double        mass_flow_scale  = 0.0;
    mutable bool          mass_initialized = false;

    // ida_statistics.csv state: bdf_steps[k] = accepted steps at BDF order k
    // (k = 1..5), attributed at each output_step callback.
    std::array<long int, 6> bdf_steps{};
    long int                bdf_prev_nst = 0;

    // ida_step_history.csv state: cumulative counters seen at the previous
    // output, so each record can store the per-interval differences.
    std::vector<IDAOutputRecord> ida_history;
    std::ofstream                ida_history_log;
    long int                     prev_nni  = 0;
    long int                     prev_ncfn = 0;
    long int                     prev_netf = 0;
    long int                     prev_nre  = 0;
    long int                     prev_nls  = 0;
        std::ofstream ida_internal_log;
    long int      is_prev_nst   = 0;
    long int      is_prev_nni   = 0;
    long int      is_prev_ncfn  = 0;
    long int      is_prev_netf  = 0;
    long int      is_prev_nls   = 0;
    double        is_t_accepted = 0.0;

    // -----------------------------------------------------------------------
    // Internal helpers — geometry
    // -----------------------------------------------------------------------
    bool
    is_terminal_boundary(const types::boundary_id bid) const
    {
      return terminal_boundary_ids.count(bid) > 0;
    }

    bool
    is_junction_face(const CellId &cid, const unsigned int f) const
    {
      return all_junction_faces.count(std::make_pair(cid, f)) > 0;
    }

    Tensor<1, spacedim>
    compute_directional_vector(
      const typename DoFHandler<dim, spacedim>::active_cell_iterator &cell)
      const
    {
      return (cell->vertex(1) - cell->vertex(0)) /
             cell->vertex(1).distance(cell->vertex(0));
    }

    double
    compute_tangent_normal_product(
      const typename DoFHandler<dim, spacedim>::active_cell_iterator &cell,
      const Tensor<1, spacedim> &normal) const
    {
      return compute_directional_vector(cell) * normal;
    }

    // -----------------------------------------------------------------------
    // Internal helpers — physics / constitutive law
    // -----------------------------------------------------------------------

    // Local diastolic cross-sectional area at a point x of a cell, obtained by
    // interpolating the radius linearly in arc length between the inlet radius
    // r_in and the outlet radius r_out of the parent vessel.  The arc-length
    // bounds are the global ones, so the value does not depend on the
    // partitioning.
    //
    // The pressure law is evaluated pointwise with this a_d(x): the momentum
    // equation is written for U with flux U^2/2 + p(A, a_d(x))/rho, so the
    // taper needs no extra source term as long as every evaluation of p --
    // volume quadrature points and both states of a face flux -- uses a_d at
    // the point where p is evaluated.
    //
    // If r_in == r_out == 0 (no taper data), fall back to the vessel-level a_d.
    double
    compute_a_d_at_point(
      const typename DoFHandler<dim, spacedim>::active_cell_iterator &cell,
      const Point<spacedim>                                          &x) const
    {
      const unsigned int vid = cell->material_id();
      const auto        &vpp = vessel_map.at(vid);

      if (vpp.r_in <= 0.0 || vpp.r_out <= 0.0)
        return vpp.a_d;

      const auto &[s_min, s_max] = vessel_s_bounds.at(vid);

      const Tensor<1, spacedim> d_hat = compute_directional_vector(cell);

      const double s = x * d_hat;

      // Clamp so that round-off at the vessel ends cannot extrapolate.
      const double xi_geometry =
        (s_max - s_min > 1e-14) ?
          std::clamp((s - s_min) / (s_max - s_min), 0.0, 1.0) :
          0.5;

      const double r = vpp.r_in + xi_geometry * (vpp.r_out - vpp.r_in);

      return numbers::PI * r * r;
    }

    // Physical location of the support point of local DoF i (the FE is
    // nodal, and the 1-D cells are straight segments).
    Point<spacedim>
    support_point(
      const typename DoFHandler<dim, spacedim>::active_cell_iterator &cell,
      const unsigned int                                              i) const
    {
      const double xi = fe_->unit_support_point(i)[0];
      return cell->vertex(0) + xi * (cell->vertex(1) - cell->vertex(0));
    }

    // a_d at the cell centroid.  Only meant for point probes at the cell
    // midpoint; residuals and Jacobians use compute_a_d_at_point().
    double
    compute_a_d_local(
      const typename DoFHandler<dim, spacedim>::active_cell_iterator &cell)
      const
    {
      return compute_a_d_at_point(cell, cell->center());
    }

    double
    compute_a_d_at_face(
      const typename DoFHandler<dim, spacedim>::active_cell_iterator &cell,
      const unsigned int face_no) const
    {
      return compute_a_d_at_point(cell, cell->vertex(face_no));
    }

    double
    compute_beta_p(const double E, const double h_wall) const
    {
      return (4.0 * std::sqrt(numbers::PI) / 3.0) * E * h_wall;
    }

    // ---- Physics helpers taking an explicit a_d_local
    // ------------------------

    double
    compute_pressure_value(const double       A,
                           const unsigned int vid,
                           const double       a_d_local) const
    {
      const auto  &vpp  = vessel_map.at(vid);
      const double beta = compute_beta_p(vpp.E, vpp.h_wall);
      return vpp.p0 + beta / a_d_local * (std::sqrt(A) - std::sqrt(a_d_local)) +
             vpp.p_d;
    }

    double
    compute_pressure_derivative(const double       A,
                                const unsigned int vid,
                                const double       a_d_local) const
    {
      const auto  &vpp    = vessel_map.at(vid);
      const double beta_p = compute_beta_p(vpp.E, vpp.h_wall);
      return beta_p / (2.0 * a_d_local * std::sqrt(std::max(A, 1e-30)));
    }

    double
    compute_wave_speed(const double       A,
                       const unsigned int vid,
                       const double       a_d_local) const
    {
      const double A_safe = std::max(A, 1e-10);
      return std::sqrt(A_safe / par["rho"] *
                       compute_pressure_derivative(A_safe, vid, a_d_local));
    }

    double
    compute_wave_speed_derivative(const double       A,
                                  const unsigned int vid,
                                  const double       a_d_local) const
    {
      const double A_safe = std::max(A, 1e-10);
      return compute_wave_speed(A_safe, vid, a_d_local) * par["m"] /
             (2.0 * A_safe);
    }

    // ---- Cell-iterator wrappers (compute a_d_local internally)
    // ---------------

    double
    compute_pressure_value(
      const double                                                    A,
      const typename DoFHandler<dim, spacedim>::active_cell_iterator &cell)
      const
    {
      return compute_pressure_value(A,
                                    cell->material_id(),
                                    compute_a_d_local(cell));
    }

    double
    compute_pressure_derivative(
      const double                                                    A,
      const typename DoFHandler<dim, spacedim>::active_cell_iterator &cell)
      const
    {
      return compute_pressure_derivative(A,
                                         cell->material_id(),
                                         compute_a_d_local(cell));
    }

    double
    compute_wave_speed(
      const double                                                    A,
      const typename DoFHandler<dim, spacedim>::active_cell_iterator &cell)
      const
    {
      return compute_wave_speed(A,
                                cell->material_id(),
                                compute_a_d_local(cell));
    }

    double
    compute_wave_speed_derivative(
      const double                                                    A,
      const typename DoFHandler<dim, spacedim>::active_cell_iterator &cell)
      const
    {
      return compute_wave_speed_derivative(A,
                                           cell->material_id(),
                                           compute_a_d_local(cell));
    }

    // ---- Wrappers for call sites that only have a vessel id
    // ------------------

    double
    compute_pressure_value(const double A, const unsigned int vid) const
    {
      return compute_pressure_value(A, vid, vessel_map.at(vid).a_d);
    }

    double
    compute_pressure_derivative(const double A, const unsigned int vid) const
    {
      return compute_pressure_derivative(A, vid, vessel_map.at(vid).a_d);
    }

    double
    compute_wave_speed(const double A, const unsigned int vid) const
    {
      return compute_wave_speed(A, vid, vessel_map.at(vid).a_d);
    }

    double
    compute_wave_speed_derivative(const double A, const unsigned int vid) const
    {
      return compute_wave_speed_derivative(A, vid, vessel_map.at(vid).a_d);
    }

    // -----------------------------------------------------------------------
    // Internal helpers — face-trace access
    // -----------------------------------------------------------------------

    // Canonical key of a face.  For interior faces the cell with the
    // lexicographically smaller CellId owns the key.  CellId comparison is
    // partitioning independent, so all ranks agree on the canonical side of a
    // face straddling a subdomain boundary.
    std::pair<CellId, unsigned int>
    canonical_face_key(
      const typename DoFHandler<dim, spacedim>::active_cell_iterator &cell,
      const unsigned int face_no) const;

    // Read (A_hat, U_hat) from a ghosted vector.
    void
    get_face_trace(
      const VectorType                                               &y,
      const typename DoFHandler<dim, spacedim>::active_cell_iterator &cell,
      const unsigned int                                              face_no,
      double                                                         &A_hat,
      double &U_hat) const;

    // -----------------------------------------------------------------------
    // Internal helpers — physical fluxes (scalar projections)
    // -----------------------------------------------------------------------

    double
    scalar_area_flux(const double bn, const double A, const double U) const
    {
      return A * U * bn;
    }

    double
    scalar_momentum_flux(const double bn,
                         const double U,
                         const double pressure,
                         const double rho) const
    {
      return (0.5 * U * U + pressure / rho) * bn;
    }

    // Linearised (Jacobian) versions
    double
    scalar_area_flux_jac(const double bn,
                         const double A,
                         const double U,
                         const double dA,
                         const double dU) const
    {
      return (A * dU + U * dA) * bn;
    }

    double
    scalar_momentum_flux_jac(const double bn,
                             const double c2_over_A, // 1/rho dp/dA = c^2/A
                             const double U,
                             const double dA,
                             const double dU) const
    {
      return (c2_over_A * dA + U * dU) * bn;
    }

    double
    compute_LF_penalty(const double       A_L,
                       const double       A_R,
                       const double       U_L,
                       const double       U_R,
                       const double       bn_L,
                       const double       bn_R,
                       const unsigned int vid_L,
                       const unsigned int vid_R,
                       const double       a_d_L,
                       const double       a_d_R) const
    {
      const double cL = compute_wave_speed(A_L, vid_L, a_d_L);
      const double cR = compute_wave_speed(A_R, vid_R, a_d_R);
      return std::max({std::abs((U_L - cL) * bn_L),
                       std::abs((U_L + cL) * bn_L),
                       std::abs((U_R - cR) * bn_R),
                       std::abs((U_R + cR) * bn_R)});
    }

    // -----------------------------------------------------------------------
    // Internal helpers — numerical fluxes
    // -----------------------------------------------------------------------

    std::array<double, 2>
    hll_flux(double       bn_L,
             double       bn_R,
             double       A_L,
             double       U_L,
             double       A_R,
             double       U_R,
             unsigned int vid_L,
             unsigned int vid_R,
             double       a_d_L,
             double       a_d_R) const;

    std::array<double, 2>
    hll_hdg_flux(double bn_L,
                 double /*bn_R*/,
                 double A_L,
                 double U_L,
                 double A_R,
                 double U_R,
                 unsigned int /*vid_L*/,
                 unsigned int vid_R,
                 double /*a_d_L*/,
                 double a_d_R) const;

    std::array<double, 2>
    lf_flux(double       bn_L,
            double       bn_R,
            double       A_L,
            double       U_L,
            double       A_R,
            double       U_R,
            unsigned int vid_L,
            unsigned int vid_R,
            double       a_d_L,
            double       a_d_R) const;

    std::array<double, 2>
    numerical_flux(double       bn_L,
                   double       bn_R,
                   double       A_L,
                   double       U_L,
                   double       A_R,
                   double       U_R,
                   unsigned int vid_L,
                   unsigned int vid_R,
                   double       a_d_L,
                   double       a_d_R) const
    {
      if (numerical_flux_type == NumericalFluxType::HLL)
        {
          return hll_flux(
            bn_L, bn_R, A_L, U_L, A_R, U_R, vid_L, vid_R, a_d_L, a_d_R);
        }
      else if (numerical_flux_type == NumericalFluxType::HLL_HDG)
        {
          return hll_hdg_flux(
            bn_L, bn_R, A_L, U_L, A_R, U_R, vid_L, vid_R, a_d_L, a_d_R);
        }
      else
        {
          return lf_flux(
            bn_L, bn_R, A_L, U_L, A_R, U_R, vid_L, vid_R, a_d_L, a_d_R);
        }
    }

    std::array<double, 2>
    hll_flux_jac(double       bn_L,
                 double       bn_R,
                 double       A_L,
                 double       U_L,
                 double       A_R,
                 double       U_R,
                 double       dA_L,
                 double       dU_L,
                 double       dA_R,
                 double       dU_R,
                 unsigned int vid_L,
                 unsigned int vid_R,
                 const double a_d_L,
                 const double a_d_R) const;

    std::array<double, 2>
    hll_hdg_flux_jac(double bn_L,
                     double /*bn_R*/,
                     double A_L,
                     double U_L,
                     double A_R,
                     double U_R,
                     double dA_L,
                     double dU_L,
                     double dA_R,
                     double dU_R,
                     unsigned int /*vid_L*/,
                     unsigned int vid_R,
                     const double /*a_d_L*/,
                     const double a_d_R) const;

    std::array<double, 2>
    lf_flux_jac(double       bn_L,
                double       bn_R,
                double       A_L,
                double       U_L,
                double       A_R,
                double       U_R,
                double       dA_L,
                double       dU_L,
                double       dA_R,
                double       dU_R,
                unsigned int vid_L,
                unsigned int vid_R,
                const double a_d_L,
                const double a_d_R) const;

    std::array<double, 2>
    numerical_flux_jac(double       bn_L,
                       double       bn_R,
                       double       A_L,
                       double       U_L,
                       double       A_R,
                       double       U_R,
                       double       dA_L,
                       double       dU_L,
                       double       dA_R,
                       double       dU_R,
                       unsigned int vid_L,
                       unsigned int vid_R,
                       const double a_d_L,
                       const double a_d_R) const
    {
      if (numerical_flux_type == NumericalFluxType::HLL)
        {
          return hll_flux_jac(bn_L,
                              bn_R,
                              A_L,
                              U_L,
                              A_R,
                              U_R,
                              dA_L,
                              dU_L,
                              dA_R,
                              dU_R,
                              vid_L,
                              vid_R,
                              a_d_L,
                              a_d_R);
        }
      else if (numerical_flux_type == NumericalFluxType::HLL_HDG)
        {
          return hll_hdg_flux_jac(bn_L,
                                  bn_R,
                                  A_L,
                                  U_L,
                                  A_R,
                                  U_R,
                                  dA_L,
                                  dU_L,
                                  dA_R,
                                  dU_R,
                                  vid_L,
                                  vid_R,
                                  a_d_L,
                                  a_d_R);
        }
      else
        {
          return lf_flux_jac(bn_L,
                             bn_R,
                             A_L,
                             U_L,
                             A_R,
                             U_R,
                             dA_L,
                             dU_L,
                             dA_R,
                             dU_R,
                             vid_L,
                             vid_R,
                             a_d_L,
                             a_d_R);
        }
    }

    // -----------------------------------------------------------------------
    // Assembly sub-routines
    //
    // Each routine loops over the locally owned cells (or over the locally
    // relevant ones and guards on row ownership, where the coupling requires
    // it), reads from the ghosted vectors, and writes only rows it owns.  The
    // caller closes the write with a single compress().
    // -----------------------------------------------------------------------

    // Cell residuals: volume integrals plus flux through the face traces.
    void
    assemble_cell_residuals(const double t, const VectorType &y, VectorType &F);

    // Trace equations for interior faces (Riemann-invariant continuity).
    void
    assemble_trace_interior_equations(const VectorType &y, VectorType &F);

    // Trace equations for boundary faces (inflow Q / RCR / reflection).
    void
    assemble_trace_boundary_equations(const double      t,
                                      const VectorType &y,
                                      VectorType       &F);

    // Trace equations for junction faces (mass conservation, total-head
    // continuity, and Riemann compatibility per vessel).
    void
    assemble_trace_junction_equations(const VectorType &y, VectorType &F);

    // Continuity rows for the duplicate side of each ordinary interior face:
    // F[a_dup] = A_hat_dup - A_hat_canon, likewise for U.
    void
    assemble_trace_continuity_equations(const VectorType &y, VectorType &F);

    // Jacobian blocks
    void
    assemble_jacobian_cell_block(const double t, const VectorType &y);

    void
    assemble_jacobian_trace_interior_block(const VectorType &y);

    void
    assemble_jacobian_trace_boundary_block(const double t, const VectorType &y);

    void
    assemble_jacobian_trace_junction_block(const VectorType &y);

    void
    assemble_jacobian_trace_continuity_block();

    // -----------------------------------------------------------------------
    // Sparsity
    //
    // The pattern is built as a DynamicSparsityPattern over the locally
    // relevant rows and then reduced with
    // SparsityTools::distribute_sparsity_pattern(), so that every rank ends up
    // knowing the entries of the rows it owns.
    // -----------------------------------------------------------------------
    void
    build_cell_sparsity(DynamicSparsityPattern &dsp);
    void
    build_trace_sparsity(DynamicSparsityPattern &dsp);
    void
    build_junction_sparsity(DynamicSparsityPattern &dsp);
    void
    build_rcr_sparsity(DynamicSparsityPattern &dsp);
    void
    build_trace_continuity_sparsity(DynamicSparsityPattern &dsp);
    void
    build_extended_sparsity_pattern();

    friend void ::test();
  };

} // namespace MetricFlowX

#endif // METRIC_FLOW_X_BLOOD_FLOW_SYSTEM_H