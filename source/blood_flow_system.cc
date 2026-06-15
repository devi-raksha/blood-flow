/* --------------------------------------------------------------------------
 * Blood Flow Simulation in 1D (dim=1, spacedim=3) using DG for cell
 * unknowns and global trace unknowns (A_hat, U_hat) on every face.
 *
 * Key architectural:
 *   - (A_hat, U_hat) on all faces (interior, boundary, junction) are
 *     explicit global DOFs appended after the cell block.
 *   - Junction coupling is written as residual equations for those DOFs;
 *     no nested local Newton solve is needed.
 *   - Solution vector layout: [ cell block (DGQ) | trace block ]
 * --------------------------------------------------------------------------
 */
#include "blood_flow_system.h"

#include <deal.II/base/function_parser.h>
#include <deal.II/base/types.h>

#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_direct.h>

#include <deal.II/meshworker/mesh_loop.h>

#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/vector_tools.h>

#include <algorithm>
#include <cmath>
#include <iomanip>

#include "vtk_utils.h"

// Main class implementation
template <int dim, int spacedim>
BloodFlowSystem<dim, spacedim>::BloodFlowSystem()
  : par("Blood Flow Parameters",
        {"rho", "mu", "xi", "m", "Rt"},
        {1060, 0.004, 2.0, 0.5, 0.5},
        {"Density",
         "Viscosity coefficient",
         "Profile constant for friction term",
         "Tube law exponent",
         "Reflection coefficient at outflow boundary"})
  , triangulation()
  , dof_handler(triangulation)
  , fe(nullptr)
  , rhs_function("Functions",
                 "0.0; 0.0",
                 "RHS expression",
                 par,
                 dealii::FunctionParser<spacedim>::default_variable_names() +
                   ",t")
  , exact_solution("Functions",
                   "1e-4; 0.0",
                   "Exact solution",
                   par,
                   dealii::FunctionParser<spacedim>::default_variable_names() +
                     ",t")
  , inflow_function("Functions", "0.0", "Inflow function", par, "x,t")
  , computing_timer(std::cout, TimerOutput::summary, TimerOutput::wall_times)
{
  add_parameter("Finite element degree", fe_degree);
  add_parameter("Problem constants", constants);
  add_parameter("Output filename", output_filename);
  add_parameter("Use direct solver", use_direct_solver);
  add_parameter("Number of refinement cycles", n_refinement_cycles);
  add_parameter("Number of global refinement", n_global_refinements);
  add_parameter("Theta (penalty parameter)", theta);
  add_parameter("Theta Boundary (stability parameter)", theta_bd);
  add_parameter("Verbosity (console depth)", verbosity);
  add_parameter("Output directory", output_directory);
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

  deallog.depth_console(verbosity);
  deallog.depth_file(verbosity);

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

// ============================================================================
// detect_junctions
// ============================================================================
template <int dim, int spacedim>
void
BloodFlowSystem<dim, spacedim>::detect_junctions()
{
  junctions.clear();
  all_junction_faces.clear();

  // vertex -> list of (cell, local_face_no) pairs
  std::map<unsigned int,
           std::vector<
             std::pair<typename DoFHandler<dim, spacedim>::active_cell_iterator,
                       unsigned int>>>
    vertex_to_half_faces;

  for (const auto &cell : dof_handler.active_cell_iterators())
    for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v)
      vertex_to_half_faces[cell->vertex_index(v)].emplace_back(cell, v);

  for (const auto &[v_idx, half_faces] : vertex_to_half_faces)
    {
      const std::size_t n_inc = half_faces.size();

      // Classify the face types based on how many cells meet at the vertex:
      // if n_inc == 1                -> boundary (inlet / terminal).  Skip.
      //  n_inc == 2, same vessel id  -> ordinary interior face skiping.
      //  n_inc == 2, but different ids -> two vessels joined end-to-end (an
      //                                    "in-line" connection). Treat as junction like 2 way junction.
      // n_inc >= 3                  ->   Junction.
      if (n_inc == 1)
        continue; // boundary

      if (n_inc == 2)
        {
          const unsigned int vid_a = half_faces[0].first->material_id();
          const unsigned int vid_b = half_faces[1].first->material_id();
          if (vid_a == vid_b)
            continue; // same vessel -> ordinary interior face, not a junction
          
        }

      JunctionInfo J;
      J.location = triangulation.get_vertices()[v_idx];

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
    }
  std::cout << "Detected " << junctions.size()
            << " junctions (including 2-way in-line vessel connections)."
            << std::endl;
}


// ============================================================================
// canonical_face_key :How faces are identified functions
//
// Interior face: key = the half-face whose cell has the smaller CellId.
// Boundary / junction face:  key = the unique owning cell.
// ============================================================================
template <int dim, int spacedim>
std::pair<CellId, unsigned int>
BloodFlowSystem<dim, spacedim>::canonical_face_key(
  const typename DoFHandler<dim, spacedim>::active_cell_iterator &cell,
  const unsigned int                                              face_no) const
{
  if (cell->face(face_no)->at_boundary() ||
      is_junction_face(cell->id(), face_no))
    return {cell->id(), face_no};
  const auto        &nb         = cell->neighbor(face_no);
  const unsigned int nb_face_no = cell->neighbor_of_neighbor(
    face_no); // function finds which local face of the neighboring cell
              // connects back to the current cell

  if (cell->id() < nb->id())
    return {cell->id(), face_no};

  return {nb->id(), nb_face_no};
}

// ============================================================================
// build_face_dof_map : For every face-
// what are the global indices of (A_hat,U_hat)
//
// Assigns two consecutive global DOF indices to every unique face:
//   index 2k   -> A_hat
//   index 2k+1 -> U_hat
// Indices start at n_cell_dofs (== dof_handler.n_dofs()).
// ============================================================================
template <int dim, int spacedim>
void
BloodFlowSystem<dim, spacedim>::build_face_dof_map()
{
  TimerOutput::Scope timer(computing_timer, "build_face_dof_map");

  face_dof_map.clear();
  n_cell_dofs  = dof_handler.n_dofs();
  n_trace_dofs = 0;

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
        {
          const auto key = canonical_face_key(cell, f);

          if (face_dof_map.count(key))
            continue; // already registered from the other side

          const types::global_dof_index a_idx = n_cell_dofs + n_trace_dofs;
          const types::global_dof_index u_idx = n_cell_dofs + n_trace_dofs + 1;

          face_dof_map[key] = FaceTraceDof{a_idx, u_idx};
          n_trace_dofs += 2;
        }
    }

  n_total_dofs = n_cell_dofs + n_trace_dofs;

  std::cout << "Face DOF map: n_cell=" << n_cell_dofs
            << "  n_trace=" << n_trace_dofs << "  n_total=" << n_total_dofs
            << std::endl;
}

// ==========================================================================
// build rcr dof map : For every RCR terminal boundary, assign a global DOF index to the capacitor pressure Pc.
// =========================================================================
template <int dim, int spacedim>
void
BloodFlowSystem<dim, spacedim>::build_rcr_dof_map()
{
  rcr_pc_dof.clear();
  n_trace_end = n_cell_dofs + n_trace_dofs;
  n_rcr_dofs  = 0;

  if (outlet_type == "RCR")
    {
      std::set<types::boundary_id> seen;
      for (const auto &cell : dof_handler.active_cell_iterators())
        for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
          {
            if (!cell->face(f)->at_boundary())
              continue;
            if (is_junction_face(cell->id(), f))
              continue;
            const types::boundary_id bid = cell->face(f)->boundary_id();
            if (bid == 0)
              continue; // inflow
            if (!rcr_map.count(bid))
              continue;
            if (rcr_map.at(bid).C <= 0.0)
              continue; // single-R: no capacitor unknown
            if (!seen.insert(bid).second)
              continue;

            rcr_pc_dof[bid] = n_trace_end + n_rcr_dofs;
            ++n_rcr_dofs;
          }
    }

  n_total_dofs = n_trace_end + n_rcr_dofs;
  std::cout << "RCR capacitor DOFs: n_rcr=" << n_rcr_dofs
            << "  n_total=" << n_total_dofs << std::endl;
}

// ============================================================================
// get_face_trace : it extracts trace values from global HDG solution
// ============================================================================
template <int dim, int spacedim>
void
BloodFlowSystem<dim, spacedim>::get_face_trace(
  const Vector<double>                                           &y,
  const typename DoFHandler<dim, spacedim>::active_cell_iterator &cell,
  const unsigned int                                              face_no,
  double                                                         &A_hat,
  double                                                         &U_hat) const
{
  const auto key = canonical_face_key(cell, face_no);

  const auto it = face_dof_map.find(key);
  Assert(it != face_dof_map.end(),
         ExcMessage("Face not found in face_dof_map."));

  A_hat = y[it->second.a_hat_dof];
  U_hat = y[it->second.u_hat_dof];
}

// ============================================================================
// build_extended_sparsity_pattern: it constructs the sparsity pattern of the
// Jacobian J = (J_cc  J_ct;   J_tc J_tt) where c=cell dofs, t=trace dofs
//
// Builds a DynamicSparsityPattern for the full n_total × n_total system.
// Coupling rules:
//   Cell i <-> cell j       if they share a face  (standard DG)
//   Cell i <-> trace (f)    for every face f of cell i
//   Trace (f) <-> cell i    for every cell incident to face f
//   Trace (f) <-> trace (g) for every pair of traces sharing a junction vertex
// ============================================================================
template <int dim, int spacedim>
void
BloodFlowSystem<dim, spacedim>::build_extended_sparsity_pattern()
{
  TimerOutput::Scope timer(computing_timer, "build_extended_sparsity_pattern");

  DynamicSparsityPattern dsp(n_total_dofs, n_total_dofs);

  // ---- (1a) Cell–cell coupling via make_flux_sparsity_pattern ---------------
  // make_flux_sparsity_pattern requires sparsity.n_rows() ==
  // dof_handler.n_dofs()
  // (== n_cell_dofs).  Build it on a separate cell-sized DSP and copy entries
  // into the extended one so the row/column indices stay in [0, n_cell_dofs).
  {
    DynamicSparsityPattern cell_dsp(n_cell_dofs, n_cell_dofs);
    DoFTools::make_flux_sparsity_pattern(dof_handler,
                                         cell_dsp); // standard DG sparsity

    for (const auto &entry : cell_dsp)
      dsp.add(entry.row(), entry.column());
  }

  // ---- (1b) Cell–trace and trace–trace couplings ---------------------------
  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      std::vector<types::global_dof_index> cell_dofs(fe->n_dofs_per_cell());
      cell->get_dof_indices(cell_dofs);

      for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
        {
          const auto key = canonical_face_key(cell, f);
          const auto it  = face_dof_map.find(key);
          Assert(it != face_dof_map.end(), ExcInternalError());

          const types::global_dof_index a_hat = it->second.a_hat_dof;
          const types::global_dof_index u_hat = it->second.u_hat_dof;

          // Cell DOFs <-> their face trace DOFs (both blocks, both directions)
          for (const types::global_dof_index ci : cell_dofs)
            {
              dsp.add(ci, a_hat);
              dsp.add(ci, u_hat);
              dsp.add(a_hat, ci);
              dsp.add(u_hat, ci);
            }

          // Trace self-coupling (needed for trace–trace Jacobian rows)
          dsp.add(a_hat, a_hat);
          dsp.add(a_hat, u_hat);
          dsp.add(u_hat, a_hat);
          dsp.add(u_hat, u_hat);

          // For interior faces: trace rows also couple to the neighbour's cell
          // DOFs (trace equation R uses cell values from both sides)
          if (!cell->face(f)->at_boundary())
            {
              const auto                          &nb = cell->neighbor(f);
              std::vector<types::global_dof_index> nb_dofs(
                fe->n_dofs_per_cell());
              nb->get_dof_indices(nb_dofs);

              for (const types::global_dof_index ni : nb_dofs)
                {
                  dsp.add(a_hat, ni); // J(R_t, w_{l/r})
                  dsp.add(u_hat, ni);
                }
            }
        }
    }

  // ---- (2) Junction: all K trace-pairs couple to each other ----------------
  for (const auto &J : junctions)
    {
      const unsigned int K = J.n_vessels();

      std::vector<std::pair<types::global_dof_index, types::global_dof_index>>
        tdofs(K);

      for (unsigned int i = 0; i < K; ++i)
        {
          const auto key =
            canonical_face_key(J.half_faces[i].cell, J.half_faces[i].face_no);
          const auto it = face_dof_map.find(key);
          Assert(it != face_dof_map.end(), ExcInternalError());
          tdofs[i] = {it->second.a_hat_dof, it->second.u_hat_dof};
        }

      for (unsigned int i = 0; i < K; ++i)
        for (unsigned int j = 0; j < K; ++j)
          {
            dsp.add(tdofs[i].first, tdofs[j].first);
            dsp.add(tdofs[i].first, tdofs[j].second);
            dsp.add(tdofs[i].second, tdofs[j].first);
            dsp.add(tdofs[i].second, tdofs[j].second);
          }
    }

    // ---- (3) RCR capacitor DOFs: couple to their face traces ----------------
  for (const auto &cell : dof_handler.active_cell_iterators())
    for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
      {
        if (!cell->face(f)->at_boundary())
          continue;
        if (is_junction_face(cell->id(), f))
          continue;
        const auto pit = rcr_pc_dof.find(cell->face(f)->boundary_id());
        if (pit == rcr_pc_dof.end())
          continue;

        const types::global_dof_index pc = pit->second;
        const auto &td = face_dof_map.at(canonical_face_key(cell, f));

        dsp.add(pc, pc);
        dsp.add(pc, td.a_hat_dof); // dR_pc/dÂ
        dsp.add(pc, td.u_hat_dof); // dR_pc/dÛ
        dsp.add(td.a_hat_dof, pc); // res_A now depends on Pc
      }
  sparsity_pattern.copy_from(dsp);
}

// ============================================================================
// setup_system
// ============================================================================
template <int dim, int spacedim>
void
BloodFlowSystem<dim, spacedim>::setup_system()
{
  TimerOutput::Scope timer(computing_timer, "setup_system");

  // ---- vessel map ----------------------------------------------------------
  vessel_map.clear();
  for (const auto &cell : triangulation.active_cell_iterators())
    {
      const unsigned int vid = cell->material_id();
      if (vessel_map.count(vid))
        continue;
      VesselPhysicalProperties vp;
      vp.a0           = cell_a0[vid];
      vp.E            = cell_E[vid];
      vp.h_wall       = cell_h_wall[vid];
      vp.p_d          = cell_p_d[vid];
      vp.p0           = cell_p0[vid];
      vp.a_d          = cell_a_d[vid];
      vp.L            = cell_L[vid];
      vp.r_d          = cell_r_d[vid];
      vp.r_in  = (cell_r_in.size() > vid) ? cell_r_in[vid] : 0.0;
      vp.r_out = (cell_r_out.size() > vid) ? cell_r_out[vid] : 0.0;

      vessel_map[vid] = vp;
    }

  
  // ---- vessel arc-length bounds ---------------------------------------
  vessel_s_bounds.clear();

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      const unsigned int vid = cell->material_id();

      const Tensor<1, spacedim> d_hat = compute_directional_vector(cell);

      const double s0 = cell->vertex(0) * d_hat;
      const double s1 = cell->vertex(1) * d_hat;

      const double smin_cell = std::min(s0, s1);
      const double smax_cell = std::max(s0, s1);

      if (!vessel_s_bounds.count(vid))
        vessel_s_bounds[vid] = std::make_pair(smin_cell, smax_cell);
      else
        {
          vessel_s_bounds[vid].first =
            std::min(vessel_s_bounds[vid].first, smin_cell);

          vessel_s_bounds[vid].second =
            std::max(vessel_s_bounds[vid].second, smax_cell);
        }
    }
  // ---- FE space (cell unknowns only) ---------------------------------------
  if (!fe)
    fe = std::make_unique<FESystem<dim, spacedim>>(
      FE_DGQ<dim, spacedim>(fe_degree), 2);

  dof_handler.distribute_dofs(*fe);

  // ---- junction detection + face DOF map -----------------------------------
  junctions.clear();
  all_junction_faces.clear();
  detect_junctions();
  build_face_dof_map();
  build_rcr_dof_map();
  // ---- sparsity + matrices --------------------------------------------------
  build_extended_sparsity_pattern();

  jacobian_matrix.reinit(sparsity_pattern);
  linear_system_matrix.reinit(sparsity_pattern);

  // Solution vector covers cell + trace DOFs
  solution.reinit(n_total_dofs);
  solution_dot.reinit(n_total_dofs);
  pressure.reinit(n_total_dofs);
  theoretical_peak.reinit(n_total_dofs);

  // ---- terminal boundary IDs -----------------------------------------------
  terminal_boundary_ids.clear();
  for (const auto &cell : triangulation.active_cell_iterators())
    for (unsigned int f : cell->face_indices())
      if (cell->face(f)->at_boundary())
        {
          const unsigned int bid = cell->face(f)->boundary_id();
          if (rcr_map.count(bid))
            terminal_boundary_ids.insert(bid);
        }
  if (outlet_type == "RCR")
    AssertThrow(!terminal_boundary_ids.empty(),
                ExcMessage("No terminal boundaries found."));

}

// ============================================================================
// initialize_terminal_capacitors
// ============================================================================
template <int dim, int spacedim>
void
BloodFlowSystem<dim, spacedim>::initialize_terminal_capacitors()
{
  terminal_Pc_storage.clear();

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
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
            {
              terminal_Pc_storage.try_emplace(bid, vpp.p_d);
              std::cout << "  [Init] Boundary " << bid << " (Vessel " << vid
                        << ") Pc = " << vpp.p_d << " Pa\n";
            }
        }
    }
}

// ============================================================================
// compute_initial_solution
// ============================================================================
template <int dim, int spacedim>
void
BloodFlowSystem<dim, spacedim>::compute_initial_solution(Vector<double> &dst,
                                                         const double /*t*/)
{
  TimerOutput::Scope timer(computing_timer, "compute_initial_solution");

  dst.reinit(n_total_dofs);

  // ---- cell block -----------------------------------------------------------
  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      // const unsigned int vid = cell->material_id();
      // const auto        &vpp = vessel_map.at(vid);
      const double a_d_local = compute_a_d_local(cell);
      std::vector<types::global_dof_index> ldofs(fe->n_dofs_per_cell());
      cell->get_dof_indices(ldofs);

      for (unsigned int i = 0; i < fe->n_dofs_per_cell(); ++i)
        {
          const unsigned int comp = fe->system_to_component_index(i).first;
          dst[ldofs[i]]           = (comp == 0) ? a_d_local : 0.0;
        }
    }

  // ---- trace block --------------------------------------------------------
  std::set<std::pair<CellId, unsigned int>> visited;
  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      // const unsigned int vid = cell->material_id();
      // const auto        &vpp = vessel_map.at(vid);
      
      for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
        {
          const auto key = canonical_face_key(cell, f);
          if (visited.count(key))
            continue;
          visited.insert(key);

          const FaceTraceDof &td = face_dof_map.at(key);

          if (cell->face(f)->at_boundary() && !is_junction_face(cell->id(), f))
            {
              const types::boundary_id bid = cell->face(f)->boundary_id();
              const double ad_local = compute_a_d_at_face(cell, f);
              if (bid == 0) // inflow: A_hat*U_hat = Q_in(0)
                {
                  dst[td.a_hat_dof] = ad_local;
                  dst[td.u_hat_dof] = 0.0; // consistent velocity
                }
              else // RCR or reflection outlet: already consistent with U=0
                {
                  dst[td.a_hat_dof] = ad_local;
                  dst[td.u_hat_dof] = 0.0;
                }
            }
          else if (!cell->face(f)->at_boundary() &&
                   !is_junction_face(cell->id(), f))
            {
              const double ad_local = compute_a_d_local(cell);
              // Interior: average area, zero velocity (flux = A*U*bn = 0 ✓)
              const unsigned int vid_nb = cell->neighbor(f)->material_id();
              dst[td.a_hat_dof] =
                0.5 * (ad_local + compute_a_d_at_face(cell->neighbor(f), f));
              dst[td.u_hat_dof] = 0.0;
            }
          else
            {
              const double ad_local = compute_a_d_at_face(cell, f);
              // Junction: each vessel uses its own diastolic area
              dst[td.a_hat_dof] = ad_local;
              dst[td.u_hat_dof] = 0.0;
            }
        }
    }

  for (const auto &cell : dof_handler.active_cell_iterators())
    for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
      {
        if (!cell->face(f)->at_boundary())
          continue;
        if (is_junction_face(cell->id(), f))
          continue;
        const auto pit = rcr_pc_dof.find(cell->face(f)->boundary_id());
        if (pit == rcr_pc_dof.end())
          continue;
        dst[pit->second] = vessel_map.at(cell->material_id()).p_d;
      }
}

// ============================================================================
// initialize_trace_unknowns
// ============================================================================
template <int dim, int spacedim>
void
BloodFlowSystem<dim, spacedim>::initialize_trace_unknowns(Vector<double> &sol,
                                                          const double    t)
{
  TimerOutput::Scope timer(computing_timer, "initialize_trace_unknowns");

  Assert(sol.size() == n_total_dofs,
         ExcDimensionMismatch(sol.size(), n_total_dofs));

  const double tol      = 1.0e-9; // ||F_trace||_inf convergence threshold
  const int    max_iter = 50;

  std::cout << "\n=== initialize_trace_unknowns (Newton) ===\n";

  // ── Private Newton matrix and solver ──────────────────────────────────────
  SparseMatrix<double> newton_matrix(sparsity_pattern);
  SparseDirectUMFPACK  newton_solver;

  for (int iter = 0; iter < max_iter; ++iter)
    {
      // ── Step 1 ─ Assemble G(yhat) = F_trace(y_cell^0, yhat) ───────────────────
      Vector<double> G(n_total_dofs); // zero-initialised
      Vector<double> zero_ydot(n_total_dofs);
      assemble_trace_interior_equations(sol, G);
      assemble_trace_boundary_equations(t, sol, G);
      assemble_trace_junction_equations(sol, G);
      assemble_rcr_capacitor_equations(sol, zero_ydot, G);

      // ── Convergence check ─────────────────────────────────────────────────
      double gnorm_inf = 0.0;
      double gnorm_l2  = 0.0;
      for (types::global_dof_index i = n_cell_dofs; i < n_total_dofs; ++i)
        {
          const double val = std::abs(G[i]);
          gnorm_inf        = std::max(gnorm_inf, val);
          gnorm_l2 += val * val;
        }
      gnorm_l2 = std::sqrt(gnorm_l2);

      std::cout << "  iter " << std::setw(3) << iter
                << "  ||G||_inf = " << std::scientific << std::setprecision(4)
                << gnorm_inf << "  ||G||_l2 = " << gnorm_l2 << "\n";

      if (gnorm_inf < tol)
        {
          std::cout << "  Converged in " << iter << " Newton iteration(s).\n";
          break;
        }

      if (iter == max_iter - 1)
        {
          std::cerr
            << "WARNING: initialize_trace_unknowns did not converge.\n"
            << "         tol=" << tol << "  ||G||_inf=" << gnorm_inf
            << "  after " << max_iter << " iterations.\n"
            << "         The DAE integrator may still recover via its own\n"
            << "         consistent-IC step, but check your initial data.\n";
          break;
        }

      // ── Step 2 ─ Assemble J_tt = dF_trace/dyhat ─────────────────────────────
      newton_matrix = 0.0;

      assemble_jacobian_trace_interior_block(
        sol); // writes into jacobian_matrix
      assemble_jacobian_trace_boundary_block(t, sol);
      assemble_jacobian_trace_junction_block(sol);
      assemble_jacobian_rcr_capacitor_block(sol);
      for (types::global_dof_index i = n_cell_dofs; i < n_total_dofs; ++i)
        {
          const bool is_pc_row = (i >= n_trace_end);
          // Iterate over all columns with non-zero entries in this row
          for (auto it = jacobian_matrix.begin(i); it != jacobian_matrix.end(i);
               ++it)
            {
              double val = it->value();
              if (is_pc_row)
                val = -val; // undo sign convention meant for
                            // assemble_jacobian's global *=-1
              newton_matrix.add(i, it->column(), val);
            }
        }

      // Stamp cell diagonal with 1 so UMFPACK does not encounter zero pivots
      for (const auto &cell : dof_handler.active_cell_iterators())
        {
          std::vector<types::global_dof_index> ldofs(fe->n_dofs_per_cell());
          cell->get_dof_indices(ldofs);
          for (const auto idx : ldofs)
            newton_matrix.set(idx, idx, 1.0);
        }

  
      jacobian_matrix = 0.0;

      // ── Step 3 ─ Build Newton RHS: b = −G ─────────────────────────────────
      //
      //   newton_matrix · Δy = b
      //
      //   b[i] = −G[i]   for trace DOFs  (i ≥ n_cell_dofs)
      //   b[i] = 0        for cell DOFs   (i <  n_cell_dofs)
      //                   → identity rows give Δy_cell = 0
      Vector<double> rhs(n_total_dofs); // zero-initialised
      for (types::global_dof_index i = n_cell_dofs; i < n_total_dofs; ++i)
        rhs[i] = -G[i];

      // ── Step 4 ─ Solve and update trace DOFs ──────────────────────────────
      newton_solver.initialize(newton_matrix);
      Vector<double> delta(n_total_dofs);
      newton_solver.vmult(delta, rhs);

      // Apply correction only to trace DOFs; cell DOFs are never changed.
      for (types::global_dof_index i = n_cell_dofs; i < n_total_dofs; ++i)
        sol[i] += delta[i];
    } // end Newton loop

  // ── Step 5 ─ Final per-type diagnostic ────────────────────────────────────

  {
    Vector<double> F_final(n_total_dofs);
    Vector<double> zero_ydot(n_total_dofs);
    assemble_trace_interior_equations(sol, F_final);
    assemble_trace_boundary_equations(t, sol, F_final);
    assemble_trace_junction_equations(sol, F_final);
    assemble_rcr_capacitor_equations(sol, zero_ydot, F_final);
    double g_int = 0.0, g_bnd = 0.0, g_jnc = 0.0; 

    std::set<std::pair<CellId, unsigned int>> visited;
    for (const auto &cell : dof_handler.active_cell_iterators())
      for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
        {
          const auto key = canonical_face_key(cell, f);
          if (visited.count(key))
            continue;
          visited.insert(key);

          const FaceTraceDof &td = face_dof_map.at(key);
          const double        ra = std::abs(F_final[td.a_hat_dof]);
          const double        ru = std::abs(F_final[td.u_hat_dof]);
          const double        r  = std::max(ra, ru);

          if (is_junction_face(cell->id(), f))
            g_jnc = std::max(g_jnc, r);
          else if (cell->face(f)->at_boundary())
            g_bnd = std::max(g_bnd, r);
          else
            g_int = std::max(g_int, r);
        }
        double g_pc = 0.0;
    for (const auto &[bid, pc_dof] : rcr_pc_dof)
      g_pc = std::max(g_pc, std::abs(F_final[pc_dof]));
    std::cout << "    capacitor = " << g_pc << "\n";

    std::cout << "  Final ||F_trace||_inf by type:\n"
              << "    interior  = " << std::scientific << std::setprecision(4)
              << g_int << "\n"
              << "    boundary  = " << g_bnd << "\n"
              << "    junction  = " << g_jnc << "\n"
              << "==========================================\n\n";
  }
}

// ============================================================================
// build_per_cell_mass
//
// For every active cell K, compute both:
//   per_cell_mass_    : forward M_K  — used in assemble_residual (M*ydot term)
//                       and assemble_jacobian (alpha*M term)
//   per_cell_mass_inv : M_K^{-1}    — used in computing consistent
//                       initial solution_dot = M_K^{-1} * F_cell(y0)
//
// IDA receives the true DAE residual F(t,y,ydot) = M*ydot - R(y) directly.
// ============================================================================

template <int dim, int spacedim>
void
BloodFlowSystem<dim, spacedim>::build_per_cell_mass_inv()
{
  TimerOutput::Scope timer(computing_timer, "build_per_cell_mass_inv");

  per_cell_mass_inv.clear();

  const QGauss<dim>       quad(fe_degree + 1);
  FEValues<dim, spacedim> fev(*fe, quad, update_values | update_JxW_values);

  const unsigned int n_dofs = fe->n_dofs_per_cell();

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      fev.reinit(cell);
      FullMatrix<double> M(n_dofs, n_dofs);

      // FESystem(FE_DGQ, 2) shape functions are component-specific:
      // phi_i is nonzero only for its component, so M(i,j) = 0 whenever
      // component(i)!= component(j).  We must skip cross-component pairs;
      // otherwise M has zero rows/columns and gauss_jordan() aborts.

      for (unsigned int q = 0; q < fev.n_quadrature_points; ++q)
        for (unsigned int i = 0; i < n_dofs; ++i)
          {
            const unsigned int ci = fe->system_to_component_index(i).first;
            for (unsigned int j = 0; j < n_dofs; ++j)
              {
                const unsigned int cj = fe->system_to_component_index(j).first;
                if (ci != cj)
                  continue; // cross-component integral is identically zero
                M(i, j) +=
                  fev.shape_value(i, q) * fev.shape_value(j, q) * fev.JxW(q);
              }
          }

      // Invert in place (small dense block-diagonal matrix)
      per_cell_mass_[cell->id()] = M;
      M.gauss_jordan();
      per_cell_mass_inv[cell->id()] = std::move(M);
    }
}

// ============================================================================
// CSV probe output for 37-artery benchmark validation
// (Boileau et al. 2015 / Alastruey et al. 2011)
//
// Outputs P [kPa] and Q [ml/s] at the midpoint of 8 selected vessels.
//
// Add these members to blood_flow_system.h:
//
//   std::ofstream csv_P_, csv_Q_;
//   std::vector<std::pair<unsigned int, Point<spacedim>>> probe_targets_;
//   // (vessel_id, coarse_midpoint) for each probe
//
//   void open_csv_files();
//   void write_csv_row(const double t, const Vector<double> &sol);
//   void close_csv_files();
// ============================================================================

// ---------- Probe definitions -----------------------------------------------
// vessel_id from the 37-artery VTK, label for CSV header
struct ProbeSpec
{
  unsigned int vessel_id;
  const char  *label;
  double       mid_x, mid_y, mid_z; // coarse-mesh midpoint (from VTK)
};

// static const std::vector<ProbeSpec> PROBES_37 = {
//   {9, "AorticArchII", -0.106500, -0.108000, 0.0},
//   {16, "ThoracicAortaII", 0.000000, -0.384000, 0.0},
//   {10, "LSubclavianI", -0.456450, -0.142050, 0.0},
//   {28, "RIliacFemoralII", 0.624900, -1.668300, 0.0},
//   {13, "LUlnar", -1.435950, -0.314550, 0.0},
//   {33, "RAnteriorTibial", 1.275000, -2.560650, 0.0},
//   {6, "RUlnar", 1.303950, -0.302550, 0.0},
//   {20, "Splenic", 0.3820, -0.4347, 0.0},       // avg of vessel 20 and 21 midpoints
// };

static const std::vector<ProbeSpec> PROBES_37 = {
  {0, "AorticArchI", 0.000000, -0.037207, 0.0},
  {35, "ThoAortaIII", 0.013168, -0.158469, 0.0},
  {54, "AbdomAortaV", 0.117062, -0.377010, 0.0},
  {4, "RightCCA", 0.065320, -0.127091, 0.0},
  {51, "RightRenal", 0.092285, -0.322537, 0.0},
  {55, "RightCommonIliac", 0.153520, -0.423788, 0.0},
  {15, "RightICA", 0.166650, -0.160293, 0.0},
  {9, "RightRadial", 0.434901, -0.451982, 0.0},
  {58, "RightInternalIliac", 0.212420, -0.463480, 0.0},
  {13, "RightPostInteross", 0.438728, -0.462092, 0.0},
  {61, "RightFemoralII", 0.379985, -0.665805, 0.0},
  {63, "RightAntTibial", 0.776402, -0.905581, 0.0},
};

// ---------- open_csv_files --------------------------------------------------
template <int dim, int spacedim>
void
BloodFlowSystem<dim, spacedim>::open_csv_files()
{
  const std::string dir =
    output_directory + (output_directory.empty() ? "" : "/");

  // Build header
  std::string hdr = "time_s";
  for (const auto &p : PROBES_37)
    hdr += std::string(",") + p.label;
  hdr += "\n";

  csv_P_.open(dir + "HDG_IDA_P.csv");
  csv_P_ << hdr;

  csv_Q_.open(dir + "HDG_IDA_Q.csv");
  csv_Q_ << hdr;

  // Build probe_targets_: for each probe, find the refined cell whose
  // midpoint is closest to the coarse midpoint AND has the right material_id.
  probe_targets_.clear();
  for (const auto &spec : PROBES_37)
    {
      Point<spacedim> target;
      target[0] = spec.mid_x;
      target[1] = spec.mid_y;
      if (spacedim > 2)
        target[2] = spec.mid_z;

      probe_targets_.emplace_back(spec.vessel_id, target);
    }
}

// ---------- write_csv_row ---------------------------------------------------
// Call this at each output timestep after the solution is updated.
template <int dim, int spacedim>
void
BloodFlowSystem<dim, spacedim>::write_csv_row(const double          t,
                                              const Vector<double> &sol)
{
  csv_P_ << std::scientific << std::setprecision(8) << t;
  csv_Q_ << std::scientific << std::setprecision(8) << t;

  const unsigned int dofs_per_cell = fe->n_dofs_per_cell();

  for (const auto &[target_vid, target_pt] : probe_targets_)
    {
      // Find the refined cell closest to target_pt with matching material_id
      double best_dist = std::numeric_limits<double>::max();
      typename DoFHandler<dim, spacedim>::active_cell_iterator best_cell;
      bool                                                     found = false;

      for (const auto &cell : dof_handler.active_cell_iterators())
        {
          if (cell->material_id() != target_vid)
            continue;

          const Point<spacedim> mid = cell->center();
          const double          d   = mid.distance(target_pt);
          if (d < best_dist)
            {
              best_dist = d;
              best_cell = cell;
              found     = true;
            }
        }

      if (!found)
        {
          csv_P_ << ",NaN";
          csv_Q_ << ",NaN";
          continue;
        }

      // Evaluate (A, U) at cell midpoint using DG shape functions
      // For DGQ1: midpoint is at reference coordinate xi = 0.5
      const Point<dim> xi_mid = (dim == 1) ? Point<dim>(0.5) : Point<dim>();

      std::vector<types::global_dof_index> ldofs(dofs_per_cell);
      best_cell->get_dof_indices(ldofs);

      // Evaluate each shape function at xi_mid
      double A_val = 0.0, U_val = 0.0;
      for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
          const double       phi  = fe->shape_value(i, xi_mid);
          const unsigned int comp = fe->system_to_component_index(i).first;
          if (comp == 0) // area
            A_val += sol[ldofs[i]] * phi;
          else // velocity
            U_val += sol[ldofs[i]] * phi;
        }

      // Compute pressure [Pa] and convert to [kPa]
      const unsigned int vid   = best_cell->material_id();
      // const double       P_Pa  = compute_pressure_value(A_val, vid);
      const double       P_Pa  = compute_pressure_value(A_val, vid, compute_a_d_local(best_cell));
      const double       P_kPa = P_Pa * 1.0e-3;
     // Compute flow rate Q = A * U [m³/s] and convert to [ml/s]
      // 1 m³/s = 1e6 ml/s
      
      const double Q_m3s  = A_val * U_val;
      const double Q_mlps = Q_m3s * 1.0e6;

      csv_P_ << "," << P_kPa;
      csv_Q_ << "," << Q_mlps;
    }

  csv_P_ << "\n";
  csv_Q_ << "\n";
  csv_P_.flush();
  csv_Q_.flush();
}

// ---------- close_csv_files -------------------------------------------------
template <int dim, int spacedim>
void
BloodFlowSystem<dim, spacedim>::close_csv_files()
{
  if (csv_P_.is_open())
    csv_P_.close();
  if (csv_Q_.is_open())
    csv_Q_.close();
}

  // ============================================================================
  // HLL flux (residual)
  // ============================================================================
  template <int dim, int spacedim>
  std::array<double, 2> BloodFlowSystem<dim, spacedim>::hll_flux(
    const double       bn_L,
    const double       bn_R,
    const double       A_L,
    const double       U_L,
    const double       A_R,
    const double       U_R,
    const unsigned int vid_L,
    const unsigned int vid_R,
  const double ad_L,
  const double ad_R) const
  {
    const double c_L   = compute_wave_speed(A_L, vid_L, ad_L);
    const double c_R   = compute_wave_speed(A_R, vid_R, ad_R);
    // const double U_bar = 0.5 * (U_L + U_R);
    // const double c_bar = 0.5 * (c_L + c_R);
    // const double s_L   = U_bar - c_bar;
    // const double s_R   = U_bar + c_bar;
    const double s_L = std::min(U_L - c_L, U_R - c_R);

    const double s_R = std::max(U_L + c_L, U_R + c_R);

    const double FAL = scalar_area_flux(bn_L, A_L, U_L);
    const double FUL = scalar_momentum_flux(bn_L,
                                            U_L,
                                            compute_pressure_value(A_L, vid_L, ad_L),
                                            par["rho"]);
    const double FAR = scalar_area_flux(bn_R, A_R, U_R);
    const double FUR = scalar_momentum_flux(bn_R,
                                            U_R,
                                            compute_pressure_value(A_R, vid_R, ad_R),
                                            par["rho"]);

    if (s_L >= 0.0)
      return {{FAL, FUL}};
    if (s_R <= 0.0)
      return {{FAR, FUR}};

    const double inv = 1.0 / (s_R - s_L);
    return {{(s_R * FAL - s_L * FAR + s_R * s_L * (A_R - A_L)) * inv,
             (s_R * FUL - s_L * FUR + s_R * s_L * (U_R - U_L)) * inv}};
  }

  // ============================================================================
  // HLL flux Jacobian (linearised w.r.t. trial perturbations dA_L, dU_L, …)
  // ============================================================================
  template <int dim, int spacedim>
  std::array<double, 2> BloodFlowSystem<dim, spacedim>::hll_flux_jac(
    const double       bn_L,
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
    // const double U_bar = 0.5 * (U_L + U_R);
    // const double c_bar = 0.5 * (c_L + c_R);
    // const double s_L   = U_bar - c_bar;
    // const double s_R   = U_bar + c_bar;
    const double s_L = std::min(U_L - c_L, U_R - c_R);

    const double s_R = std::max(U_L + c_L, U_R + c_R);

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

  // template <int dim, int spacedim>
  // std::array<double, 2>
  // BloodFlowSystem<dim, spacedim>::hll_flux_jac(const double       bn_L,
  //                                              const double       bn_R,
  //                                              const double       A_L,
  //                                              const double       U_L,
  //                                              const double       A_R,
  //                                              const double       U_R,
  //                                              const double       dA_L,
  //                                              const double       dU_L,
  //                                              const double       dA_R,
  //                                              const double       dU_R,
  //                                              const unsigned int vid_L,
  //                                              const unsigned int vid_R,
  //                                              const double       ad_L,
  //                                              const double       ad_R) const
  // {
  //   const double c_L = compute_wave_speed(A_L, vid_L, ad_L);
  //   const double c_R = compute_wave_speed(A_R, vid_R, ad_R);

  //   const double s_L = std::min(U_L - c_L, U_R - c_R);
  //   const double s_R = std::max(U_L + c_L, U_R + c_R);

  //   const double c2L_over_AL = c_L * c_L / A_L;
  //   const double c2R_over_AR = c_R * c_R / A_R;

  //   // Physical-flux Jacobians (unchanged)
  //   const double FAL_j = scalar_area_flux_jac(bn_L, A_L, U_L, dA_L, dU_L);
  //   const double FUL_j =
  //     scalar_momentum_flux_jac(bn_L, c2L_over_AL, U_L, dA_L, dU_L);
  //   const double FAR_j = scalar_area_flux_jac(bn_R, A_R, U_R, dA_R, dU_R);
  //   const double FUR_j =
  //     scalar_momentum_flux_jac(bn_R, c2R_over_AR, U_R, dA_R, dU_R);

  //   // Supersonic branches: flux = one-sided physical flux, no s-dependence
  //   if (s_L >= 0.0)
  //     return {{FAL_j, FUL_j}};
  //   if (s_R <= 0.0)
  //     return {{FAR_j, FUR_j}};

  //   // ---- Linearise the wave speeds (the previously-frozen part) ----
  //   const double dc_L = compute_wave_speed_derivative(A_L, vid_L, ad_L) * dA_L;
  //   const double dc_R = compute_wave_speed_derivative(A_R, vid_R, ad_R) * dA_R;

  //   // s_L = min(U_L - c_L, U_R - c_R) : pick the branch that is actually the
  //   // min
  //   const double ds_L =
  //     ((U_L - c_L) <= (U_R - c_R)) ? (dU_L - dc_L) : (dU_R - dc_R);

  //   // s_R = max(U_L + c_L, U_R + c_R) : pick the branch that is actually the
  //   // max
  //   const double ds_R =
  //     ((U_L + c_L) >= (U_R + c_R)) ? (dU_L + dc_L) : (dU_R + dc_R);

  //   const double D   = s_R - s_L;
  //   const double inv = 1.0 / D;
  //   const double dD  = ds_R - ds_L;

  //   // ---- Area component ----
  //   {
  //     // Physical fluxes F at current state (needed for ds_R*F_L - ds_L*F_R
  //     // term)
  //     const double FAL = scalar_area_flux(bn_L, A_L, U_L);
  //     const double FAR = scalar_area_flux(bn_R, A_R, U_R);

  //     const double N  = s_R * FAL - s_L * FAR + s_R * s_L * (A_R - A_L);
  //     const double dN = ds_R * FAL + s_R * FAL_j - ds_L * FAR - s_L * FAR_j +
  //                       (ds_R * s_L + s_R * ds_L) * (A_R - A_L) +
  //                       s_R * s_L * (dA_R - dA_L);

  //     const double dFA = (dN * D - N * dD) * inv * inv;

  //     // ---- Momentum component ----
  //     const double FUL = scalar_momentum_flux(
  //       bn_L, U_L, compute_pressure_value(A_L, vid_L, ad_L), par["rho"]);
  //     const double FUR = scalar_momentum_flux(
  //       bn_R, U_R, compute_pressure_value(A_R, vid_R, ad_R), par["rho"]);

  //     const double M  = s_R * FUL - s_L * FUR + s_R * s_L * (U_R - U_L);
  //     const double dM = ds_R * FUL + s_R * FUL_j - ds_L * FUR - s_L * FUR_j +
  //                       (ds_R * s_L + s_R * ds_L) * (U_R - U_L) +
  //                       s_R * s_L * (dU_R - dU_L);

  //     const double dFU = (dM * D - M * dD) * inv * inv;

  //     return {{dFA, dFU}};
  //   }
  // }

  // HLL-HDG flux based on Paper "Hybridisable discontinuous Galerkin
  // formulation of compressible flows "

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
    // Stabilization from TRACE state only (eq. 37 of Vila-Perez et al.)
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
    const double FUL = scalar_momentum_flux(bn_L,
                                            U_L,
                                            compute_pressure_value(A_L, vid_L, ad_L),
                                            par["rho"]);
    const double FAR = scalar_area_flux(bn_R, A_R, U_R);
    const double FUR = scalar_momentum_flux(bn_R,
                                            U_R,
                                            compute_pressure_value(A_R, vid_R, ad_R),
                                            par["rho"]);
    const double alpha =
      theta * compute_LF_penalty(A_L, A_R, U_L, U_R, bn_L, bn_R, vid_L, vid_R, ad_L, ad_R);

    return {{0.5 * (FAL + FAR) - 0.5 * alpha * (A_R - A_L),
             0.5 * (FUL + FUR) - 0.5 * alpha * (U_R - U_L)}};
  }

  // ============================================================================
  // Lax–Friedrichs flux Jacobian
  // ============================================================================
  template <int dim, int spacedim>
  std::array<double, 2> BloodFlowSystem<dim, spacedim>::lf_flux_jac(
    const double       bn_L,
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
      theta * compute_LF_penalty(A_L, A_R, U_L, U_R, bn_L, bn_R, vid_L, vid_R, ad_L, ad_R);

    return {{0.5 * (FAL_j + FAR_j) - 0.5 * alpha * (dA_R - dA_L),
             0.5 * (FUL_j + FUR_j) - 0.5 * alpha * (dU_R - dU_L)}};
  }

  // ============================================================================
  // assemble_cell_residuals
  //
  // For each cell K:
  //   R_A = \int_K [ F_A(A,U) · \gradφ ] dK
  //         − \Sigma_f  hat{F}_A(A,U ; A_hat,U_hat) [[φ]]  (trace from
  //         face_dof_map)
  //         + source
  //   R_U similarly, with viscous friction source term.
  //
  // FEValues::get_function_values internally indexes via
  // cell->get_dof_indices(), which only produces indices in [0, n_cell_dofs).
  // We must pass a vector of exactly size n_cell_dofs; the trace block of y is
  // never needed here.
  // ============================================================================
  template <int dim, int spacedim>
  void BloodFlowSystem<dim, spacedim>::assemble_cell_residuals(
    const double t, const Vector<double> &y, Vector<double> &F)
  {
    TimerOutput::Scope timer(computing_timer, "assemble_cell_residuals");

    // Extract the cell sub-block once.  All FEValues::get_function_values calls
    // below receive y_cell (size n_cell_dofs) — consistent with dof_handler.
    Vector<double> y_cell(n_cell_dofs);
    for (types::global_dof_index i = 0; i < n_cell_dofs; ++i)
      y_cell[i] = y[i];

    const FEValuesExtractors::Scalar area_extractor(0);
    const FEValuesExtractors::Scalar velocity_extractor(1);

    const QGauss<dim>     quad_cell(fe->tensor_degree() + 1);
    const QGauss<dim - 1> quad_face(fe->tensor_degree() + 1);

    FEValues<dim, spacedim> fev(*fe,
                                quad_cell,
                                update_values | update_gradients |
                                  update_quadrature_points | update_JxW_values);
    FEFaceValues<dim, spacedim> fef(*fe,
                                    quad_face,
                                    update_values | update_JxW_values |
                                      update_normal_vectors);

    rhs_function.set_time(t);

    const double rho = par["rho"];
    const double eta = 2.0 * (par["xi"] + 2.0) * numbers::PI * par["mu"] / rho;

    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        const unsigned int vid    = cell->material_id();
        const unsigned int n_dofs = fe->n_dofs_per_cell();
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
            const double              A = std::max(A_h[q], 1e-10);
            const double              U = U_h[q];
            const double                P = compute_pressure_value(A, vid , compute_a_d_local(cell));
            //const double              P = compute_pressure_value(A, vid);
            const Tensor<1, spacedim> b = compute_directional_vector(cell);

            const double rhs_A =
              rhs_function.value(fev.get_quadrature_points()[q], 0);
            const double rhs_U =
              rhs_function.value(fev.get_quadrature_points()[q], 1);

            for (unsigned int i = 0; i < n_dofs; ++i)
              {
                const unsigned int comp =
                  fe->system_to_component_index(i).first;

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
            const auto &normals = fef.get_normal_vectors();
            const auto &JxW     = fef.get_JxW_values();
            // if (is_junction_face(cell->id(), f) && cell->material_id() == 0)
            //   {
            //     double A_h = 0, U_h = 0;
            //     get_face_trace(y, cell, f, A_h, U_h);
            //     std::cout << "JUNCTION FACE cell=" << cell->id() << " f=" << f
            //               << " vid=" << cell->material_id() << " A_hat=" << A_h
            //               << " U_hat=" << U_h << "\n";
            //   }
            // Interior cell values at the face quadrature point
            std::vector<double> Ah_q(fef.n_quadrature_points);
            std::vector<double> Uh_q(fef.n_quadrature_points);
            fef[area_extractor].get_function_values(y_cell, Ah_q);
            fef[velocity_extractor].get_function_values(y_cell, Uh_q);

            // Global face-trace values (read directly from trace block of y)
            double A_hat = 0.0, U_hat = 0.0;
            get_face_trace(y, cell, f, A_hat, U_hat);
            A_hat = std::max(A_hat, 1e-10);

            for (unsigned int q = 0; q < fef.n_quadrature_points; ++q)
              {
                const double A_in = std::max(Ah_q[q], 1e-10);
                const double U_in = Uh_q[q];
                const double bn =
                  compute_tangent_normal_product(cell, normals[q]);

                // numerical_flux designed for left/right states.
                // In current implementation there is no physical neighbour cell
                // on the other side so we are ;

                const auto [FA, FU] =
                  numerical_flux(bn, bn, A_in, U_in, A_hat, U_hat, vid, vid, ad_local, ad_face);

                for (unsigned int i = 0; i < n_dofs; ++i)
                  {
                    const unsigned int comp =
                      fe->system_to_component_index(i).first;
                    cell_rhs(i) -=
                      (comp == 0 ? FA : FU) * fef.shape_value(i, q) * JxW[q];
                  }
              }
          }

        // ---- Scatter into global F
        // -------------------------------------------
        for (unsigned int i = 0; i < n_dofs; ++i)
          F[ldofs[i]] += cell_rhs(i);
      }
  }

  // ============================================================================
  // assemble_trace_interior_equations
  //
  // For each unique interior (non-junction) face shared by cells L and R,
  // enforce conservation of numerical flux:
  //
  //   Fhat_L(W_L, What) + Fhat_R(W_R, What) = 0
  //
  // where
  //   R_A = FA_L + Fa_R
  //   R_U = FU_L + FU_R
  // and W_L, W_R are the interior states at the face from the left and right
  // cells, respectively, and W_hat is the trace state at the face (from
  // face_dof_map).
  // ============================================================================

  template <int dim, int spacedim>
  void BloodFlowSystem<dim, spacedim>::assemble_trace_interior_equations(
    const Vector<double> &y, Vector<double> &F)
  {
    TimerOutput::Scope timer(computing_timer,
                             "assemble_trace_interior_equations");

    // Cell-block sub-vector for FEFaceValues (size must equal
    // dof_handler.n_dofs())
    Vector<double> y_cell(n_cell_dofs);
    for (types::global_dof_index i = 0; i < n_cell_dofs; ++i)
      y_cell[i] = y[i];

    const FEValuesExtractors::Scalar area_extractor(0);
    const FEValuesExtractors::Scalar velocity_extractor(1);

    const QGauss<dim - 1> quad_face(1); // each 1-D face is a single 0-D point
    FEFaceValues<dim, spacedim> fef(*fe,
                                    quad_face,
                                    update_values | update_normal_vectors);

    // Avoid double assembly of the same face from left and right cells
    std::set<std::pair<CellId, unsigned int>> processed;

    for (const auto &cell : dof_handler.active_cell_iterators())
      {
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

            const auto        &nb   = cell->neighbor(f);
            const unsigned int nb_f = cell->neighbor_of_neighbor(f);

            const unsigned int vid_L = cell->material_id();
            const unsigned int vid_R = nb->material_id();
            const double       ad_L  = compute_a_d_local(cell);
            const double       ad_R  = compute_a_d_local(nb);
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

            // Left HDG type flux residuals (face integrals with interior state
            // from left cell) numerical_flux is designed for left/right states.
            // The trace acts as the exterior state, therefore use
            // opposite orientation on the trace side.
            const auto [FA_L, FU_L] =
              numerical_flux(bn_L, bn_L, A_L, U_L, A_hat, U_hat, vid_L, vid_R, ad_L, ad_face);

            // Right HDG type flux residuals (face integrals with interior state
            // from right cell)
            const auto [FA_R, FU_R] =
              numerical_flux(bn_R, bn_R, A_R, U_R, A_hat, U_hat, vid_R, vid_L, ad_R, ad_face);

            // Trace dofs
            const FaceTraceDof &td = face_dof_map.at(key);

            // HDG type transmission equations
            // sum of left and right fluxes must be zero
            F[td.a_hat_dof] += FA_L + FA_R;
            F[td.u_hat_dof] += FU_L + FU_R;
          }
      }
  }

  // ============================================================================
  // assemble_trace_boundary_equations
  //
  // Each boundary face (non-junction) gets two residual equations for its
  // trace pair (A_hat, U_hat):
  //
  //   bid == 0  (inflow):
  //     R_a = A_hat * U_hat − Q_in(t) = 0          (prescribed volumetric flow)
  //     R_u = [U_hat − 4(c_hat − c0)] − W2_int = 0 (outgoing Riemann compat.)
  //           W2_int = U_int − 4(c_int − c0)
  //
  //   bid != 0 + RCR:
  //     R_a = P(A_hat) − [R1 * A_hat*u_hat + Pc] = 0         (Windkessel
  //     pressure BC) R_u = [U_hat + 4(c_hat − c0)] − W1_int = 0 (incoming
  //     Riemann compat.)
  //           W1_int = U_int + 4(c_int − c0)
  //
  //   bid != 0 + Reflection:
  //     R_a = [U_hat + 4(c_hat − c0)] − W1_int = 0 (forward compat.)
  //     R_u = [U_hat − 4(c_hat − c0)] − W2_tgt = 0 (backward: −Rt * W1_int)
  // ============================================================================
  template <int dim, int spacedim>
  void BloodFlowSystem<dim, spacedim>::assemble_trace_boundary_equations(
    const double t, const Vector<double> &y, Vector<double> &F)
  {
    TimerOutput::Scope timer(computing_timer,
                             "assemble_trace_boundary_equations");

    // Cell-block sub-vector for FEFaceValues
    Vector<double> y_cell(n_cell_dofs);
    for (types::global_dof_index i = 0; i < n_cell_dofs; ++i)
      y_cell[i] = y[i];

    const FEValuesExtractors::Scalar area_extractor(0);
    const FEValuesExtractors::Scalar velocity_extractor(1);

    const QGauss<dim - 1>       quad_face(1);
    FEFaceValues<dim, spacedim> fef(*fe, quad_face, update_values);

    inflow_function.set_time(t);

    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
          {
            if (!cell->face(f)->at_boundary())
              continue;
            if (is_junction_face(cell->id(), f))
              continue;

            const types::boundary_id bid = cell->face(f)->boundary_id();
            const unsigned int       vid = cell->material_id();
            const auto              &vpp = vessel_map.at(vid);

            // Interior cell value at face — use y_cell, not y
            fef.reinit(cell, f);
            std::vector<double> A_int_v(1), U_int_v(1);
            fef[area_extractor].get_function_values(y_cell, A_int_v);
            fef[velocity_extractor].get_function_values(y_cell, U_int_v);

            const double A_int = std::max(A_int_v[0], 1e-10);
            const double U_int = U_int_v[0];
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
                    // const double Pc = terminal_Pc_storage.at(bid);
                    // res_A =
                    //   compute_pressure_value(A_hat, vid, a_d_local) - (rcr.R1 * Q + Pc);
                    const double Pc =
                      y[rcr_pc_dof.at(bid)]; 
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

            const FaceTraceDof &td =
              face_dof_map.at(canonical_face_key(cell, f));
            F[td.a_hat_dof] = res_A; 
            F[td.u_hat_dof] = res_U;
          }
      }
  }

  // ============================================================================
  // assemble_rcr_capacitance_equations
  // For each RCR outlet, enforce the capacitance relation:
  //   dP/dt = (P - R2*Q - P_out) / (R1*C)
  // where P = P(A_hat) is the pressure at the outlet face (from trace state), Q = A_hat * U_hat
  // ============================================================================
  template <int dim, int spacedim>
  void
  BloodFlowSystem<dim, spacedim>::assemble_rcr_capacitor_equations(
    const Vector<double> &y,
    const Vector<double> &ydot,
    Vector<double>       &F)
  {
    if (rcr_pc_dof.empty())
      return;

    for (const auto &cell : dof_handler.active_cell_iterators())
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
          const double Pc     = y[pc_dof];
          const double Pc_dot = ydot[pc_dof];

          // C*Ṗc - Q + (Pc - P_out)/R2 = 0
          F[pc_dof] = rcr.C * Pc_dot - Q + (Pc - rcr.P_out) / rcr.R2;
        }
  }

  // ============================================================================
  // assemble_trace_junction_equations
  //
  // At a K-way junction the 2K trace unknowns {A_hat_i, U_hat_i} for
  // i = 0 … K−1 must satisfy: ( here K>=2 , K=2 , 2 way junction otherwise K>2  multi-way junction)
  //
  //   (a) Mass conservation  (1 equation):
  //         sum_i [ s_i * A_hat_i * U_hat_i ] = 0
  //
  //   (b) Total-head continuity  (K−1 equations):
  //         H_0 − H_i = 0,   H_i = U_hat_i^2/2 + P(A_hat_i)/ρ
  //
  //   (c) Riemann compatibility  (K equations):
  //         U_hat_i + s_i * 4(c_hat_i − c0_i) − W_i = 0
  //         where  W_i = U_int_i + s_i * 4(c_int_i − c0_i)
  //                is the outgoing Riemann invariant from cell i's interior.
  //
  // Total: 1 + (K−1) + K = 2K equations.
  //
  // Row assignment to avoid collision:
  //   a_idx[0]          -> (a) mass conservation
  //   u_idx[0..K−2]     -> (b) total-head continuity for vessels 1..K−1
  //   u_idx[K−1]        -> (c) Riemann compat. for vessel 0
  //   a_idx[1..K−1]     -> (c) Riemann compat. for vessels 1..K−1
  //
  // FEFaceValues::get_function_values uses y_cell (size n_cell_dofs).
  // Trace values are read directly from the trace block of y.
  // ============================================================================
  template <int dim, int spacedim>
  void BloodFlowSystem<dim, spacedim>::assemble_trace_junction_equations(
    const Vector<double> &y, Vector<double> &F)
  {
    TimerOutput::Scope timer(computing_timer,
                             "assemble_trace_junction_equations");

    if (junctions.empty())
      return;

    // Cell-block sub-vector for FEFaceValues
    Vector<double> y_cell(n_cell_dofs);
    for (types::global_dof_index i = 0; i < n_cell_dofs; ++i)
      y_cell[i] = y[i];

    const FEValuesExtractors::Scalar area_extractor(0);
    const FEValuesExtractors::Scalar velocity_extractor(1);

    const QGauss<dim - 1>       quad_face(1);
    FEFaceValues<dim, spacedim> fef(*fe, quad_face, update_values);

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
            const auto        &vpp = vessel_map.at(vid);

            // Interior cell value at the junction face — use y_cell
            fef.reinit(hf.cell, hf.face_no);
            std::vector<double> Av(1), Uv(1);
            fef[area_extractor].get_function_values(y_cell, Av);
            fef[velocity_extractor].get_function_values(y_cell, Uv);

            A_int[i] = std::max(Av[0], A_min);
            U_int[i] = Uv[0];
            s[i]  = hf.orientation; // +1 if junction is at face 1 (right end)
            const double a_d_face = compute_a_d_at_face(hf.cell, hf.face_no);
            c0[i] = compute_wave_speed(a_d_face, vid, a_d_face);

            // Outgoing Riemann invariant from cell i toward the junction
            const double c_i = compute_wave_speed(A_int[i], vid, a_d_face);
            W[i] = U_int[i] + static_cast<double>(s[i]) * 4.0 * (c_i - c0[i]);

            // Trace DOF indices and current trace values (from trace block of
            // y)
            const FaceTraceDof &td =
              face_dof_map.at(canonical_face_key(hf.cell, hf.face_no));
            a_idx[i] = td.a_hat_dof;
            u_idx[i] = td.u_hat_dof;

            A_hat[i] = std::max(y[a_idx[i]], A_min);
            U_hat[i] = y[u_idx[i]];
            c_hat[i] = compute_wave_speed(A_hat[i], vid, a_d_face);
          }

        // (a) Mass conservation -> row a_idx[0]
        {
          double mass_res = 0.0;
          for (unsigned int i = 0; i < K; ++i)
            mass_res += static_cast<double>(s[i]) * A_hat[i] * U_hat[i];
          F[a_idx[0]] = mass_res;
        }

        // (b) Total-head continuity: H_0 − H_i = 0 -> rows u_idx[0..K−2]
        {
          const double a_d0 = compute_a_d_at_face(J.half_faces[0].cell, J.half_faces[0].face_no);
          const double H0 =
            0.5 * U_hat[0] * U_hat[0] +
            compute_pressure_value(A_hat[0],
                                   J.half_faces[0].cell->material_id(), a_d0) /
              rho;

          for (unsigned int i = 1; i < K; ++i)
            {
              const double a_di = compute_a_d_at_face(J.half_faces[i].cell,
                                                      J.half_faces[i].face_no);
              const double Hi =
                0.5 * U_hat[i] * U_hat[i] +
                compute_pressure_value(A_hat[i],
                                       J.half_faces[i].cell->material_id(), a_di) /
                  rho;
              F[u_idx[i - 1]] = H0 - Hi;
            }
        }

        // (c) Riemann compatibility: U_hat_i + s_i*4(c_hat_i−c0_i) − W_i = 0
        //     vessel 0 -> row u_idx[K−1]
        //     vessel i>=1 -> row a_idx[i]
        for (unsigned int i = 0; i < K; ++i)
          {
            const double compatibility_res =
              U_hat[i] + static_cast<double>(s[i]) * 4.0 * (c_hat[i] - c0[i]) -
              W[i];

            const types::global_dof_index row =
              (i == 0) ? u_idx[K - 1] : a_idx[i];

            F[row] = compatibility_res;
          }
      }
  }
  
  // ============================================================================
  // assemble_residual
  //
  // ARKode calls this to evaluate  f(t, y)  where the system is
  //   dy/dt = f(t, y)   with M_ARKode = I.
  //
  // For cell DOFs: f_cell = M_K^-1 * F_cell(y)  (apply per-cell inverse mass)
  // For trace DOFs: f_trace = F_trace(y)         (algebraic — returned as-is)
  //
  // ARKode drives  dy/dt = f(t,y) and enforces  F_trace = 0 as a stiff
  // algebraic constraint through its implicit solver.
  // ============================================================================
  template <int dim, int spacedim>
  void BloodFlowSystem<dim, spacedim>::assemble_residual(
    const double          t,
    const Vector<double> &y,
    const Vector<double> &ydot,
    Vector<double>       &residual)
  {
    TimerOutput::Scope timer(computing_timer, "assemble_residual");
    deallog.push("assemble_residual");
    deallog << "t=" << t << std::endl;
    AssertDimension(y.size(), n_total_dofs);
    AssertDimension(ydot.size(), n_total_dofs);
    residual.reinit(n_total_dofs);

    // ---- Assemble raw residuals for all DOFs --------------------------------
    assemble_cell_residuals(t, y, residual);
    assemble_trace_interior_equations(y, residual);
    assemble_trace_boundary_equations(t, y, residual);
    assemble_trace_junction_equations(y, residual);
    // ---- Apply M_K^-1 to cell rows ------------------------------------------
    // The cell block of F currently holds  F_cell(y).
    const unsigned int n_dofs = fe->n_dofs_per_cell();
    Vector<double>     local_F(n_dofs), local_Mydot(n_dofs), local_res(n_dofs);

    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        std::vector<types::global_dof_index> ldofs(n_dofs);
        cell->get_dof_indices(ldofs);

        for (unsigned int i = 0; i < n_dofs; ++i)
          {
            local_F(i)     = residual[ldofs[i]]; // F_cell
            local_Mydot(i) = ydot[ldofs[i]];
          }
        per_cell_mass_.at(cell->id()).vmult(local_res, local_Mydot); // M_k*ydot
        for (unsigned int i = 0; i < n_dofs; ++i)
          {
            residual[ldofs[i]] = local_res(i) - local_F(i); // M ydot - R_cell
          }
      }
    for (types::global_dof_index i = n_cell_dofs; i < n_trace_end; ++i)
      {
        residual[i] = -residual[i]; // algebraic constraint: F_trace = 0 ->
                                    // residual = -F_trace
      }
    // RCR capacitor rows: differential, already in M*ydot - R form
    assemble_rcr_capacitor_equations(y, ydot, residual);
    deallog.pop();
  }

  // ============================================================================
  // assemble_jacobian_cell_block
  //
  // Differentiates the cell residuals w.r.t. cell DOFs (block 1,1) and
  // w.r.t. trace DOFs (block 1,2).
  // ============================================================================
  template <int dim, int spacedim>
  void BloodFlowSystem<dim, spacedim>::assemble_jacobian_cell_block(
    const double t, const Vector<double> &y)
  {
    TimerOutput::Scope timer(computing_timer, "assemble_jacobian_cell_block");

    Vector<double> y_cell(n_cell_dofs);
    for (types::global_dof_index i = 0; i < n_cell_dofs; ++i)
      y_cell[i] = y[i];

    const FEValuesExtractors::Scalar area_extractor(0);
    const FEValuesExtractors::Scalar velocity_extractor(1);

    const QGauss<dim>     quad_cell(fe->tensor_degree() + 1);
    const QGauss<dim - 1> quad_face(fe->tensor_degree() + 1);

    FEValues<dim, spacedim> fev(
      *fe, quad_cell, update_values | update_gradients | update_JxW_values);
    FEFaceValues<dim, spacedim> fef(*fe,
                                    quad_face,
                                    update_values | update_JxW_values |
                                      update_normal_vectors);

    const double rho = par["rho"];
    const double eta = 2.0 * (par["xi"] + 2.0) * numbers::PI * par["mu"] / rho;
    (void)t;

    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        const unsigned int vid    = cell->material_id();
        const unsigned int n_dofs = fe->n_dofs_per_cell();

        fev.reinit(cell);
        
        const auto &JxW = fev.get_JxW_values();
        const double ad_local = compute_a_d_local(cell);
        std::vector<types::global_dof_index> ldofs(n_dofs);
        cell->get_dof_indices(ldofs);

        std::vector<double> A_h(fev.n_quadrature_points);
        std::vector<double> U_h(fev.n_quadrature_points);
        fev[area_extractor].get_function_values(y_cell, A_h);
        fev[velocity_extractor].get_function_values(y_cell, U_h);

        FullMatrix<double> cell_matrix(n_dofs, n_dofs);
   
        // ---- Block (1,1) volume - cell_matrix -------------------------------
        for (unsigned int q = 0; q < fev.n_quadrature_points; ++q)
          {
            const double A    = std::max(A_h[q], 1e-10);
            const double U    = U_h[q];
            const double dpdA = compute_pressure_derivative(A, vid, ad_local);
            const double c2_A = A / rho * dpdA;

            for (unsigned int i = 0; i < n_dofs; ++i)
              {
                const unsigned int ci = fe->system_to_component_index(i).first;
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

        // ---- Block (1,1) face + Block (1,2) ---------------------------------
        // Face cell-cell -> cell_matrix 
        
        for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
          {
            fef.reinit(cell, f);
            const auto &normals = fef.get_normal_vectors();
            const auto &JxW_f   = fef.get_JxW_values();
            const double ad_face = compute_a_d_at_face(cell, f);
            std::vector<double> Ah_q(fef.n_quadrature_points);
            std::vector<double> Uh_q(fef.n_quadrature_points);
            fef[area_extractor].get_function_values(y_cell, Ah_q);
            fef[velocity_extractor].get_function_values(y_cell, Uh_q);

            double A_hat_cur = 0.0, U_hat_cur = 0.0;
            get_face_trace(y, cell, f, A_hat_cur, U_hat_cur);
            const double A_hat = std::max(A_hat_cur, 1e-10);
            const double U_hat = U_hat_cur;

            const auto  key = canonical_face_key(cell, f);
            const auto &td  = face_dof_map.at(key);

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
                              fe->system_to_component_index(i).first;
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
                              fe->system_to_component_index(i).first;
                            const double phi = fef.shape_value(i, q);
                            jacobian_matrix.add(ldofs[i],
                                                trace_col,
                                                -(comp == 0 ? dFA : dFU) * phi *
                                                  JxW_f[q]);
                          }
                      }
                  }
              }
        // Single scatter: volume + face cell-cell -> jacobian_matrix

        for (unsigned int i = 0; i < n_dofs; ++i)
          for (unsigned int j = 0; j < n_dofs; ++j)
            jacobian_matrix.add(ldofs[i], ldofs[j], cell_matrix(i, j));
      }
  }

  // ============================================================================
  // assemble_jacobian_trace_interior_block
  //
  //   R_A = F_hat_A(bn_L,bn_L, A_L,U_L, A_hat,U_hat)
  //         + F_hat_A(bn_R,bn_R, A_R,U_R, A_hat,U_hat)
  //   R_U = F_hat_U(bn_L,bn_L, A_L,U_L, A_hat,U_hat)
  //         + F_hat_U(bn_R,bn_R, A_R,U_R, A_hat,U_hat)
  //
  // Blocks filled:
  //   (trace, cell_L)  : dR / d w_L  via left HLL flux jacobian
  //   (trace, cell_R)  : dR / d w_R  via right HLL flux jacobian
  //   (trace, trace)   : dR / d(A_hat, U_hat) summed from both fluxes
  // ============================================================================
  template <int dim, int spacedim>
  void BloodFlowSystem<dim, spacedim>::assemble_jacobian_trace_interior_block(
    const Vector<double> &y)
  {
    TimerOutput::Scope timer(computing_timer,
                             "assemble_jacobian_trace_interior_block");

    Vector<double> y_cell(n_cell_dofs);
    for (types::global_dof_index i = 0; i < n_cell_dofs; ++i)
      y_cell[i] = y[i];

    const FEValuesExtractors::Scalar area_extractor(0);
    const FEValuesExtractors::Scalar velocity_extractor(1);

    const QGauss<dim - 1>       quad_face(1);
    FEFaceValues<dim, spacedim> fef(*fe,
                                    quad_face,
                                    update_values | update_normal_vectors);

    std::set<std::pair<CellId, unsigned int>> processed;

    for (const auto &cell : dof_handler.active_cell_iterators())
      {
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

            const auto        &nb    = cell->neighbor(f);
            const unsigned int nb_f  = cell->neighbor_of_neighbor(f);
            const unsigned int vid_L = cell->material_id();
            const unsigned int vid_R = nb->material_id();
            const double ad_L = compute_a_d_local(cell);
            const double ad_R = compute_a_d_local(nb);
            const double       ad_face = compute_a_d_at_face(cell, f);
            // ---- Left cell: state and normal --------------------------------
            fef.reinit(cell, f);
            const double bn_L =
              compute_tangent_normal_product(cell, fef.get_normal_vectors()[0]);

            std::vector<double> A_Lv(1), U_Lv(1);
            fef[area_extractor].get_function_values(y_cell, A_Lv);
            fef[velocity_extractor].get_function_values(y_cell, U_Lv);
            const double A_L = std::max(A_Lv[0], 1e-10);
            const double U_L = U_Lv[0];

            std::vector<types::global_dof_index> ldofs_L(fe->n_dofs_per_cell());
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

            std::vector<types::global_dof_index> ldofs_R(fe->n_dofs_per_cell());
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
            // Linearise F_hat(bn_L,bn_L, A_L,U_L, A_hat,U_hat) w.r.t. w_L.
            // Interior = L (trial nonzero), exterior = trace (trial zero).
            // ================================================================
            fef.reinit(cell, f);
            for (unsigned int j = 0; j < fe->n_dofs_per_cell(); ++j)
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
                                    ad_L, ad_face);

                jacobian_matrix.add(a_row, ldofs_L[j], dFA);
                jacobian_matrix.add(u_row, ldofs_L[j], dFU);
              }

            // ================================================================
            // Block (trace, cell_R): dR/dw_R
            // Linearise F_hat(bn_R,bn_R, A_R,U_R, A_hat,U_hat) w.r.t. w_R.
            // Interior = R (trial nonzero), exterior = trace (trial zero).
            // ================================================================
            fef.reinit(nb, nb_f);
            for (unsigned int j = 0; j < fe->n_dofs_per_cell(); ++j)
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
                                    ad_R, ad_face);

                jacobian_matrix.add(a_row, ldofs_R[j], dFA);
                jacobian_matrix.add(u_row, ldofs_R[j], dFU);
              }

            // ================================================================
            // Block (trace, trace): dR/d(A_hat, U_hat)
            // Trace appears as the EXTERIOR state in both fluxes.
            // Contribution from left flux: trial on exterior = (dA_hat,
            // dU_hat). Contribution from right flux: same trial. Sum both
            // contributions.
            // ================================================================
            // Unit perturbations in A_hat and U_hat:
            for (const auto &[dA_hat_trial, dU_hat_trial, col] :
                 {std::tuple{1.0, 0.0, td.a_hat_dof},
                  std::tuple{0.0, 1.0, td.u_hat_dof}})
              {
                // Left flux: exterior trial = (dA_hat_trial, dU_hat_trial)
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
                                    ad_L, ad_face);

                // Right flux: exterior trial = (dA_hat_trial, dU_hat_trial)
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
                                    ad_R, ad_face);

                jacobian_matrix.add(a_row, col, dFA_L + dFA_R);
                jacobian_matrix.add(u_row, col, dFU_L + dFU_R);
              }
          }
      }
  }

  // ============================================================================
  // assemble_jacobian_trace_boundary_block
  // ============================================================================
  template <int dim, int spacedim>
  void BloodFlowSystem<dim, spacedim>::assemble_jacobian_trace_boundary_block(
    const double t, const Vector<double> &y)
  {
    TimerOutput::Scope timer(computing_timer,
                             "assemble_jacobian_trace_boundary_block");

    Vector<double> y_cell(n_cell_dofs);
    for (types::global_dof_index i = 0; i < n_cell_dofs; ++i)
      y_cell[i] = y[i];

    const FEValuesExtractors::Scalar area_extractor(0);
    const FEValuesExtractors::Scalar velocity_extractor(1);

    const QGauss<dim - 1>       quad_face(1);
    FEFaceValues<dim, spacedim> fef(*fe, quad_face, update_values);

    (void)t;

    for (const auto &cell : dof_handler.active_cell_iterators())
      {
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

            const double A_int = std::max(A_int_v[0], 1e-10);
            const double a_d_local = compute_a_d_at_face(cell, f);
            // const double U_int  = U_int_v[0];
            // const double c0     = compute_wave_speed(vpp.a_d, vid);
            const double dc_int = compute_wave_speed_derivative(A_int, vid, a_d_local);

            double A_hat_cur = 0.0, U_hat_cur = 0.0;
            get_face_trace(y, cell, f, A_hat_cur, U_hat_cur);
            const double A_hat    = std::max(A_hat_cur, 1e-10);
            const double dc_hat   = compute_wave_speed_derivative(A_hat, vid, a_d_local);
            const double dPdA_hat = compute_pressure_derivative(A_hat, vid, a_d_local);

            const auto                    key   = canonical_face_key(cell, f);
            const auto                   &td    = face_dof_map.at(key);
            const types::global_dof_index a_row = td.a_hat_dof;
            const types::global_dof_index u_row = td.u_hat_dof;

            std::vector<types::global_dof_index> ldofs(fe->n_dofs_per_cell());
            cell->get_dof_indices(ldofs);

            if (bid == 0) // inflow
              {
                // R1 = A_hat * U_hat - Q_in(t)
                // R2 = U_hat - 4(c_hat - c0) - W2_int
                // where W2_int = U_int - 4(c_int - c0)

                // \partialR1/ \partialA_hat, \partialR1/ \partialU_hat
                jacobian_matrix.add(a_row, a_row, U_hat_cur);
                jacobian_matrix.add(a_row, u_row, A_hat);

                // \partialR2/\partialA_hat, \partialR2/\partialU_hat
                jacobian_matrix.add(u_row, a_row, -4.0 * dc_hat);
                jacobian_matrix.add(u_row, u_row, 1.0);

                // \partialR2/\partial(A_int, U_int) — block (2,1)
                for (unsigned int j = 0; j < fe->n_dofs_per_cell(); ++j)
                  {
                    const double phi_A = fef[area_extractor].value(j, 0);
                    const double phi_U = fef[velocity_extractor].value(j, 0);
                    // \partialW2_int/\partialw_int = phi_U - 4*dc_int*phi_A
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
                    for (unsigned int j = 0; j < fe->n_dofs_per_cell(); ++j)
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
                    // dres_A/dA_hat = dP/dA_hat - R2*U_hat   (same form as RCR
                    // with R2) dres_A/dU_hat = -R2*A_hat               (R2
                    // replaces R1) Pc term drops out (no capacitor, no time
                    // derivative)
                    jacobian_matrix.add(a_row,
                                        a_row,
                                        dPdA_hat - rcr.R2 * U_hat_cur);
                    jacobian_matrix.add(a_row, u_row, -rcr.R2 * A_hat);

                    // dres_U/dA_hat = 4*dc_hat
                    // dres_U/dU_hat = 1
                    // (identical to RCR — res_U has no R dependence)
                    jacobian_matrix.add(u_row, a_row, 4.0 * dc_hat);
                    jacobian_matrix.add(u_row, u_row, 1.0);

                    // dres_U/d(A_int, U_int)
                    for (unsigned int j = 0; j < fe->n_dofs_per_cell(); ++j)
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

                // \partialR1/\partialA_hat, \partialR1/\partialU_hat
                jacobian_matrix.add(a_row, a_row, 4.0 * dc_hat);
                jacobian_matrix.add(a_row, u_row, 1.0);

                // \partialR2/\partialA_hat, \partialR2/\partialU_hat
                jacobian_matrix.add(u_row, a_row, -4.0 * dc_hat);
                jacobian_matrix.add(u_row, u_row, 1.0);

                // \partialR1 and \partialR2 /\partial(A_int,U_int) via W1_int
                for (unsigned int j = 0; j < fe->n_dofs_per_cell(); ++j)
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

//================================================
// assemble_rcr_capacitor_equations in jacobian assembly
// RCR capacitor equations are ODEs, so their Jacobian contributions are just
// the derivatives of the residual w.r.t. the capacitor state DOFs (no coupling
// to cell DOFs, no dependence on trace DOFs).
//=========================================================
  template <int dim, int spacedim>
  void
  BloodFlowSystem<dim, spacedim>::assemble_jacobian_rcr_capacitor_block(
    const Vector<double> &y)
  {
    if (rcr_pc_dof.empty())
      return;

    for (const auto &cell : dof_handler.active_cell_iterators())
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

  // ============================================================================
  // assemble_jacobian_trace_junction_block
  //
  // Differentiates the junction residuals (mass conservation, total-head
  // continuity, Riemann compatibility) w.r.t. all cell and trace DOFs.
  // ============================================================================
  template <int dim, int spacedim>
  void BloodFlowSystem<dim, spacedim>::assemble_jacobian_trace_junction_block(
    const Vector<double> &y)
  {
    TimerOutput::Scope timer(computing_timer,
                             "assemble_jacobian_trace_junction_block");

    Vector<double> y_cell(n_cell_dofs);
    for (types::global_dof_index i = 0; i < n_cell_dofs; ++i)
      y_cell[i] = y[i];

    const FEValuesExtractors::Scalar area_extractor(0);
    const FEValuesExtractors::Scalar velocity_extractor(1);

    const QGauss<dim - 1>       quad_face(1);
    FEFaceValues<dim, spacedim> fef(*fe, quad_face, update_values);

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

            A_int[i]    = std::max(Av[0], A_min);
            U_int[i]    = Uv[0];
            orient[i]   = hf.orientation;
            
              const double a_d_face = compute_a_d_at_face(hf.cell, hf.face_no);
              c0[i] = compute_wave_speed(a_d_face, vid, a_d_face);
              dc_int_v[i] =
                compute_wave_speed_derivative(A_int[i], vid, a_d_face);
            

            // c0[i]       = compute_wave_speed(vessel_map.at(vid).a_d, vid);
            // dc_int_v[i] = compute_wave_speed_derivative(A_int[i], vid);

            const auto  key = canonical_face_key(hf.cell, hf.face_no);
            const auto &td  = face_dof_map.at(key);
            a_row[i]        = td.a_hat_dof;
            u_row[i]        = td.u_hat_dof;

            A_hat[i]    = std::max(y[a_row[i]], A_min);
            U_hat[i]    = y[u_row[i]];
            
            c_hat_v[i]  = compute_wave_speed(A_hat[i], vid, a_d_face);
            dc_hat_v[i] = compute_wave_speed_derivative(A_hat[i], vid, a_d_face);
            dP_hat[i]   = compute_pressure_derivative(A_hat[i], vid, a_d_face);

            cell_dofs[i].resize(fe->n_dofs_per_cell());
            hf.cell->get_dof_indices(cell_dofs[i]);
          }

        // Row a_row[0]: mass conservation \sum s_i A_hat_i U_hat_i = 0
        for (unsigned int i = 0; i < K; ++i)
          {
            const double s = static_cast<double>(orient[i]);
            jacobian_matrix.add(a_row[0], a_row[i], s * U_hat[i]);
            jacobian_matrix.add(a_row[0], u_row[i], s * A_hat[i]);
          }

        // Rows u_row[0..K-2]: H_0 − H_i = 0
        for (unsigned int i = 1; i < K; ++i)
          {
            // \partialH_0/\partialA_hat_0 , \partialH_0/\partialU_hat_0
            jacobian_matrix.add(u_row[i - 1], a_row[0], dP_hat[0] / rho);
            jacobian_matrix.add(u_row[i - 1], u_row[0], U_hat[0]);
            // \partial(-H_i)/\partialA_hat_i, \partial(-H_i)/\partialU_hat_i
            jacobian_matrix.add(u_row[i - 1], a_row[i], -dP_hat[i] / rho);
            jacobian_matrix.add(u_row[i - 1], u_row[i], -U_hat[i]);
          }

        // Compat row for vessel 0 -> u_row[K-1]
        // Compat row for vessel i>=1 -> a_row[i]
        for (unsigned int i = 0; i < K; ++i)
          {
            const double                  s = static_cast<double>(orient[i]);
            const types::global_dof_index rr =
              (i == 0) ? u_row[K - 1] : a_row[i];

            // \partial/\partialA_hat_i, \partial/\partialU_hat_i of (U_hat_i +
            // s*4(c_hat_i - c0_i) - W_i)
            jacobian_matrix.add(rr, a_row[i], s * 4.0 * dc_hat_v[i]);
            jacobian_matrix.add(rr, u_row[i], 1.0);

            // \partial(-W_i)/\partial(A_int_i, U_int_i)
            const unsigned int vid = J.half_faces[i].cell->material_id();
            fef.reinit(J.half_faces[i].cell, J.half_faces[i].face_no);

            for (unsigned int j = 0; j < fe->n_dofs_per_cell(); ++j)
              {
                const double phi_A = fef[area_extractor].value(j, 0);
                const double phi_U = fef[velocity_extractor].value(j, 0);
                // \partialW_i/\partialw = phi_U + s*4*dc_int_v[i]*phi_A
                const double dW = phi_U + s * 4.0 * dc_int_v[i] * phi_A;
                jacobian_matrix.add(rr, cell_dofs[i][j], -dW);
              }
            (void)vid;
          }
      }
  }

  // ============================================================================
  // assemble_jacobian
  //
  // Builds  J = d f(t,y) / d y  where f = [M_K^-1 F_cell ; F_trace].
  //
  // Cell rows:  J_cell_rows = M_K^-1 * (d F_cell / d y)
  // Trace rows: J_trace_rows = d F_trace / d y  (unchanged)
  //
  // We assemble d F / d y first, then apply M_K^-1 row-by-row to the
  // cell rows only.
  // ============================================================================
  template <int dim, int spacedim>
  void BloodFlowSystem<dim, spacedim>::assemble_jacobian(
    const double          t,
    const Vector<double> &y,
    const Vector<double> & /*ydot*/,
    const double alpha)
  {
    TimerOutput::Scope timer(computing_timer, "assemble_jacobian");
    deallog.push("assemble_jacobian");
    deallog << "t=" << t << std::endl;

    AssertDimension(y.size(), n_total_dofs);
    jacobian_matrix = 0;

    // ---- Assemble raw d F / d y ---------------------------------------------
    assemble_jacobian_cell_block(t, y);
    assemble_jacobian_trace_interior_block(y);
    assemble_jacobian_trace_boundary_block(t, y);
    assemble_jacobian_trace_junction_block(y);
    assemble_jacobian_rcr_capacitor_block(y);

    // ---- Negate: dF/dy = −dR/dy -------------------------------------------
    // Residual is F = M*ydot − R_cell (cell) and F = −R_trace (trace)
    // so dF/dy = −dR/dy for both blocks.
    jacobian_matrix *= -1.0;

    // ---- Add alpha*M_K to cell-block (dF/dydot term)
    // --------------------------- J_IDA = dF/dy + alpha · dF/dẏ = −dR/dy +
    // alpha · M_block M_block is M_K on cell rows, zero on trace rows.

    const unsigned int n_dofs = fe->n_dofs_per_cell();

    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        std::vector<types::global_dof_index> ldofs(n_dofs);
        cell->get_dof_indices(ldofs);

        const FullMatrix<double> &M_K = per_cell_mass_.at(cell->id());

        for (unsigned int i = 0; i < n_dofs; ++i)
          for (unsigned int j = 0; j < n_dofs; ++j)
            jacobian_matrix.add(ldofs[i], ldofs[j], alpha * M_K(i, j));
      }

    // alpha * C on capacitor diagonal (the dF/dẏ term for Pc rows)
    for (const auto &[bid, pc_dof] : rcr_pc_dof)
  {
      jacobian_matrix.add(pc_dof, pc_dof, alpha * rcr_map.at(bid).C);
  }
    deallog.pop();
  }

  // ============================================================================
  // compute_pressure
  // ============================================================================
  template <int dim, int spacedim>
  void BloodFlowSystem<dim, spacedim>::compute_pressure(const Vector<double> &y,
                                                        Vector<double> &p) const
  {
    TimerOutput::Scope timer(computing_timer, "compute_pressure");
    AssertDimension(y.size(), n_total_dofs);

    p.reinit(n_total_dofs);

    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        std::vector<types::global_dof_index> ldofs(fe->n_dofs_per_cell());
        cell->get_dof_indices(ldofs);

        for (unsigned int i = 0; i < fe->n_dofs_per_cell(); ++i)
          if (fe->system_to_component_index(i).first == 0)
            {
              const double A = y[ldofs[i]];
              // p[ldofs[i]]    = compute_pressure_value(A, cell->material_id());
              p[ldofs[i]] = compute_pressure_value(A,
                                                   cell->material_id(),
                                                   compute_a_d_local(cell));
            }
      }
  }

  // ============================================================================
  // compute_theoretical_peak
  // ============================================================================
  template <int dim, int spacedim>
  void BloodFlowSystem<dim, spacedim>::compute_theoretical_peak(Vector<double> &
                                                                tp) const
  {
    tp.reinit(n_total_dofs);

    const double xi  = par["xi"];
    const double mu  = par["mu"];
    const double rho = par["rho"];

    const auto  &props = vessel_map.at(0);
    const double Ad    = props.a_d;
    const double c0    = compute_wave_speed(Ad, 0);

    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        std::vector<types::global_dof_index> ldofs(fe->n_dofs_per_cell());
        cell->get_dof_indices(ldofs);

        const double x = cell->center()[0];
        const double val =
          std::exp(-(xi + 2.0) * numbers::PI * mu * x / (c0 * Ad * rho));

        for (unsigned int i = 0; i < fe->n_dofs_per_cell(); ++i)
          tp[ldofs[i]] = val;
      }
  }

  // ============================================================================
  // output_results
  // ============================================================================
  template <int dim, int spacedim>
  void BloodFlowSystem<dim, spacedim>::output_results(
    const Vector<double> &y,
    const Vector<double> &pressure_vec,
    const Vector<double> &tp,
    const unsigned int    cycle) const
  {
    TimerOutput::Scope timer(computing_timer, "output_results");

    const std::string rel =
      output_filename + "-" + std::to_string(cycle) + ".vtu";
    const std::string fname =
      output_directory + (output_directory.empty() ? "" : "/") + rel;

    // Extract cell-block sub-vector for DataOut
    Vector<double> cell_sol(n_cell_dofs);
    Vector<double> cell_p(n_cell_dofs);
    Vector<double> cell_tp(n_cell_dofs);
    for (types::global_dof_index i = 0; i < n_cell_dofs; ++i)
      {
        cell_sol[i] = y[i];
        cell_p[i]   = pressure_vec[i];
        cell_tp[i]  = tp[i];
      }

    DataOut<dim, spacedim> data_out;
    data_out.attach_dof_handler(dof_handler);

    std::vector<std::string> names = {"area", "velocity"};
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      interp(2, DataComponentInterpretation::component_is_scalar);

    data_out.add_data_vector(cell_sol,
                             names,
                             DataOut<dim, spacedim>::type_dof_data,
                             interp);

    names[0] = "pressure";
    names[1] = "unused";
    data_out.add_data_vector(cell_p,
                             names,
                             DataOut<dim, spacedim>::type_dof_data,
                             interp);

    names[0] = "theoretical_peak";
    names[1] = "unused1";
    data_out.add_data_vector(cell_tp,
                             names,
                             DataOut<dim, spacedim>::type_dof_data,
                             interp);

    data_out.build_patches();

    std::ofstream out(fname);
    data_out.write_vtu(out);

    static std::vector<std::pair<double, std::string>> pvd_records;
    pvd_records.emplace_back(time, rel);

    std::ofstream pvd(output_directory + (output_directory.empty() ? "" : "/") +
                      output_filename + ".pvd");
    DataOutBase::write_pvd_record(pvd, pvd_records);

    std::cout << "  Wrote <" << fname << ">" << std::endl;
  }

  // ============================================================================
  // compute_errors
  // ============================================================================
  template <int dim, int spacedim>
  void BloodFlowSystem<dim, spacedim>::compute_errors(const unsigned int k)
  {
    TimerOutput::Scope timer(computing_timer, "compute_errors");

    const ComponentSelectFunction<spacedim> area_mask(0, 1.0, 2);
    const ComponentSelectFunction<spacedim> vel_mask(1, 1.0, 2);

    Vector<float> diff(triangulation.n_active_cells());

    // Work on the cell sub-vector only
    Vector<double> cell_sol(n_cell_dofs);
    for (types::global_dof_index i = 0; i < n_cell_dofs; ++i)
      cell_sol[i] = solution[i];

    exact_solution.set_time(time);

    auto l2_error = [&](const ComponentSelectFunction<spacedim> &mask) {
      VectorTools::integrate_difference(dof_handler,
                                        cell_sol,
                                        exact_solution,
                                        diff,
                                        QGauss<dim>(fe_degree + 3),
                                        VectorTools::L2_norm,
                                        &mask);
      return VectorTools::compute_global_error(triangulation,
                                               diff,
                                               VectorTools::L2_norm);
    };
    auto h1_error = [&](const ComponentSelectFunction<spacedim> &mask) {
      VectorTools::integrate_difference(dof_handler,
                                        cell_sol,
                                        exact_solution,
                                        diff,
                                        QGauss<dim>(fe_degree + 3),
                                        VectorTools::H1_seminorm,
                                        &mask);
      return VectorTools::compute_global_error(triangulation,
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

    std::cout << std::scientific << std::setprecision(3)
              << "=== Errors t=" << time << " cycle " << k + 1 << " ===\n"
              << " A  L2=" << AL2 << " rate=" << rate(prev_AL2, AL2) << "\n"
              << " A  H1=" << AH1 << " rate=" << rate(prev_AH1, AH1) << "\n"
              << " U  L2=" << UL2 << " rate=" << rate(prev_UL2, UL2) << "\n"
              << " U  H1=" << UH1 << " rate=" << rate(prev_UH1, UH1) << "\n"
              << " DoFs cell=" << n_cell_dofs << " trace=" << n_trace_dofs
              << " total=" << n_total_dofs << "\n"
              << std::string(60, '=') << "\n";

    prev_AL2 = AL2;
    prev_AH1 = AH1;
    prev_UL2 = UL2;
    prev_UH1 = UH1;
  }

  // ============================================================================
  // run
  // ============================================================================
  template <int dim, int spacedim>
  void BloodFlowSystem<dim, spacedim>::run()
  {
    std::cout << "=== Blood Flow HDG, polynomial degree p = " << fe_degree
              << " (cell DOFs + global face traces) ===\n";
    for (unsigned int cycle = 0; cycle < n_refinement_cycles; ++cycle)
      {
        std::cout << "\n--- Cycle " << cycle << " ---\n";

        if (cycle == 0)
          {
            // Load mesh and VTK data
            dealii::GridIn<dim, spacedim> grid_in;
            grid_in.attach_triangulation(triangulation);
            std::cout << "Reading VTK file: " << vtk_file_path << std::endl;
            std::ifstream mesh_file(vtk_file_path);
            grid_in.read_vtk(mesh_file);

            VTKUtils::read_cell_data(vtk_file_path,
                                     "vessel_id",
                                     cell_vessel_ids);
            VTKUtils::read_cell_data(vtk_file_path, "a0", cell_a0);
            VTKUtils::read_cell_data(vtk_file_path, "a_d", cell_a_d);
            VTKUtils::read_cell_data(vtk_file_path, "E", cell_E);
            VTKUtils::read_cell_data(vtk_file_path, "h_wall", cell_h_wall);
            VTKUtils::read_cell_data(vtk_file_path, "p_d", cell_p_d);
            VTKUtils::read_cell_data(vtk_file_path, "p0", cell_p0);
            VTKUtils::read_cell_data(vtk_file_path, "L", cell_L);
            VTKUtils::read_cell_data(vtk_file_path, "r_d", cell_r_d);

            // Tapered vessel radii; skip if absent.
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
            VTKUtils::read_vertex_data(vtk_file_path,
                                       "boundary_id",
                                       point_boundary_id);

            // Set material IDs and boundary IDs from VTK point data
            {
              unsigned int cell_idx = 0;
              for (auto &cell : triangulation.active_cell_iterators())
                {
                  cell->set_material_id(
                    static_cast<unsigned int>(cell_vessel_ids[cell_idx]));

                  for (unsigned int f = 0;
                       f < GeometryInfo<dim>::faces_per_cell;
                       ++f)
                    if (cell->face(f)->at_boundary())
                      {
                        const unsigned int v_idx =
                          cell->face(f)->vertex_index(0);
                        cell->face(f)->set_boundary_id(
                          static_cast<types::boundary_id>(
                            point_boundary_id[v_idx]));
                      }
                  ++cell_idx;
                }
            }

            // Populate RCR map from coarse mesh
            rcr_map.clear();
            for (const auto &cell : triangulation.active_cell_iterators())
              for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell;
                   ++f)
                if (cell->face(f)->at_boundary())
                  {
                    const types::boundary_id bid = cell->face(f)->boundary_id();
                    const unsigned int       v = cell->face(f)->vertex_index(0);

                    RCRPhysics rcr;
                    rcr.R1    = point_R1[v];
                    rcr.R2    = point_R2[v];
                    rcr.C     = point_C[v];
                    rcr.P_out = point_P_out[v];

                    if (rcr.R2 > 0.0)
                      rcr_map[bid] = rcr;
                  }

            triangulation.refine_global(n_global_refinements);
          }
        else
          {
            triangulation.refine_global(1);
          }

        setup_system();

        for (const auto &[vid, vp] : vessel_map)
          {
            const double beta =
              (4.0 / 3.0) * std::sqrt(numbers::PI) * vp.E * vp.h_wall;

            // c_in: wave speed evaluated at the inlet diastolic area
            const double a_in =
              (vp.r_in > 0.0) ? numbers::PI * vp.r_in * vp.r_in : vp.a_d;
            const double c_in = std::sqrt(beta / (2.0 * par["rho"] * a_in)) *
                                std::pow(a_in, 0.25);

            // c_out: wave speed evaluated at the outlet diastolic area
            const double a_out =
              (vp.r_out > 0.0) ? numbers::PI * vp.r_out * vp.r_out : vp.a_d;
            const double c_out = std::sqrt(beta / (2.0 * par["rho"] * a_out)) *
                                 std::pow(a_out, 0.25);

            std::cout << "Vessel " << vid << ": c_in = " << c_in << " m/s"
                      << ", c_out = " << c_out << " m/s\n";
          }

        open_csv_files();
        initialize_terminal_capacitors();
        compute_theoretical_peak(theoretical_peak);
        // Build per-cell mass inverses (replaces global mass matrix)
        build_per_cell_mass_inv();
        compute_initial_solution(solution, ida_parameters.initial_time);
        initialize_trace_unknowns(solution, ida_parameters.initial_time);
        time = ida_parameters.initial_time;

        auto check_jacobian_fd = [this](const Vector<double> &y0) {
          const double t     = time;
          const double eps   = 1e-7;
          const double alpha = 0.0;

          Vector<double> ydot0(n_total_dofs);
          ydot0 = 0.0;

          // Assemble Jacobian

          assemble_jacobian(t, y0, ydot0, alpha);
          // Random direction
          Vector<double> v(n_total_dofs);

          std::srand(0);
          for (unsigned int i = 0; i < n_total_dofs; ++i)
            v[i] = 2.0 * std::rand() / double(RAND_MAX) - 1.0;

          // y+eps*v and y-eps*v
          Vector<double> yp(y0);
          Vector<double> ym(y0);

          yp.add(eps, v);
          ym.add(-eps, v);
          // Residuals
          Vector<double> Fp(n_total_dofs);
          Vector<double> Fm(n_total_dofs);

          assemble_residual(t, yp, ydot0, Fp);
          assemble_residual(t, ym, ydot0, Fm);

          // Central FD approximation:
          // (F(y+eps*v)-F(y-eps*v))/(2 eps)
          Vector<double> fd(Fp);
          fd -= Fm;
          fd /= (2.0 * eps);

          // J*v
          Vector<double> Jv(n_total_dofs);
          jacobian_matrix.vmult(Jv, v);

          // Difference
          Vector<double> diff(Jv);
          diff -= fd;

          const double abs_err = diff.l2_norm();
          const double rel_err = abs_err / std::max(1.0, fd.l2_norm());
          double                  worst   = 0.0;
          types::global_dof_index worst_i = 0;
          for (types::global_dof_index i = 0; i < n_total_dofs; ++i)
            {
              const double d = std::abs(Jv[i] - fd[i]);
              if (d > worst)
                {
                  worst   = d;
                  worst_i = i;
                }
            }
          std::cout << "worst row i=" << worst_i
                    << (worst_i < n_cell_dofs ? " (CELL)" : " (TRACE)")
                    << "  |Jv-fd|=" << worst << "  Jv=" << Jv[worst_i]
                    << "  fd=" << fd[worst_i] << "\n";
          // classify worst_i: which trace equation owns it?
          for (const auto &cell : dof_handler.active_cell_iterators())
            for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
              {
                const auto key = canonical_face_key(cell, f);
                const auto it  = face_dof_map.find(key);
                if (it == face_dof_map.end())
                  continue;
                if (it->second.a_hat_dof == worst_i ||
                    it->second.u_hat_dof == worst_i)
                  {
                    const bool is_a = (it->second.a_hat_dof == worst_i);
                    std::cout
                      << "row " << worst_i << " is "
                      << (is_a ? "A_hat" : "U_hat") << " of face (cell "
                      << cell->id() << ", f=" << f
                      << ")  vid=" << cell->material_id()
                      << (cell->face(f)->at_boundary() ? "  BOUNDARY" : "")
                      << (is_junction_face(cell->id(), f) ? "  JUNCTION" : "")
                      << (!cell->face(f)->at_boundary() &&
                              !is_junction_face(cell->id(), f) ?
                            "  INTERIOR" :
                            "")
                      << "\n";
                    goto done;
                  }
              }
      done:;
        std::cout << "\n=====================================\n";
        std::cout << "Central FD Jacobian check\n";
        std::cout << "eps        = " << eps << "\n";
        std::cout << "||Jv-FD||  = " << abs_err << "\n";
        std::cout << "||FD||     = " << fd.l2_norm() << "\n";
        std::cout << "relative   = " << rel_err << "\n";
        std::cout << "=====================================\n";
        };
        check_jacobian_fd(solution);
       // open_csv_files();

        // ---- IDA setup
        // -----------------------------------------------------------
        ida_parameters.ic_type =
          SUNDIALS::IDA<Vector<double>>::AdditionalData::use_y_diff;
        // ida_parameters.ic_type =
        //   SUNDIALS::IDA<Vector<double>>::AdditionalData::use_y_dot;
        SUNDIALS::IDA<Vector<double>> ida(ida_parameters);

        // vectoor allocation
        ida.reinit_vector = [this](Vector<double> &v) {
          v.reinit(n_total_dofs);
        };

        ida.differential_components = [this]() -> IndexSet {
          IndexSet is(n_total_dofs);
          is.add_range(0, n_cell_dofs); // only cell DOFs carry d/dt
          if (n_rcr_dofs > 0)
            is.add_range(n_trace_end, n_total_dofs);
          return is;                    // trace DOFs are algebraic
        };

        // Residual F(t, y, ydot) = 0
        ida.residual = [this](const double          t,
                              const Vector<double> &y,
                              const Vector<double> &ydot,
                              Vector<double>       &res) -> int {
          assemble_residual(t, y, ydot, res);
          return 0;
        };

        // Jacobian J = dF/dy + alpha · dF/dẏ
        ida.setup_jacobian = [this](const double          t,
                                    const Vector<double> &y,
                                    const Vector<double> &ydot,
                                    const double          alpha) -> int {
          TimerOutput::Scope ts(computing_timer, "setup_jacobian");
          assemble_jacobian(t, y, ydot, alpha);
          linear_system_matrix.copy_from(jacobian_matrix);
          linear_solver.initialize(linear_system_matrix);
          return 0;
        };

        // Linear solve: J * z = r
        ida.solve_with_jacobian = [this](const Vector<double> &r,
                                         Vector<double>       &z,
                                         const double /*tol*/) -> int {
          TimerOutput::Scope ts(computing_timer, "solve_with_jacobian");
          linear_solver.vmult(z, r);
          return 0;
        };

        // Output at each accepted step
        ida.output_step = [this](const double          t,
                                 const Vector<double> &sol,
                                 const Vector<double> & /*ydot*/,
                                 const unsigned int step_number) {
          const double dt =
            (time > 0.0) ? (t - time) : ida_parameters.initial_step_size;
          time = t;
          // update_terminal_pressures(dt, sol);
          compute_pressure(sol, pressure);
          compute_theoretical_peak(theoretical_peak);
          output_results(sol, pressure, theoretical_peak, step_number);
          write_csv_row(t, sol);
        };

        // ---- Compute consistent initial ydot
        // ------------------------------------

        solution_dot = 0.0;
        // {
        //   Vector<double> F0(n_total_dofs);
        //   assemble_cell_residuals(ida_parameters.initial_time, solution, F0);
        //   //trace equations are zero at initial state, by doing initialize_trace_unknow() we don't need them 

        //   const unsigned int n_dpc = fe->n_dofs_per_cell();
        //   Vector<double>     loc_F(n_dpc), loc_ydot(n_dpc);
        //   for (const auto &cell : dof_handler.active_cell_iterators())
        //     {
        //       std::vector<types::global_dof_index> ldofs(n_dpc);
        //       cell->get_dof_indices(ldofs);
        //       for (unsigned int i = 0; i < n_dpc; ++i)
        //         loc_F(i) = F0[ldofs[i]];
        //       per_cell_mass_inv.at(cell->id()).vmult(loc_ydot, loc_F);
        //       for (unsigned int i = 0; i < n_dpc; ++i)
        //         solution_dot[ldofs[i]] = loc_ydot(i);
        //     }

        //   std::cout << "Initial ||ydot_cell||_inf = "
        //             << *std::max_element(solution_dot.begin(),
        //                                  solution_dot.begin() + n_cell_dofs,
        //                                  [](double a, double b) {
        //                                    return std::abs(a) < std::abs(b);
        //                                  })
        //             << "\n";
        // }

        // ---- Solve
        // ---------------------------------------------------------------
        time = ida_parameters.initial_time;
        ida.solve_dae(solution, solution_dot);
        close_csv_files();
        compute_pressure(solution, pressure);
        compute_errors(cycle);
      }
  }

  // Explicit instantiation
  template class BloodFlowSystem<1, 3>;