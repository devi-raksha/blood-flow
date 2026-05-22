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

  this->enter_subsection("ARKOde parameters");
  this->enter_my_subsection(this->prm);
  arkode_parameters.add_parameters(this->prm);
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
      if (half_faces.size() <= 2)
        continue; // ordinary interior vertex or boundary — not a junction

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

  std::cout << "Detected " << junctions.size() << " junctions." << std::endl;
}

// ============================================================================
// canonical_face_key :How faces are identified functions
//
// Interior face: key = the half-face whose cell has the smaller CellId.
// Boundary / junction face: key = the unique owning cell.
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
// build_extended_sparsity_pattern: it constructs the sparsity pattern of the Jacobian
// J = (J_cc  J_ct;   J_tc J_tt) where c=cell dofs, t=trace dofs
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
                  dsp.add(a_hat, ni);  // J(R_t, w_{l/r}) 
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
      vessel_map[vid] = vp;
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

  // ---- sparsity + matrices --------------------------------------------------
  build_extended_sparsity_pattern();

  jacobian_matrix.reinit(sparsity_pattern);
  linear_system_matrix.reinit(sparsity_pattern);

  // Solution vector covers cell + trace DOFs
  solution.reinit(n_total_dofs);
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

  AssertThrow(!terminal_boundary_ids.empty(),
              ExcMessage(
                "No terminal boundaries found. Check RCR IDs vs mesh IDs."));
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
      const unsigned int vid = cell->material_id();
      const auto        &vpp = vessel_map.at(vid);

      std::vector<types::global_dof_index> ldofs(fe->n_dofs_per_cell());
      cell->get_dof_indices(ldofs);

      for (unsigned int i = 0; i < fe->n_dofs_per_cell(); ++i)
        {
          const unsigned int comp = fe->system_to_component_index(i).first;
          dst[ldofs[i]]           = (comp == 0) ? vpp.a_d : 0.0;
        }
    }

  // ---- trace block: diastolic area, zero velocity -------------------------
  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      const unsigned int vid = cell->material_id();
      const auto        &vpp = vessel_map.at(vid);

      for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
        {
          const auto key = canonical_face_key(cell, f);
          if (!face_dof_map.count(key))
            continue;

          const FaceTraceDof &td = face_dof_map.at(key);
          if (dst[td.a_hat_dof] == 0.0) // first write wins
            {
              dst[td.a_hat_dof] = vpp.a_d;
              dst[td.u_hat_dof] = 0.0;
            }
        }
    }
}

// ============================================================================
// build_per_cell_mass_inv
//
// For every active cell K, compute M_K^-1: the inverse of the local
// (n_dofs_per_cell x n_dofs_per_cell) mass matrix.
//
// This replaces a global sparse mass matrix entirely.  ARKode sees
// M_ARKode = I because we fold M_K^-1 into the RHS and Jacobian ourselves.
// No mass-matrix callbacks are registered.
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
      M.gauss_jordan();
      per_cell_mass_inv[cell->id()] = std::move(M);
    }
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
                                         const unsigned int vid_R) const
{
  const double c_L   = compute_wave_speed(A_L, vid_L);
  const double c_R   = compute_wave_speed(A_R, vid_R);
  const double U_bar = 0.5 * (U_L + U_R);
  const double c_bar = 0.5 * (c_L + c_R);
  const double s_L   = U_bar - c_bar;
  const double s_R   = U_bar + c_bar;

  const double FAL = scalar_area_flux(bn_L, A_L, U_L);
  const double FUL = scalar_momentum_flux(bn_L,
                                          U_L,
                                          compute_pressure_value(A_L, vid_L),
                                          par["rho"]);
  const double FAR = scalar_area_flux(bn_R, A_R, U_R);
  const double FUR = scalar_momentum_flux(bn_R,
                                          U_R,
                                          compute_pressure_value(A_R, vid_R),
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
                                             const unsigned int vid_R) const
{
  const double c_L   = compute_wave_speed(A_L, vid_L);
  const double c_R   = compute_wave_speed(A_R, vid_R);
  const double U_bar = 0.5 * (U_L + U_R);
  const double c_bar = 0.5 * (c_L + c_R);
  const double s_L   = U_bar - c_bar;
  const double s_R   = U_bar + c_bar;

  const double c2L_over_AL = c_L * c_L;
  const double c2R_over_AR = c_R * c_R;

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
                                        const unsigned int vid_R) const
{
  const double FAL = scalar_area_flux(bn_L, A_L, U_L);
  const double FUL = scalar_momentum_flux(bn_L,
                                          U_L,
                                          compute_pressure_value(A_L, vid_L),
                                          par["rho"]);
  const double FAR = scalar_area_flux(bn_R, A_R, U_R);
  const double FUR = scalar_momentum_flux(bn_R,
                                          U_R,
                                          compute_pressure_value(A_R, vid_R),
                                          par["rho"]);
  const double alpha =
    theta * compute_LF_penalty(A_L, A_R, U_L, U_R, bn_L, bn_R, vid_L, vid_R);

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
                                            const unsigned int vid_R) const
{
  const double c2L = compute_wave_speed(A_L, vid_L);
  const double c2R = compute_wave_speed(A_R, vid_R);

  const double FAL_j = scalar_area_flux_jac(bn_L, A_L, U_L, dA_L, dU_L);
  const double FUL_j =
    scalar_momentum_flux_jac(bn_L, c2L * c2L, U_L, dA_L, dU_L);
  const double FAR_j = scalar_area_flux_jac(bn_R, A_R, U_R, dA_R, dU_R);
  const double FUR_j =
    scalar_momentum_flux_jac(bn_R, c2R * c2R, U_R, dA_R, dU_R);
  const double alpha =
    theta * compute_LF_penalty(A_L, A_R, U_L, U_R, bn_L, bn_R, vid_L, vid_R);

  return {{0.5 * (FAL_j + FAR_j) - 0.5 * alpha * (dA_R - dA_L),
           0.5 * (FUL_j + FUR_j) - 0.5 * alpha * (dU_R - dU_L)}};
}

// ============================================================================
// assemble_cell_residuals
//
// For each cell K:
//   R_A = \int_K [ F_A(A,U) · \gradφ ] dK
//         − \Sigma_f  hat{F}_A(A,U ; A_hat,U_hat) [[φ]]  (trace from face_dof_map)
//         + source
//   R_U similarly, with viscous friction source term.
//
// FEValues::get_function_values internally indexes via cell->get_dof_indices(),
// which only produces indices in [0, n_cell_dofs).  We must pass a vector of
// exactly size n_cell_dofs; the trace block of y is never needed here.
// ============================================================================
template <int dim, int spacedim>
void
BloodFlowSystem<dim, spacedim>::assemble_cell_residuals(const double          t,
                                                        const Vector<double> &y,
                                                        Vector<double>       &F)
{
  TimerOutput::Scope timer(computing_timer, "assemble_cell_residuals");

  // Extract the cell sub-block once.  All FEValues::get_function_values calls
  // below receive y_cell (size n_cell_dofs) — consistent with dof_handler.
  Vector<double> y_cell(n_cell_dofs);
  for (types::global_dof_index i = 0; i < n_cell_dofs; ++i)
    y_cell[i] = y[i];

  const FEValuesExtractors::Scalar Aex(0);
  const FEValuesExtractors::Scalar Uex(1);

  const QGauss<dim>     quad_cell(fe->tensor_degree() + 1);
  const QGauss<dim - 1> quad_face(fe->tensor_degree() + 1);

  FEValues<dim, spacedim> fev(*fe,
                              quad_cell,
                              update_values | update_gradients |
                                update_quadrature_points | update_JxW_values);
  FEFaceValues<dim, spacedim> fef(
    *fe, quad_face, update_values | update_JxW_values | update_normal_vectors);

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

      std::vector<double> A_h(fev.n_quadrature_points);
      std::vector<double> U_h(fev.n_quadrature_points);
      fev[Aex].get_function_values(y_cell, A_h);
      fev[Uex].get_function_values(y_cell, U_h);

      Vector<double> cell_rhs(n_dofs);

      // ---- Volume integral --------------------------------------------------
      for (unsigned int q = 0; q < fev.n_quadrature_points; ++q)
        {
          const double              A = std::max(A_h[q], 1e-10);
          const double              U = U_h[q];
          const double              P = compute_pressure_value(A, vid);
          const Tensor<1, spacedim> b = compute_directional_vector(cell);

          const double rhs_A =
            rhs_function.value(fev.get_quadrature_points()[q], 0);
          const double rhs_U =
            rhs_function.value(fev.get_quadrature_points()[q], 1);

          for (unsigned int i = 0; i < n_dofs; ++i)
            {
              const unsigned int comp = fe->system_to_component_index(i).first;

              if (comp == 0)
                {
                  cell_rhs(i) += (rhs_A * fev[Aex].value(i, q) +
                                  A * U * (b * fev[Aex].gradient(i, q))) *
                                 fev.JxW(q);
                }
              else
                {
                  const double phi = fev[Uex].value(i, q);
                  cell_rhs(i) +=
                    (rhs_U * phi +
                     (0.5 * U * U + P / rho) * (b * fev[Uex].gradient(i, q)) -
                     eta * U / A * phi) *
                    fev.JxW(q);
                }
            }
        }

      // ---- Face flux: interior cell value vs. global face trace -------------
      for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
        {
          fef.reinit(cell, f);

          const auto &normals = fef.get_normal_vectors();
          const auto &JxW     = fef.get_JxW_values();

          // Interior cell values at the face quadrature point
          std::vector<double> Ah_q(fef.n_quadrature_points);
          std::vector<double> Uh_q(fef.n_quadrature_points);
          fef[Aex].get_function_values(y_cell, Ah_q);
          fef[Uex].get_function_values(y_cell, Uh_q);

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

              const auto [FA, FU] =
                numerical_flux(bn, bn, A_in, U_in, A_hat, U_hat, vid, vid);

              for (unsigned int i = 0; i < n_dofs; ++i)
                {
                  const unsigned int comp =
                    fe->system_to_component_index(i).first;
                  cell_rhs(i) -=
                    (comp == 0 ? FA : FU) * fef.shape_value(i, q) * JxW[q];
                }
            }
        }

      // ---- Scatter into global F -------------------------------------------
      for (unsigned int i = 0; i < n_dofs; ++i)
        F[ldofs[i]] += cell_rhs(i);
    }
}

// ============================================================================
// assemble_trace_interior_equations
//
// For each unique interior (non-junction) face shared by cells L and R,
// write two Riemann-invariant residuals for the trace pair (A_hat, U_hat):
//
//   R_a = [ U_hat + 4(c_hat − c0) ] − W1_L = 0
//   R_u = [ U_hat − 4(c_hat − c0) ] − W2_R = 0
//
// where
//   W1_L = U_L + 4(c_L − c0)   outgoing (rightward) Riemann wave from left
//   W2_R = U_R − 4(c_R − c0)   outgoing (leftward)  Riemann wave from right
//   c0   = diastolic wave speed (reference, per-vessel)
//
// Solving R_a = R_u = 0 recovers the unique star state at the face.
//
// FEFaceValues::get_function_values requires a vector of size n_cell_dofs;
// we pass y_cell, not the full y.
// ============================================================================
template <int dim, int spacedim>
void
BloodFlowSystem<dim, spacedim>::assemble_trace_interior_equations(
  const Vector<double> &y,
  Vector<double>       &F)
{
  TimerOutput::Scope timer(computing_timer,
                           "assemble_trace_interior_equations");

  // Cell-block sub-vector for FEFaceValues (size must equal
  // dof_handler.n_dofs())
  Vector<double> y_cell(n_cell_dofs);
  for (types::global_dof_index i = 0; i < n_cell_dofs; ++i)
    y_cell[i] = y[i];

  const FEValuesExtractors::Scalar Aex(0);
  const FEValuesExtractors::Scalar Uex(1);

  const QGauss<dim - 1> quad_face(1); // each 1-D face is a single 0-D point
  FEFaceValues<dim, spacedim> fef(*fe, quad_face, update_values);

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

          // Interior cell values at face — use y_cell, not y
          fef.reinit(cell, f);
          std::vector<double> A_L_v(1), U_L_v(1);
          fef[Aex].get_function_values(y_cell, A_L_v);
          fef[Uex].get_function_values(y_cell, U_L_v);

          fef.reinit(nb, nb_f);
          std::vector<double> A_R_v(1), U_R_v(1);
          fef[Aex].get_function_values(y_cell, A_R_v);
          fef[Uex].get_function_values(y_cell, U_R_v);

          const double A_L = std::max(A_L_v[0], 1e-10);
          const double U_L = U_L_v[0];
          const double A_R = std::max(A_R_v[0], 1e-10);
          const double U_R = U_R_v[0];

          // Current trace (read directly from trace block of y)
          double A_hat_cur = 0.0, U_hat_cur = 0.0;
          get_face_trace(y, cell, f, A_hat_cur, U_hat_cur);
          const double A_hat = std::max(A_hat_cur, 1e-10);

          // Diastolic reference wave speeds
          const double c0_L =
            compute_wave_speed(vessel_map.at(vid_L).a_d, vid_L);
          const double c0_R =
            compute_wave_speed(vessel_map.at(vid_R).a_d, vid_R);
          const double c0 = 0.5 * (c0_L + c0_R);

          const double c_L   = compute_wave_speed(A_L, vid_L);
          const double c_R   = compute_wave_speed(A_R, vid_R);
          const double c_hat = compute_wave_speed(A_hat, vid_L);

          // Outgoing Riemann invariants from each side
          const double W1_L =
            U_L + 4.0 * (c_L - c0); // rightward from left cell
          const double W2_R =
            U_R - 4.0 * (c_R - c0); // leftward  from right cell

          const FaceTraceDof &td = face_dof_map.at(key);
          F[td.a_hat_dof]        = (U_hat_cur + 4.0 * (c_hat - c0)) - W1_L;
          F[td.u_hat_dof]        = (U_hat_cur - 4.0 * (c_hat - c0)) - W2_R;
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
//     R_a = P(A_hat) − [R1 * A_hat*u_hat + Pc] = 0         (Windkessel pressure BC)
//     R_u = [U_hat + 4(c_hat − c0)] − W1_int = 0 (incoming Riemann compat.)
//           W1_int = U_int + 4(c_int − c0)
//
//   bid != 0 + Reflection:
//     R_a = [U_hat + 4(c_hat − c0)] − W1_int = 0 (forward compat.)
//     R_u = [U_hat − 4(c_hat − c0)] − W2_tgt = 0 (backward: −Rt * W1_int)
// ============================================================================
template <int dim, int spacedim>
void
BloodFlowSystem<dim, spacedim>::assemble_trace_boundary_equations(
  const double          t,
  const Vector<double> &y,
  Vector<double>       &F)
{
  TimerOutput::Scope timer(computing_timer,
                           "assemble_trace_boundary_equations");

  // Cell-block sub-vector for FEFaceValues
  Vector<double> y_cell(n_cell_dofs);
  for (types::global_dof_index i = 0; i < n_cell_dofs; ++i)
    y_cell[i] = y[i];

  const FEValuesExtractors::Scalar Aex(0);
  const FEValuesExtractors::Scalar Uex(1);

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
          fef[Aex].get_function_values(y_cell, A_int_v);
          fef[Uex].get_function_values(y_cell, U_int_v);

          const double A_int = std::max(A_int_v[0], 1e-10);
          const double U_int = U_int_v[0];
          const double c0    = compute_wave_speed(vpp.a_d, vid);
          const double c_int = compute_wave_speed(A_int, vid);

          // Current face trace (from trace block of y)
          double A_hat_cur = 0.0, U_hat_cur = 0.0;
          get_face_trace(y, cell, f, A_hat_cur, U_hat_cur);
          const double A_hat = std::max(A_hat_cur, 1e-10);
          const double c_hat = compute_wave_speed(A_hat, vid);

          double res_A = 0.0, res_U = 0.0;

          if (bid == 0) // inflow
            {
              const double Q_in   = inflow_function.value(Point<1>(t));
              const double W2_int = U_int - 4.0 * (c_int - c0);

              res_A = A_hat_cur * U_hat_cur - Q_in;
              res_U = (U_hat_cur - 4.0 * (c_hat - c0)) - W2_int;
            }
          else if (outlet_type == "RCR" && rcr_map.count(bid) &&
                   rcr_map.at(bid).R1 > 0.0)
            {
              const auto  &rcr    = rcr_map.at(bid);
              const double Pc     = terminal_Pc_storage.at(bid);
              const double Q      = A_hat_cur * U_hat_cur;
              const double W1_int = U_int + 4.0 * (c_int - c0);

              res_A = compute_pressure_value(A_hat, vid) - (rcr.R1 * Q + Pc);
              res_U = (U_hat_cur + 4.0 * (c_hat - c0)) - W1_int;
            }
          else // Reflection outlet
            {
              const double Rt     = par["Rt"];
              const double W1_int = U_int + 4.0 * (c_int - c0);
              const double W2_tgt = -Rt * W1_int;

              res_A = (U_hat_cur + 4.0 * (c_hat - c0)) - W1_int;
              res_U = (U_hat_cur - 4.0 * (c_hat - c0)) - W2_tgt;
            }

          const FaceTraceDof &td = face_dof_map.at(canonical_face_key(cell, f));
          F[td.a_hat_dof]        = res_A;
          F[td.u_hat_dof]        = res_U;
        }
    }
}

// ============================================================================
// assemble_trace_junction_equations
//
// At a K-way junction the 2K trace unknowns {A_hat_i, U_hat_i} for
// i = 0 … K−1 must satisfy:
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
// Total: 1 + (K−1) + K = 2K equations — exactly the same system that the
// original code solved via a nested Newton inside junction_solver.solve().
//
// Why Riemann invariants, not kinematic decoupling?
//   * The Riemann invariant W_i is the characteristic quantity that travels
//     *out* of cell i toward the junction.  Requiring the trace to satisfy
//     U_hat_i + s_i·4(c_hat_i−c0_i) = W_i is the exact characteristic
//     matching condition at a wave boundary — it is the same as enforcing
//     that no spurious reflection is introduced at the junction face.
//   * Kinematic decoupling (U_hat_i = U_cell_i) would force the trace to
//     equal the one-sided interior value, which misses the effect of the
//     other K−1 vessels and produces a wrong star state.
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
void
BloodFlowSystem<dim, spacedim>::assemble_trace_junction_equations(
  const Vector<double> &y,
  Vector<double>       &F)
{
  TimerOutput::Scope timer(computing_timer,
                           "assemble_trace_junction_equations");

  if (junctions.empty())
    return;

  // Cell-block sub-vector for FEFaceValues
  Vector<double> y_cell(n_cell_dofs);
  for (types::global_dof_index i = 0; i < n_cell_dofs; ++i)
    y_cell[i] = y[i];

  const FEValuesExtractors::Scalar Aex(0);
  const FEValuesExtractors::Scalar Uex(1);

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
          fef[Aex].get_function_values(y_cell, Av);
          fef[Uex].get_function_values(y_cell, Uv);

          A_int[i] = std::max(Av[0], A_min);
          U_int[i] = Uv[0];
          s[i]     = hf.orientation; // +1 if junction is at face 1 (right end)
          c0[i]    = compute_wave_speed(vpp.a_d, vid);

          // Outgoing Riemann invariant from cell i toward the junction
          const double c_i = compute_wave_speed(A_int[i], vid);
          W[i] = U_int[i] + static_cast<double>(s[i]) * 4.0 * (c_i - c0[i]);

          // Trace DOF indices and current trace values (from trace block of y)
          const FaceTraceDof &td =
            face_dof_map.at(canonical_face_key(hf.cell, hf.face_no));
          a_idx[i] = td.a_hat_dof;
          u_idx[i] = td.u_hat_dof;

          A_hat[i] = std::max(y[a_idx[i]], A_min);
          U_hat[i] = y[u_idx[i]];
          c_hat[i] = compute_wave_speed(A_hat[i], vid);
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
        const double H0 =
          0.5 * U_hat[0] * U_hat[0] +
          compute_pressure_value(A_hat[0],
                                 J.half_faces[0].cell->material_id()) /
            rho;

        for (unsigned int i = 1; i < K; ++i)
          {
            const double Hi =
              0.5 * U_hat[i] * U_hat[i] +
              compute_pressure_value(A_hat[i],
                                     J.half_faces[i].cell->material_id()) /
                rho;
            F[u_idx[i - 1]] = H0 - Hi;
          }
      }

      // (c) Riemann compatibility: U_hat_i + s_i*4(c_hat_i−c0_i) − W_i = 0
      //     vessel 0 -> row u_idx[K−1]
      //     vessel i≥1 -> row a_idx[i]
      for (unsigned int i = 0; i < K; ++i)
        {
          const double compat_res =
            U_hat[i] + static_cast<double>(s[i]) * 4.0 * (c_hat[i] - c0[i]) -
            W[i];

          const types::global_dof_index row =
            (i == 0) ? u_idx[K - 1] : a_idx[i];

          F[row] = compat_res;
        }
    }
}

// ============================================================================
// assemble_implicit_function
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
void
BloodFlowSystem<dim, spacedim>::assemble_implicit_function(
  const double          t,
  const Vector<double> &y,
  Vector<double>       &F)
{
  TimerOutput::Scope timer(computing_timer, "assemble_implicit_function");
  deallog.push("assemble_implicit_function");
  deallog << "t=" << t << std::endl;

  AssertDimension(y.size(), n_total_dofs);
  F.reinit(n_total_dofs);

  // ---- Assemble raw residuals for all DOFs --------------------------------
  assemble_cell_residuals(t, y, F);
  assemble_trace_interior_equations(y, F);
  assemble_trace_boundary_equations(t, y, F);
  assemble_trace_junction_equations(y, F);

  // ---- Apply M_K^-1 to cell rows ------------------------------------------
  // The cell block of F currently holds  F_cell(y).  
  const unsigned int n_dofs = fe->n_dofs_per_cell();
  Vector<double>     local_F(n_dofs);
  Vector<double>     local_MiF(n_dofs);

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      std::vector<types::global_dof_index> ldofs(n_dofs);
      cell->get_dof_indices(ldofs);

      for (unsigned int i = 0; i < n_dofs; ++i)
        local_F(i) = F[ldofs[i]];

      per_cell_mass_inv.at(cell->id()).vmult(local_MiF, local_F);

      for (unsigned int i = 0; i < n_dofs; ++i)
        F[ldofs[i]] = local_MiF(i);
    }

  deallog.pop();
}

// ============================================================================
// assemble_jacobian_cell_block
//
// Differentiates the cell residuals w.r.t. cell DOFs (block 1,1) and
// w.r.t. trace DOFs (block 1,2).
// ============================================================================
template <int dim, int spacedim>
void
BloodFlowSystem<dim, spacedim>::assemble_jacobian_cell_block(
  const double          t,
  const Vector<double> &y)
{
  TimerOutput::Scope timer(computing_timer, "assemble_jacobian_cell_block");

  // Cell-block sub-vector for FEValues::get_function_values
  Vector<double> y_cell(n_cell_dofs);
  for (types::global_dof_index i = 0; i < n_cell_dofs; ++i)
    y_cell[i] = y[i];

  const FEValuesExtractors::Scalar Aex(0);
  const FEValuesExtractors::Scalar Uex(1);

  const QGauss<dim>     quad_cell(fe->tensor_degree() + 1);
  const QGauss<dim - 1> quad_face(fe->tensor_degree() + 1);

  FEValues<dim, spacedim> fev(
    *fe, quad_cell, update_values | update_gradients | update_JxW_values);
  FEFaceValues<dim, spacedim> fef(
    *fe, quad_face, update_values | update_JxW_values | update_normal_vectors);

  const double rho = par["rho"];
  const double eta = 2.0 * (par["xi"] + 2.0) * numbers::PI * par["mu"] / rho;

  (void)t;

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      const unsigned int vid    = cell->material_id();
      const unsigned int n_dofs = fe->n_dofs_per_cell();

      fev.reinit(cell);
      std::vector<types::global_dof_index> ldofs(n_dofs);
      cell->get_dof_indices(ldofs);

      std::vector<double> A_h(fev.n_quadrature_points);
      std::vector<double> U_h(fev.n_quadrature_points);
      fev[Aex].get_function_values(y_cell, A_h);
      fev[Uex].get_function_values(y_cell, U_h);

      FullMatrix<double> cell_mat(n_dofs, n_dofs);

      // ---- Block (1,1): cell–cell derivatives (volume) --------------------
      for (unsigned int q = 0; q < fev.n_quadrature_points; ++q)
        {
          const double A    = std::max(A_h[q], 1e-10);
          const double U    = U_h[q];
          const double dpdA = compute_pressure_derivative(A, vid);
          const double c2_A = A / rho * dpdA; // c^2

          for (unsigned int i = 0; i < n_dofs; ++i)
            {
              const unsigned int ci = fe->system_to_component_index(i).first;
              const Tensor<1, spacedim> grad_phiA = fev[Aex].gradient(i, q);
              const Tensor<1, spacedim> grad_phiU = fev[Uex].gradient(i, q);
              const double              phi_U     = fev[Uex].value(i, q);

              for (unsigned int j = 0; j < n_dofs; ++j)
                {
                  const double              trial_A = fev[Aex].value(j, q);
                  const double              trial_U = fev[Uex].value(j, q);
                  const Tensor<1, spacedim> b =
                    compute_directional_vector(cell);

                  double contrib = 0.0;
                  if (ci == 0) // row: area equation
                    contrib = (U * trial_A + A * trial_U) * (b * grad_phiA);
                  else // row: velocity equation
                    contrib =
                      (c2_A / A * trial_A + U * trial_U) * (b * grad_phiU) -
                      eta * (trial_U / A - U * trial_A / (A * A)) * phi_U;

                  cell_mat(i, j) += contrib * fev.JxW(q);
                }
            }
        }

      // Scatter cell–cell block
      for (unsigned int i = 0; i < n_dofs; ++i)
        for (unsigned int j = 0; j < n_dofs; ++j)
          jacobian_matrix.add(ldofs[i], ldofs[j], cell_mat(i, j));

      // ---- Block (1,2): cell–trace derivatives (face fluxes) --------------
      for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
        {
          fef.reinit(cell, f);
          const auto &normals = fef.get_normal_vectors();
          const auto &JxW     = fef.get_JxW_values();

          std::vector<double> Ah_q(fef.n_quadrature_points);
          std::vector<double> Uh_q(fef.n_quadrature_points);
          fef[Aex].get_function_values(y_cell, Ah_q);
          fef[Uex].get_function_values(y_cell, Uh_q);

          double A_hat_cur = 0.0, U_hat_cur = 0.0;
          get_face_trace(y, cell, f, A_hat_cur, U_hat_cur);
          const double A_hat = std::max(A_hat_cur, 1e-10);

          const auto  key = canonical_face_key(cell, f);
          const auto &td  = face_dof_map.at(key);

          for (unsigned int q = 0; q < fef.n_quadrature_points; ++q)
            {
              const double A_in = std::max(Ah_q[q], 1e-10);
              const double U_in = Uh_q[q];
              const double bn =
                compute_tangent_normal_product(cell, normals[q]);

              // ∂F/∂(A_int,U_int) — cell block (1,1) face contributions
              for (unsigned int j = 0; j < n_dofs; ++j)
                {
                  const double dA = fef[Aex].value(j, q);
                  const double dU = fef[Uex].value(j, q);

                  const auto [dFA, dFU] = numerical_flux_jac(bn,
                                                             bn,
                                                             A_in,
                                                             U_in,
                                                             A_hat,
                                                             U_hat_cur,
                                                             dA,
                                                             dU,
                                                             0.0,
                                                             0.0,
                                                             vid,
                                                             vid);

                  for (unsigned int i = 0; i < n_dofs; ++i)
                    {
                      const unsigned int comp =
                        fe->system_to_component_index(i).first;
                      const double phi = fef.shape_value(i, q);
                      jacobian_matrix.add(ldofs[i],
                                          ldofs[j],
                                          -(comp == 0 ? dFA : dFU) * phi *
                                            JxW[q]);
                    }
                }

              // ∂F/∂(A_hat,U_hat) — cross block (1,2)
              for (const auto trace_col : {td.a_hat_dof, td.u_hat_dof})
                {
                  const bool   is_a_col = (trace_col == td.a_hat_dof);
                  const double dA_hat   = is_a_col ? 1.0 : 0.0;
                  const double dU_hat   = is_a_col ? 0.0 : 1.0;

                  const auto [dFA, dFU] = numerical_flux_jac(bn,
                                                             bn,
                                                             A_in,
                                                             U_in,
                                                             A_hat,
                                                             U_hat_cur,
                                                             0.0,
                                                             0.0,
                                                             dA_hat,
                                                             dU_hat,
                                                             vid,
                                                             vid);

                  for (unsigned int i = 0; i < n_dofs; ++i)
                    {
                      const unsigned int comp =
                        fe->system_to_component_index(i).first;
                      const double phi = fef.shape_value(i, q);
                      jacobian_matrix.add(ldofs[i],
                                          trace_col,
                                          -(comp == 0 ? dFA : dFU) * phi *
                                            JxW[q]);
                    }
                }
            }
        }
    }
}

// ============================================================================
// assemble_jacobian_trace_interior_block
//
// Differentiates the interior-face trace equations w.r.t. cell DOFs (2,1)
// and trace DOFs (2,2).
//
// Trace equations (from assemble_trace_interior_equations):
//   R1 = U_hat + 4(c_hat - c0) - [U_L + 4(c_L - c0)]   = 0
//   R2 = U_hat - 4(c_hat - c0) - [U_R - 4(c_R - c0)]   = 0
//
// ∂R1/∂A_L  = -4 * dc_L/dA_L
// ∂R1/∂U_L  = -1
// ∂R1/∂A_hat = 4 * dc_hat/dA_hat    (from both R1 and R2)
// ∂R1/∂U_hat = 1
// (Similarly for R2)
// ============================================================================
template <int dim, int spacedim>
void
BloodFlowSystem<dim, spacedim>::assemble_jacobian_trace_interior_block(
  const Vector<double> &y)
{
  TimerOutput::Scope timer(computing_timer,
                           "assemble_jacobian_trace_interior_block");

  Vector<double> y_cell(n_cell_dofs);
  for (types::global_dof_index i = 0; i < n_cell_dofs; ++i)
    y_cell[i] = y[i];

  const FEValuesExtractors::Scalar Aex(0);
  const FEValuesExtractors::Scalar Uex(1);

  const QGauss<dim - 1>       quad_face(1);
  FEFaceValues<dim, spacedim> fef(*fe, quad_face, update_values);

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

          fef.reinit(cell, f);
          std::vector<double> A_Lv(1), U_Lv(1);
          fef[Aex].get_function_values(y_cell, A_Lv);
          fef[Uex].get_function_values(y_cell, U_Lv);

          fef.reinit(nb, nb_f);
          std::vector<double> A_Rv(1), U_Rv(1);
          fef[Aex].get_function_values(y_cell, A_Rv);
          fef[Uex].get_function_values(y_cell, U_Rv);

          const double A_L = std::max(A_Lv[0], 1e-10);
          const double A_R = std::max(A_Rv[0], 1e-10);
         // const double c0_L =
          //  compute_wave_speed(vessel_map.at(vid_L).a_d, vid_L);
          //const double c0_R =
          //  compute_wave_speed(vessel_map.at(vid_R).a_d, vid_R);
         // const double c0 = 0.5 * (c0_L + c0_R);

          const double dc_L = compute_wave_speed_derivative(A_L, vid_L);
          const double dc_R = compute_wave_speed_derivative(A_R, vid_R);

          double A_hat_cur = 0.0, U_hat_cur = 0.0;
          get_face_trace(y, cell, f, A_hat_cur, U_hat_cur);
          const double A_hat  = std::max(A_hat_cur, 1e-10);
          const double dc_hat = compute_wave_speed_derivative(A_hat, vid_L);

          const auto                   &td    = face_dof_map.at(key);
          const types::global_dof_index a_row = td.a_hat_dof;
          const types::global_dof_index u_row = td.u_hat_dof;

          // ∂R1/∂(A_hat, U_hat)
          jacobian_matrix.add(a_row, a_row, 4.0 * dc_hat);
          jacobian_matrix.add(a_row, u_row, 1.0);

          // ∂R2/∂(A_hat, U_hat)
          jacobian_matrix.add(u_row, a_row, -4.0 * dc_hat);
          jacobian_matrix.add(u_row, u_row, 1.0);

          // ∂R1/∂(A_L, U_L): need cell DOFs at face quadrature point
          {
            fef.reinit(cell, f);
            std::vector<types::global_dof_index> ldofs_L(fe->n_dofs_per_cell());
            cell->get_dof_indices(ldofs_L);

            for (unsigned int j = 0; j < fe->n_dofs_per_cell(); ++j)
              {
                const double phi_A = fef[Aex].value(j, 0);
                const double phi_U = fef[Uex].value(j, 0);
                // ∂W1_L/∂w_L = phi_U + 4*dc_L*phi_A
                const double dW1_dw = phi_U + 4.0 * dc_L * phi_A;
                // ∂R1/∂w_L = -∂W1_L/∂w_L
                jacobian_matrix.add(a_row, ldofs_L[j], -dW1_dw);
              }
          }

          // ∂R2/∂(A_R, U_R)
          {
            fef.reinit(nb, nb_f);
            std::vector<types::global_dof_index> ldofs_R(fe->n_dofs_per_cell());
            nb->get_dof_indices(ldofs_R);

            for (unsigned int j = 0; j < fe->n_dofs_per_cell(); ++j)
              {
                const double phi_A = fef[Aex].value(j, 0);
                const double phi_U = fef[Uex].value(j, 0);
                // ∂W2_R/∂w_R = phi_U - 4*dc_R*phi_A
                const double dW2_dw = phi_U - 4.0 * dc_R * phi_A;
                jacobian_matrix.add(u_row, ldofs_R[j], -dW2_dw);
              }
          }
        }
    }
}

// ============================================================================
// assemble_jacobian_trace_boundary_block
// ============================================================================
template <int dim, int spacedim>
void
BloodFlowSystem<dim, spacedim>::assemble_jacobian_trace_boundary_block(
  const double          t,
  const Vector<double> &y)
{
  TimerOutput::Scope timer(computing_timer,
                           "assemble_jacobian_trace_boundary_block");

  Vector<double> y_cell(n_cell_dofs);
  for (types::global_dof_index i = 0; i < n_cell_dofs; ++i)
    y_cell[i] = y[i];

  const FEValuesExtractors::Scalar Aex(0);
  const FEValuesExtractors::Scalar Uex(1);

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
          //const auto              &vpp = vessel_map.at(vid);

          fef.reinit(cell, f);
          std::vector<double> A_int_v(1), U_int_v(1);
          fef[Aex].get_function_values(y_cell, A_int_v);
          fef[Uex].get_function_values(y_cell, U_int_v);

          const double A_int  = std::max(A_int_v[0], 1e-10);
          // const double U_int  = U_int_v[0];
          // const double c0     = compute_wave_speed(vpp.a_d, vid);
          const double dc_int = compute_wave_speed_derivative(A_int, vid);

          double A_hat_cur = 0.0, U_hat_cur = 0.0;
          get_face_trace(y, cell, f, A_hat_cur, U_hat_cur);
          const double A_hat    = std::max(A_hat_cur, 1e-10);
          const double dc_hat   = compute_wave_speed_derivative(A_hat, vid);
          const double dPdA_hat = compute_pressure_derivative(A_hat, vid);

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

              // ∂R1/∂A_hat, ∂R1/∂U_hat
              jacobian_matrix.add(a_row, a_row, U_hat_cur);
              jacobian_matrix.add(a_row, u_row, A_hat);

              // ∂R2/∂A_hat, ∂R2/∂U_hat
              jacobian_matrix.add(u_row, a_row, -4.0 * dc_hat);
              jacobian_matrix.add(u_row, u_row, 1.0);

              // ∂R2/∂(A_int, U_int) — block (2,1)
              for (unsigned int j = 0; j < fe->n_dofs_per_cell(); ++j)
                {
                  const double phi_A = fef[Aex].value(j, 0);
                  const double phi_U = fef[Uex].value(j, 0);
                  // ∂W2_int/∂w_int = phi_U - 4*dc_int*phi_A
                  jacobian_matrix.add(u_row,
                                      ldofs[j],
                                      -(phi_U - 4.0 * dc_int * phi_A));
                }
            }
          else if (outlet_type == "RCR" && rcr_map.count(bid) &&
                   rcr_map.at(bid).R1 > 0.0)
            {
              const auto &rcr = rcr_map.at(bid);

              // R1 = P(A_hat) - R1*(A_hat*U_hat) - Pc
              // R2 = U_hat + 4(c_hat - c0) - W1_int

              // ∂R1/∂A_hat, ∂R1/∂U_hat
              jacobian_matrix.add(a_row, a_row, dPdA_hat - rcr.R1 * U_hat_cur);
              jacobian_matrix.add(a_row, u_row, -rcr.R1 * A_hat);

              // ∂R2/∂A_hat, ∂R2/∂U_hat
              jacobian_matrix.add(u_row, a_row, 4.0 * dc_hat);
              jacobian_matrix.add(u_row, u_row, 1.0);

              // ∂R2/∂(A_int, U_int)
              for (unsigned int j = 0; j < fe->n_dofs_per_cell(); ++j)
                {
                  const double phi_A = fef[Aex].value(j, 0);
                  const double phi_U = fef[Uex].value(j, 0);
                  jacobian_matrix.add(u_row,
                                      ldofs[j],
                                      -(phi_U + 4.0 * dc_int * phi_A));
                }
            }
          else // Reflection
            {
              const double Rt = par["Rt"];

              // R1 = U_hat + 4(c_hat - c0) - W1_int
              // R2 = U_hat - 4(c_hat - c0) - (-Rt * W1_int)

              // ∂R1/∂A_hat, ∂R1/∂U_hat
              jacobian_matrix.add(a_row, a_row, 4.0 * dc_hat);
              jacobian_matrix.add(a_row, u_row, 1.0);

              // ∂R2/∂A_hat, ∂R2/∂U_hat
              jacobian_matrix.add(u_row, a_row, -4.0 * dc_hat);
              jacobian_matrix.add(u_row, u_row, 1.0);

              // ∂R1 and ∂R2 /∂(A_int,U_int) via W1_int
              for (unsigned int j = 0; j < fe->n_dofs_per_cell(); ++j)
                {
                  const double phi_A = fef[Aex].value(j, 0);
                  const double phi_U = fef[Uex].value(j, 0);
                  const double dW1   = phi_U + 4.0 * dc_int * phi_A;
                  jacobian_matrix.add(a_row, ldofs[j], -dW1);
                  jacobian_matrix.add(u_row, ldofs[j], Rt * dW1);
                }
            }
        }
    }
}

// ============================================================================
// assemble_jacobian_trace_junction_block
//
// Differentiates the junction residuals (mass conservation, total-head
// continuity, Riemann compatibility) w.r.t. all cell and trace DOFs.
// ============================================================================
template <int dim, int spacedim>
void
BloodFlowSystem<dim, spacedim>::assemble_jacobian_trace_junction_block(
  const Vector<double> &y)
{
  TimerOutput::Scope timer(computing_timer,
                           "assemble_jacobian_trace_junction_block");

  Vector<double> y_cell(n_cell_dofs);
  for (types::global_dof_index i = 0; i < n_cell_dofs; ++i)
    y_cell[i] = y[i];

  const FEValuesExtractors::Scalar Aex(0);
  const FEValuesExtractors::Scalar Uex(1);

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
          fef[Aex].get_function_values(y_cell, Av);
          fef[Uex].get_function_values(y_cell, Uv);

          A_int[i]    = std::max(Av[0], A_min);
          U_int[i]    = Uv[0];
          orient[i]   = hf.orientation;
          c0[i]       = compute_wave_speed(vessel_map.at(vid).a_d, vid);
          dc_int_v[i] = compute_wave_speed_derivative(A_int[i], vid);

          const auto  key = canonical_face_key(hf.cell, hf.face_no);
          const auto &td  = face_dof_map.at(key);
          a_row[i]        = td.a_hat_dof;
          u_row[i]        = td.u_hat_dof;

          A_hat[i]    = std::max(y[a_row[i]], A_min);
          U_hat[i]    = y[u_row[i]];
          c_hat_v[i]  = compute_wave_speed(A_hat[i], vid);
          dc_hat_v[i] = compute_wave_speed_derivative(A_hat[i], vid);
          dP_hat[i]   = compute_pressure_derivative(A_hat[i], vid);

          cell_dofs[i].resize(fe->n_dofs_per_cell());
          hf.cell->get_dof_indices(cell_dofs[i]);
        }

      // Row a_row[0]: mass conservation ∑ s_i A_hat_i U_hat_i = 0
      for (unsigned int i = 0; i < K; ++i)
        {
          const double s = static_cast<double>(orient[i]);
          jacobian_matrix.add(a_row[0], a_row[i], s * U_hat[i]);
          jacobian_matrix.add(a_row[0], u_row[i], s * A_hat[i]);
        }

      // Rows u_row[0..K-2]: H_0 − H_i = 0
      for (unsigned int i = 1; i < K; ++i)
        {
          // ∂H_0/∂A_hat_0 , ∂H_0/∂U_hat_0
          jacobian_matrix.add(u_row[i - 1], a_row[0], dP_hat[0] / rho);
          jacobian_matrix.add(u_row[i - 1], u_row[0], U_hat[0]);
          // ∂(-H_i)/∂A_hat_i, ∂(-H_i)/∂U_hat_i
          jacobian_matrix.add(u_row[i - 1], a_row[i], -dP_hat[i] / rho);
          jacobian_matrix.add(u_row[i - 1], u_row[i], -U_hat[i]);
        }

      // Compat row for vessel 0 -> u_row[K-1]
      // Compat row for vessel i≥1 -> a_row[i]
      for (unsigned int i = 0; i < K; ++i)
        {
          const double                  s  = static_cast<double>(orient[i]);
          const types::global_dof_index rr = (i == 0) ? u_row[K - 1] : a_row[i];

          // ∂/∂A_hat_i, ∂/∂U_hat_i of (U_hat_i + s*4(c_hat_i - c0_i) - W_i)
          jacobian_matrix.add(rr, a_row[i], s * 4.0 * dc_hat_v[i]);
          jacobian_matrix.add(rr, u_row[i], 1.0);

          // ∂(-W_i)/∂(A_int_i, U_int_i)
          const unsigned int vid = J.half_faces[i].cell->material_id();
          fef.reinit(J.half_faces[i].cell, J.half_faces[i].face_no);

          for (unsigned int j = 0; j < fe->n_dofs_per_cell(); ++j)
            {
              const double phi_A = fef[Aex].value(j, 0);
              const double phi_U = fef[Uex].value(j, 0);
              // ∂W_i/∂w = phi_U + s*4*dc_int_v[i]*phi_A
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
void
BloodFlowSystem<dim, spacedim>::assemble_jacobian(
  const double          t,
  const Vector<double> &y,
  const Vector<double> & /*Mydot*/)
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

  // ---- Apply M_K^-1 to the cell rows of the Jacobian ---------------------
  // For cell K with local DOFs {r_0,...,r_p}, rows r_i in jacobian_matrix
  // represent  d F_cell|_K / d y.  Replace them with
  //   (M_K^-1 * (d F_cell|_K / d y))_i  for all columns j.
  //
  // We do this by extracting each cell's dense row block, multiplying by
  // M_K^-1, and writing back.  The column extent covers all n_total_dofs
  // columns that can be non-zero for that cell's rows (from sparsity).
  const unsigned int n_dofs = fe->n_dofs_per_cell();

  // Collect all column indices that appear in at least one cell row
  // (needed to iterate the sparse row efficiently).
  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      std::vector<types::global_dof_index> ldofs(n_dofs);
      cell->get_dof_indices(ldofs);

      const FullMatrix<double> &Minv = per_cell_mass_inv.at(cell->id());

      // Gather all column indices that are structurally non-zero in ANY of
      // the rows belonging to this cell.
      std::vector<types::global_dof_index> col_indices;
      for (unsigned int i = 0; i < n_dofs; ++i)
        for (auto it = jacobian_matrix.begin(ldofs[i]);
             it != jacobian_matrix.end(ldofs[i]);
             ++it)
          col_indices.push_back(it->column());

      // Unique-sort so we process each column once
      std::sort(col_indices.begin(), col_indices.end());
      col_indices.erase(std::unique(col_indices.begin(), col_indices.end()),
                        col_indices.end());

      // For each column j, extract the local column vector, multiply by
      // M_K^-1, and scatter back.
      Vector<double> col_in(n_dofs), col_out(n_dofs);
      for (const types::global_dof_index j : col_indices)
        {
          for (unsigned int i = 0; i < n_dofs; ++i)
            col_in(i) = jacobian_matrix.el(ldofs[i], j);

          Minv.vmult(col_out, col_in);

          for (unsigned int i = 0; i < n_dofs; ++i)
            jacobian_matrix.set(ldofs[i], j, col_out(i));
        }
    }

  deallog.pop();
}

// ============================================================================
// update_terminal_pressures
// ============================================================================
template <int dim, int spacedim>
void
BloodFlowSystem<dim, spacedim>::update_terminal_pressures(
  const double          dt,
  const Vector<double> &y)
{
  // Accumulate volumetric flow at each terminal face from trace unknowns
  for (const auto &[bid, Pc_old] : terminal_Pc_storage)
    {
      const auto &rcr = rcr_map.at(bid);

      // Find the face trace for this boundary
      double A_hat = 0.0, U_hat = 0.0;
      for (const auto &cell : dof_handler.active_cell_iterators())
        for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
          if (cell->face(f)->at_boundary() &&
              cell->face(f)->boundary_id() == bid)
            {
              get_face_trace(y, cell, f, A_hat, U_hat);
            }

      const double Q     = A_hat * U_hat;
      const double denom = 1.0 + dt / (rcr.C * rcr.R2);
      const double Pc_new =
        (Pc_old + (dt / rcr.C) * Q + (dt / (rcr.C * rcr.R2)) * rcr.P_out) /
        denom;

      terminal_Pc_storage[bid] = Pc_new;
    }
}

// ============================================================================
// compute_pressure
// ============================================================================
template <int dim, int spacedim>
void
BloodFlowSystem<dim, spacedim>::compute_pressure(const Vector<double> &y,
                                                 Vector<double>       &p) const
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
            p[ldofs[i]]    = compute_pressure_value(A, cell->material_id());
          }
    }
}

// ============================================================================
// compute_theoretical_peak
// ============================================================================
template <int dim, int spacedim>
void
BloodFlowSystem<dim, spacedim>::compute_theoretical_peak(
  Vector<double> &tp) const
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
void
BloodFlowSystem<dim, spacedim>::output_results(
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
  std::vector<DataComponentInterpretation::DataComponentInterpretation> interp(
    2, DataComponentInterpretation::component_is_scalar);

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
void
BloodFlowSystem<dim, spacedim>::compute_errors(const unsigned int k)
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
    return (k == 0 || prev == 0.0) ? 0.0 : std::log(prev / cur) / std::log(2.0);
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
void
BloodFlowSystem<dim, spacedim>::run()
{
  std::cout << "=== Blood Flow HDG  (cell DOFs + global face traces) ===\n";

  for (unsigned int cycle = 0; cycle < n_refinement_cycles; ++cycle)
    {
      std::cout << "\n--- Cycle " << cycle << " ---\n";

      if (cycle == 0)
        {
          // Load mesh and VTK data
          dealii::GridIn<dim, spacedim> grid_in;
          grid_in.attach_triangulation(triangulation);
          std::ifstream mesh_file(vtk_file_path);
          grid_in.read_vtk(mesh_file);

          VTKUtils::read_cell_data(vtk_file_path, "vessel_id", cell_vessel_ids);
          VTKUtils::read_cell_data(vtk_file_path, "a0", cell_a0);
          VTKUtils::read_cell_data(vtk_file_path, "a_d", cell_a_d);
          VTKUtils::read_cell_data(vtk_file_path, "E", cell_E);
          VTKUtils::read_cell_data(vtk_file_path, "h_wall", cell_h_wall);
          VTKUtils::read_cell_data(vtk_file_path, "p_d", cell_p_d);
          VTKUtils::read_cell_data(vtk_file_path, "p0", cell_p0);
          VTKUtils::read_cell_data(vtk_file_path, "L", cell_L);
          VTKUtils::read_cell_data(vtk_file_path, "r_d", cell_r_d);

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

                for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell;
                     ++f)
                  if (cell->face(f)->at_boundary())
                    {
                      const unsigned int v_idx = cell->face(f)->vertex_index(0);
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
            for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
              if (cell->face(f)->at_boundary())
                {
                  const types::boundary_id bid = cell->face(f)->boundary_id();
                  const unsigned int       v   = cell->face(f)->vertex_index(0);

                  RCRPhysics rcr;
                  rcr.R1    = point_R1[v];
                  rcr.R2    = point_R2[v];
                  rcr.C     = point_C[v];
                  rcr.P_out = point_P_out[v];

                  if (rcr.R1 > 0.0)
                    rcr_map[bid] = rcr;
                }

          triangulation.refine_global(n_global_refinements);
        }
      else
        {
          triangulation.refine_global(1);
        }

      setup_system();
      initialize_terminal_capacitors();
      compute_theoretical_peak(theoretical_peak);
      // Build per-cell mass inverses (replaces global mass matrix)
      build_per_cell_mass_inv();
      compute_initial_solution(solution, arkode_parameters.initial_time);
      time = arkode_parameters.initial_time;

      // ARKode setup
      // No mass-matrix callbacks: ARKode sees M = I.
      // The cell equations carry M_K^-1 already folded into implicit_function
      // and assemble_jacobian.  Trace equations are algebraic (f_trace = 0).
      SUNDIALS::ARKode<Vector<double>> ode(arkode_parameters);

      ode.implicit_function =
        [this](const double t, const Vector<double> &y, Vector<double> &F) {
          assemble_implicit_function(t, y, F);
        };

      ode.jacobian_times_setup = [this](const double          t,
                                        const Vector<double> &y,
                                        const Vector<double> &Mydot) {
        assemble_jacobian(t, y, Mydot);
      };

      ode.jacobian_times_vector = [this](const Vector<double> &v,
                                         Vector<double>       &Jv,
                                         const double,
                                         const Vector<double> &,
                                         const Vector<double> &) {
        TimerOutput::Scope t(computing_timer, "jacobian_times_vector");
        jacobian_matrix.vmult(Jv, v);
      };

      // Preconditioner approximates (I - gamma * J)^-1.
      // No mass matrix: linear_system_matrix = I - gamma * J.
      ode.jacobian_preconditioner_setup = [this](const double,
                                                 const Vector<double> &,
                                                 const Vector<double> &,
                                                 const int    jok,
                                                 int         &jcur,
                                                 const double gamma) {
        TimerOutput::Scope t(computing_timer, "preconditioner_setup");
        if (jok == SUNFALSE)
          {
            // I - gamma * J: start from identity then subtract
            linear_system_matrix = 0.0;
            for (types::global_dof_index i = 0; i < n_total_dofs; ++i)
              linear_system_matrix.set(i, i, 1.0);
            linear_system_matrix.add(-gamma, jacobian_matrix);
            linear_solver.initialize(linear_system_matrix);
            jcur       = SUNTRUE;
            current_dt = std::abs(gamma);
          }
        else
          {
            jcur = SUNFALSE;
          }
      };

      ode.jacobian_preconditioner_solve = [this](const double,
                                                 const Vector<double> &,
                                                 const Vector<double> &,
                                                 const Vector<double> &r,
                                                 Vector<double>       &z,
                                                 const double,
                                                 const double,
                                                 const int) {
        TimerOutput::Scope t(computing_timer, "preconditioner_solve");
        linear_solver.vmult(z, r);
      };

      ode.output_step = [this](const double          t,
                               const Vector<double> &sol,
                               const unsigned int /*step_number*/) {
        const unsigned int actual_step =
          static_cast<unsigned int>((t - arkode_parameters.initial_time) /
                                    arkode_parameters.output_period);
        time = t;
        compute_pressure(sol, pressure);
        compute_theoretical_peak(theoretical_peak);
        output_results(sol, pressure, theoretical_peak, actual_step);
      };

      while (time < arkode_parameters.final_time)
        {
          const unsigned int n_steps =
            ode.solve_ode_incrementally(solution,
                                        time + arkode_parameters.output_period,
                                        true);

          std::cout << "  ARKode steps: " << n_steps << "  t=" << time << "\n";

          update_terminal_pressures(current_dt, solution);
          time += arkode_parameters.output_period;
        }

      compute_pressure(solution, pressure);
      compute_errors(cycle);
    }
}

// Explicit instantiation
template class BloodFlowSystem<1, 3>;