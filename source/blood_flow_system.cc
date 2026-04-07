/* --------------------------------------------------------------------------
 * Blood Flow Simulation in 1D using DG method in space with Lax Friedruch and
 * HLL fluxes and SUNDIALS ARKode for time integration.
 * Author: Luca Heltai, Raksha Devi
 * --------------------------------------------------------------------------
 */
#include "blood_flow_system.h"

#include <deal.II/base/function_parser.h>
#include <deal.II/base/types.h>

#include <deal.II/grid/cell_data.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/tria.h>
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

#include "../include/junction_solver.h"
#include "vtk_utils.h"

// Main class implementation
template <int dim, int spacedim>
BloodFlowSystem<dim, spacedim>::BloodFlowSystem()
  : par("Blood Flow Parameters",
        {"rho",
         "a0",
         "a_d",
         "E",
         "h_wall",
         "p0",
         "p_d",
         "mu",
         "xi",
         "r0",
         "m",
         "Rt",
         "R1",
         "R2",
         "C",
         "P_out",
         "L"},

        {1.06,
         3.141592653589793e-4,
         0.28274e-4,
         1.0e6,
         1.0e-3,
         0.0,
         0.0,
         0.0,
         9.0,
         9.99e-3,
         0.5,
         0,
         1.0e3,
         1.0e4,
         1.0e-6,
         0.0,
         10.0},

        {"Density",
         "Reference cross-sectional area",
         "Area at diastolic pressure",
         "Elastic modulus of the vessel wall",
         "Wall thickness",
         "Reference pressure",
         "Diastolic pressure",
         "Viscosity coefficient",
         "Profile constant for friction term",
         "Reference radius",
         "Tube law exponent",
         "Reflection coefficient at outflow boundary",
         "Resistance of proximal capacitor",
         "Resistance of distal capacitor",
         "Capacitance of terminal capacitor",
         "Pressure at terminal boundary",
         "Length of the vessel"})
  , triangulation()
  , dof_handler(triangulation)
  , fe(nullptr)
  , initial_condition(
      "Functions",
      "1e-4; 0.0",
      "Initial condition",
      par,
      dealii::FunctionParser<spacedim>::default_variable_names() + ",t")
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
  //, inflow_function("Functions", "0.0", "Inflow function", par, "t")
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

  this->enter_subsection("ARKOde parameters");
  this->enter_my_subsection(this->prm);
  arkode_parameters.add_parameters(this->prm);
  this->leave_my_subsection(this->prm);
  this->leave_subsection();
}

// ========================================================================
// INITIALIZE PARAMETERS
// ========================================================================

template <int dim, int spacedim>
void
BloodFlowSystem<dim, spacedim>::initialize_params(const std::string &filename)
{
  TimerOutput::Scope timer(computing_timer, "initialize_params");
  ParameterAcceptor::initialize(filename,
                                "last_used_parameters.prm",
                                ParameterHandler::Short,
                                this->prm,
                                ParameterHandler::Short);

  deallog.depth_console(verbosity);
  deallog.depth_file(verbosity);

  exact_solution.update_constants(par);
  initial_condition.update_constants(par);
  rhs_function.update_constants(par);
  inflow_function.update_constants(par);

  if (numerical_flux_type_str == "HLL")
    numerical_flux_type = NumericalFluxType::HLL;
  else if (numerical_flux_type_str == "LAX_FRIEDRICHS")
    numerical_flux_type = NumericalFluxType::LAX_FRIEDRICHS;
  else if (numerical_flux_type_str == "HLL_SYMPY")
    numerical_flux_type = NumericalFluxType::HLL_SYMPY;
  else
    AssertThrow(false,
                ExcMessage("Unknown numerical flux type: " +
                           numerical_flux_type_str));
  deallog << "Selected numerical flux: " << numerical_flux_type_str
          << std::endl;
}

// For terminal boundary conditions, we need to initialize the storage for the
// capacitor pressures
template <int dim, int spacedim>
bool
BloodFlowSystem<dim, spacedim>::is_terminal_boundary(
  const dealii::types::boundary_id bid) const
{
  return terminal_boundary_ids.count(bid) > 0;
}

// ========================================================================
// For general (K>=3) junction detection
// ========================================================================

template <int dim, int spacedim>
void
BloodFlowSystem<dim, spacedim>::detect_junctions()
{
  junctions.clear();
  all_junction_faces.clear();

  // 1. Count how many cells are attached to each vertex
  // In 1D, a vertex with more than 2 cells attached is a junction.
  // A vertex with 2 cells is just a standard connection.
  std::map<unsigned int,
           std::vector<
             std::pair<typename DoFHandler<dim, spacedim>::active_cell_iterator,
                       unsigned int>>>
    vertex_to_faces;

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v)
        {
          unsigned int v_index = cell->vertex_index(v);
          // In 1D, vertex index 0 is face 0, vertex index 1 is face 1
          vertex_to_faces[v_index].push_back({cell, v});
        }
    }

  // 2. Iterate through the map and find junctions
  for (const auto &[v_idx, attached_faces] : vertex_to_faces)
    {
      // In a bifurcation, 3 cells meet at 1 vertex.
      // If it's 2, it's just a continuous pipe (usually handled by deal.II
      // automatically).
      // If we want to solve the junction equations manually,
      // use >= 2 then every inlet vertex is treated as junction and we solve
      // the junction equations at inlets as well.
      if (attached_faces.size() > 2)
        {
          JunctionInfo<dim, spacedim> J;
          J.point = triangulation.get_vertices()[v_idx];

          for (const auto &face_info : attached_faces)
            {
              JunctionFace<dim, spacedim> jf;
              jf.cell        = face_info.first;
              jf.face_no     = face_info.second; // In 1D, face_no == vertex_no
              jf.orientation = (jf.face_no == 1) ? 1 : -1;

              J.faces.push_back(jf);

              // Mark for assembly
              all_junction_faces.insert(
                std::make_pair(jf.cell->id(), jf.face_no));
            }
          junctions.push_back(J);
        }
    }

  std::cout << "Detected " << junctions.size() << " junctions." << std::endl;
}


// For solution at junction

template <int dim, int spacedim>
void
BloodFlowSystem<dim, spacedim>::compute_junction_residual(
  const JunctionState               &X,
  const std::vector<double>         &W_in,
  const JunctionInfo<dim, spacedim> &junction,
  Vector<double>                    &R) const
{
  const unsigned int K    = junction.n_vessels();
  const double       rho  = par["rho"];
  const double       Amin = 1e-10;

  // 1. Mass Conservation: Sum(s_i * A_i * U_i) = 0
  R[0] = 0.0;
  for (unsigned int i = 0; i < K; ++i)
    {
      double s = static_cast<double>(junction.faces[i].orientation);
      R[0] += s * X.A(i) * X.U(i);
    }

  // 2. Pressure Continuity: H_0 - H_i = 0
  double H0 = 0.5 * std::pow(X.U(0), 2) + compute_pressure_value(X.A(0)) / rho;
  for (unsigned int i = 1; i < K; ++i)
    {
      double Hi =
        0.5 * std::pow(X.U(i), 2) + compute_pressure_value(X.A(i)) / rho;
      R[i] = H0 - Hi;
    }

  // 3. Compatibility: U_i + s_i * 4c_i - W_in_i = 0
  for (unsigned int i = 0; i < K; ++i)
    {
      double s = static_cast<double>(junction.faces[i].orientation);
      double c = compute_wave_speed(std::max(X.A(i), Amin));
      R[K + i] = (X.U(i) + s * 4.0 * c) - W_in[i];
    }
}

template <int dim, int spacedim>
void
BloodFlowSystem<dim, spacedim>::compute_junction_jacobian(
  const JunctionState               &X,
  const JunctionInfo<dim, spacedim> &junction,
  FullMatrix<double>                &J) const
{
  const unsigned int K    = junction.n_vessels();
  const double       rho  = par["rho"];
  const double       Amin = 1e-10;
  J                       = 0.0;

  for (unsigned int i = 0; i < K; ++i)
    {
      const double s    = static_cast<double>(junction.faces[i].orientation);
      const double Ai   = std::max(X.A(i), Amin);
      const double Ui   = X.U(i);
      const double dPdA = compute_pressure_derivative(Ai);
      const double dcdA = compute_wave_speed_derivative(Ai);

      // Row 0: Mass Conservation derivatives
      J(0, 2 * i)     = s * Ui; // dR/dAi
      J(0, 2 * i + 1) = s * Ai; // dR/dUi

      // Rows 1 to K-1: Pressure Continuity
      if (i == 0)
        { // Derivatives of H0 relative to A0, U0
          for (unsigned int row = 1; row < K; ++row)
            {
              J(row, 0) = dPdA / rho;
              J(row, 1) = Ui;
            }
        }
      else
        { // Derivatives of -Hi relative to Ai, Ui
          J(i, 2 * i)     = -dPdA / rho;
          J(i, 2 * i + 1) = -Ui;
        }

      // Rows K to 2K-1: Compatibility
      unsigned int row_c  = K + i;
      J(row_c, 2 * i)     = s * 4.0 * dcdA;
      J(row_c, 2 * i + 1) = 1.0;
    }
}

template <int dim, int spacedim>
void
BloodFlowSystem<dim, spacedim>::get_trace(
  const Vector<double>                                           &vec,
  const typename DoFHandler<dim, spacedim>::active_cell_iterator &cell,
  unsigned int                                                    face_no,
  double                                                         &A,
  double                                                         &U) const
{
  // Simple trace evaluator for DG (using midpoint of the 1D face/vertex)
  const FEValuesExtractors::Scalar area_extractor(0);
  const FEValuesExtractors::Scalar velocity_extractor(1);

  // 1D face is a 0D point, QGauss<0>(1) is a single point
  QGauss<dim - 1>             quad(1);
  FEFaceValues<dim, spacedim> fe_face(*fe, quad, update_values);
  fe_face.reinit(cell, face_no);

  std::vector<double> A_vals(1), U_vals(1);
  fe_face[area_extractor].get_function_values(vec, A_vals);
  fe_face[velocity_extractor].get_function_values(vec, U_vals);

  A = A_vals[0];
  U = U_vals[0];
}

// ---------------------------------------------------------------------------------
// Assemble the residual contributions from the junctions by applying
// the numerical fluxes at the junction faces using the star state as the
// exterior trace.
// ---------------------------------------------------------------------------------

template <int dim, int spacedim>
void
BloodFlowSystem<dim, spacedim>::assemble_junction_residual(
  const Vector<double> &y,
  Vector<double>       &rhs)
{
  if (junctions.empty())
    return;

  const FEValuesExtractors::Scalar area_extractor(0);
  const FEValuesExtractors::Scalar velocity_extractor(1);

  QGauss<dim - 1> quad(1);

  FEFaceValues<dim, spacedim> fe_face(
    *fe, quad, update_values | update_JxW_values | update_normal_vectors);

  const double A_min = 1e-10;

  for (const auto &J : junctions)
    {
      const unsigned int K = J.faces.size();

      std::vector<double> A(K), U(K), W_in(K);

      // =====================================================
      // 1. Extract interior traces
      // =====================================================
      for (unsigned int i = 0; i < K; ++i)
        {
          get_trace(y, J.faces[i].cell, J.faces[i].face_no, A[i], U[i]);

          const double s = static_cast<double>(J.faces[i].orientation);

          const double c = compute_wave_speed(std::max(A[i], A_min));

          // incoming invariant
          W_in[i] = U[i] + s * 4.0 * c;
        }

      // =====================================================
      // 2. Solve junction nonlinear system
      // =====================================================
      JunctionState X0;
      X0.X.reinit(2 * K);

      for (unsigned int i = 0; i < K; ++i)
        {
          X0.A(i) = A[i];
          X0.U(i) = U[i];
        }

      JunctionState X = junction_solver.solve(X0, W_in, J, *this);

      // =====================================================
      // 3. Assemble DG flux using star state
      // =====================================================
      for (unsigned int face_idx = 0; face_idx < K; ++face_idx)
        {
          const auto &fd = J.faces[face_idx];

          fe_face.reinit(fd.cell, fd.face_no);

          const auto &normals = fe_face.get_normal_vectors();
          const auto &JxW     = fe_face.get_JxW_values();

          const unsigned int n_q = fe_face.n_quadrature_points;

          std::vector<double> Ah(n_q), Uh(n_q);

          fe_face[area_extractor].get_function_values(y, Ah);
          fe_face[velocity_extractor].get_function_values(y, Uh);

          std::vector<types::global_dof_index> dofs(fe->n_dofs_per_cell());
          fd.cell->get_dof_indices(dofs);

          for (unsigned int q = 0; q < n_q; ++q)
            {
              const double Ah_safe = std::max(Ah[q], A_min);

              const double A_star = std::max(X.A(face_idx), A_min);
              const double U_star = X.U(face_idx);

              auto [FA, FU] = numerical_flux(
                compute_tangent_normal_product(fd.cell, normals[q]),
                compute_tangent_normal_product(fd.cell, normals[q]),
                Ah_safe,
                Uh[q],
                A_star,
                U_star);

              for (unsigned int i = 0; i < fe->n_dofs_per_cell(); ++i)
                {
                  const unsigned int comp =
                    fe->system_to_component_index(i).first;

                  const double phi_i = fe_face.shape_value(i, q);

                  rhs(dofs[i]) -= (comp == 0 ? FA : FU) * phi_i * JxW[q];
                }
            }
        }
    }
}

// .........................................................
// Assemble the Jacobian contributions through Arkode from the junctions by
// applying the chain rule and the numerical fluxes at the junction faces. This
// is a complex process that involves computing the sensitivity of the star
// state to variations in the interior states
// and then applying the Jacobian of the numerical flux with respect to these
// variations.
// ................................................................

template <int dim, int spacedim>
void
BloodFlowSystem<dim, spacedim>::assemble_junction_jacobian(
  const Vector<double> &y)
{
  if (fe->dofs_per_vertex > 0 || junctions.empty())
    return;
  TimerOutput::Scope timer(computing_timer, "assemble_junction_jacobian");
  const FEValuesExtractors::Scalar area_extractor(0);
  const FEValuesExtractors::Scalar velocity_extractor(1);

  QGauss<dim - 1> quad(1);

  FEFaceValues<dim, spacedim> fe_face(
    *fe, quad, update_values | update_JxW_values | update_normal_vectors);

  const double A_min = 1e-10;

  for (const auto &J : junctions)
    {
      const unsigned int K = J.faces.size();

      std::vector<double> A(K), U(K), W_in(K);

      // ----------------------------------------------------
      // 1. Extract traces from ARKode state y
      // ----------------------------------------------------
      for (unsigned int i = 0; i < K; ++i)
        {
          get_trace(y, J.faces[i].cell, J.faces[i].face_no, A[i], U[i]);

          const double s = static_cast<double>(J.faces[i].orientation);
          const double c = compute_wave_speed(std::max(A[i], A_min));

          W_in[i] = U[i] + s * 4.0 * c;
        }

      // ----------------------------------------------------
      // 2. Solve junction nonlinear system
      // ----------------------------------------------------
      JunctionState X0;
      X0.X.reinit(2 * K);

      for (unsigned int i = 0; i < K; ++i)
        {
          X0.A(i) = A[i];
          X0.U(i) = U[i];
        }

      JunctionState X = junction_solver.solve(X0, W_in, J, *this);

      // ----------------------------------------------------
      // 3. Junction Jacobian inverse
      // ----------------------------------------------------
      FullMatrix<double> JX(2 * K, 2 * K);
      compute_junction_jacobian(X, J, JX);

      FullMatrix<double> JX_inv = JX;
      JX_inv.gauss_jordan();

      // ----------------------------------------------------
      // 4. Assemble each junction face
      // ----------------------------------------------------
      for (unsigned int face_idx = 0; face_idx < K; ++face_idx)
        {
          const auto &fd = J.faces[face_idx];

          fe_face.reinit(fd.cell, fd.face_no);

          const auto &normals = fe_face.get_normal_vectors();
          const auto &JxW     = fe_face.get_JxW_values();

          const unsigned int n_q = fe_face.n_quadrature_points;

          std::vector<double> Ah(n_q), Uh(n_q);

          fe_face[area_extractor].get_function_values(y, Ah);
          fe_face[velocity_extractor].get_function_values(y, Uh);

          std::vector<types::global_dof_index> dofs(fe->n_dofs_per_cell());
          fd.cell->get_dof_indices(dofs);

          for (unsigned int q = 0; q < n_q; ++q)
            {
              const double Ah_safe = std::max(Ah[q], A_min);

              const double A_star = std::max(X.A(face_idx), A_min);
              const double U_star = X.U(face_idx);

              for (unsigned int j = 0; j < fe->n_dofs_per_cell(); ++j)
                {
                  const double phi_A_j = fe_face[area_extractor].value(j, q);

                  const double phi_U_j =
                    fe_face[velocity_extractor].value(j, q);

                  const double bn_L =
                    compute_tangent_normal_product(fd.cell, normals[q]);
                  const double bn_R = bn_L;
                  ;
                  // --------------------------------------------------
                  // (a) Direct DG derivative
                  // --------------------------------------------------
                  auto [dFA_dwh, dFU_dwh] = numerical_flux_jacobian(bn_L,
                                                                    bn_R,
                                                                    Ah_safe,
                                                                    Uh[q],
                                                                    A_star,
                                                                    U_star,
                                                                    phi_A_j,
                                                                    phi_U_j,
                                                                    0.0,
                                                                    0.0);


                  // --------------------------------------------------
                  // (b) Star state sensitivity
                  // --------------------------------------------------
                  const double s     = static_cast<double>(fd.orientation);
                  const double dc_dA = compute_wave_speed_derivative(Ah_safe);

                  const double dWin_dwh = phi_U_j + s * 4.0 * dc_dA * phi_A_j;

                  Vector<double> dX_dWin(2 * K);

                  for (unsigned int r = 0; r < 2 * K; ++r)
                    dX_dWin[r] = JX_inv(r, K + face_idx);

                  // --------------------------------------------------
                  // (c) Chain rule via star state
                  // --------------------------------------------------
                  auto [dFA_chain, dFU_chain] =
                    numerical_flux_jacobian(bn_L,
                                            bn_R,
                                            Ah_safe,
                                            Uh[q],
                                            A_star,
                                            U_star,
                                            0.0,
                                            0.0,
                                            dX_dWin[2 * face_idx] * dWin_dwh,
                                            dX_dWin[2 * face_idx + 1] *
                                              dWin_dwh);

                  for (unsigned int i = 0; i < fe->n_dofs_per_cell(); ++i)
                    {
                      const unsigned int comp =
                        fe->system_to_component_index(i).first;

                      const double phi_i = fe_face.shape_value(i, q);

                      jacobian_matrix.add(dofs[i],
                                          dofs[j],
                                          -((comp == 0 ? dFA_dwh + dFA_chain :
                                                         dFU_dwh + dFU_chain)) *
                                            phi_i * JxW[q]);
                    }
                }
            }
        }
    }
}


// ========================================================================
// SETUP SYSTEM
// ========================================================================

template <int dim, int spacedim>
void
BloodFlowSystem<dim, spacedim>::setup_system()
{
  TimerOutput::Scope timer(computing_timer, "setup_system");

  VTKUtils::read_vtk(
    "/home/rakshad/blood-flow/notebooks/bifurcation_physics.vtk",
    dof_handler,
    vtk_cell_data,
    cell_data_names);

  VTKUtils::read_data(
    "/home/rakshad/blood-flow/notebooks/bifurcation_physics.vtk",
    vtk_point_data);

  // --- 2. EXTRACT VESSEL PHYSICS (CELL DATA) ---
  vessel_map.clear();
  for (const auto &cell : triangulation.active_cell_iterators())
    {
      unsigned int v_id = cell->user_index(); // The 'vessel_id' from VTK
      if (vessel_map.find(v_id) == vessel_map.end())
        {
          VesselPhysicalProperties vp;
          // Helper to find column index by name (e.g., "E", "a0")
          vp.a0 = VTKUtils::get_cell_value(v_id,
                                           "a0",
                                           vtk_cell_data,
                                           cell_data_names);
          vp.E =
            VTKUtils::get_cell_value(v_id, "E", vtk_cell_data, cell_data_names);
          vp.h_wall        = VTKUtils::get_cell_value(v_id,
                                               "h_wall",
                                               vtk_cell_data,
                                               cell_data_names);
          vp.p_d           = VTKUtils::get_cell_value(v_id,
                                            "p_d",
                                            vtk_cell_data,
                                            cell_data_names);
          vessel_map[v_id] = vp;
        }
    }

  // --- 3. EXTRACT RCR PHYSICS (POINT DATA) ---
  rcr_map.clear();
  terminal_boundary_ids.clear();
  for (const auto &cell : triangulation.active_cell_iterators())
    for (unsigned int f : cell->face_indices())
      if (cell->face(f)->at_boundary())
        {
          unsigned int b_id  = cell->face(f)->boundary_id();
          unsigned int v_idx = cell->face(f)->vertex_index(0);

          RCRPhysics rcr;
          rcr.R1    = VTKUtils::get_point_value(v_idx,
                                             "R1",
                                             vtk_point_data,
                                             point_data_names);
          rcr.R2    = VTKUtils::get_point_value(v_idx,
                                             "R2",
                                             vtk_point_data,
                                             point_data_names);
          rcr.C     = VTKUtils::get_point_value(v_idx,
                                            "C",
                                            vtk_point_data,
                                            point_data_names);
          rcr.P_out = VTKUtils::get_point_value(v_idx,
                                                "P_out",
                                                vtk_point_data,
                                                point_data_names);

          // If R1 > 0, it's a Windkessel outlet
          if (rcr.R1 > 0)
            {
              rcr_map[b_id] = rcr;
              terminal_boundary_ids.insert(b_id);
              // Initialize Pc storage for this specific boundary
              terminal_Pc_storage[b_id] = 0.0;
            }
        }

  if (!fe)
    {
      // Two-component system: A (area) and U (velocity)
      fe = std::make_unique<FESystem<dim, spacedim>>(
        FE_DGQ<dim, spacedim>(fe_degree), 2);
    }

  dof_handler.distribute_dofs(*fe);

  junctions.clear();
  all_junction_faces.clear();
  detect_junctions();

  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_flux_sparsity_pattern(dof_handler, dsp);
  sparsity_pattern.copy_from(dsp);
  jacobian_matrix.reinit(sparsity_pattern); // for Newton iteration
  mass_matrix.reinit(sparsity_pattern);
  linear_system_matrix.reinit(sparsity_pattern);
  solution.reinit(dof_handler.n_dofs());
  pressure.reinit(dof_handler.n_dofs());
  theoretical_peak.reinit(dof_handler.n_dofs());
}

// ========================================================================
// ASSEMBLE Jacobian Linearization
// ========================================================================
template <int dim, int spacedim>
void
BloodFlowSystem<dim, spacedim>::assemble_jacobian(const double          t,
                                                  const Vector<double> &y,
                                                  const Vector<double> &Mydot)
{
  TimerOutput::Scope timer(computing_timer, "assemble_jacobian");
  deallog.push("assemble_jacobian");
  deallog << "Called assemble_jacobian t=" << t << std::endl;

  using Iterator = typename DoFHandler<dim, spacedim>::active_cell_iterator;

  (void)Mydot;

  jacobian_matrix = 0;

  const FEValuesExtractors::Scalar area_extractor(0);
  const FEValuesExtractors::Scalar velocity_extractor(1);

  // ========== CELL WORKER ==========
  auto cell_worker = [&](const Iterator                      &cell,
                         BloodFlowScratchData<dim, spacedim> &scratch_data,
                         BloodFlowCopyData                   &copy_data) {
    const unsigned int n_dofs =
      scratch_data.fe_values.get_fe().n_dofs_per_cell();
    copy_data.reinit(cell, n_dofs);
    scratch_data.fe_values.reinit(cell);

    const auto &fe_v = scratch_data.fe_values;
    const auto &JxW  = fe_v.get_JxW_values();

    // Current Newton Iteration Values
    std::vector<double> current_area(fe_v.n_quadrature_points);
    std::vector<double> current_velocity(fe_v.n_quadrature_points);

    fe_v[area_extractor].get_function_values(y, current_area);
    fe_v[velocity_extractor].get_function_values(y, current_velocity);

    for (unsigned int point = 0; point < fe_v.n_quadrature_points; ++point)
      {
        // Compute pressure derivative for wave speed
        const double dpdA = compute_pressure_derivative(current_area[point]);
        const double c_squared = current_area[point] / par["rho"] * dpdA;
        const double eta =
          2 * (par["xi"] + 2) * numbers::PI * par["mu"] / par["rho"];

        for (unsigned int i = 0; i < n_dofs; ++i)
          {
            // const auto test_area     = fe_v[area_extractor].value(i, point);
            const auto test_velocity = fe_v[velocity_extractor].value(i, point);

            const auto test_area_grad = fe_v[area_extractor].gradient(i, point);
            const auto test_velocity_grad =
              fe_v[velocity_extractor].gradient(i, point);

            for (unsigned int j = 0; j < n_dofs; ++j)
              {
                // ===== AREA EQUATION Jacobian =====
                const auto trial_area = fe_v[area_extractor].value(j, point);
                const auto trial_velocity =
                  fe_v[velocity_extractor].value(j, point);

                // Flux Jacobian: -b·∇(H_A * trial_W)
                // H_A = [U, A] so H_A·[trial_A, trial_U] = U*trial_A +
                // A*trial_U
                const auto flux_jacobian_A =
                  compute_physical_area_jacobian_flux(cell,
                                                      current_area[point],
                                                      trial_velocity,
                                                      current_velocity[point],
                                                      trial_area);

                copy_data.cell_matrix(i, j) +=
                  flux_jacobian_A * test_area_grad * JxW[point];

                // ===== MOMENTUM EQUATION Jacobian=====
                // Flux Jacobian: -b·∇(H_U * trial_W)
                // H_U = [c^2 /A, U] so H_U·[trial_A, trial_U] =
                // (c^2/A)*trial_A + U*trial_U
                const auto flux_jacobian_U =
                  compute_physical_momentum_jacobian_flux(
                    cell,
                    c_squared,
                    current_area[point],
                    trial_area,
                    current_velocity[point],
                    trial_velocity);

                copy_data.cell_matrix(i, j) +=
                  flux_jacobian_U * test_velocity_grad * JxW[point];

                // Reaction (viscosity): ∫ c U phi_U dx
                copy_data.cell_matrix(i, j) -=
                  eta * test_velocity *
                  (trial_velocity / current_area[point] -
                   current_velocity[point] /
                     (current_area[point] * current_area[point])) *
                  JxW[point];
              }
          }
      }
  };

  // ========== FACE WORKER (Interior Faces) - WITH HLL FLUX Jacobian ==========
  auto face_worker = [&](const Iterator                      &cell,
                         const unsigned int                   f,
                         const unsigned int                   sf,
                         const Iterator                      &ncell,
                         const unsigned int                   nf,
                         const unsigned int                   nsf,
                         BloodFlowScratchData<dim, spacedim> &scratch,
                         BloodFlowCopyData                   &copy) {
    FEInterfaceValues<dim, spacedim> &fe_iv = scratch.fe_interface_values;
    fe_iv.reinit(cell, f, sf, ncell, nf, nsf);

    const auto &JxW        = fe_iv.get_JxW_values();
    const auto &normals    = fe_iv.get_fe_face_values(0).get_normal_vectors();
    const unsigned int n_q = fe_iv.get_fe_face_values(0).n_quadrature_points;

    // Current Newton iterate values on both sides
    std::vector<double> current_area(n_q), current_velocity(n_q),
      current_area_neighbor(n_q), current_velocity_neighbor(n_q);

    fe_iv.get_fe_face_values(0)[area_extractor].get_function_values(
      y, current_area);
    fe_iv.get_fe_face_values(0)[velocity_extractor].get_function_values(
      y, current_velocity);
    fe_iv.get_fe_face_values(1)[area_extractor].get_function_values(
      y, current_area_neighbor);
    fe_iv.get_fe_face_values(1)[velocity_extractor].get_function_values(
      y, current_velocity_neighbor);

    copy.face_data.emplace_back();
    auto              &face = copy.face_data.back();
    const unsigned int nd   = fe_iv.n_current_interface_dofs();
    face.joint_dof_indices  = fe_iv.get_interface_dof_indices();
    face.cell_matrix.reinit(nd, nd);
    face.cell_rhs.reinit(nd);

    for (unsigned int q = 0; q < n_q; ++q)
      {
        double bn_L = compute_tangent_normal_product(cell, normals[q]);
        double bn_R = compute_tangent_normal_product(ncell, normals[q]);

        for (unsigned int j = 0; j < nd; ++j)
          {
            const auto trial_area = fe_iv[area_extractor].value(0, j, q);
            const auto trial_area_neighbor =
              fe_iv[area_extractor].value(1, j, q);
            const auto trial_velocity =
              fe_iv[velocity_extractor].value(0, j, q);
            const auto trial_velocity_neighbor =
              fe_iv[velocity_extractor].value(1, j, q);

            // ===== HLL JACOBIAN FLUX (for Jacobian matrix) =====
            auto [flux_A_jac, flux_U_jac] =
              numerical_flux_jacobian(bn_L,
                                      bn_R,
                                      current_area[q],
                                      current_velocity[q],
                                      current_area_neighbor[q],
                                      current_velocity_neighbor[q],
                                      trial_area,               // trial_A_L
                                      trial_velocity,           // trial_U_L
                                      trial_area_neighbor,      // trial_A_R
                                      trial_velocity_neighbor); // trial_U_R


            for (unsigned int i = 0; i < nd; ++i)
              {
                // Area equation contribution to Jacobian
                face.cell_matrix(i, j) -=
                  flux_A_jac * fe_iv[area_extractor].jump_in_values(i, q) *
                  JxW[q];

                // Momentum equation contribution to Jacobian
                face.cell_matrix(i, j) -=
                  flux_U_jac * fe_iv[velocity_extractor].jump_in_values(i, q) *
                  JxW[q];
              }
          }
      }
  };

  //========== BOUNDARY WORKER Jacobian - WITH HLL FLUX ==========
  exact_solution.set_time(t);
  auto boundary_worker_with_characteristics =
    [&](const Iterator                      &cell,
        const unsigned int                   face_no,
        BloodFlowScratchData<dim, spacedim> &scratch,
        BloodFlowCopyData                   &copy) {
      scratch.fe_interface_values.reinit(cell, face_no);
      const auto &fe_face = scratch.fe_interface_values.get_fe_face_values(0);

      const auto        &JxW     = fe_face.get_JxW_values();
      const auto        &normals = fe_face.get_normal_vectors();
      const unsigned int n_q     = fe_face.n_quadrature_points;

      // Interior trace (current Newton iterate)
      std::vector<double> current_area(n_q), current_velocity(n_q);
      fe_face[area_extractor].get_function_values(y, current_area);
      fe_face[velocity_extractor].get_function_values(y, current_velocity);

      // Boundary values
      exact_solution.set_time(t);
      std::vector<Vector<double>> bc(n_q, Vector<double>(2));
      exact_solution.vector_value_list(fe_face.get_quadrature_points(), bc);

      // Determine which boundary this is (BID=0 is inlet, BID=1 is outlet)
      unsigned int boundary_id = cell->face(face_no)->boundary_id();

      for (unsigned int q = 0; q < n_q; ++q)
        {
          const double A_bc = bc[q](0);
          const double U_bc = bc[q](1);

          // Interior state
          const double A_int = current_area[q];
          const double U_int = current_velocity[q];

          const double c_int = compute_wave_speed(A_int);

          // ===== CHARACTERISTIC SPEEDS =====
          const double lambda1 = (U_int - c_int);
          const double lambda2 = (U_int + c_int);

          // ===== BOUNDARY CLASSIFICATION =====
          // Count incoming characteristics
          int incoming_count = 0;
          if (lambda1 <= 0.0)
            incoming_count++;
          if (lambda2 <= 0.0)
            incoming_count++;

          bool is_subcritical_inflow =
            (incoming_count == 1) && (lambda1 <= 0.0);
          bool is_subcritical_outflow =
            (incoming_count == 1) && (lambda2 <= 0.0);
          bool is_supercritical_inflow  = (incoming_count == 2);
          bool is_supercritical_outflow = (incoming_count == 0);

          // ===== DETERMINE EXTERIOR STATE (A_ext, U_ext) =====
          double A_ext, U_ext;

          if (is_subcritical_inflow)
            {
              // Subcritical: first characteristic is incoming
              A_ext = A_bc;  // Impose area at boundary
              U_ext = U_int; // velocity from interior
            }
          else if (is_subcritical_outflow)
            {
              // Subcritical: second characteristic is incoming
              A_ext = A_int; // area from interior
              U_ext = U_bc;  // Impose velocity at boundary
            }
          else if (is_supercritical_inflow)
            {
              // Supercritical inflow: both characteristics enter
              A_ext = A_bc; // Impose area
              U_ext = U_bc; // Impose velocity
            }
          else if (is_supercritical_outflow)
            {
              // Supercritical outflow: no characteristics enter
              A_ext = A_int; // interior area
              U_ext = U_int; // interior velocity
            }
          else
            {
              Assert(false, ExcMessage("Unclassified boundary condition."));
            }

          double bn_L = compute_tangent_normal_product(cell, normals[q]);
          double bn_R = bn_L; // boundary face, same normal
          // ===== JACOBIAN ASSEMBLY AT BOUNDARY =====
          for (unsigned int j = 0; j < fe_face.get_fe().dofs_per_cell; ++j)
            {
              const double trial_A = fe_face[area_extractor].value(j, q);
              const double trial_U = fe_face[velocity_extractor].value(j, q);

              // ===== HLL JACOBIAN AT BOUNDARY =====
              // Only trial functions on interior side are active (j comes from
              // interior cell) Exterior state does not change with trial
              // functions
              auto [flux_A_jac, flux_U_jac] = numerical_flux_jacobian(
                bn_L,
                bn_R,
                A_int,   // interior current (left)
                U_int,   // interior current (left)
                A_ext,   // exterior (right) - fixed by boundary condition
                U_ext,   // exterior (right) - fixed by boundary condition
                trial_A, // trial_A_L (only interior varies)
                trial_U, // trial_U_L (only interior varies)
                0.0,     // trial_A_R = 0 (exterior is fixed)
                0.0);    // trial_U_R = 0 (exterior is fixed)


              // Test function loop
              for (unsigned int i = 0; i < fe_face.get_fe().dofs_per_cell; ++i)
                {
                  const double test_A = fe_face[area_extractor].value(i, q);
                  const double test_U = fe_face[velocity_extractor].value(i, q);

                  // Area equation Jacobian
                  copy.cell_matrix(i, j) -= flux_A_jac * test_A * JxW[q];

                  // Momentum equation Jacobian
                  copy.cell_matrix(i, j) -= flux_U_jac * test_U * JxW[q];
                }
            }

          deallog.push("boundary_debug");
          deallog << "BID=" << boundary_id << " q=" << q << " | A_int=" << A_int
                  << " U_int=" << U_int << " c=" << c_int
                  << " | lambda1=" << lambda1 << " lambda2=" << lambda2
                  << " | inlet=" << is_subcritical_inflow
                  << " outlet=" << is_subcritical_outflow
                  << " | A_ext=" << A_ext << " U_ext=" << U_ext << std::endl;
          deallog.pop();
        }
    };

  //========== BOUNDARY WORKER – CONSISTENT A/U RIEMANN BOUNDARY ==========
  const double dt                                = this->current_dt;
  auto         boundary_worker_riemann_invariant = [&](
                                             const Iterator    &cell,
                                             const unsigned int face_no,
                                             BloodFlowScratchData<dim, spacedim>
                                                               &scratch,
                                             BloodFlowCopyData &copy) {
    if (all_junction_faces.count({cell->id(), face_no}) > 0)
      return;

    scratch.fe_interface_values.reinit(cell, face_no);
    const auto &fe_face = scratch.fe_interface_values.get_fe_face_values(0);

    const auto        &JxW     = fe_face.get_JxW_values();
    const auto        &normals = fe_face.get_normal_vectors();
    const unsigned int n_q     = fe_face.n_quadrature_points;

    const double A_min = 1e-12;
    // const double dt_p = 1e-4;

    const unsigned int boundary_id = cell->face(face_no)->boundary_id();

    std::vector<double> Ah(n_q), Uh(n_q);
    fe_face[area_extractor].get_function_values(y, Ah);
    fe_face[velocity_extractor].get_function_values(y, Uh);

    for (unsigned int q = 0; q < n_q; ++q)
      {
        const double A_L     = std::max(Ah[q], A_min);
        const double U_L     = Uh[q];
        const double c_L     = compute_wave_speed(A_L);
        const double dc_dA_L = compute_wave_speed_derivative(A_L);
        const double Ad      = par["a_d"];
        const double c0      = compute_wave_speed(Ad);

        const double W1_L =
          U_L + 4.0 * (c_L - c0); // OUTGOING FOR TERMINAL BOUNDARY
        const double W2_L = U_L - 4.0 * (c_L - c0); // INCOMING FOR INLET

        double A_star = A_L;
        double U_star = U_L;

        FullMatrix<double> J_boundary_local(2, 2);

        if (boundary_id == 0) // INFLOW: Prescribed Q_in(t)
          {
            const double t_shifted = t; // + 0.055;
            inflow_function.set_time(t_shifted);
            const double Qin = inflow_function.value(Point<1>(t_shifted));
            for (unsigned int it = 0; it < 20; ++it)
              {
                const double A     = std::max(A_star, A_min);
                const double cA    = compute_wave_speed(A);
                const double dc_dA = compute_wave_speed_derivative(A);

                // R[0]: Prescribed Flow: A*U - Qin = 0
                // R[1]: Backward Invariant: U - 4c - W2_L = 0
                Vector<double> R(2);
                R[0] = A * U_star - Qin;
                R[1] = 4.0 * (cA - c0) + W2_L - U_star;


                FullMatrix<double> J(2, 2);
                J(0, 0) = U_star;
                J(0, 1) = A;
                J(1, 0) = 4.0 * dc_dA;
                J(1, 1) = -1.0;

                J_boundary_local = J; // Store for sensitivity
                J.gauss_jordan();
                Vector<double> delta(2);
                J.vmult(delta, R);

                A_star -= delta[0];
                U_star -= delta[1];
                if (delta.l2_norm() < 1e-12)
                  break;
              }
          }
        else // ================= OUTLET BOUNDARY =================
          {
            if (this->outlet_type == "RCR")
              {
                // ================= CHOICE: RCR =================
                const auto  &rcr = rcr_map.at(boundary_id);
                const double R1  = rcr.R1;
                const double R2  = rcr.R2;
                const double C   = rcr.C;


                double Pc_old = 0.0;
                try
                  {
                    Pc_old = terminal_Pc_storage.at(boundary_id);
                  }
                catch (const std::out_of_range &e)
                  {
                    AssertThrow(
                      false,
                      ExcMessage(
                        "Boundary ID not found in Pc storage. Initialize it in setup_system."));
                  }

                for (unsigned int it = 0; it < 20; ++it)
                  {
                    const double A  = std::max(A_star, 1e-12);
                    const double cA = compute_wave_speed(A);
                    const double Pe = compute_pressure_value(A);
                    const double U  = W1_L - 4.0 * (cA - c0);
                    const double Q  = A * U;
                    const double dQdA =
                      U + A * (-4.0 * compute_wave_speed_derivative(A));

                    // DISCRETIZATION: Use current_dt from ARKode
                    const double Pc_star =
                      (Pc_old + (current_dt / C) * (Q + par["P_out"] / R2)) /
                      (1.0 + dt / (C * R2));
                    const double dPc_star_dA =
                      (dt / C) * (dQdA) / (1.0 + dt / (C * R2));

                    const double F     = Pe - (R1 * Q + Pc_star);
                    const double dF_dA = compute_pressure_derivative(A) -
                                         (R1 * dQdA + dPc_star_dA);

                    A_star -= F / dF_dA;
                    if (std::abs(F / A_L) < 1e-12)
                      break;
                  }
                U_star = W1_L - 4.0 * (compute_wave_speed(A_star) - c0);
                const double dPc_dA_partial =
                  (dt / C) * U_star / (1.0 + dt / (C * R2));
                // Final state sync and Jacobian for sensitivity (Step 4)

                J_boundary_local(0, 0) = compute_pressure_derivative(A_star) -
                                         R1 * U_star - dPc_dA_partial;

                J_boundary_local(0, 1) = -R1 * A_star;
                J_boundary_local(1, 0) =
                  4.0 * compute_wave_speed_derivative(A_star);
                J_boundary_local(1, 1) = 1.0;
              }
            else // (outlet_type == "Rt")
              {
                // ================= CHOICE: Rt (Reflection) =================
                const double Rt = par["Rt"];
                const double W2_target =
                  -Rt * W1_L; // Simplified reflection target

                for (unsigned int it = 0; it < 20; ++it)
                  {
                    const double A     = std::max(A_star, A_min);
                    const double cA    = compute_wave_speed(A);
                    const double dc_dA = compute_wave_speed_derivative(A);

                    // System: W1 invariant is fixed, W2 invariant is target
                    // Eq1: U + 4(c-c0) - W1_L = 0
                    // Eq2: U - 4(c-c0) - W2_target = 0
                    Vector<double> R(2);
                    R[0] = U_star + 4.0 * (cA - c0) - W1_L;
                    R[1] = U_star - 4.0 * (cA - c0) - W2_target;

                    FullMatrix<double> J(2, 2);
                    J(0, 0) = 4.0 * dc_dA;
                    J(0, 1) = 1.0;
                    J(1, 0) = -4.0 * dc_dA;
                    J(1, 1) = 1.0;

                    J.gauss_jordan();
                    Vector<double> delta(2);
                    J.vmult(delta, R);

                    A_star -= delta[0];
                    U_star -= delta[1];
                    if (delta.l2_norm() < 1e-12)
                      break;
                  }

                // Local Jacobian for sensitivity (Step 4)
                J_boundary_local(0, 0) =
                  4.0 * compute_wave_speed_derivative(A_star);
                J_boundary_local(0, 1) = 1.0;
                J_boundary_local(1, 0) =
                  -4.0 * compute_wave_speed_derivative(A_star);
                J_boundary_local(1, 1) = 1.0;
              }
          }

        // ---------- 4. Newton-Consistent Jacobian (Sensitivity) ----------
        J_boundary_local.gauss_jordan();

        for (unsigned int j = 0; j < fe->n_dofs_per_cell(); ++j)
          {
            const double dAL_trial = fe_face[area_extractor].value(j, q);
            const double dUL_trial = fe_face[velocity_extractor].value(j, q);

            Vector<double> dRb_dInterior(2);
            if (boundary_id == 0) // Sensitivity to W2
              {
                dRb_dInterior[0] = 0;
                dRb_dInterior[1] = -(dUL_trial - 4.0 * dc_dA_L * dAL_trial);
              }
            else // Sensitivity to W1
              {
                dRb_dInterior[0] = 0;
                dRb_dInterior[1] = -(dUL_trial + 4.0 * dc_dA_L * dAL_trial);
              }

            Vector<double> dX_star(2);
            J_boundary_local.vmult(dX_star, dRb_dInterior);
            dX_star *= -1.0;

            double bn_L = compute_tangent_normal_product(cell, normals[q]);
            double bn_R = bn_L;

            auto [dFA_dir, dFU_dir]     = numerical_flux_jacobian(bn_L,
                                                              bn_R,
                                                              A_L,
                                                              U_L,
                                                              A_star,
                                                              U_star,
                                                              dAL_trial,
                                                              dUL_trial,
                                                              0.0,
                                                              0.0);
            auto [dFA_chain, dFU_chain] = numerical_flux_jacobian(bn_L,
                                                                  bn_R,
                                                                  A_L,
                                                                  U_L,
                                                                  A_star,
                                                                  U_star,
                                                                  0.0,
                                                                  0.0,
                                                                  dX_star[0],
                                                                  dX_star[1]);

            for (unsigned int i = 0; i < fe->n_dofs_per_cell(); ++i)
              {
                const unsigned int comp =
                  fe->system_to_component_index(i).first;
                const double phi = fe_face.shape_value(i, q);
                if (comp == 0)
                  copy.cell_matrix(i, j) -=
                    (dFA_dir + dFA_chain) * phi * JxW[q];
                else
                  copy.cell_matrix(i, j) -=
                    (dFU_dir + dFU_chain) * phi * JxW[q];
              }
          }
      }
  };

  // Copier lambda
  const AffineConstraints<double> constraints;
  auto                            copier = [&](const BloodFlowCopyData &c) {
    constraints.distribute_local_to_global(c.cell_matrix,
                                           c.local_dof_indices,
                                           jacobian_matrix);

    for (auto &cdf : c.face_data)
      {
        constraints.distribute_local_to_global(cdf.cell_matrix,
                                               cdf.joint_dof_indices,
                                               jacobian_matrix);
      }
  };

  // Execute mesh loop
  const QGauss<dim>     quadrature(fe->tensor_degree() + 1);
  const QGauss<dim - 1> quadrature_face(fe->tensor_degree() + 1);

  BloodFlowScratchData<dim, spacedim> scratch_data(*fe,
                                                   quadrature,
                                                   quadrature_face);
  BloodFlowCopyData                   copy_data;

  if (use_riemann_invariants)
    {
      MeshWorker::mesh_loop(dof_handler.begin_active(),
                            dof_handler.end(),
                            cell_worker,
                            copier,
                            scratch_data,
                            copy_data,
                            MeshWorker::assemble_own_cells |
                              MeshWorker::assemble_boundary_faces |
                              MeshWorker::assemble_own_interior_faces_once,
                            boundary_worker_riemann_invariant,
                            face_worker);
    }
  else
    {
      MeshWorker::mesh_loop(dof_handler.begin_active(),
                            dof_handler.end(),
                            cell_worker,
                            copier,
                            scratch_data,
                            copy_data,
                            MeshWorker::assemble_own_cells |
                              MeshWorker::assemble_boundary_faces |
                              MeshWorker::assemble_own_interior_faces_once,
                            boundary_worker_with_characteristics,
                            face_worker);
    }

  assemble_junction_jacobian(y);
  deallog.pop();
}

// ========================================================================
// ASSEMBLE IMPLICIT FUNCTION (WITHOUT dot_y TERM)
// ========================================================================
template <int dim, int spacedim>
void
BloodFlowSystem<dim, spacedim>::assemble_implicit_function(
  const double          t,
  const Vector<double> &y,
  Vector<double>       &Mydot)
{
  TimerOutput::Scope timer(computing_timer, "assemble_implicit_function");
  deallog.push("assemble_implicit_function");
  deallog << "Called assemble_implicit_function t=" << t << std::endl;

  using Iterator = typename DoFHandler<dim, spacedim>::active_cell_iterator;

  Mydot = 0;

  const FEValuesExtractors::Scalar area_extractor(0);
  const FEValuesExtractors::Scalar velocity_extractor(1);

  // ========== CELL WORKER ==========
  auto cell_worker = [&](const Iterator                      &cell,
                         BloodFlowScratchData<dim, spacedim> &scratch_data,
                         BloodFlowCopyData                   &copy_data) {
    const unsigned int n_dofs =
      scratch_data.fe_values.get_fe().n_dofs_per_cell();
    copy_data.reinit(cell, n_dofs);
    scratch_data.fe_values.reinit(cell);

    const auto &fe_v     = scratch_data.fe_values;
    const auto &JxW      = fe_v.get_JxW_values();
    const auto &q_points = fe_v.get_quadrature_points();

    std::vector<double> current_area(fe_v.n_quadrature_points);
    std::vector<double> current_velocity(fe_v.n_quadrature_points);

    fe_v[area_extractor].get_function_values(y, current_area);
    fe_v[velocity_extractor].get_function_values(y, current_velocity);

    rhs_function.set_time(t);

    for (unsigned int point = 0; point < fe_v.n_quadrature_points; ++point)
      {
        const double rhs_A_value = rhs_function.value(q_points[point], 0);
        const double rhs_U_value = rhs_function.value(q_points[point], 1);

        const double current_pressure =
          compute_pressure_value(current_area[point]);

        const auto current_flux_A =
          compute_physical_area_flux(cell,
                                     current_area[point],
                                     current_velocity[point]);

        const auto current_flux_U =
          compute_physical_momentum_flux(cell,
                                         current_velocity[point],
                                         current_velocity[point],
                                         current_pressure);
        const double eta =
          2 * (par["xi"] + 2) * numbers::PI * par["mu"] / par["rho"];

        for (unsigned int i = 0; i < n_dofs; ++i)
          {
            const auto test_area     = fe_v[area_extractor].value(i, point);
            const auto test_velocity = fe_v[velocity_extractor].value(i, point);

            const auto test_area_grad = fe_v[area_extractor].gradient(i, point);
            const auto test_velocity_grad =
              fe_v[velocity_extractor].gradient(i, point);

            copy_data.cell_rhs(i) += (
                                       // Area equation
                                       rhs_A_value * test_area +
                                       current_flux_A * test_area_grad
                                       // Momentum equation
                                       + rhs_U_value * test_velocity +
                                       current_flux_U * test_velocity_grad -
                                       eta * current_velocity[point] /
                                         current_area[point] * test_velocity) *
                                     JxW[point];
          }
      }
  };

  // ========== FACE WORKER (Interior Faces) - WITH HLL FLUX ==========
  auto face_worker = [&](const Iterator                      &cell,
                         const unsigned int                   f,
                         const unsigned int                   sf,
                         const Iterator                      &ncell,
                         const unsigned int                   nf,
                         const unsigned int                   nsf,
                         BloodFlowScratchData<dim, spacedim> &scratch,
                         BloodFlowCopyData                   &copy) {
    FEInterfaceValues<dim, spacedim> &fe_iv = scratch.fe_interface_values;
    fe_iv.reinit(cell, f, sf, ncell, nf, nsf);

    const auto &JxW        = fe_iv.get_JxW_values();
    const auto &normals    = fe_iv.get_fe_face_values(0).get_normal_vectors();
    const unsigned int n_q = fe_iv.get_fe_face_values(0).n_quadrature_points;

    std::vector<double> current_area(n_q), current_velocity(n_q),
      current_area_neighbor(n_q), current_velocity_neighbor(n_q);

    fe_iv.get_fe_face_values(0)[area_extractor].get_function_values(
      y, current_area);
    fe_iv.get_fe_face_values(0)[velocity_extractor].get_function_values(
      y, current_velocity);
    fe_iv.get_fe_face_values(1)[area_extractor].get_function_values(
      y, current_area_neighbor);
    fe_iv.get_fe_face_values(1)[velocity_extractor].get_function_values(
      y, current_velocity_neighbor);
    copy.face_data.emplace_back();
    auto              &face = copy.face_data.back();
    const unsigned int nd   = fe_iv.n_current_interface_dofs();
    face.joint_dof_indices  = fe_iv.get_interface_dof_indices();
    face.cell_matrix.reinit(nd, nd);
    face.cell_rhs.reinit(nd);

    for (unsigned int q = 0; q < n_q; ++q)
      {
        double bn_L = compute_tangent_normal_product(cell, normals[q]);
        double bn_R = compute_tangent_normal_product(ncell, normals[q]);

        auto [flux_A, flux_U] = numerical_flux(bn_L,
                                               bn_R,
                                               current_area[q],
                                               current_velocity[q],
                                               current_area_neighbor[q],
                                               current_velocity_neighbor[q]);


        for (unsigned int i = 0; i < nd; ++i)
          {
            face.cell_rhs(i) -=
              flux_A * fe_iv[area_extractor].jump_in_values(i, q) * JxW[q];

            face.cell_rhs(i) -=
              flux_U * fe_iv[velocity_extractor].jump_in_values(i, q) * JxW[q];
          }
      }
  };

  //========== BOUNDARY WORKER - WITH HLL FLUX ==========
  exact_solution.set_time(t);
  auto boundary_worker_with_characteristics =
    [&](const Iterator                      &cell,
        const unsigned int                   face_no,
        BloodFlowScratchData<dim, spacedim> &scratch,
        BloodFlowCopyData                   &copy) {
      scratch.fe_interface_values.reinit(cell, face_no);
      const auto &fe_face = scratch.fe_interface_values.get_fe_face_values(0);

      const auto        &JxW     = fe_face.get_JxW_values();
      const auto        &normals = fe_face.get_normal_vectors();
      const unsigned int n_q     = fe_face.n_quadrature_points;

      std::vector<double> current_area(n_q), current_velocity(n_q);
      fe_face[area_extractor].get_function_values(y, current_area);
      fe_face[velocity_extractor].get_function_values(y, current_velocity);

      exact_solution.set_time(t);
      std::vector<Vector<double>> bc(n_q, Vector<double>(2));
      exact_solution.vector_value_list(fe_face.get_quadrature_points(), bc);

      unsigned int boundary_id = cell->face(face_no)->boundary_id();

      for (unsigned int q = 0; q < n_q; ++q)
        {
          const double A_bc = bc[q](0);
          const double U_bc = bc[q](1);

          const double A_int = current_area[q];
          const double U_int = current_velocity[q];

          const double c_int = compute_wave_speed(A_int);

          const double lambda1 = (U_int - c_int);
          const double lambda2 = (U_int + c_int);

          int incoming_count = 0;
          if (lambda1 <= 0.0)
            incoming_count++;
          if (lambda2 <= 0.0)
            incoming_count++;

          bool is_subcritical_inflow =
            (incoming_count == 1) && (lambda1 <= 0.0);
          bool is_subcritical_outflow =
            (incoming_count == 1) && (lambda2 <= 0.0);
          bool is_supercritical_inflow  = (incoming_count == 2);
          bool is_supercritical_outflow = (incoming_count == 0);

          double A_ext = 0, U_ext = 0;

          if (is_subcritical_inflow)
            {
              A_ext = A_bc;
              U_ext = U_int;
            }
          else if (is_subcritical_outflow)
            {
              A_ext = A_int;
              U_ext = U_bc;
            }
          else if (is_supercritical_inflow)
            {
              A_ext = A_bc;
              U_ext = U_bc;
            }
          else if (is_supercritical_outflow)
            {
              A_ext = A_int;
              U_ext = U_int;
            }
          else
            {
              Assert(false, ExcMessage("Unclassified boundary condition."));
            }

          double bn_L = compute_tangent_normal_product(cell, normals[q]);
          double bn_R = bn_L; // boundary face, same normal
          auto [flux_A, flux_U] =
            numerical_flux(bn_L, bn_R, A_int, U_int, A_ext, U_ext);

          const double F_area_boundary     = flux_A;
          const double F_momentum_boundary = flux_U;

          for (unsigned int i = 0; i < fe_face.get_fe().dofs_per_cell; ++i)
            {
              const double test_A = fe_face[area_extractor].value(i, q);
              const double test_U = fe_face[velocity_extractor].value(i, q);

              copy.cell_rhs(i) -= F_area_boundary * test_A * JxW[q];
              copy.cell_rhs(i) -= F_momentum_boundary * test_U * JxW[q];
            }

          deallog.push("boundary_debug");
          deallog << "BID=" << boundary_id << " q=" << q << " | A_int=" << A_int
                  << " U_int=" << U_int << " c=" << c_int
                  << " | lambda1=" << lambda1 << " lambda2=" << lambda2
                  << " | inlet=" << is_subcritical_inflow
                  << " outlet=" << is_subcritical_outflow
                  << " | A_ext=" << A_ext << " U_ext=" << U_ext << std::endl;
          deallog.pop();
        }
    };

  //========== BOUNDARY WORKER – CONSISTENT A/U RIEMANN BOUNDARY ==========
  const double dt                                = this->current_dt;
  auto         boundary_worker_riemann_invariant = [&](
                                             const Iterator    &cell,
                                             const unsigned int face_no,
                                             BloodFlowScratchData<dim, spacedim>
                                                               &scratch,
                                             BloodFlowCopyData &copy) {
    if (all_junction_faces.count({cell->id(), face_no}) > 0)
      return;

    scratch.fe_interface_values.reinit(cell, face_no);
    const auto &fe_face = scratch.fe_interface_values.get_fe_face_values(0);

    const auto        &JxW     = fe_face.get_JxW_values();
    const auto        &normals = fe_face.get_normal_vectors();
    const unsigned int n_q     = fe_face.n_quadrature_points;

    const double A_min = 1e-12;
    // const double dt_p = 1e-4;

    const unsigned int boundary_id = cell->face(face_no)->boundary_id();

    std::vector<double> Ah(n_q), Uh(n_q);
    fe_face[area_extractor].get_function_values(y, Ah);
    fe_face[velocity_extractor].get_function_values(y, Uh);

    for (unsigned int q = 0; q < n_q; ++q)
      {
        const double A_L = std::max(Ah[q], A_min);
        const double U_L = Uh[q];
        const double c_L = compute_wave_speed(A_L);
        const double Ad  = par["a_d"];
        const double c0  = compute_wave_speed(Ad);

        const double W1_L =
          U_L + 4.0 * (c_L - c0); // OUTGOING FOR TERMINAL BOUNDARY
        const double W2_L = U_L - 4.0 * (c_L - c0); // INCOMING FOR INLET

        double A_star = A_L;
        double U_star = U_L;

        FullMatrix<double> J_boundary_local(2, 2);
        if (boundary_id == 0) // INFLOW: Prescribed Q_in(t)
          {
            const double t_shifted = t; // + 0.055;
            inflow_function.set_time(t_shifted);
            const double Qin = inflow_function.value(Point<1>(t_shifted));
            // std::cout << "t=" << t << " Qin=" << Qin << std::endl;
            for (unsigned int it = 0; it < 20; ++it)
              {
                const double A     = std::max(A_star, A_min);
                const double cA    = compute_wave_speed(A);
                const double dc_dA = compute_wave_speed_derivative(A);

                // R[0]: Prescribed Flow: A*U - Qin = 0
                // R[1]: Backward Invariant: U - 4c - W2_L = 0
                Vector<double> R(2);
                R[0] = A * U_star - Qin;
                R[1] = U_star - 4.0 * (cA - c0) - W2_L;

                FullMatrix<double> J(2, 2);
                J(0, 0) = U_star;
                J(0, 1) = A;
                J(1, 0) = -4.0 * dc_dA;
                J(1, 1) = 1.0;

                J_boundary_local = J; // Store for sensitivity
                J.gauss_jordan();
                Vector<double> delta(2);
                J.vmult(delta, R);

                A_star -= delta[0];
                U_star -= delta[1];
                if (delta.l2_norm() < 1e-12)
                  break;
              }
          }
        else // ================= OUTLET BOUNDARY =================
          {
            if (this->outlet_type == "RCR")
              {
                // ================= CHOICE: RCR =================
                const auto  &rcr = rcr_map.at(boundary_id);
                const double R1  = rcr.R1;
                const double R2  = rcr.R2;
                const double C   = rcr.C;

                double Pc_old = 0.0;
                try
                  {
                    Pc_old = terminal_Pc_storage.at(boundary_id);
                  }
                catch (const std::out_of_range &e)
                  {
                    AssertThrow(
                      false,
                      ExcMessage(
                        "Boundary ID not found in Pc storage. Initialize it in setup_system."));
                  }

                for (unsigned int it = 0; it < 20; ++it)
                  {
                    const double A  = std::max(A_star, 1e-12);
                    const double cA = compute_wave_speed(A);
                    const double Pe = compute_pressure_value(A);
                    const double U  = W1_L - 4.0 * (cA - c0);
                    const double Q  = A * U;
                    const double dQdA =
                      U + A * (-4.0 * compute_wave_speed_derivative(A));

                    // DISCRETIZATION: Use dt from ARKode
                    const double Pc_star =
                      (Pc_old + (dt / C) * (Q + par["P_out"] / R2)) /
                      (1.0 + dt / (C * R2));
                    const double dPc_star_dA =
                      (dt / C) * (dQdA) / (1.0 + dt / (C * R2));

                    const double F     = Pe - (R1 * Q + Pc_star);
                    const double dF_dA = compute_pressure_derivative(A) -
                                         (R1 * dQdA + dPc_star_dA);

                    A_star -= F / dF_dA;
                    if (std::abs(F / A_L) < 1e-12)
                      break;
                  }
                U_star = W1_L - 4.0 * (compute_wave_speed(A_star) - c0);
              }
            else // (outlet_type == "Rt")
              {
                // ================= CHOICE: Rt (Reflection) =================
                const double Rt = par["Rt"];
                const double W2_target =
                  -Rt * W1_L; // Simplified reflection target

                for (unsigned int it = 0; it < 20; ++it)
                  {
                    const double A     = std::max(A_star, A_min);
                    const double cA    = compute_wave_speed(A);
                    const double dc_dA = compute_wave_speed_derivative(A);

                    // System: W1 invariant is fixed, W2 invariant is target
                    // Eq1: U + 4(c-c0) - W1_L = 0
                    // Eq2: U - 4(c-c0) - W2_target = 0
                    Vector<double> R(2);
                    R[0] = U_star + 4.0 * (cA - c0) - W1_L;
                    R[1] = U_star - 4.0 * (cA - c0) - W2_target;

                    FullMatrix<double> J(2, 2);
                    J(0, 0) = 4.0 * dc_dA;
                    J(0, 1) = 1.0;
                    J(1, 0) = -4.0 * dc_dA;
                    J(1, 1) = 1.0;

                    J.gauss_jordan();
                    Vector<double> delta(2);
                    J.vmult(delta, R);

                    A_star -= delta[0];
                    U_star -= delta[1];
                    if (delta.l2_norm() < 1e-12)
                      break;
                  }
              }
          }

        // ---------- 4. Newton-Consistent Jacobian (Sensitivity) ----------

        const double bn_L = compute_tangent_normal_product(cell, normals[q]);
        auto [flux_A, flux_U] =
          numerical_flux(bn_L, bn_L, A_L, U_L, A_star, U_star);

        for (unsigned int i = 0; i < fe->n_dofs_per_cell(); ++i)
          {
            const double test_A = fe_face[area_extractor].value(i, q);
            const double test_U = fe_face[velocity_extractor].value(i, q);
            copy.cell_rhs(i) -= (flux_A * test_A + flux_U * test_U) * JxW[q];
          }
      }
  };

  const AffineConstraints<double> constraints;
  auto                            copier = [&](const BloodFlowCopyData &c) {
    constraints.distribute_local_to_global(c.cell_rhs,
                                           c.local_dof_indices,
                                           Mydot);

    for (auto &cdf : c.face_data)
      {
        constraints.distribute_local_to_global(cdf.cell_rhs,
                                               cdf.joint_dof_indices,
                                               Mydot);
      }
  };

  const QGauss<dim>     quadrature(fe->tensor_degree() + 1);
  const QGauss<dim - 1> quadrature_face(fe->tensor_degree() + 1);

  BloodFlowScratchData<dim, spacedim> scratch_data(*fe,
                                                   quadrature,
                                                   quadrature_face);
  BloodFlowCopyData                   copy_data;

  if (use_riemann_invariants)
    {
      MeshWorker::mesh_loop(dof_handler.begin_active(),
                            dof_handler.end(),
                            cell_worker,
                            copier,
                            scratch_data,
                            copy_data,
                            MeshWorker::assemble_own_cells |
                              MeshWorker::assemble_boundary_faces |
                              MeshWorker::assemble_own_interior_faces_once,
                            boundary_worker_riemann_invariant,
                            face_worker);
    }
  else
    {
      MeshWorker::mesh_loop(dof_handler.begin_active(),
                            dof_handler.end(),
                            cell_worker,
                            copier,
                            scratch_data,
                            copy_data,
                            MeshWorker::assemble_own_cells |
                              MeshWorker::assemble_boundary_faces |
                              MeshWorker::assemble_own_interior_faces_once,
                            boundary_worker_with_characteristics,
                            face_worker);
    }

  assemble_junction_residual(y, Mydot);
  deallog.pop();
}

// ========================================================================
// OUTPUT RESULTS
// ========================================================================

template <int dim, int spacedim>
void
BloodFlowSystem<dim, spacedim>::output_results(
  const Vector<double> &y,
  const Vector<double> &pressure_vec,
  const Vector<double> &theoretical_peak,
  const unsigned int    time_step_number) const
{
  TimerOutput::Scope timer(computing_timer, "output_results");
  const std::string  rel_filename =
    output_filename + "-" + std::to_string(time_step_number) + ".vtu";
  const std::string filename =
    output_directory + (output_directory.empty() ? "" : "/") + rel_filename;
  std::cout << "  Writing solution to <" << filename << ">" << std::endl;
  std::ofstream output(filename);

  DataOut<dim, spacedim> data_out;
  data_out.attach_dof_handler(dof_handler);

  std::vector<std::string> solution_names(2);
  solution_names[0] = "area";
  solution_names[1] = "velocity";

  std::vector<DataComponentInterpretation::DataComponentInterpretation>
    data_component_interpretation(
      2, DataComponentInterpretation::component_is_scalar);

  data_out.add_data_vector(y,
                           solution_names,
                           DataOut<dim, spacedim>::type_dof_data,
                           data_component_interpretation);
  solution_names[0] = "pressure";
  solution_names[1] = "unused";
  data_out.add_data_vector(pressure_vec,
                           solution_names,
                           DataOut<dim, spacedim>::type_dof_data,
                           data_component_interpretation);
  solution_names[0] = "theoretical_peak";
  solution_names[1] = "unused1";
  data_out.add_data_vector(theoretical_peak,
                           solution_names,
                           DataOut<dim, spacedim>::type_dof_data,
                           data_component_interpretation);

  data_out.build_patches();
  data_out.write_vtu(output);

  // Also write the pvd record
  static std::vector<std::pair<double, std::string>> pvd_output_records;
  pvd_output_records.push_back(std::make_pair(time, rel_filename));
  std::ofstream pvd_output(output_directory +
                           (output_directory.empty() ? "" : "/") +
                           output_filename + ".pvd");
  DataOutBase::write_pvd_record(pvd_output, pvd_output_records);
}

template <int dim, int spacedim>
void
BloodFlowSystem<dim, spacedim>::update_peak_pressure()
{
  const double Ad  = par["a_d"];
  const double rho = par["rho"];
  const double c0  = compute_wave_speed(Ad);

  const double Q_peak = 1e-6;

  const double Zc = (rho * c0) / Ad;

  P_peak = Q_peak * Zc;
  std::cout << "initial wave speed c0 = " << c0 << std::endl;
}

// Theoretical peak pressure independent of time, used for comparison.

template <int dim, int spacedim>
void
BloodFlowSystem<dim, spacedim>::compute_theoretical_peak(
  Vector<double> &theoretical_peak) const
{
  theoretical_peak = 0;

  const double xi = par["xi"];
  const double mu = par["mu"];
  const double Ad = par["a_d"];
  const double c0 = compute_wave_speed(Ad);

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      std::vector<types::global_dof_index> dof_indices(fe->n_dofs_per_cell());
      cell->get_dof_indices(dof_indices);

      const double x = cell->center()[0];

      const double val =
        std::exp(-(xi + 2.0) * numbers::PI * mu * x / (c0 * Ad * par["rho"]));

      for (unsigned int i = 0; i < fe->n_dofs_per_cell(); ++i)
        theoretical_peak[dof_indices[i]] = val;
    }
}

// COMPUTE PRESSURE
// ========================================================================

template <int dim, int spacedim>
void
BloodFlowSystem<dim, spacedim>::compute_pressure(
  const Vector<double> &y,
  Vector<double>       &pressure_vec) const
{
  TimerOutput::Scope timer(computing_timer, "compute_pressure");
  AssertDimension(y.size(), dof_handler.n_dofs());
  AssertDimension(pressure_vec.size(), dof_handler.n_dofs());

  pressure_vec = 0;

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      std::vector<types::global_dof_index> dof_indices(fe->n_dofs_per_cell());
      cell->get_dof_indices(dof_indices);

      for (unsigned int i = 0; i < fe->n_dofs_per_cell(); ++i)
        {
          const unsigned int component = fe->system_to_component_index(i).first;
          if (component == 0) // Area component
            {
              const double area            = y[dof_indices[i]];
              pressure_vec[dof_indices[i]] = compute_pressure_value(
                area); // for single vessel test divide by P_peak
            }
        }
    }
}

// store the value of (terminal pressure)Pc​ from the previous time step. This
// is necessary for the implicit update of the capacitor pressure in the RCR
// boundary condition.
//  By keeping track of (terminal pressure)Pc​ at each terminal boundary, we
//  can ensure that the time integration of the capacitor pressure is consistent
//  and stable across time steps.

template <int dim, int spacedim>
void
BloodFlowSystem<dim, spacedim>::initialize_terminal_capacitors()
{
  terminal_Pc_storage.clear();

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      const unsigned int v_id    = cell->user_index();
      const auto        &v_props = vessel_map.at(v_id);
      const double       pd      = v_props.p_d;
      for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
        {
          if (!cell->face(f)->at_boundary())
            continue;

          if (all_junction_faces.count({cell->id(), f}))
            continue;

          const auto bid = cell->face(f)->boundary_id();

          if (bid != 0 && this->is_terminal_boundary(bid))
            {
              terminal_Pc_storage[bid] = pd;
              std::cout << "  [Init] Boundary " << bid << " (Vessel " << v_id
                        << ") initial Pc set to " << pd << " Pa" << std::endl;
              // terminal_Pc_storage.try_emplace(bid, par["p_d"]);
              //           inflow_function.set_time(0.0);
              // const double Q0 = inflow_function.value(Point<1>(0.0));

              // const double Pc0 = par["P_out"] + par["R2"] * Q0;

              // terminal_Pc_storage[bid] = Pc0;
            }
        }
    }
}


template <int dim, int spacedim>
void
BloodFlowSystem<dim, spacedim>::update_terminal_pressures(
  const double          dt,
  const Vector<double> &evaluation_point)
{
  // ===================== 1. Initialize flow accumulator =====================
  std::map<unsigned int, double> Q_accumulated;
  for (const auto &[bid, Pc] : terminal_Pc_storage)
    Q_accumulated[bid] = 0.0;

  // ===================== 2. FE setup =====================
  const FEValuesExtractors::Scalar area(0);
  const FEValuesExtractors::Scalar velocity(1);

  QGauss<dim - 1> face_quadrature(fe->degree + 1);

  FEFaceValues<dim, spacedim> fe_face(*fe,
                                      face_quadrature,
                                      update_values | update_JxW_values |
                                        update_normal_vectors);

  const unsigned int n_q = face_quadrature.size();

  std::vector<double> A_values(n_q);
  std::vector<double> U_values(n_q);

  // ===================== 3. Compute total flow at outlets
  // =====================
  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      if (!cell->is_locally_owned())
        continue;

      for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
        {
          if (!cell->face(f)->at_boundary())
            continue;

          const unsigned int bid = cell->face(f)->boundary_id();

          if (!this->is_terminal_boundary(bid))
            continue;

          fe_face.reinit(cell, f);

          fe_face[area].get_function_values(evaluation_point, A_values);
          fe_face[velocity].get_function_values(evaluation_point, U_values);

          const auto &JxW = fe_face.get_JxW_values();

          for (unsigned int q = 0; q < n_q; ++q)
            {
              const double Q_local = A_values[q] * U_values[q] * JxW[q];

              Q_accumulated[bid] += Q_local;

              // Optional debug
              if (q == 0)
                std::cout << "DEBUG: A=" << A_values[q] << " U=" << U_values[q]
                          << " Q=" << Q_local << std::endl;
            }
        }
    }

  // ===================== 4. Update Windkessel pressure =====================

  for (auto bid : terminal_boundary_ids)
    {
      if (!terminal_Pc_storage.count(bid))
        continue;

      double      &Pc_old = terminal_Pc_storage[bid];
      const double Q_tot  = Q_accumulated[bid];

      const double C     = par["C"];
      const double R2    = par["R2"];
      const double P_out = par["P_out"];

      const double Pc_new =
        (Pc_old + (dt / C) * (Q_tot + P_out / R2)) / (1.0 + dt / (C * R2));

      // Store updated value
      Pc_old = Pc_new;

      // Debug output
      if (std::abs(Q_tot) > 1e-10)
        std::cout << "Step Update: Boundary " << bid << " Q=" << Q_tot
                  << " Pc: " << Pc_old << std::endl;
    }
}
// ========================================================================
// COMPUTE ERRORS
// ========================================================================


template <int dim, int spacedim>
void
BloodFlowSystem<dim, spacedim>::compute_errors(unsigned int k)
{
  TimerOutput::Scope timer(computing_timer, "compute_errors");
  const ComponentSelectFunction<spacedim> area_mask(0, 1.0, 2);
  const ComponentSelectFunction<spacedim> velocity_mask(1, 1.0, 2);

  Vector<float> difference_per_cell(triangulation.n_active_cells());

  // // Create exact solution at current time
  // ExactSolutionBloodFlow<spacedim> exact_solution;
  exact_solution.set_time(time);
  // Area L2 error
  VectorTools::integrate_difference(dof_handler,
                                    solution,
                                    exact_solution,
                                    difference_per_cell,
                                    QGauss<dim>(fe_degree + 3),
                                    VectorTools::L2_norm,
                                    &area_mask);
  const double Area_L2_error =
    VectorTools::compute_global_error(triangulation,
                                      difference_per_cell,
                                      VectorTools::L2_norm);

  // Area H1 error
  VectorTools::integrate_difference(dof_handler,
                                    solution,
                                    exact_solution,
                                    difference_per_cell,
                                    QGauss<dim>(fe_degree + 3),
                                    VectorTools::H1_seminorm,
                                    &area_mask);
  const double Area_H1_error =
    VectorTools::compute_global_error(triangulation,
                                      difference_per_cell,
                                      VectorTools::H1_seminorm);

  // Velocity L2 error
  VectorTools::integrate_difference(dof_handler,
                                    solution,
                                    exact_solution,
                                    difference_per_cell,
                                    QGauss<dim>(fe_degree + 3),
                                    VectorTools::L2_norm,
                                    &velocity_mask);
  const double Velocity_L2_error =
    VectorTools::compute_global_error(triangulation,
                                      difference_per_cell,
                                      VectorTools::L2_norm);

  // Velocity H1 error
  VectorTools::integrate_difference(dof_handler,
                                    solution,
                                    exact_solution,
                                    difference_per_cell,
                                    QGauss<dim>(fe_degree + 3),
                                    VectorTools::H1_seminorm,
                                    &velocity_mask);
  const double Velocity_H1_error =
    VectorTools::compute_global_error(triangulation,
                                      difference_per_cell,
                                      VectorTools::H1_seminorm);

  // Variables to store previous errors for convergence rate calculation
  static double last_Area_L2_error     = 0;
  static double last_Area_H1_error     = 0;
  static double last_Velocity_L2_error = 0;
  static double last_Velocity_H1_error = 0;

  // Output results with convergence rates
  std::cout << std::scientific << std::setprecision(3);
  std::cout << "=== Error Analysis at Time t = " << time
            << " (Refinement Level " << k + 1 << ") ===" << std::endl;

  std::cout << " Area L2 error:      " << std::setw(12) << Area_L2_error
            << "   Conv_rate: " << std::setw(6)
            << (k == 0 ?
                  0.0 :
                  std::log(last_Area_L2_error / Area_L2_error) / std::log(2.0))
            << std::endl;

  std::cout << " Area H1 error:      " << std::setw(12) << Area_H1_error
            << "   Conv_rate: " << std::setw(6)
            << (k == 0 ?
                  0.0 :
                  std::log(last_Area_H1_error / Area_H1_error) / std::log(2.0))
            << std::endl;

  std::cout << " Velocity L2 error:  " << std::setw(12) << Velocity_L2_error
            << "   Conv_rate: " << std::setw(6)
            << (k == 0 ? 0.0 :
                         std::log(last_Velocity_L2_error / Velocity_L2_error) /
                           std::log(2.0))
            << std::endl;

  std::cout << " Velocity H1 error:  " << std::setw(12) << Velocity_H1_error
            << "   Conv_rate: " << std::setw(6)
            << (k == 0 ? 0.0 :
                         std::log(last_Velocity_H1_error / Velocity_H1_error) /
                           std::log(2.0))
            << std::endl;

  std::cout << " DoFs: " << dof_handler.n_dofs() << "   h ≈ "
            << 1.0 / triangulation.n_active_cells() << std::endl;
  std::cout << std::string(70, '=') << std::endl;

  // Update previous error values
  last_Area_L2_error     = Area_L2_error;
  last_Area_H1_error     = Area_H1_error;
  last_Velocity_L2_error = Velocity_L2_error;
  last_Velocity_H1_error = Velocity_H1_error;
}


// create a simple bifurcation network for testing junction conditions and RCR
// boundaries. The geometry is defined by a set of nodes and edges, where each
// edge represents a vessel segment. The network includes one inlet, one
// bifurcation, and multiple outlets to test the implementation of boundary
// conditions and junctions in the blood flow system.

template <int dim, int spacedim>
void
BloodFlowSystem<dim, spacedim>::create_vascular_network()
{
  // ---------------- Nodes ----------------
  std::vector<Point<spacedim>> nodes = {
    {0.0, 0.0, 0.0},   // 0: inlet (aorta start)
    {0.086, 0.0, 0.0}, // 1: bifurcation (8.6 cm)

    {0.086 + 0.085, 0.05, 0.0}, // 2: iliac 1 (8.5 cm)
    {0.086 + 0.085, -0.05, 0.0} // 3: iliac 2
  };

  // ---------------- Edges ----------------
  std::vector<std::pair<int, int>> edges = {
    {0, 1}, // aorta
    {1, 2}, // iliac branch 1
    {1, 3}  // iliac branch 2
  };

  std::vector<CellData<dim>> cells(edges.size());

  for (unsigned int i = 0; i < edges.size(); ++i)
    {
      cells[i].vertices[0] = edges[i].first;
      cells[i].vertices[1] = edges[i].second;
    }

  SubCellData subcelldata;
  triangulation.create_triangulation(nodes, cells, subcelldata);
}

// ========================================================================
// RUN CONVERGENCE STUDY WITH NEWTON ITERATION
// ========================================================================

template <int dim, int spacedim>
void
BloodFlowSystem<dim, spacedim>::run()
{
  std::cout << "=== CONVERGENCE STUDY for DG" << fe_degree
            << " ===" << std::endl;

  for (unsigned int cycle = 0; cycle < n_refinement_cycles; ++cycle)
    {
      std::cout << "\n--- Refinement Cycle " << cycle << " ---" << std::endl;

      if (cycle == 0)
        {
          triangulation.clear();

          // if (!use_junction_mesh)
          //   {
          //     // ================= SINGLE VESSEL =================
          //     const double L = par["L"];
          //     GridGenerator::hyper_cube(triangulation, 0.0, L);
          //     // tagging logic for terminal boundaries
          //     terminal_boundary_ids.clear();
          //     // unsigned int outlet_count = 0;

          //     for (auto &cell : triangulation.active_cell_iterators())
          //       {
          //         for (unsigned int f = 0;
          //              f < GeometryInfo<dim>::faces_per_cell;
          //              ++f)
          //           {
          //             if (cell->face(f)->at_boundary())
          //               {
          //                 Point<3> face_center = cell->face(f)->center();

          //                 // 1. Identify Inlet (Root) at (0,0,0)
          //                 if (face_center.distance(Point<3>(0.0, 0.0, 0.0)) <
          //                     1e-8)
          //                   {
          //                     cell->face(f)->set_boundary_id(0);
          //                     std::cout << "  [Mesh] Tagged INLET (ID 0) at "
          //                               << face_center << std::endl;
          //                   }
          //                 // 2. Identify ALL other boundaries as Outlets
          //                 else
          //                   {
          //                     // This will now catch x = -0.5 AND x = 2.0
          //                     cell->face(f)->set_boundary_id(1);
          //                     terminal_boundary_ids.insert(1);
          //                     // outlet_count++;
          //                     std::cout << "  [Mesh] Tagged OUTLET (ID 1) at
          //                     "
          //                               << face_center << std::endl;
          //                   }
          //               }
          //           }
          //       }
          //     triangulation.refine_global(n_global_refinements);
          //   }

          // else
          //   {
          //     // ================= JUNCTION NETWORK =================


          //     if constexpr (dim == 1 && spacedim == 3)
          //       {
          //         create_vascular_network();

          //         // ================= TAGGING LOGIC for terminal boundary

          //         terminal_boundary_ids.clear();
          //         // unsigned int outlet_count = 0;

          //         for (auto &cell : triangulation.active_cell_iterators())
          //           {
          //             for (unsigned int f = 0;
          //                  f < GeometryInfo<dim>::faces_per_cell;
          //                  ++f)
          //               {
          //                 if (cell->face(f)->at_boundary())
          //                   {
          //                     Point<3> face_center = cell->face(f)->center();

          //                     // 1. Identify Inlet (Root) at (0,0,0)
          //                     if (face_center.distance(Point<3>(-0.5, 0.5,
          //                     0)) <
          //                         1e-8)
          //                       {
          //                         cell->face(f)->set_boundary_id(0);
          //                         std::cout
          //                           << "  [Mesh] Tagged INLET (ID 0) at "
          //                           << face_center << std::endl;
          //                       }
          //                     // 2. Identify ALL other boundaries as Outlets
          //                     else
          //                       {
          //                         // This will now catch x = -0.5 AND x = 2.0
          //                         cell->face(f)->set_boundary_id(1);
          //                         terminal_boundary_ids.insert(1);
          //                         // outlet_count++;
          //                         std::cout
          //                           << "  [Mesh] Tagged OUTLET (ID 1) at "
          //                           << face_center << std::endl;
          //                       }
          //                   }
          //               }
          //           }
          //         triangulation.refine_global(n_global_refinements);
          //       }
          //   }



          GridIn<dim, spacedim> grid_in;
          grid_in.attach_triangulation(triangulation);
          std::ifstream input_file(
            "/home/rakshad/blood-flow/notebooks/bifurcation_physics.vtk");

          // AssertThrow(input_file.is_open(),
          //             ExcMessage("Could not open VTK mesh file: " +
          //             par["mesh_filename"]));

          grid_in.read_vtk(input_file);
          triangulation.refine_global(n_global_refinements);

          // 2. Identify Boundary IDs (Inlet vs Outlets)
          // This logic is now identical for both single vessel and junctions
          terminal_boundary_ids.clear();

          for (const auto &cell : triangulation.active_cell_iterators())
            {
              for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell;
                   ++f)
                {
                  if (cell->face(f)->at_boundary())
                    {
                      const types::boundary_id bid =
                        cell->face(f)->boundary_id();

                      // INFLOW: In your VTK logic, Inlet is ID 0
                      if (bid == 0)
                        {
                          std::cout << "  [Mesh] Recognized INLET (ID 0) at "
                                    << cell->face(f)->center() << std::endl;
                        }
                      // OUTFLOW: IDs 1, 1 are treated as terminals
                      else if (bid == 1)
                        {
                          terminal_boundary_ids.insert(bid);
                          std::cout << "  [Mesh] Recognized OUTLET (ID " << bid
                                    << ") at " << cell->face(f)->center()
                                    << std::endl;
                        }
                    }
                }
            }
          // triangulation.refine_global(n_global_refinements);
        }
      else
        {
          triangulation.refine_global(1);
        }

      setup_system();
      std::cout << "Terminal boundary IDs: ";
      for (auto id : terminal_boundary_ids)
        std::cout << id << " ";
      std::cout << std::endl;
      initialize_terminal_capacitors();
      update_peak_pressure();
      compute_theoretical_peak(theoretical_peak);
      compute_initial_solution(solution, arkode_parameters.initial_time);

      time = arkode_parameters.initial_time;

      SUNDIALS::ARKode<Vector<double>> ode(arkode_parameters);

      ode.mass_times_setup = [this](const double) {
        deallog.push("mass_times_setup");
        deallog << "Called mass_times_setup" << std::endl;
        assemble_mass_matrix();
        deallog.pop();
      };
      ode.mass_times_vector =
        [this](const double, const Vector<double> &v, Vector<double> &Mv) {
          deallog.push("mass_times_vector");
          deallog << "Called mass_times_vector" << std::endl;
          mass_matrix.vmult(Mv, v);
          deallog.pop();
        };
      ode.solve_mass =
        [this](SUNDIALS::SundialsOperator<Vector<double>> &,
               SUNDIALS::SundialsPreconditioner<Vector<double>> &,
               Vector<double>       &x,
               const Vector<double> &b,
               double) {
          TimerOutput::Scope timer(computing_timer, "solve_mass");
          deallog.push("solve_mass");
          deallog << "Called solve_mass" << std::endl;
          mass_solver.vmult(x, b);
          deallog.pop();
        };

      ode.implicit_function =
        [this](const double t, const Vector<double> &y, Vector<double> &Mydot) {
          deallog.push("implicit_function");
          deallog << "Called implicit_function t=" << t << std::endl;
          assemble_implicit_function(t, y, Mydot);
          deallog.pop();
        };

      ode.jacobian_times_setup = [this](const double          t,
                                        const Vector<double> &y,
                                        const Vector<double> &Mydot) {
        deallog.push("jacobian_times_setup");
        deallog << "Called jacobian_times_setup t=" << t << std::endl;
        assemble_jacobian(t, y, Mydot);
        deallog.pop();
      };

      ode.jacobian_times_vector = [this](const Vector<double> &v,
                                         Vector<double>       &Jv,
                                         const double,
                                         const Vector<double> &,
                                         const Vector<double> &) {
        TimerOutput::Scope timer(computing_timer, "jacobian_times_vector");
        deallog.push("jacobian_times_vector");
        deallog << "Called jacobian_times_vector" << std::endl;
        jacobian_matrix.vmult(Jv, v);
        deallog.pop();
      };

      ode.jacobian_preconditioner_setup = [this](const double,
                                                 const Vector<double> &,
                                                 const Vector<double> &,
                                                 const int    jok,
                                                 int         &jcur,
                                                 const double gamma) {
        TimerOutput::Scope timer(computing_timer,
                                 "jacobian_preconditioner_setup");
        deallog.push("jacobian_preconditioner_setup");
        deallog << "Called jacobian_preconditioner_setup gamma=" << gamma
                << " jok=" << jok << std::endl;
        if (jok == SUNFALSE)
          {
            linear_system_matrix.copy_from(mass_matrix);
            linear_system_matrix.add(-gamma, jacobian_matrix);
            linear_solver.initialize(linear_system_matrix);
            jcur = SUNTRUE;
          }
        else
          {
            jcur = SUNFALSE;
          }
        deallog.pop();
      };

      ode.jacobian_preconditioner_solve = [this](const double,
                                                 const Vector<double> &,
                                                 const Vector<double> &,
                                                 const Vector<double> &r,
                                                 Vector<double>       &z,
                                                 const double          gamma,
                                                 const double,
                                                 const int) {
        TimerOutput::Scope timer(computing_timer,
                                 "jacobian_preconditioner_solve");
        deallog.push("jacobian_preconditioner_solve");
        deallog << "Called jacobian_preconditioner_solve gamma=" << gamma
                << std::endl;
        (void)gamma;
        linear_solver.vmult(z, r);
        deallog.pop();
      };

      ode.output_step = [this](const double          t,
                               const Vector<double> &sol,
                               const unsigned int    step_number) {
        const unsigned int actual_step_number =
          (t - arkode_parameters.initial_time) /
          arkode_parameters.output_period;

        this->current_dt = (actual_step_number == 0) ? 0.0 : (t - this->time);
        update_terminal_pressures(this->current_dt, sol);
        // 2. Update the RCR pressures using the converged solution 'sol'
        double P_inlet      = 0.0;
        double P_outlet     = 0.0;
        bool   found_inlet  = false;
        bool   found_outlet = false;

        for (const auto &cell : dof_handler.active_cell_iterators())
          {
            if (!cell->is_locally_owned())
              continue;

            for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
              {
                if (!cell->face(f)->at_boundary())
                  continue;

                const unsigned int bid = cell->face(f)->boundary_id();

                std::vector<types::global_dof_index> dof_indices(
                  fe->n_dofs_per_cell());
                cell->get_dof_indices(dof_indices);

                for (unsigned int i = 0; i < fe->n_dofs_per_cell(); ++i)
                  {
                    const unsigned int comp =
                      fe->system_to_component_index(i).first;

                    if (comp == 0) // area
                      {
                        const double A = sol[dof_indices[i]];
                        const double P = compute_pressure_value(A);

                        if (bid == 0 && !found_inlet)
                          {
                            P_inlet     = P;
                            found_inlet = true;
                          }
                        else if (bid == 1 && !found_outlet)
                          {
                            P_outlet     = P;
                            found_outlet = true;
                          }
                      }
                  }
              }
          }

        if (found_inlet && found_outlet)
          {
            const double deltaP = P_inlet - P_outlet;

            std::cout << "t = " << t << "  Pin = " << P_inlet
                      << "  Pout = " << P_outlet
                      << "  dP(Kpa) = " << deltaP / 1000 << std::endl;
          }


        for (const auto &cell : dof_handler.active_cell_iterators())
          {
            const double x = cell->center()[0];

            // midpoint check (L = 0.126 → midpoint = 0.063)
            if (std::abs(x - par["L"] / 2.0) < 1e-3)
              {
                std::vector<types::global_dof_index> dof_indices(
                  fe->n_dofs_per_cell());
                cell->get_dof_indices(dof_indices);

                for (unsigned int i = 0; i < fe->n_dofs_per_cell(); ++i)
                  {
                    const unsigned int comp =
                      fe->system_to_component_index(i).first;

                    if (comp == 0) // area component
                      {
                        const double A = sol[dof_indices[i]];
                        const double p = compute_pressure_value(A);

                        std::cout << "t = " << t << "  A = " << A
                                  << "  Raw pressure = " << p << std::endl;
                      }
                  }
              }
          }
        deallog.push("output_step");
        deallog << "Called output_step t=" << t
                << " step_number=" << actual_step_number << std::endl;
        time = t;
        compute_pressure(sol, pressure);
        compute_theoretical_peak(theoretical_peak);
        output_results(sol, pressure, theoretical_peak, actual_step_number);
        deallog.pop();
      };


      time = arkode_parameters.initial_time;
      while (time < arkode_parameters.final_time)
        {
          const unsigned int n_timesteps =
            ode.solve_ode_incrementally(solution,
                                        time + arkode_parameters.output_period,
                                        true);
          std::cout << "  ARKode intermediate steps: " << n_timesteps
                    << std::endl;
          time += arkode_parameters.output_period;
        }
      compute_pressure(solution, pressure);
      compute_errors(cycle);
    }
}

template <int dim, int spacedim>
inline void
BloodFlowSystem<dim, spacedim>::assemble_mass_matrix()
{
  TimerOutput::Scope timer(computing_timer, "assemble_mass_matrix");
  MatrixTools::create_mass_matrix(dof_handler,
                                  QGauss<dim>(fe_degree + 1),
                                  mass_matrix);
  mass_solver.initialize(mass_matrix);
}

template <int dim, int spacedim>
void
BloodFlowSystem<dim, spacedim>::compute_initial_solution(Vector<double> &dst,
                                                         const double    t)
{
  TimerOutput::Scope timer(computing_timer, "compute_initial_solution");
  initial_condition.set_time(t);
  AffineConstraints<double> constraints;
  constraints.close();
  VectorTools::project(dof_handler,
                       constraints,
                       QGauss<dim>(fe_degree + 1),
                       initial_condition,
                       dst);
}

// Explicit instantiation
template class BloodFlowSystem<1, 3>;
