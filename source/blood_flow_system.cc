/* --------------------------------------------------------------------------
 *
 * This is the implementation file for the BloodFlowSystem class,
 *  which models blood flow in a 1D vessel network embedded in 3D
 * using the discontinuous finite element method.
 */
#include "blood_flow_system.h"

#include <deal.II/base/function_parser.h>
#include <deal.II/base/types.h>

#include <deal.II/grid/cell_data.h>
#include <deal.II/grid/grid_generator.h>
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

// Main class implementation
template <int dim, int spacedim>
BloodFlowSystem<dim, spacedim>::BloodFlowSystem()
  : par("Blood Flow Parameters",
        {"rho",
         "a0",
         "mu",
         "p0",
         "eta_c",
         "r0",
         "m",
         "Rt",
         "R1",
         "R2",
         "C",
         "P_out"},
        {1.06,
         3.141592653589793e-4,
         1.0e6,
         0.0,
         1.0,
         9.99e-3,
         0.5,
         0.0,
         0.0,
         0.0,
         0.0,
         0.0},
        {"Density",
         "Reference cross-sectional area",
         "Elastic modulus of the vessel wall",
         "Reference pressure",
         "Viscosity coefficient",
         "Reference radius",
         "Tube law exponent",
         "Reflection coefficient at outflow boundary",
         "Resistance of proximal capacitor",
         "Resistance of distal capacitor",
         "Capacitance of terminal capacitor",
         "Pressure at terminal boundary"})
  , triangulation()
  , dof_handler(triangulation)
  , fe(nullptr)
  , outlet_type("RCR")
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

  , inflow_function("Functions", "0.0", "Inflow function", par, "t")
{
  add_parameter("Finite element degree", fe_degree);
  add_parameter("Problem constants", constants);
  add_parameter("Output filename", output_filename);
  add_parameter("Use direct solver", use_direct_solver);
  add_parameter("Number of refinement cycles", n_refinement_cycles);
  add_parameter("Number of global refinement", n_global_refinements);
  add_parameter("Time step", time_step);
  add_parameter("Final time", final_time);
  add_parameter("Theta (penalty parameter)", theta);
  add_parameter("Theta Boundary (stability parameter)", theta_bd);
  add_parameter("Omega (relaxation parameter)", omega);
  // add_parameter("Picard iteroutlet_typeations", max_picard_iterations);
  add_parameter("Newton iterations", max_newton_iterations);
  add_parameter("Newton tolerance", newton_tolerance);
  add_parameter("Use Riemann Invariants", use_riemann_invariants);
  add_parameter("Use junction mesh", use_junction_mesh);
  // Use Patterns::Selection to strictly enforce "RCR" or "Rt"
  add_parameter("Outlet boundary condition type", outlet_type);
}

// ========================================================================
// INITIALIZE PARAMETERS
// ========================================================================

template <int dim, int spacedim>
void
BloodFlowSystem<dim, spacedim>::initialize_params(const std::string &filename)
{
  ParameterAcceptor::initialize(filename,
                                "last_used_parameters.prm",
                                ParameterHandler::Short,
                                this->prm,
                                ParameterHandler::Short);

  exact_solution.update_constants(par);
  initial_condition.update_constants(par);
  rhs_function.update_constants(par);
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
// For junction detection
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
      // automatically). If you want to solve the junction equations manually,
      // use >= 2.
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
  const std::vector<double>         &W_in,
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

template <int dim, int spacedim>
void
BloodFlowSystem<dim, spacedim>::assemble_junction_terms()
{
  if (fe->dofs_per_vertex > 0 || junctions.empty())
    return;

  const FEValuesExtractors::Scalar area_extractor(0);
  const FEValuesExtractors::Scalar velocity_extractor(1);

  QGauss<dim - 1>             quad(1);
  FEFaceValues<dim, spacedim> fe_face(
    *fe, quad, update_values | update_JxW_values | update_normal_vectors);

  const double A_min = 1e-10;

  for (const auto &J : junctions)
    {
      const unsigned int  K = J.faces.size();
      std::vector<double> A(K), U(K), W_in(K);

      // 1. Get current traces and compute Invariants
      for (unsigned int i = 0; i < K; ++i)
        {
          get_trace(solution, J.faces[i].cell, J.faces[i].face_no, A[i], U[i]);

          double s = static_cast<double>(J.faces[i].orientation);
          // std::cout << "sign " << s << ", Face " << i
          //            << std::endl;
          double c = compute_wave_speed(std::max(A[i], A_min));

          // The characteristic entering the junction: W = U + s*4c
          W_in[i] = U[i] + s * 4.0 * c;
        }

      // 2. Solve junction system for Star State X*
      JunctionState X0;
      X0.X.reinit(2 * K);
      for (unsigned int i = 0; i < K; ++i)
        {
          X0.A(i) = A[i];
          X0.U(i) = U[i];
        }

      JunctionState X = junction_solver.solve(X0, W_in, J, *this);

      // 3. Local Junction Jacobian Inverse
      FullMatrix<double> JX(2 * K, 2 * K);
      compute_junction_jacobian(X, J, W_in, JX);
      FullMatrix<double> JX_inv = JX;
      JX_inv.gauss_jordan();

      // 4. Assemble each participating face
      auto assemble_face = [&](unsigned int face_idx) {
        const auto &fd = J.faces[face_idx];
        fe_face.reinit(fd.cell, fd.face_no);

        const auto        &normals = fe_face.get_normal_vectors();
        const auto        &JxW     = fe_face.get_JxW_values();
        const unsigned int n_q     = fe_face.n_quadrature_points;

        std::vector<double> Ah(n_q), Uh(n_q);
        fe_face[area_extractor].get_function_values(solution, Ah);
        fe_face[velocity_extractor].get_function_values(solution, Uh);

        std::vector<types::global_dof_index> dofs(fe->n_dofs_per_cell());
        fd.cell->get_dof_indices(dofs);

        for (unsigned int q = 0; q < n_q; ++q)
          {
            const double Ah_safe = std::max(Ah[q], A_min);
            const double A_star  = std::max(X.A(face_idx), A_min);
            const double U_star  = X.U(face_idx);

            auto [FA, FU] = HLL_residual_flux(
              fd.cell, fd.cell, Ah_safe, Uh[q], A_star, U_star, normals[q]);

            // --- Residual Assembly ---
            for (unsigned int i = 0; i < fe->n_dofs_per_cell(); ++i)
              {
                const unsigned int comp =
                  fe->system_to_component_index(i).first;
                const double phi_i = fe_face.shape_value(i, q);
                residual_vector(dofs[i]) -=
                  (comp == 0 ? FA : FU) * phi_i * JxW[q];
              }

            // --- Jacobian Assembly ---
            for (unsigned int j = 0; j < fe->n_dofs_per_cell(); ++j)
              {
                const double phi_A_j = fe_face[area_extractor].value(j, q);
                const double phi_U_j = fe_face[velocity_extractor].value(j, q);

                // (a) Direct DG part (Star state held constant)
                auto [dFA_dwh, dFU_dwh] = HLL_jacobian_flux(fd.cell,
                                                            fd.cell,
                                                            Ah_safe,
                                                            Uh[q],
                                                            A_star,
                                                            U_star,
                                                            phi_A_j,
                                                            phi_U_j,
                                                            0.0,
                                                            0.0,
                                                            normals[q]);

                // (b) Sensitivity of Star state to interior via Chain Rule
                // dWin/dwh
                const double s        = static_cast<double>(fd.orientation);
                const double dc_dA    = compute_wave_speed_derivative(Ah_safe);
                const double dWin_dwh = phi_U_j + s * 4.0 * dc_dA * phi_A_j;

                // dX*/dWin = -inv(JX) * (dRj/dWin). Note: dRj/dWin = -1 for
                // compatibility rows. Thus dX*/dWin = inv(JX)[col K+face_idx]
                Vector<double> dX_dWin(2 * K);
                for (unsigned int r = 0; r < 2 * K; ++r)
                  dX_dWin[r] = JX_inv(r, K + face_idx);

                // (c) Chain-rule Jacobian via HLL (Sensitivity to Star State)
                auto [dFA_chain, dFU_chain] =
                  HLL_jacobian_flux(fd.cell,
                                    fd.cell,
                                    Ah_safe,
                                    Uh[q],
                                    A_star,
                                    U_star,
                                    0.0,
                                    0.0,
                                    dX_dWin[2 * face_idx] * dWin_dwh,
                                    dX_dWin[2 * face_idx + 1] * dWin_dwh,
                                    normals[q]);

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
      };

      for (unsigned int i = 0; i < K; ++i)
        assemble_face(i);
    }
}

// ========================================================================
// SETUP SYSTEM
// ========================================================================

template <int dim, int spacedim>
void
BloodFlowSystem<dim, spacedim>::setup_system()
{
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
  solution.reinit(dof_handler.n_dofs());
  solution_old.reinit(dof_handler.n_dofs());
  residual_vector.reinit(dof_handler.n_dofs()); // for Newton residual
  newton_update.reinit(dof_handler.n_dofs());   // for Newton update
  pressure.reinit(dof_handler.n_dofs());
  theoretical_peak.reinit(dof_handler.n_dofs());  // for comparison with benchmark paper 
}

// ========================================================================
// ASSEMBLE Jacobian Linearization and Residual
// ========================================================================
template <int dim, int spacedim>
void
BloodFlowSystem<dim, spacedim>::assemble_system()
{
  using Iterator = typename DoFHandler<dim, spacedim>::active_cell_iterator;

  jacobian_matrix = 0;
  residual_vector = 0;

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

    // Current Newton Iteration Values
    std::vector<double> current_area(fe_v.n_quadrature_points);
    std::vector<double> current_velocity(fe_v.n_quadrature_points);

    // Previous time Iteration Values
    std::vector<double> old_area(fe_v.n_quadrature_points);
    std::vector<double> old_velocity(fe_v.n_quadrature_points);

    fe_v[area_extractor].get_function_values(solution, current_area);
    fe_v[velocity_extractor].get_function_values(solution, current_velocity);

    fe_v[area_extractor].get_function_values(solution_old, old_area);
    fe_v[velocity_extractor].get_function_values(solution_old, old_velocity);

    // const auto b_vec = compute_directional_vector(cell);

    rhs_function.set_time(time);

    for (unsigned int point = 0; point < fe_v.n_quadrature_points; ++point)
      {
        const double rhs_A_value = rhs_function.value(q_points[point], 0);
        const double rhs_U_value = rhs_function.value(q_points[point], 1);

        // Compute explicit pressure and derivative
        const double current_pressure =
          compute_pressure_value(current_area[point]);
        const double dpdA = compute_pressure_derivative(current_area[point]);
        const double c_squared = current_area[point] / par["rho"] * dpdA;

        // flux for residual computation flux_A = U*A, Flux_U= U^2/2+ P/rho
        const auto current_flux_A =
          compute_physical_area_flux(cell,
                                     current_area[point],
                                     current_velocity[point]);

        const auto current_flux_U =
          compute_physical_momentum_flux(cell,
                                         current_velocity[point],
                                         current_velocity[point],
                                         current_pressure);

        for (unsigned int i = 0; i < n_dofs; ++i)
          {
            const auto test_area     = fe_v[area_extractor].value(i, point);
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

                // Mass term: (1/dt) * trial_A *
                // fe_face[area_extractor].value(i, q)
                copy_data.cell_matrix(i, j) +=
                  (1.0 / time_step) * trial_area * test_area * JxW[point];

                // Flux Jacobian: -b·∇(H_A * trial_W)
                // H_A = [U, A] so H_A·[trial_A, trial_U] = U*trial_A +
                // A*trial_U
                const auto flux_jacobian_A =
                  compute_physical_area_jacobian_flux(cell,
                                                      current_area[point],
                                                      trial_velocity,
                                                      current_velocity[point],
                                                      trial_area);

                copy_data.cell_matrix(i, j) -=
                  flux_jacobian_A * test_area_grad * JxW[point];

                // ===== MOMENTUM EQUATION Jacobian=====
                // Mass term: (1/dt) * trial_U *
                // fe_face[velocity_extractor].value(i, q)
                copy_data.cell_matrix(i, j) += (1.0 / time_step) *
                                               trial_velocity * test_velocity *
                                               JxW[point];

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

                copy_data.cell_matrix(i, j) -=
                  flux_jacobian_U * test_velocity_grad * JxW[point];

                // Reaction (viscosity): ∫ c U phi_U dx
                copy_data.cell_matrix(i, j) +=
                  par["eta_c"] * test_velocity * trial_velocity * JxW[point];
              }

            // RHS: source terms
            copy_data.cell_rhs(i) -=
              (
                // Area equation
                -rhs_A_value * test_area +
                1. / time_step * (current_area[point] - old_area[point]) *
                  test_area -
                current_flux_A * test_area_grad
                // Momentum equation
                - rhs_U_value * test_velocity +
                1. / time_step *
                  (current_velocity[point] - old_velocity[point]) *
                  test_velocity -
                current_flux_U * test_velocity_grad +
                par["eta_c"] * current_velocity[point] * test_velocity) *
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
    // is face a junction?
    if (all_junction_faces.count({cell->id(), f}) > 0 ||
        all_junction_faces.count({ncell->id(), nf}) > 0)
      return; // Skip junction faces

    FEInterfaceValues<dim, spacedim> &fe_iv = scratch.fe_interface_values;
    fe_iv.reinit(cell, f, sf, ncell, nf, nsf);

    const auto &JxW        = fe_iv.get_JxW_values();
    const auto &normals    = fe_iv.get_fe_face_values(0).get_normal_vectors();
    const unsigned int n_q = fe_iv.get_fe_face_values(0).n_quadrature_points;

    // Current Newton iterate values on both sides
    std::vector<double> current_area(n_q), current_velocity(n_q),
      current_area_neighbor(n_q), current_velocity_neighbor(n_q);

    fe_iv.get_fe_face_values(0)[area_extractor].get_function_values(
      solution, current_area);
    fe_iv.get_fe_face_values(0)[velocity_extractor].get_function_values(
      solution, current_velocity);
    fe_iv.get_fe_face_values(1)[area_extractor].get_function_values(
      solution, current_area_neighbor);
    fe_iv.get_fe_face_values(1)[velocity_extractor].get_function_values(
      solution, current_velocity_neighbor);

    copy.face_data.emplace_back();
    auto              &face = copy.face_data.back();
    const unsigned int nd   = fe_iv.n_current_interface_dofs();
    face.joint_dof_indices  = fe_iv.get_interface_dof_indices();
    face.cell_matrix.reinit(nd, nd);
    face.cell_rhs.reinit(nd);

    for (unsigned int q = 0; q < n_q; ++q)
      {
        for (unsigned int i = 0; i < nd; ++i)
          {
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
                auto [flux_A_jac_hll, flux_U_jac_hll] =
                  HLL_jacobian_flux(cell,
                                    ncell,
                                    current_area[q],
                                    current_velocity[q],
                                    current_area_neighbor[q],
                                    current_velocity_neighbor[q],
                                    trial_area,              // trial_A_L
                                    trial_velocity,          // trial_U_L
                                    trial_area_neighbor,     // trial_A_R
                                    trial_velocity_neighbor, // trial_U_R
                                    normals[q]);

                // Area equation contribution to Jacobian
                face.cell_matrix(i, j) +=
                  flux_A_jac_hll * fe_iv[area_extractor].jump_in_values(i, q) *
                  JxW[q];

                // Momentum equation contribution to Jacobian
                face.cell_matrix(i, j) +=
                  flux_U_jac_hll *
                  fe_iv[velocity_extractor].jump_in_values(i, q) * JxW[q];
              }

            // ===== HLL RESIDUAL FLUX (for RHS) =====
            auto [flux_A_hll, flux_U_hll] =
              HLL_residual_flux(cell,
                                ncell,
                                current_area[q],
                                current_velocity[q],
                                current_area_neighbor[q],
                                current_velocity_neighbor[q],
                                normals[q]);

            // ===== RHS: AREA FLUX =====
            face.cell_rhs(i) -=
              flux_A_hll * fe_iv[area_extractor].jump_in_values(i, q) * JxW[q];

            // ===== RHS: MOMENTUM FLUX =====
            face.cell_rhs(i) -= flux_U_hll *
                                fe_iv[velocity_extractor].jump_in_values(i, q) *
                                JxW[q];
          }
      }
  };

  //========== BOUNDARY WORKER - WITH HLL FLUX ==========
  exact_solution.set_time(time);
  auto boundary_worker_with_characteristics =
    [&](const Iterator                      &cell,
        const unsigned int                   face_no,
        BloodFlowScratchData<dim, spacedim> &scratch,
        BloodFlowCopyData                   &copy) {
      if (all_junction_faces.count({cell->id(), face_no}) > 0)
        return; // Skip boundary faces

      // Standard boundary condition treatment
      scratch.fe_interface_values.reinit(cell, face_no);
      const auto &fe_face = scratch.fe_interface_values.get_fe_face_values(0);

      const auto        &JxW     = fe_face.get_JxW_values();
      const auto        &normals = fe_face.get_normal_vectors();
      const unsigned int n_q     = fe_face.n_quadrature_points;

      // Interior trace (current Newton iterate)
      std::vector<double> current_area(n_q), current_velocity(n_q);
      fe_face[area_extractor].get_function_values(solution, current_area);
      fe_face[velocity_extractor].get_function_values(solution,
                                                      current_velocity);

      // Boundary values
      exact_solution.set_time(time);
      std::vector<Vector<double>> bc(n_q, Vector<double>(2));
      exact_solution.vector_value_list(fe_face.get_quadrature_points(), bc);

      // Determine which boundary this is (BID=0 is inlet, BID=1 is outlet)
      // unsigned int boundary_id = cell->face(face_no)->boundary_id();

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
                             // ---- WEAK OUTFLOW STABILIZATION ----
              const double eps = 1e-10;

              for (unsigned int i = 0; i < fe_face.get_fe().dofs_per_cell; ++i)
                {
                  copy.cell_matrix(i, i) += eps;
                }
            }
          else
            {
              Assert(false, ExcMessage("Unclassified boundary condition."));
            }

          // ===== HLL RESIDUAL FLUX AT BOUNDARY =====
          // For boundary: treat exterior as "right" state and interior as
          // "left" state in HLL flux
          auto [flux_A_boundary, flux_U_boundary] =
            HLL_residual_flux(cell,
                              cell,
                              A_int, // interior (left)
                              U_int, // interior (left)
                              A_ext, // exterior (right)
                              U_ext, // exterior (right)
                              normals[q]);

          const double F_area_boundary     = flux_A_boundary;
          const double F_momentum_boundary = flux_U_boundary;

          // ===== JACOBIAN ASSEMBLY =====
          for (unsigned int j = 0; j < fe_face.get_fe().dofs_per_cell; ++j)
            {
              const double trial_A = fe_face[area_extractor].value(j, q);
              const double trial_U = fe_face[velocity_extractor].value(j, q);

              // ===== HLL JACOBIAN AT BOUNDARY =====
              // Only trial functions on interior side are active (j comes from
              // interior cell) Exterior state does not change with trial
              // functions
              auto [flux_A_jac_boundary, flux_U_jac_boundary] =
                HLL_jacobian_flux(
                  cell,
                  cell,
                  A_int,   // interior current (left)
                  U_int,   // interior current (left)
                  A_ext,   // exterior (right) - fixed by boundary condition
                  U_ext,   // exterior (right) - fixed by boundary condition
                  trial_A, // trial_A_L (only interior varies)
                  trial_U, // trial_U_L (only interior varies)
                  0.0,     // trial_A_R = 0 (exterior is fixed)
                  0.0,     // trial_U_R = 0 (exterior is fixed)
                  normals[q]);

              // Test function loop
              for (unsigned int i = 0; i < fe_face.get_fe().dofs_per_cell; ++i)
                {
                  const double test_A = fe_face[area_extractor].value(i, q);
                  const double test_U = fe_face[velocity_extractor].value(i, q);

                  // Area equation Jacobian
                  copy.cell_matrix(i, j) +=
                    flux_A_jac_boundary * test_A * JxW[q];

                  // Momentum equation Jacobian
                  copy.cell_matrix(i, j) +=
                    flux_U_jac_boundary * test_U * JxW[q];
                }
            }

          // ===== RHS ASSEMBLY =====
          for (unsigned int i = 0; i < fe_face.get_fe().dofs_per_cell; ++i)
            {
              const double test_A = fe_face[area_extractor].value(i, q);
              const double test_U = fe_face[velocity_extractor].value(i, q);

              // Area RHS
              copy.cell_rhs(i) -= F_area_boundary * test_A * JxW[q];

              // Momentum RHS
              copy.cell_rhs(i) -= F_momentum_boundary * test_U * JxW[q];
            }

          // ===== DEBUG OUTPUT =====
          // deallog.push("boundary_debug");
          // deallog << "BID=" << boundary_id << " q=" << q << " | A_int=" << A_int
          //         << " U_int=" << U_int << " c=" << c_int
          //         << " | lambda1=" << lambda1 << " lambda2=" << lambda2
          //         << " | inlet=" << is_subcritical_inflow
          //         << " outlet=" << is_subcritical_outflow << " | A_ext=" << A_ext
          //         << " U_ext=" << U_ext << std::endl;
          // deallog.pop();
        }
    };

  //========== BOUNDARY WORKER – CONSISTENT A/U RIEMANN BOUNDARY ==========
  auto boundary_worker_riemann_invariant = [&](
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

    const double       A_min       = 1e-12;
    const double       dt          = time_step;
    const unsigned int boundary_id = cell->face(face_no)->boundary_id();

    std::vector<double> Ah(n_q), Uh(n_q);
    fe_face[area_extractor].get_function_values(solution, Ah);
    fe_face[velocity_extractor].get_function_values(solution, Uh);

    for (unsigned int q = 0; q < n_q; ++q)
      {
        const double A_L     = std::max(Ah[q], A_min);
        const double U_L     = Uh[q];
        const double c_L     = compute_wave_speed(A_L);
        const double dc_dA_L = compute_wave_speed_derivative(A_L);
        const double A0      = par["a0"];
        const double c0      = compute_wave_speed(A0);

        const double W1_L =
          U_L + 4.0 * (c_L - c0); // OUTGOING FOR TERMINAL BOUNDARY
        const double W2_L = U_L - 4.0 * (c_L - c0); // INCOMING FOR INLET

        double             A_star = A_L;
        double             U_star = U_L;
        FullMatrix<double> J_boundary_local(2, 2);
        FullMatrix<double> J_interior_local(2, 2);

        if (boundary_id == 0) // INFLOW: Prescribed Q_in(t)
          {
            inflow_function.set_time(time);
            const double Qin = inflow_function.value(Point<1>(0.0));
            for (unsigned int it = 0; it < 20; ++it)
              {
                const double A     = std::max(A_star, A_min);
                const double cA    = compute_wave_speed(A);
                const double dc_dA = compute_wave_speed_derivative(A);

                // R[0]: Prescribed Flow: A*U - Qin = 0
                // R[1]: Backward Invariant: U - 4c - W2_L = 0
                Vector<double> R(2);
                R[0] = A * U_star - Qin;
                R[1] =
                  U_star - 4.0 * (cA - compute_wave_speed(par["a0"])) - W2_L;

                FullMatrix<double> J(2, 2);
                J(0, 0) = U_star;
                J(0, 1) = A;
                J(1, 0) = -4.0 * dc_dA;
                J(1, 1) = 1.0;

                // J_boundary_local = J; // Store for sensitivity
                J.gauss_jordan();
                Vector<double> delta(2);
                J.vmult(delta, R);

                A_star -= delta[0];
                U_star -= delta[1];
                if (delta.l2_norm() < 1e-12)
                  break;
              }
            J_boundary_local(0, 0) = U_star;
            J_boundary_local(0, 1) = A_star;
            J_boundary_local(1, 0) =
              -4.0 * compute_wave_speed_derivative(A_star);
            J_boundary_local(1, 1) = 1.0;

            J_interior_local(1, 0) = 4.0 * dc_dA_L; // - d(W2_L)/dAL
            J_interior_local(1, 1) = -1.0;          // - d(W2_L)/dUL
          }
        else // ================= OUTLET BOUNDARY =================
          {
            // 1. Retrieve the choice from your ParameterHandler
            // std::string outlet_type;
            // this->prm.enter_subsection("Blood Flow Parameters");
            // outlet_type = this->prm.get("Outlet boundary condition type");
            // this->prm.leave_subsection();

            if (this->outlet_type == "RCR")
              {
                // ================= CHOICE: RCR =================
                const double R1     = par["R1"];
                const double R2     = par["R2"];
                const double C      = par["C"];
                double       Pc_old = 0.0;
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
                    const double A      = std::max(A_star, A_min);
                    const double cA     = compute_wave_speed(A);
                    const double dc_dA  = compute_wave_speed_derivative(A);
                    const double Pe     = compute_pressure_value(A);
                    const double dPe_dA = compute_pressure_derivative(A);

                    // Link U to A via the outgoing Riemann Invariant (W1)
                    U_star                 = W1_L - 4.0 * (cA - c0);
                    const double dUstar_dA = -4.0 * dc_dA;
                    const double dQ_dA     = U_star + A * dUstar_dA;

                    // Implicit Capacitor Pressure update
                    const double Pc_star =
                      (Pc_old + (dt / C) * (A * U_star + par["P_out"] / R2)) /
                      (1.0 + dt / (C * R2));
                    const double dPc_star_dA =
                      (dt / C * dQ_dA) / (1.0 + dt / (C * R2));

                    // Residual: Balance at proximal resistance R1
                    const double F     = R1 * (A * U_star) - (Pe - Pc_star);
                    const double dF_dA = R1 * dQ_dA - dPe_dA + dPc_star_dA;

                    A_star -= F / dF_dA;
                    if (std::abs(F / A_L) < 1e-12)
                      break;
                  }

                // Final state sync and Jacobian for sensitivity (Step 4)
                U_star = W1_L - 4.0 * (compute_wave_speed(A_star) - c0);
                J_boundary_local(0, 0) =
                  compute_pressure_derivative(A_star) / R1 - U_star;
                J_boundary_local(0, 1) = -A_star;
                J_boundary_local(1, 0) =
                  4.0 * compute_wave_speed_derivative(A_star);
                J_boundary_local(1, 1) = 1.0;

                // Interior sensitivity: Only R0 depends on W1_L
                J_interior_local(0, 0) = -4.0 * dc_dA_L;
                J_interior_local(0, 1) = -1.0;
                J_interior_local(1, 0) = 0.0;
                J_interior_local(1, 1) = 0.0;
              }
            else // (outlet_type == "Rt")
              {
                // ================= CHOICE: Rt (Reflection) =================
                const double Rt   = par["Rt"];
                const double W1_L = U_L + 4.0 * (c_L - c0);
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

                J_interior_local(0, 0) = -4.0 * dc_dA_L;
                J_interior_local(0, 1) = -1.0;
                // R1 = U* - 4c* - (-Rt * (UL + 4cL)) -> dR1/dAL = Rt*4*dcdAL,
                // dR1/dUL = Rt
                J_interior_local(1, 0) = par["Rt"] * 4.0 * dc_dA_L;
                J_interior_local(1, 1) = par["Rt"];
              }
          }
        // ---------- 3. Numerical Flux (Residual) ----------
        auto [FA, FU] =
          HLL_residual_flux(cell, cell, A_L, U_L, A_star, U_star, normals[q]);

        for (unsigned int i = 0; i < fe->n_dofs_per_cell(); ++i)
          {
            const unsigned int comp = fe->system_to_component_index(i).first;
            const double       phi  = fe_face.shape_value(i, q);
            if (comp == 0)
              copy.cell_rhs(i) -= FA * phi * JxW[q];
            else
              copy.cell_rhs(i) -= FU * phi * JxW[q];
          }

        // ---------- 4. Newton-Consistent Jacobian (Sensitivity) ----------
        J_boundary_local.gauss_jordan();

        for (unsigned int j = 0; j < fe->n_dofs_per_cell(); ++j)
          {
            const double dAL_trial = fe_face[area_extractor].value(j, q);
            const double dUL_trial = fe_face[velocity_extractor].value(j, q);

            Vector<double> dRb_dInt(2);
            dRb_dInt[0] = J_interior_local(0, 0) * dAL_trial +
                          J_interior_local(0, 1) * dUL_trial;
            dRb_dInt[1] = J_interior_local(1, 0) * dAL_trial +
                          J_interior_local(1, 1) * dUL_trial;

            // dX_star = - (dRb/dX_star)^-1 * (dRb/dX_L) * dX_L
            Vector<double> dX_star(2);
            J_boundary_local.vmult(dX_star, dRb_dInt);
            dX_star *= -1.0;

            auto [dFA_dir, dFU_dir]     = HLL_jacobian_flux(cell,
                                                        cell,
                                                        A_L,
                                                        U_L,
                                                        A_star,
                                                        U_star,
                                                        dAL_trial,
                                                        dUL_trial,
                                                        0.0,
                                                        0.0,
                                                        normals[q]);
            auto [dFA_chain, dFU_chain] = HLL_jacobian_flux(cell,
                                                            cell,
                                                            A_L,
                                                            U_L,
                                                            A_star,
                                                            U_star,
                                                            0.0,
                                                            0.0,
                                                            dX_star[0],
                                                            dX_star[1],
                                                            normals[q]);

            for (unsigned int i = 0; i < fe->n_dofs_per_cell(); ++i)
              {
                const unsigned int comp =
                  fe->system_to_component_index(i).first;
                const double phi = fe_face.shape_value(i, q);
                if (comp == 0)
                  copy.cell_matrix(i, j) +=
                    (dFA_dir + dFA_chain) * phi * JxW[q];
                else
                  copy.cell_matrix(i, j) +=
                    (dFU_dir + dFU_chain) * phi * JxW[q];
              }
          }
      }c
  };


  // Copier lambda
  const AffineConstraints<double> constraints;
  auto                            copier = [&](const BloodFlowCopyData &c) {
    constraints.distribute_local_to_global(c.cell_matrix,
                                           c.cell_rhs,
                                           c.local_dof_indices,
                                           jacobian_matrix,
                                           residual_vector);

    for (auto &cdf : c.face_data)
      {
        constraints.distribute_local_to_global(cdf.cell_matrix,
                                               cdf.cell_rhs,
                                               cdf.joint_dof_indices,
                                               jacobian_matrix,
                                               residual_vector);
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


  assemble_junction_terms();
}

// ========================================================================
// SOLVE LINEAR SYSTEM
// ========================================================================

template <int dim, int spacedim>
void
BloodFlowSystem<dim, spacedim>::solve()
{
  if (use_direct_solver)
    {
#ifdef DEAL_II_WITH_MUMPS
      SparseDirectMUMPS inverse;
#else
      SparseDirectUMFPACK inverse;
#endif

      inverse.initialize(jacobian_matrix);
      inverse.vmult(newton_update, residual_vector);
    }
  else
    {
      SolverControl               solver_control(1000, 1e-14);
      SolverGMRES<Vector<double>> solver(solver_control);
      PreconditionSSOR<>          preconditioner;
      preconditioner.initialize(jacobian_matrix, 1.4);
      solver.solve(jacobian_matrix,
                   newton_update,
                   residual_vector,
                   preconditioner);
      std::cout << "    Solver converged in " << solver_control.last_step()
                << " iterations." << std::endl;
    }
}

template <int dim, int spacedim>
void
BloodFlowSystem<dim, spacedim>::output_results(const unsigned int cycle) const
{
  const std::string filename =
    output_filename + "-" + std::to_string(cycle) + ".vtu";
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

  data_out.add_data_vector(solution,
                           solution_names,
                           DataOut<dim, spacedim>::type_dof_data,
                           data_component_interpretation);
  solution_names[0] = "pressure";
  solution_names[1] = "unused";
  data_out.add_data_vector(pressure,
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
  pvd_output_records.push_back(std::make_pair(time, filename));
  std::ofstream pvd_output(output_filename + ".pvd");
  DataOutBase::write_pvd_record(pvd_output, pvd_output_records);
}

// ========================================================================
// COMPUTE PRESSURE
// ========================================================================

template <int dim, int spacedim>
void
BloodFlowSystem<dim, spacedim>::compute_pressure()
{
  pressure         = 0;
  theoretical_peak = 0;

  const double gamma = 9.0;          // friction parameter
  const double mu    = par["eta_c"]; // viscosity
  const double A0    = par["a0"];
  const double c0    = compute_wave_speed(A0); // wave speed



  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      std::vector<types::global_dof_index> dof_indices(fe->n_dofs_per_cell());
      cell->get_dof_indices(dof_indices);
      const double x = cell->center()[0];
      for (unsigned int i = 0; i < fe->n_dofs_per_cell(); ++i)
        {
          const unsigned int component = fe->system_to_component_index(i).first;
          if (component == 0) // Area component
            {
              const double area        = solution[dof_indices[i]];
              pressure[dof_indices[i]] = compute_pressure_value(area);
              theoretical_peak[dof_indices[i]] =
                pressure[dof_indices[i]] *
                std::exp(-(gamma + 2.0) * numbers::PI * mu * x / (c0 * A0));
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
    for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
      {
        if (!cell->face(f)->at_boundary())
          continue;

        if (all_junction_faces.count({cell->id(), f}))
          continue;

        const auto bid = cell->face(f)->boundary_id();

        if (this->is_terminal_boundary(bid))
          terminal_Pc_storage.try_emplace(bid, par["p0"]);
      }
}
// ========================================================================
// COMPUTE ERRORS
// ========================================================================


template <int dim, int spacedim>
void
BloodFlowSystem<dim, spacedim>::compute_errors(unsigned int k)
{
  const ComponentSelectFunction<spacedim> area_mask(0, 1.0, 2);
  const ComponentSelectFunction<spacedim> velocity_mask(1, 1.0, 2);

  Vector<float> difference_per_cell(triangulation.n_active_cells());

  // // Create exact solution at current time
  // ExactSolutionBloodFlow<spacedim> exact_solution;
  exact_solution.set_time(time);
  initial_condition.set_time(time);
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


// ========================================================================
// Create vascular network
// ========================================================================
template <int dim, int spacedim>
void
BloodFlowSystem<dim, spacedim>::create_vascular_network()
{
  std::vector<Point<spacedim>> nodes = {{-0.5, 0.5, 0.0},
                                        {0.0, 0.0, 0.0},

                                        {-0.5, -0.5, 0.0},

                                        {0.5, 0.0, 0.0},

                                        {1.0, 0.0, 0.0},

                                        {1.5, 0.5, 0.0},
                                        {1.5, -0.5, 0.0},

                                        {2.0, 0.75, 0.0},
                                        {2.0, 0.25, 0.0},
                                        {2.0, -0.25, 0.0},
                                        {2.0, -0.75, 0.0}};


  std::vector<std::pair<int, int>> edges = {{0, 1},
                                            {2, 1},
                                            {1, 3},

                                            {3, 4},

                                            {4, 5},
                                            {4, 6},

                                            {5, 7},
                                            {5, 8},

                                            {6, 9},
                                            {6, 10}};


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
BloodFlowSystem<dim, spacedim>::run_convergence_study()
{
  std::cout << "=== CONVERGENCE STUDY for DG" << fe_degree
            << " ===" << std::endl;

  for (unsigned int cycle = 0; cycle < n_refinement_cycles; ++cycle)
    {
      std::cout << "\n--- Refinement Cycle " << cycle << " ---" << std::endl;

      if (cycle == 0)
        {
          triangulation.clear();

          if (!use_junction_mesh)
            {
              // ================= SINGLE VESSEL =================
              GridGenerator::hyper_cube(triangulation, 0.0, 10.0);
              //tagging logic for terminal boundaries
              terminal_boundary_ids.clear();
              unsigned int outlet_count = 0;

              for (auto &cell : triangulation.active_cell_iterators())
                {
                  for (unsigned int f = 0;
                       f < GeometryInfo<dim>::faces_per_cell;
                       ++f)
                    {
                      if (cell->face(f)->at_boundary())
                        {
                          Point<3> face_center = cell->face(f)->center();

                          // 1. Identify Inlet (Root) at (0,0,0)
                          if (face_center.distance(Point<3>(0.0, 0.0, 0.0)) <
                              1e-8)
                            {
                              cell->face(f)->set_boundary_id(0);
                              std::cout << "  [Mesh] Tagged INLET (ID 0) at "
                                        << face_center << std::endl;
                            }
                          // 2. Identify ALL other boundaries as Outlets
                          else
                            {
                              // This will now catch x = -0.5 AND x = 2.0
                              cell->face(f)->set_boundary_id(1);
                              terminal_boundary_ids.insert(1);
                              outlet_count++;
                              std::cout << "  [Mesh] Tagged OUTLET (ID 1) at "
                                        << face_center << std::endl;
                            }
                        }
                    }
                }
              triangulation.refine_global(n_global_refinements);
            }

          else
            {
              // ================= JUNCTION NETWORK =================


              if constexpr (dim == 1 && spacedim == 3)
                {
                  create_vascular_network();

                  // ================= TAGGING LOGIC for terminal boundary

                  terminal_boundary_ids.clear();
                  unsigned int outlet_count = 0;

                  for (auto &cell : triangulation.active_cell_iterators())
                    {
                      for (unsigned int f = 0;
                           f < GeometryInfo<dim>::faces_per_cell;
                           ++f)
                        {
                          if (cell->face(f)->at_boundary())
                            {
                              Point<3> face_center = cell->face(f)->center();

                              // 1. Identify Inlet (Root) at (0,0,0)
                              if (face_center.distance(Point<3>(-0.5, 0.5, 0)) <
                                  1e-8)
                                {
                                  cell->face(f)->set_boundary_id(0);
                                  std::cout
                                    << "  [Mesh] Tagged INLET (ID 0) at "
                                    << face_center << std::endl;
                                }
                              // 2. Identify ALL other boundaries as Outlets
                              else
                                {
                                  // This will now catch x = -0.5 AND x = 2.0
                                  cell->face(f)->set_boundary_id(1);
                                  terminal_boundary_ids.insert(1);
                                  outlet_count++;
                                  std::cout
                                    << "  [Mesh] Tagged OUTLET (ID 1) at "
                                    << face_center << std::endl;
                                }
                            }
                        }
                    }
                  triangulation.refine_global(n_global_refinements);
                }
            }
        }
      else
        {
          triangulation.refine_global(1);
        }


      setup_system();
      initialize_terminal_capacitors();

      // Project initial conditions
      AffineConstraints<double> constraints;
      constraints.close();
      VectorTools::project(dof_handler,
                           constraints,
                           QGauss<dim>(fe_degree + 1),
                           initial_condition,
                           solution);


      solution_old = solution;
      compute_pressure();
      output_results(0);

      // Run time stepping to final_time
      time = 0.0;

      n_time_steps =
        static_cast<unsigned int>(std::round(final_time / time_step));
      tmp_vector.reinit(dof_handler.n_dofs());



      for (unsigned int step = 1; step <= n_time_steps; ++step)
        {
          time += time_step;
          std::cout << "Step " << step << "  t=" << time << std::endl;

          unsigned int newton_iter = 0;
          while (newton_iter < max_newton_iterations)
            {
              // A. Assemble the residual and Jacobian at the current guess
              assemble_system();

              // B. Check the residual norm BEFORE the solve
              double current_residual = residual_vector.l2_norm();

              if (newton_iter == 0)
                std::cout << "  Initial Residual: " << std::scientific
                          << current_residual << std::endl;

              // C. Check for convergence
              if (current_residual < newton_tolerance)
                {
                  std::cout << "  Newton converged in " << newton_iter
                            << " iterations." << std::endl;
                  break;
                }

              // D. Solve for update and apply it
              solve();
              solution.add(omega, newton_update);

              // E. Divergence check
              if (!std::isfinite(current_residual) || current_residual > 1e12)
                {
                  std::cout << "⚠️ Residual exploded. Halving dt.\n";
                  time_step *= 0.5;
                  return; // Or exit loop
                }

              newton_iter++;
            }

          solution_old = solution;
          compute_pressure();
          output_results(step);
        }

      // Compute errors at final time
      compute_errors(cycle);
    }
}

template <int dim, int spacedim>
inline void
BloodFlowSystem<dim, spacedim>::assemble_mass_matrix()
{
  MatrixTools::create_mass_matrix(dof_handler,
                                  QGauss<dim>(fe_degree + 1),
                                  mass_matrix);
}

// Explicit instantiation
template class BloodFlowSystem<1, 3>;