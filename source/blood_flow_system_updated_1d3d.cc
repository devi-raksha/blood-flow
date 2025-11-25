/* --------------------------------------------------------------------------
 */
#include "../include/blood_flow_system_updated_1d3d.h"

#include <deal.II/base/function_parser.h>
#include <deal.II/base/types.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_direct.h>

#include <deal.II/meshworker/mesh_loop.h>

#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/vector_tools.h>

#include <algorithm>
#include <cmath>
#include <iomanip>

// Main class implementation
template <int dim, int spacedim>
BloodFlowSystem<dim, spacedim>::BloodFlowSystem()
  : par("Blood Flow Parameters",
        {"rho", "a0", "mu", "p0", "eta_c", "r0", "m"},
        {1.06, 3.141592653589793e-4, 1.0e6, 0.0, 1.0, 9.99e-3, 0.5},
        {"Density",
         "Reference cross-sectional area",
         "Elastic modulus of the vessel wall",
         "Reference pressure",
         "Viscosity coefficient",
         "Reference radius",
         "Tube law exponent"})
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
  // add_parameter("Picard iterations", max_picard_iterations);
  add_parameter("Newton iterations", max_newton_iterations);
  add_parameter("Newton tolerance", newton_tolerance);
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

  // ========== FACE WORKER (Interior Faces) ==========
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
      solution, current_area);
    fe_iv.get_fe_face_values(0)[velocity_extractor].get_function_values(
      solution, current_velocity);
    fe_iv.get_fe_face_values(1)[area_extractor].get_function_values(
      solution, current_area_neighbor);
    fe_iv.get_fe_face_values(1)[velocity_extractor].get_function_values(
      solution, current_velocity_neighbor);

    // Previous time iterate values on both sides
    std::vector<double> old_area(n_q), old_velocity(n_q),
      old_area_neighbor(n_q), old_velocity_neighbor(n_q);
    fe_iv.get_fe_face_values(0)[area_extractor].get_function_values(
      solution_old, old_area);
    fe_iv.get_fe_face_values(0)[velocity_extractor].get_function_values(
      solution_old, old_velocity);
    fe_iv.get_fe_face_values(1)[area_extractor].get_function_values(
      solution_old, old_area_neighbor);
    fe_iv.get_fe_face_values(1)[velocity_extractor].get_function_values(
      solution_old, old_velocity_neighbor);

    copy.face_data.emplace_back();
    auto              &face = copy.face_data.back();
    const unsigned int nd   = fe_iv.n_current_interface_dofs();
    face.joint_dof_indices  = fe_iv.get_interface_dof_indices();
    face.cell_matrix.reinit(nd, nd);
    face.cell_rhs.reinit(nd);

    for (unsigned int q = 0; q < n_q; ++q)
      {
        // Compute pressures
        const double current_pressure = compute_pressure_value(current_area[q]);

        const double current_pressure_neighbor =
          compute_pressure_value(current_area_neighbor[q]);

        // Wave speeds for Lax-Friedrichs penalty
        const double cL = compute_wave_speed(current_area[q]);
        const double cR = compute_wave_speed(current_area_neighbor[q]);

        // -----------------------------------------------------------------------------
        // Lax–Friedrichs (Rusanov) numerical flux penalty parameter.
        // The constant C is chosen as the *maximum characteristic speed*
        // across the interface, i.e. the spectral radius of the normal flux
        // Jacobian.
        //
        //     C = max_{u in {u_L , u_R}}  | lambda_i( (n · partial f/ partial u)(u) ) | ,
        //
        // where lambda_i(·) are the eigenvalues of the matrix (n · ∂f/∂u),
        // i.e. the Jacobian of the physical flux projected along the
        // interface normal direction.
        //
        // For the 1D blood–flow system (A,U), the characteristic speeds are
        //     lambda_1 = U − c ,    lambda_2  = U + c ,
        // so the LF penalty is
        //     C = max( |U_L − c_L|, |U_L + c_L|,
        //               |U_R − c_R|, |U_R + c_R| ).
        //
        // Reference:
        //   Hesthaven & Warburton, "Nodal Discontinuous Galerkin Methods",
        //   Springer, 2008. Chapter 2: "Introduction to DG Methods",
        //   Section 2.3.1: Local Lax–Friedrichs (Rusanov) Flux.
        // -----------------------------------------------------------------------------

        const double bn = compute_tangent_normal_product(cell, normals[q]);
        const double bn_neighbor =
          compute_tangent_normal_product(ncell, normals[q]);
        const double beta = compute_LF_penalty(current_area[q],
                                               current_area_neighbor[q],
                                               current_velocity[q],
                                               current_velocity_neighbor[q],
                                               bn,
                                               bn_neighbor);

        // const double h = cell->measure();
        const double alpha  = theta * beta;
        const double F_area = compute_physical_area_flux(cell,
                                                         current_area[q],
                                                         current_velocity[q]) *
                              normals[q];

        const double F_area_neighbor =
          compute_physical_area_flux(ncell,
                                     current_area_neighbor[q],
                                     current_velocity_neighbor[q]) *
          normals[q];

        const double F_hat_area_numerical =
          0.5 * (F_area + F_area_neighbor) -
          0.5 * alpha * (current_area[q] - current_area_neighbor[q]);

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

                // ===== AREA FLUX Jacobian =====
                // F_hat_A = 0.5*(U_L*trial_A_L + A_L*trial_U_L +
                // U_R*trial_A_R + A_R*trial_U_R)
                //           - 0.5*alpha*(trial_A_L - trial_A_R)
                const auto flux_jac_A =
                  compute_physical_area_jacobian_flux(cell,
                                                      current_area[q],
                                                      trial_velocity,
                                                      current_velocity[q],
                                                      trial_area) *
                  normals[q];
                const auto flux_jac_A_neighbor =
                  compute_physical_area_jacobian_flux(
                    ncell,
                    current_area_neighbor[q],
                    trial_velocity_neighbor,
                    current_velocity_neighbor[q],
                    trial_area_neighbor) *
                  normals[q];

                // Lax-Friedrichs numerical flux
                const double F_hat_area =
                  0.5 * (flux_jac_A + flux_jac_A_neighbor) -
                  0.5 * alpha * (trial_area - trial_area_neighbor);

                face.cell_matrix(i, j) +=
                  F_hat_area * fe_iv[area_extractor].jump_in_values(i, q) *
                  JxW[q];

                // ===== MOMENTUM FLUX JACOBIAN =====
                const double c_sq = cL * cL;

                const double c_sq_neighbor = cR * cR;

                const double flux_jac_U =
                  compute_physical_momentum_jacobian_flux(cell,
                                                          c_sq,
                                                          current_area[q],
                                                          trial_area,
                                                          current_velocity[q],
                                                          trial_velocity) *
                  normals[q];

                const double flux_jac_U_neighbor =
                  compute_physical_momentum_jacobian_flux(
                    ncell,
                    c_sq_neighbor,
                    current_area_neighbor[q],
                    trial_area_neighbor,
                    current_velocity_neighbor[q],
                    trial_velocity_neighbor) *
                  normals[q];
                const double F_hat_U =
                  0.5 * (flux_jac_U + flux_jac_U_neighbor) -
                  0.5 * alpha * (trial_velocity - trial_velocity_neighbor);

                face.cell_matrix(i, j) +=
                  F_hat_U * fe_iv[velocity_extractor].jump_in_values(i, q) *
                  JxW[q];
              }

            // ===== RHS: AREA FLUX =====
            // F_A = U*A (at left and right states)
            // Lax-Friedrichs: 0.5*(F_A^L + F_A^R) - 0.5*alpha*(A^L - A^R)

            face.cell_rhs(i) -= F_hat_area_numerical *
                                fe_iv[area_extractor].jump_in_values(i, q) *
                                JxW[q];

            // ===== RHS: MOMENTUM FLUX - CONVECTIVE PART (U^2/2) =====
            // F_U_conv = U^2/2 + P/rho (at left and right states)
            const double F_momentum =
              compute_physical_momentum_flux(cell,
                                             current_velocity[q],
                                             current_velocity[q],
                                             current_pressure) *
              normals[q];
            const double F_momentum_neighbor =
              compute_physical_momentum_flux(ncell,
                                             current_velocity_neighbor[q],
                                             current_velocity_neighbor[q],
                                             current_pressure_neighbor) *
              normals[q];

            const double F_hat_momentum_numerical =
              0.5 * (F_momentum + F_momentum_neighbor) -
              0.5 * alpha *
                (current_velocity[q] - current_velocity_neighbor[q]);

            face.cell_rhs(i) -= F_hat_momentum_numerical *
                                fe_iv[velocity_extractor].jump_in_values(i, q) *
                                JxW[q];
          }
      }
  };

  // ========== BOUNDARY WORKER ==========
  exact_solution.set_time(time);
  auto boundary_worker = [&](const Iterator                      &cell,
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
    fe_face[area_extractor].get_function_values(solution, current_area);
    fe_face[velocity_extractor].get_function_values(solution, current_velocity);

    // Boundary values
    exact_solution.set_time(time);
    std::vector<Vector<double>> bc(n_q, Vector<double>(2));
    exact_solution.vector_value_list(fe_face.get_quadrature_points(), bc);

    // Determine which boundary this is (BID=0 is inlet, BID=1 is outlet)
    unsigned int boundary_id = cell->face(face_no)->boundary_id();
    // bool         is_inlet_boundary  = (boundary_id == 0); // based on mesh
    // bool         is_outlet_boundary = (boundary_id == 1); // based on mesh

    for (unsigned int q = 0; q < n_q; ++q)
      {
        const double A_bc = bc[q](0);
        const double U_bc = bc[q](1);

        // Interior state
        const double A_int = current_area[q];
        const double U_int = current_velocity[q];

        // Pressures
        // const double p_int = compute_pressure_value(A_int);
        // const double p_bc  = compute_pressure_value(A_bc);

        const double dpdA  = compute_pressure_derivative(A_int);
        const double c_int = compute_wave_speed(A_int);

        const double bn = compute_tangent_normal_product(cell, normals[q]);

        // ===== CHARACTERISTIC SPEEDS =====
        // In 1D: lambda1 = (U - c) * n, lambda2 = (U + c) * n
        const double lambda1 = (U_int - c_int) * bn;
        const double lambda2 = (U_int + c_int) * bn;

        // ===== BOUNDARY CLASSIFICATION =====
        // Count incoming characteristics
        int incoming_count = 0;
        if (lambda1 < 0.0)
          incoming_count++;
        if (lambda2 < 0.0)
          incoming_count++;

        bool is_subcritical           = (incoming_count == 1);
        bool is_supercritical_inflow  = (incoming_count == 2);
        bool is_supercritical_outflow = (incoming_count == 0);

        // ===== DETERMINE EXTERIOR STATE (A_ext, U_ext) =====

        double A_ext, U_ext;

        if (is_subcritical)
          {
            // Subcritical: 1 incoming characteristic

            A_ext = A_bc;  // Impose area at boundary
            U_ext = U_int; // velocity from interior
          }
        else if (is_supercritical_inflow)
          {
            // Supercritical inflow: both characteristics enter
            A_ext = A_bc; // Impose area
            U_ext = U_bc; // Impose velocity
          }
        else
          { // is_supercritical_outflow
            // Supercritical outflow: no characteristics enter

            A_ext = A_int; // interior area
            U_ext = U_int; // interior velocity
          }

        // ===== LAXFRIEDRICHS PENALTY =====
        const double an          = bn;
        const double an_neighbor = bn;
        const double beta_bd =
          compute_LF_penalty(A_int, A_ext, U_int, U_ext, an, an_neighbor);

        const double alpha_bd = theta_bd * beta_bd;

        // ===== RHS ASSEMBLY =====
        // Pressure at exterior
        const double p_ext = compute_pressure_value(A_ext);

        // Physical fluxes
        auto F_area_ext =
          compute_physical_area_flux(cell, A_ext, U_ext) * normals[q];
        auto F_momentum_ext =
          compute_physical_momentum_flux(cell, U_ext, U_ext, p_ext) *
          normals[q];


        // Rusanov dissipation
        auto F_area_dissipation     = -alpha_bd * (A_ext - A_int);
        auto F_momentum_dissipation = -alpha_bd * (U_ext - U_int);

        // Complete flux = physical flux - Rusanov dissipation
        auto F_area_bd     = F_area_ext + F_area_dissipation;
        auto F_momentum_bd = F_momentum_ext + F_momentum_dissipation;

        // ===== JACOBIAN ASSEMBLY =====
        // Trial function loop
        for (unsigned int j = 0; j < fe_face.get_fe().dofs_per_cell; ++j)
          {
            const double trial_A = fe_face[area_extractor].value(j, q);
            const double trial_U = fe_face[velocity_extractor].value(j, q);

            // Derivative of A_ext with respect to solution
            double dA_ext_dA = 0.0;
            if (is_supercritical_outflow)
              {
                dA_ext_dA = 1.0;
              }
            // If subcritical or inflow: A_ext = A_bc (prescribed), so
            // dA_ext/dA = 0

            // Derivative of U_ext with respect to solution
            double dU_ext_dU = 0.0;
            if (is_supercritical_outflow)
              {
                dU_ext_dU = 1.0; // U_ext = U_int → depends on interior velocity
              }
            // If subcritical: U_ext = U_int → dU_ext/dU = 1
            if (is_subcritical)
              {
                dU_ext_dU = 1.0;
              }
            // If inflow: U_ext = U_bc (prescribed), so dU_ext/dU = 0

            // Area equation Jacobian
            // ∂/∂A [F_area - alpha * (A_ext - A_int)]
            const double flux_jac_area_val =
              compute_physical_area_jacobian_flux(
                cell, A_ext, trial_U, U_ext, trial_A) *
                normals[q] -
              alpha_bd * (dA_ext_dA - 1.0);

            // Momentum equation Jacobian (simplified)
            const double c_sq = A_int / par["rho"] * dpdA;
            const double flux_jac_momentum_val =
              compute_physical_momentum_jacobian_flux(cell,
                                                      c_sq,
                                                      A_ext,
                                                      trial_A,
                                                      U_ext,
                                                      trial_U) *
                normals[q] -
              alpha_bd * (dU_ext_dU - 1.0); // from boundary_penalty

            // Test function loop
            for (unsigned int i = 0; i < fe_face.get_fe().dofs_per_cell; ++i)
              {
                const double test_A = fe_face[area_extractor].value(i, q);
                const double test_U = fe_face[velocity_extractor].value(i, q);

                // Area equation
                copy.cell_matrix(i, j) += flux_jac_area_val * test_A * JxW[q];

                // Momentum equation
                copy.cell_matrix(i, j) +=
                  flux_jac_momentum_val * test_U * JxW[q];
              }
          }

        // ===== Residual RHS ASSEMBLY =====
        for (unsigned int i = 0; i < fe_face.get_fe().dofs_per_cell; ++i)
          {
            const double test_A = fe_face[area_extractor].value(i, q);
            const double test_U = fe_face[velocity_extractor].value(i, q);

            // Area RHS
            copy.cell_rhs(i) -= F_area_bd * test_A * JxW[q];

            // Momentum RHS
            copy.cell_rhs(i) -= F_momentum_bd * test_U * JxW[q];
          }

        // ===== DEBUG OUTPUT =====
        //  boundary classifications
        std::cout << "BID=" << boundary_id << " q=" << q << " | A_int=" << A_int
                  << " U_int=" << U_int << " c=" << c_int
                  << " | lambda1=" << lambda1 << " lambda2=" << lambda2
                  << " | subcrit=" << is_subcritical << " | A_ext=" << A_ext
                  << " U_ext=" << U_ext << std::endl;
      }
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

  MeshWorker::mesh_loop(dof_handler.begin_active(),
                        dof_handler.end(),
                        cell_worker,
                        copier,
                        scratch_data,
                        copy_data,
                        MeshWorker::assemble_own_cells |
                          MeshWorker::assemble_boundary_faces |
                          MeshWorker::assemble_own_interior_faces_once,
                        boundary_worker,
                        face_worker);
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
      SparseDirectUMFPACK inverse;
      inverse.initialize(jacobian_matrix);           // use jacobian_matrix
      inverse.vmult(newton_update, residual_vector); // use newton_update
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
  pressure = 0;

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      std::vector<types::global_dof_index> dof_indices(fe->n_dofs_per_cell());
      cell->get_dof_indices(dof_indices);

      for (unsigned int i = 0; i < fe->n_dofs_per_cell(); ++i)
        {
          const unsigned int component = fe->system_to_component_index(i).first;
          if (component == 0) // Area component
            {
              const double area        = solution[dof_indices[i]];
              pressure[dof_indices[i]] = compute_pressure_value(area);
            }
        }
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
          // GridGenerator::hyper_cube(triangulation);
          GridGenerator::hyper_cube(triangulation, 0.0, 1); // 1 cm

          triangulation.refine_global(n_global_refinements);
        }
      else
        {
          triangulation.refine_global(1);
        }

      setup_system();

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

          // Newton iteration loop
          solution                 = solution_old; // w^(0) := W^n
          unsigned int newton_iter = 0;
          bool         converged   = false;

          // Newton loop
          do
            {
              // Step 1: Assemble system
              assemble_system();

              // Step 2: Solve for Newton update
              solve();

              // Step 3: Apply the update
              solution.add(omega, newton_update);

              // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
              // Reassemble AFTER updating solution
              // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
              assemble_system();

              // Step 4: Get residual at NEW solution
              double newton_residual_norm = residual_vector.l2_norm();

              std::cout << " Newton iter " << newton_iter
                        << "  residual = " << std::scientific
                        << std::setprecision(6) << newton_residual_norm
                        << std::endl;

              // Step 5: Check for divergence
              if (!std::isfinite(newton_residual_norm) ||
                  newton_residual_norm > 1e12)
                {
                  std::cout << "⚠️  Residual too large, halving Δt\n";
                  time_step *= 0.5;
                  break;
                }

              // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
              // Check convergence AFTER update
              // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
              if (newton_residual_norm < newton_tolerance)
                {
                  converged = true;
                  std::cout << "  Newton converged in " << newton_iter
                            << " iterations.\n";
                  break;
                }

              // Step 6: Increment counter
              ++newton_iter;

              if (newton_iter >= max_newton_iterations)
                {
                  std::cout << "  WARNING: Newton did not converge after "
                            << max_newton_iterations << " iterations.\n";
                  converged = true;
                }
            }
          while (!converged);

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