/* --------------------------------------------------------------------------
 */
#include "blood_flow_system.h"

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
  add_parameter("Theta (penalty parameter)", theta);
  add_parameter("Theta Boundary (stability parameter)", theta_bd);
  add_parameter("Verbosity (console depth)", verbosity);
  add_parameter("Output directory", output_directory);

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
  linear_system_matrix.reinit(sparsity_pattern);
  solution.reinit(dof_handler.n_dofs());
  pressure.reinit(dof_handler.n_dofs());
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
                  par["eta_c"] * test_velocity * trial_velocity * JxW[point];
              }
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
                face.cell_matrix(i, j) -=
                  flux_A_jac_hll * fe_iv[area_extractor].jump_in_values(i, q) *
                  JxW[q];

                // Momentum equation contribution to Jacobian
                face.cell_matrix(i, j) -=
                  flux_U_jac_hll *
                  fe_iv[velocity_extractor].jump_in_values(i, q) * JxW[q];
              }
          }
      }
  };

  //========== BOUNDARY WORKER - WITH HLL FLUX ==========
  exact_solution.set_time(t);
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

        bool is_subcritical_inflow  = (incoming_count == 1) && (lambda1 <= 0.0);
        bool is_subcritical_outflow = (incoming_count == 1) && (lambda2 <= 0.0);
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

        // ===== JACOBIAN ASSEMBLY AT BOUNDARY =====
        for (unsigned int j = 0; j < fe_face.get_fe().dofs_per_cell; ++j)
          {
            const double trial_A = fe_face[area_extractor].value(j, q);
            const double trial_U = fe_face[velocity_extractor].value(j, q);

            // ===== HLL JACOBIAN AT BOUNDARY =====
            // Only trial functions on interior side are active (j comes from
            // interior cell) Exterior state does not change with trial
            // functions
            auto [flux_A_jac_boundary, flux_U_jac_boundary] = HLL_jacobian_flux(
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
                copy.cell_matrix(i, j) -= flux_A_jac_boundary * test_A * JxW[q];

                // Momentum equation Jacobian
                copy.cell_matrix(i, j) -= flux_U_jac_boundary * test_U * JxW[q];
              }
          }

        deallog.push("boundary_debug");
        deallog << "BID=" << boundary_id << " q=" << q << " | A_int=" << A_int
                << " U_int=" << U_int << " c=" << c_int
                << " | lambda1=" << lambda1 << " lambda2=" << lambda2
                << " | inlet=" << is_subcritical_inflow
                << " outlet=" << is_subcritical_outflow << " | A_ext=" << A_ext
                << " U_ext=" << U_ext << std::endl;
        deallog.pop();
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

        for (unsigned int i = 0; i < n_dofs; ++i)
          {
            const auto test_area     = fe_v[area_extractor].value(i, point);
            const auto test_velocity = fe_v[velocity_extractor].value(i, point);

            const auto test_area_grad = fe_v[area_extractor].gradient(i, point);
            const auto test_velocity_grad =
              fe_v[velocity_extractor].gradient(i, point);

            copy_data.cell_rhs(i) +=
              (
                // Area equation
                rhs_A_value * test_area +
                current_flux_A * test_area_grad
                // Momentum equation
                + rhs_U_value * test_velocity +
                current_flux_U * test_velocity_grad -
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
        auto [flux_A_hll, flux_U_hll] =
          HLL_residual_flux(cell,
                            ncell,
                            current_area[q],
                            current_velocity[q],
                            current_area_neighbor[q],
                            current_velocity_neighbor[q],
                            normals[q]);

        for (unsigned int i = 0; i < nd; ++i)
          {
            face.cell_rhs(i) -=
              flux_A_hll * fe_iv[area_extractor].jump_in_values(i, q) * JxW[q];

            face.cell_rhs(i) -= flux_U_hll *
                                fe_iv[velocity_extractor].jump_in_values(i, q) *
                                JxW[q];
          }
      }
  };

  //========== BOUNDARY WORKER - WITH HLL FLUX ==========
  exact_solution.set_time(t);
  auto boundary_worker = [&](const Iterator                      &cell,
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

        bool is_subcritical_inflow  = (incoming_count == 1) && (lambda1 <= 0.0);
        bool is_subcritical_outflow = (incoming_count == 1) && (lambda2 <= 0.0);
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

        auto [flux_A_boundary, flux_U_boundary] =
          HLL_residual_flux(cell, cell, A_int, U_int, A_ext, U_ext, normals[q]);

        const double F_area_boundary     = flux_A_boundary;
        const double F_momentum_boundary = flux_U_boundary;

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
                << " outlet=" << is_subcritical_outflow << " | A_ext=" << A_ext
                << " U_ext=" << U_ext << std::endl;
        deallog.pop();
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
  const unsigned int    cycle) const
{
  const std::string rel_filename =
    output_filename + "-" + std::to_string(cycle) + ".vtu";
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

// ========================================================================
// COMPUTE PRESSURE
// ========================================================================

template <int dim, int spacedim>
void
BloodFlowSystem<dim, spacedim>::compute_pressure(
  const Vector<double> &y,
  Vector<double>       &pressure_vec) const
{
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
              pressure_vec[dof_indices[i]] = compute_pressure_value(area);
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
          GridGenerator::hyper_cube(triangulation, 0.0, 1); // 1 cm
          triangulation.refine_global(n_global_refinements);
        }
      else
        {
          triangulation.refine_global(1);
        }

      setup_system();

      // Project initial conditions at ARKode initial time
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
        deallog.push("output_step");
        deallog << "Called output_step t=" << t
                << " step_number=" << step_number << std::endl;
        time = t;
        compute_pressure(sol, pressure);
        output_results(sol, pressure, step_number);
        deallog.pop();
      };

      const unsigned int n_timesteps = ode.solve_ode(solution);
      std::cout << "  ARKode steps: " << n_timesteps << std::endl;
      time = arkode_parameters.final_time;
      compute_pressure(solution, pressure);
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
  mass_solver.initialize(mass_matrix);
}

template <int dim, int spacedim>
void
BloodFlowSystem<dim, spacedim>::compute_initial_solution(Vector<double> &dst,
                                                         const double    t)
{
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
