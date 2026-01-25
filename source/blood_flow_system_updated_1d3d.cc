/* --------------------------------------------------------------------------
 */
#include "../include/blood_flow_system_updated_1d3d.h"

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
// For junction detection
// ========================================================================

template <int dim, int spacedim>
void
BloodFlowSystem<dim, spacedim>::detect_bifurcation_junctions()
{
  junctions.clear();

  const unsigned int n_vertices = triangulation.n_vertices();

  for (unsigned int v = 0; v < n_vertices; ++v)
    {
      std::vector<typename JunctionInfo::FaceData> incident_faces;

      for (const auto &cell : dof_handler.active_cell_iterators())
        {
          for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
            {
              // face corresponds to vertex in 1D
              if (cell->face(f)->vertex(0) == triangulation.get_vertices()[v])
                {
                  typename JunctionInfo::FaceData fd;
                  fd.cell    = cell;
                  fd.face_no = f;
                  incident_faces.push_back(fd);
                }
            }
        }

      // Y-junction = exactly 3 incident cells
      if (incident_faces.size() == 3)
        {
          JunctionInfo J;
          J.point = triangulation.get_vertices()[v];

          // first = parent (temporary choice)
          J.parent = incident_faces[0];
          J.daughters.push_back(incident_faces[1]);
          J.daughters.push_back(incident_faces[2]);

          junctions.push_back(J);

          std::cout << "Detected DG Y-junction at vertex " << v << " : "
                    << J.point << std::endl;
        }
    }
}



// ========================================================================
// ASSEMBLE JUNCTION TERMS
// ========================================================================

template <int dim, int spacedim>
void
BloodFlowSystem<dim, spacedim>::assemble_junction_terms()
{
  if (fe->dofs_per_vertex > 0)
    return; // CG not supported
  if (junctions.empty())
    return;

  const FEValuesExtractors::Scalar area(0);
  const FEValuesExtractors::Scalar velocity(1);

  QGauss<dim - 1>             quad(1);
  FEFaceValues<dim, spacedim> fe_face(*fe,
                                      quad,
                                      update_values | update_JxW_values);

  for (const auto &J : junctions)
    {
      Assert(J.daughters.size() == 2, ExcInternalError());

      // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
      // 1. Extract DG traces
      // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
      auto get_trace =
        [&](const JunctionInfo::FaceData &fd, double &A, double &U) {
          fe_face.reinit(fd.cell, fd.face_no);
          const unsigned int                   q = 0;
          std::vector<types::global_dof_index> dofs(fe->n_dofs_per_cell());
          fd.cell->get_dof_indices(dofs);

          A = U = 0.0;
          for (unsigned int i = 0; i < fe->n_dofs_per_cell(); ++i)
            {
              const unsigned int comp = fe->system_to_component_index(i).first;
              const double       phi  = fe_face.shape_value(i, q);
              const double       val  = solution(dofs[i]);
              if (comp == 0)
                A += val * phi;
              if (comp == 1)
                U += val * phi;
            }
        };

      double Ap, Up, Ad[2], Ud[2];
      get_trace(J.parent, Ap, Up);
      get_trace(J.daughters[0], Ad[0], Ud[0]);
      get_trace(J.daughters[1], Ad[1], Ud[1]);

      // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
      // 2. Characteristic variables
      // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
      const double cp  = compute_wave_speed(Ap);
      const double cd1 = compute_wave_speed(Ad[0]);
      const double cd2 = compute_wave_speed(Ad[1]);

      // Incoming characteristics
      const double Wp_minus = Up - 4.0 * cp;
      const double Wd1_plus = Ud[0] + 4.0 * cd1;
      const double Wd2_plus = Ud[1] + 4.0 * cd2;

      // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
      // 3. Solve junction coupling
      // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

      // Area-weighted pressure
      double       total_area = Ap + Ad[0] + Ad[1];
      const double Pstar      = (Ap * compute_pressure_value(Ap) +
                            Ad[0] * compute_pressure_value(Ad[0]) +
                            Ad[1] * compute_pressure_value(Ad[1])) /
                           total_area;

      // Wave speeds at junction
      const double Astar_p  = compute_area_from_pressure(Pstar);
      const double Astar_d1 = compute_area_from_pressure(Pstar);
      const double Astar_d2 = compute_area_from_pressure(Pstar);

      const double cstar_p  = compute_wave_speed(Astar_p);
      const double cstar_d1 = compute_wave_speed(Astar_d1);
      const double cstar_d2 = compute_wave_speed(Astar_d2);

      // USE characteristic invariants for velocities
      const double Ustar_p  = Wp_minus + cstar_p;
      const double Ustar_d1 = Wd1_plus - cstar_d1;
      const double Ustar_d2 = Wd2_plus - cstar_d2;

      // Mass conservation from characteristics
      const double Qp  = Astar_p * Ustar_p;
      const double Qd1 = Astar_d1 * Ustar_d1;
      const double Qd2 = Astar_d2 * Ustar_d2;

      // Diagnostic: check conservation
      double mass_error = std::abs(Qp - (Qd1 + Qd2)) / (std::abs(Qp) + 1e-16);
      if (mass_error > 1e-3)
        {
          std::cout << " Mass conservation error at junction: " << mass_error
                    << "\n";
        }

      // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
      // 4. DG residual contribution
      // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
      auto assemble_face =
        [&](const JunctionInfo::FaceData &fd, double Fm, double Fp) {
          fe_face.reinit(fd.cell, fd.face_no);
          std::vector<types::global_dof_index> dofs(fe->n_dofs_per_cell());
          fd.cell->get_dof_indices(dofs);

          for (unsigned int i = 0; i < fe->n_dofs_per_cell(); ++i)
            {
              const unsigned int comp = fe->system_to_component_index(i).first;
              const double       phi  = fe_face.shape_value(i, 0);
              const double       w    = fe_face.JxW(0);

              if (comp == 1)
                residual_vector[dofs[i]] += Fm * phi * w;
              if (comp == 0)
                residual_vector[dofs[i]] += Fp * phi * w;
            }
        };

      // Parent: outgoing
      assemble_face(J.parent, +Qp, +Pstar);

      // Daughters: incoming
      assemble_face(J.daughters[0], -Qd1, -Pstar);
      assemble_face(J.daughters[1], -Qd2, -Pstar);
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

  // detect_bifurcation_junctions();

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

  // ========== FACE WORKER (Interior Faces) - WITH HLL FLUX ==========
  auto face_worker = [&](const Iterator                      &cell,
                         const unsigned int                   f,
                         const unsigned int                   sf,
                         const Iterator                      &ncell,
                         const unsigned int                   nf,
                         const unsigned int                   nsf,
                         BloodFlowScratchData<dim, spacedim> &scratch,
                         BloodFlowCopyData                   &copy) {
    // is this a junction? If yes skip

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

        // ===== HLL RESIDUAL FLUX AT BOUNDARY =====
        // For boundary: treat exterior as "right" state and interior as "left"
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
                copy.cell_matrix(i, j) += flux_A_jac_boundary * test_A * JxW[q];

                // Momentum equation Jacobian
                copy.cell_matrix(i, j) += flux_U_jac_boundary * test_U * JxW[q];
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
          // // GridGenerator::hyper_cube(triangulation, 0.0, 1); // 1 cm
          // create_y_junction_mesh();

          triangulation.clear();

          const unsigned int N = 1;

          if constexpr (dim == 1 && spacedim == 3)
            {
              std::vector<Point<3>> vertices;

              vertices.push_back(Point<3>());
              vertices.push_back(Point<3>(0, .5, 0));
              vertices.push_back(Point<3>(-.5, 1, 0));
              vertices.push_back(Point<3>(.5, 1, 0));

              std::vector<CellData<1>> cells(3);
              cells[0].vertices[0] = 0;
              cells[0].vertices[1] = 1;
              cells[1].vertices[0] = 1;
              cells[1].vertices[1] = 2;
              cells[2].vertices[0] = 1;
              cells[2].vertices[1] = 3;

              triangulation.create_triangulation(vertices,
                                                 cells,
                                                 SubCellData());
            }
          else
            {
              // 1D meshes embedded in 3D
              Triangulation<1, 3> parent, d1, d2;

              // Parent vessel: [0,1]
              GridGenerator::subdivided_hyper_rectangle(
                parent,
                std::vector<unsigned int>{N},
                Point<1>(0.0),
                Point<1>(1.0));

              // Daughter 1: [1,2]
              GridGenerator::subdivided_hyper_rectangle(
                d1, std::vector<unsigned int>{N}, Point<1>(1.0), Point<1>(2.0));

              // Daughter 2: [1,2]
              GridGenerator::subdivided_hyper_rectangle(
                d2, std::vector<unsigned int>{N}, Point<1>(1.0), Point<1>(2.0));

              // non-manifold junction
              GridGenerator::merge_triangulations(parent, d1, triangulation);
              GridGenerator::merge_triangulations(triangulation,
                                                  d2,
                                                  triangulation);
            }
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

      detect_bifurcation_junctions();
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
      if (!junctions.empty())
        {
          std::cout
            << "Skipping convergence study: junctions present "
            << "(error norms not well-defined at non-manifold points).\n";
        }
      else
        {
          compute_errors(cycle);
        }
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