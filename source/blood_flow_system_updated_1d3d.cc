/* --------------------------------------------------------------------------
 */
#include "../include/blood_flow_system_updated_1d3d.h"

#include <deal.II/base/exceptions.h>
#include <deal.II/base/function_parser.h>
#include <deal.II/base/types.h>
#include <deal.II/base/utilities.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>

#include <deal.II/meshworker/mesh_loop.h>

#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/vector_tools.h>

#include <algorithm>
#include <iomanip>

namespace dealii
{
  // Main class implementation
  template <int dim, int spacedim>
  BloodFlowSystem<dim, spacedim>::BloodFlowSystem()
    : triangulation()
    , dof_handler(triangulation)
    , fe(nullptr)
    , time_step(1.0)
    , time(0.0)
    , n_time_steps(0)
  {
    add_parameter("Finite element degree", fe_degree);
    add_parameter("Problem constants", constants);
    add_parameter("Output filename", output_filename);
    add_parameter("Use direct solver", use_direct_solver);
    add_parameter("Number of refinement cycles", n_refinement_cycles);
    add_parameter("Number of global refinement", n_global_refinements);
    add_parameter("Time step", time_step);
    add_parameter("Final time", final_time);
    add_parameter("Density (rho)", rho);
    add_parameter("Viscosity coefficient (c)", viscosity_c);
    add_parameter("Reference area", reference_area);
    add_parameter("Elastic modulus", elastic_modulus);
    add_parameter("Reference pressure", reference_pressure);
    add_parameter("Theta (penalty parameter)", theta);
    add_parameter("Eta (stability parameter)", eta);
    add_parameter("Initial condition A expression", initial_A_expression);
    add_parameter("Initial condition U expression", initial_Q_expression);
    // add_parameter("Pressure boundary expression", pressure_bc_expression);
  }

  template <int dim, int spacedim>
  void
  BloodFlowSystem<dim, spacedim>::initialize_params(const std::string &filename)
  {
    ParameterAcceptor::initialize(filename,
                                  "last_used_parameters.prm",
                                  ParameterHandler::Short);
  }

  template <int dim, int spacedim>
  void
  BloodFlowSystem<dim, spacedim>::setup_system()
  {
    if (!fe)
      {
        // Two-component system: A (area) and U (momentum)
        fe = std::make_unique<FESystem<dim, spacedim>>(
          FE_DGQ<dim, spacedim>(fe_degree), 2);


        // Initialize RHS functions with current parameters
        rhs_A_function = std::make_unique<RHS_A_BloodFlow<spacedim>>();
        rhs_Q_function = std::make_unique<RHS_Q_BloodFlow<spacedim>>();

        // Update RHS function parameters
        rhs_Q_function->set_rho(rho);
        rhs_Q_function->set_elastic_modulus(elastic_modulus);
        rhs_Q_function->set_viscosity_c(viscosity_c);

        std::string vars;
        if (spacedim == 1)
          vars = "x";
        else if (spacedim == 2)
          vars = "x,y";
        else
          vars = "x,y,z";
        std::map<std::string, double> const_map;
        initial_A.initialize(vars, initial_A_expression, const_map);
        initial_Q.initialize(vars, initial_Q_expression, const_map);
        // pressure_bc.initialize(vars, pressure_bc_expression, const_map);
      }

    dof_handler.distribute_dofs(*fe);

    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_flux_sparsity_pattern(dof_handler, dsp);
    sparsity_pattern.copy_from(dsp);
    system_matrix.reinit(sparsity_pattern);
    mass_matrix.reinit(sparsity_pattern);
    solution.reinit(dof_handler.n_dofs());
    solution_old.reinit(dof_handler.n_dofs());
    right_hand_side.reinit(dof_handler.n_dofs());
    pressure.reinit(dof_handler.n_dofs());
  }

  template <int dim, int spacedim>
  void
  BloodFlowSystem<dim, spacedim>::assemble_mass_matrix()
  {
    mass_matrix = 0;
    Vector<double> dummy_rhs(dof_handler.n_dofs());

    const FEValuesExtractors::Scalar area_extractor(0);
    const FEValuesExtractors::Scalar momentum_extractor(1);

    auto cell_worker = [&](const auto                          &cell,
                           BloodFlowScratchData<dim, spacedim> &scratch,
                           BloodFlowCopyData                   &copy) {
      const unsigned int n_dofs = scratch.fe_values.get_fe().n_dofs_per_cell();
      copy.reinit(cell, n_dofs);
      scratch.fe_values.reinit(cell);

      const auto &fe_v = scratch.fe_values;
      const auto &JxW  = fe_v.get_JxW_values();

      for (unsigned int q = 0; q < fe_v.n_quadrature_points; ++q)
        {
          for (unsigned int i = 0; i < n_dofs; ++i)
            {
              for (unsigned int j = 0; j < n_dofs; ++j)
                {
                  // Mass matrix for area component
                  copy.cell_matrix(i, j) += fe_v[area_extractor].value(i, q) *
                                            fe_v[area_extractor].value(j, q) *
                                            JxW[q];

                  // Mass matrix for momentum component
                  copy.cell_matrix(i, j) +=
                    fe_v[momentum_extractor].value(i, q) *
                    fe_v[momentum_extractor].value(j, q) * JxW[q];
                }
            }
        }
    };

    // Execute assembly
    const QGauss<dim>     quadrature(fe->tensor_degree() + 1);
    const QGauss<dim - 1> quadrature_face(fe->tensor_degree() + 1);

    BloodFlowScratchData<dim, spacedim> scratch_data(*fe,
                                                     quadrature,
                                                     quadrature_face);
    BloodFlowCopyData                   copy_data;

    const AffineConstraints<double> constraints;
    auto                            copier = [&](const BloodFlowCopyData &c) {
      constraints.distribute_local_to_global(
        c.cell_matrix, c.cell_rhs, c.local_dof_indices, mass_matrix, dummy_rhs);
    };

    MeshWorker::mesh_loop(dof_handler.begin_active(),
                          dof_handler.end(),
                          cell_worker,
                          copier,
                          scratch_data,
                          copy_data,
                          MeshWorker::assemble_own_cells);
  }

  template <int dim, int spacedim>
  void
  BloodFlowSystem<dim, spacedim>::assemble_system()
  {
    using Iterator = typename DoFHandler<dim, spacedim>::active_cell_iterator;

    system_matrix   = 0;
    right_hand_side = 0;

    // Define extractors
    const FEValuesExtractors::Scalar area_extractor(0);
    const FEValuesExtractors::Scalar momentum_extractor(1);

    // Cell worker - handles volume integral terms
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

      // Get old solution values using extractors
      std::vector<double> old_area_values(fe_v.n_quadrature_points);
      std::vector<double> old_momentum_values(fe_v.n_quadrature_points);

      fe_v[area_extractor].get_function_values(solution_old, old_area_values);
      fe_v[momentum_extractor].get_function_values(solution_old,
                                                   old_momentum_values);

      for (unsigned int point = 0; point < fe_v.n_quadrature_points; ++point)
        {
          const auto b_vec = (cell->vertex(1) - cell->vertex(0)) /
                             cell->vertex(1).distance(cell->vertex(0));
          const double A_old = old_area_values[point];
          const double Q_old = old_momentum_values[point];

          // Set time for RHS functions
          rhs_A_function->set_time(time);
          rhs_Q_function->set_time(time);

          const double rhs_A_value = rhs_A_function->value(q_points[point]);
          const double rhs_Q_value = rhs_Q_function->value(q_points[point]);

          for (unsigned int i = 0; i < n_dofs; ++i)
            {
              for (unsigned int j = 0; j < n_dofs; ++j)
                {
                  // A-Q block: implicit term in momentum equation
                  // - (b · ∇\phi_A, Q^{n+1})
                  const double b_gradA =
                    b_vec * fe_v[area_extractor].gradient(i, point);

                  copy_data.cell_matrix(i, j) -=
                    fe_v[momentum_extractor].value(j, point) * b_gradA *
                    JxW[point];

                  // pressure term in momentum equation
                  // -(b · ∇\phi_Q ,  P_old/rho *A)
                  const double P_old = compute_pressure_value<dim, spacedim>(
                    A_old, reference_area, elastic_modulus, reference_pressure);

                  copy_data.cell_matrix(i, j) -=
                    (1.0 / rho) * P_old * fe_v[area_extractor].value(j, point) *
                    fe_v[momentum_extractor].gradient(i, point) * b_vec *
                    JxW[point];

                  // - (\phi_Q, P_old /rho*b · ∇A)
                  copy_data.cell_matrix(i, j) -=
                    (1.0 / rho) * P_old *
                    fe_v[momentum_extractor].value(i, point) *
                    fe_v[area_extractor].gradient(j, point) * b_vec *
                    JxW[point];
                }

              copy_data.cell_rhs(i) +=
                1 / rho * Q_old * Q_old / A_old *
                fe_v[momentum_extractor].gradient(i, point) * b_vec *
                JxW[point];


              // Right-hand side terms
              copy_data.cell_rhs(i) +=
                rhs_A_value * fe_v[area_extractor].value(i, point) * JxW[point];
              copy_data.cell_rhs(i) +=
                rhs_Q_value * fe_v[momentum_extractor].value(i, point) *
                JxW[point];
            }
        }
    };

    // Face worker - handles interior face integrals using header flux functions
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

      // Extract solution values (old time level)
      std::vector<double> A_L(n_q), Q_L(n_q), A_R(n_q), Q_R(n_q);
      fe_iv.get_fe_face_values(0)[area_extractor].get_function_values(
        solution_old, A_L);
      fe_iv.get_fe_face_values(0)[momentum_extractor].get_function_values(
        solution_old, Q_L);
      fe_iv.get_fe_face_values(1)[area_extractor].get_function_values(
        solution_old, A_R);
      fe_iv.get_fe_face_values(1)[momentum_extractor].get_function_values(
        solution_old, Q_R);

      copy.face_data.emplace_back();
      auto              &face = copy.face_data.back();
      const unsigned int nd   = fe_iv.n_current_interface_dofs();
      face.joint_dof_indices  = fe_iv.get_interface_dof_indices();
      face.cell_matrix.reinit(nd, nd);
      face.cell_rhs.reinit(nd);

      for (unsigned int q = 0; q < n_q; ++q)
        {
          const auto   normal = normals[q];
          const double b_dot_n =
            compute_tangent_normal_product<dim, spacedim>(cell, normal);

          // Left and right states (old time level)
          const double AL_old = A_L[q], QL_old = Q_L[q];
          const double AR_old = A_R[q], QR_old = Q_R[q];

          for (unsigned int j = 0; j < nd; ++j)
            {
              // New time level basis function values
              const double AL_new = fe_iv[area_extractor].value(0, j, q);
              const double AR_new = fe_iv[area_extractor].value(1, j, q);

              // Compute numerical fluxes
              const double flux_A =
                compute_area_rusanov_flux<dim, spacedim>(AL_new,
                                                         AR_new,
                                                         AL_old,
                                                         AR_old,
                                                         reference_area,
                                                         elastic_modulus,
                                                         rho,
                                                         b_dot_n);

              const double flux_Q =
                compute_momentum_rusanov_flux<dim, spacedim>(
                  QL_old, QR_old, AL_old, AR_old, rho, b_dot_n);

              for (unsigned int i = 0; i < nd; ++i)
                {
                  // Area equation: flux_A contribution
                  face.cell_matrix(i, j) +=
                    flux_A * fe_iv[area_extractor].jump_in_values(i, q) *
                    JxW[q];

                  // Momentum equation: flux_Q contribution
                  face.cell_rhs(i) +=
                    flux_Q * fe_iv[momentum_extractor].jump_in_values(i, q) *
                    JxW[q];
                }
            }
        }
    };

    // Boundary worker with periodic and Dirichlet support
    ExactSolutionBloodFlow<spacedim> exact_solution;
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

      // Interior trace (left state)
      std::vector<double> A_L(n_q), Q_L(n_q);
      fe_face[area_extractor].get_function_values(solution_old, A_L);
      fe_face[momentum_extractor].get_function_values(solution_old, Q_L);

      copy.cell_matrix.reinit(fe_face.get_fe().dofs_per_cell,
                              fe_face.get_fe().dofs_per_cell);
      copy.cell_rhs.reinit(fe_face.get_fe().dofs_per_cell);

      // Check boundary type
      const auto boundary_id = cell->face(face_no)->boundary_id();

      if (boundary_id == 0 || boundary_id == 1) // Periodic boundaries
        {
          // For now, use exact solution as placeholder
          std::vector<Vector<double>> bc(n_q, Vector<double>(2));
          exact_solution.vector_value_list(fe_face.get_quadrature_points(), bc);

          for (unsigned int q = 0; q < n_q; ++q)
            {
              const double b_dot_n =
                compute_tangent_normal_product<dim, spacedim>(cell, normals[q]);

              // States
              const double AL_old = A_L[q], QL_old = Q_L[q];
              const double AR_old = bc[q](0),
                           QR_old = bc[q](1); // From periodic neighbor

              for (unsigned int j = 0; j < fe_face.get_fe().dofs_per_cell; ++j)
                {
                  // New time level basis values
                  const double AL_new = fe_face[area_extractor].value(j, q);
                  const double AR_new =
                    bc[q](0); // Prescribed boundary (or from periodic neighbor)

                  // Compute fluxes
                  const double flux_A =
                    compute_area_rusanov_flux<dim, spacedim>(AL_new,
                                                             AR_new,
                                                             AL_old,
                                                             AR_old,
                                                             reference_area,
                                                             elastic_modulus,
                                                             rho,
                                                             b_dot_n);

                  const double flux_Q =
                    compute_momentum_rusanov_flux<dim, spacedim>(
                      QL_old, QR_old, AL_old, AR_old, rho, b_dot_n);

                  for (unsigned int i = 0; i < fe_face.get_fe().dofs_per_cell;
                       ++i)
                    {
                      // 1) Area equation: flux_A contribution
                      copy.cell_matrix(i, j) +=
                        flux_A * fe_face[area_extractor].value(i, q) * JxW[q];

                      // 2) -<A^{n+1}P(A^n)/ρ, φ_Q> (implicit in matrix)
                      const double P_boundary_old =
                        compute_pressure_value<dim, spacedim>(
                          AR_old,
                          reference_area,
                          elastic_modulus,
                          reference_pressure);
                      copy.cell_matrix(i, j) -=
                        (AR_new * P_boundary_old / rho) *
                        fe_face[momentum_extractor].value(i, q) * JxW[q];
                    }

                  // RHS contributions

                  // 1) -<Q^{n+1}, φ_A>
                  copy.cell_rhs(j) -=
                    QR_old * fe_face[area_extractor].value(j, q) * JxW[q];

                  // 2) Momentum flux contribution
                  copy.cell_rhs(j) +=
                    flux_Q * fe_face[momentum_extractor].value(j, q) * JxW[q];
                }
            }
        }
      else // Dirichlet boundary
        {
          // Boundary data from exact solution
          std::vector<Vector<double>> bc(n_q, Vector<double>(2));
          exact_solution.vector_value_list(fe_face.get_quadrature_points(), bc);

          for (unsigned int q = 0; q < n_q; ++q)
            {
              const double b_dot_n =
                compute_tangent_normal_product<dim, spacedim>(cell, normals[q]);

              // States
              const double AL_old = A_L[q], QL_old = Q_L[q];
              const double AR_old = bc[q](0),
                           QR_old = bc[q](1); // Prescribed values

              for (unsigned int j = 0; j < fe_face.get_fe().dofs_per_cell; ++j)
                {
                  // New time level basis values
                  const double AL_new = fe_face[area_extractor].value(j, q);
                  const double AR_new = bc[q](0); // Prescribed boundary value

                  // Compute fluxes
                  const double flux_A =
                    compute_area_rusanov_flux<dim, spacedim>(AL_new,
                                                             AR_new,
                                                             AL_old,
                                                             AR_old,
                                                             reference_area,
                                                             elastic_modulus,
                                                             rho,
                                                             b_dot_n);

                  const double flux_Q =
                    compute_momentum_rusanov_flux<dim, spacedim>(
                      QL_old, QR_old, AL_old, AR_old, rho, b_dot_n);

                  for (unsigned int i = 0; i < fe_face.get_fe().dofs_per_cell;
                       ++i)
                    {
                      // Matrix contributions (implicit terms)
                      // 1) Area equation flux
                      copy.cell_matrix(i, j) +=
                        flux_A * fe_face[area_extractor].value(i, q) * JxW[q];

                      // 2) -<A^{n+1}P(A^n)/ρ, φ_Q>
                      const double P_boundary_old =
                        compute_pressure_value<dim, spacedim>(
                          AR_old,
                          reference_area,
                          elastic_modulus,
                          reference_pressure);
                      copy.cell_matrix(i, j) -=
                        (AR_new * P_boundary_old / rho) *
                        fe_face[momentum_extractor].value(i, q) * JxW[q];
                    }

                  // RHS contributions (explicit terms)
                  for (unsigned int i = 0; i < fe_face.get_fe().dofs_per_cell;
                       ++i)
                    {
                      // 1) -<Q^{n+1}, φ_A>
                      copy.cell_rhs(i) -=
                        QR_old * fe_face[area_extractor].value(i, q) * JxW[q];

                      // 2) -<Q^n*Q^n/(A^n*ρ), φ_Q> (from momentum flux)
                      copy.cell_rhs(i) +=
                        flux_Q * fe_face[momentum_extractor].value(i, q) *
                        JxW[q];
                    }
                }
            }
        }
    };

    // Copier lambda
    const AffineConstraints<double> constraints;
    auto                            copier = [&](const BloodFlowCopyData &c) {
      constraints.distribute_local_to_global(c.cell_matrix,
                                             c.cell_rhs,
                                             c.local_dof_indices,
                                             system_matrix,
                                             right_hand_side);
      for (auto &cdf : c.face_data)
        {
          constraints.distribute_local_to_global(cdf.cell_matrix,
                                                 cdf.joint_dof_indices,
                                                 system_matrix);
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

  template <int dim, int spacedim>
  void
  BloodFlowSystem<dim, spacedim>::solve()
  {
    if (use_direct_solver)
      {
        SparseDirectUMFPACK system_matrix_inverse;
        system_matrix_inverse.initialize(system_matrix);
        system_matrix_inverse.vmult(solution, right_hand_side);
      }
    else
      {
        SolverControl               solver_control(1000, 1e-14);
        SolverGMRES<Vector<double>> solver(solver_control);
        PreconditionSSOR<>          preconditioner;
        const double                omega = 1.4;
        preconditioner.initialize(system_matrix, omega);
        solver.solve(system_matrix, solution, right_hand_side, preconditioner);
        std::cout << "  Solver converged in " << solver_control.last_step()
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
    solution_names[1] = "momentum";

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
            const unsigned int component =
              fe->system_to_component_index(i).first;
            if (component == 0) // Area component
              {
                const double area = solution[dof_indices[i]];
                pressure[dof_indices[i]] =
                  compute_pressure_value<dim, spacedim>(area,
                                                        reference_area,
                                                        elastic_modulus,
                                                        reference_pressure);
              }
          }
      }
  }

  template <int dim, int spacedim>
  void
  BloodFlowSystem<dim, spacedim>::compute_errors(unsigned int k)
  {
    const ComponentSelectFunction<spacedim> area_mask(0, 1.0, 2);
    const ComponentSelectFunction<spacedim> momentum_mask(1, 1.0, 2);

    Vector<float> difference_per_cell(triangulation.n_active_cells());

    // Create exact solution at current time
    ExactSolutionBloodFlow<spacedim> exact_solution;
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

    // momentum L2 error
    VectorTools::integrate_difference(dof_handler,
                                      solution,
                                      exact_solution,
                                      difference_per_cell,
                                      QGauss<dim>(fe_degree + 3),
                                      VectorTools::L2_norm,
                                      &momentum_mask);
    const double momentum_L2_error =
      VectorTools::compute_global_error(triangulation,
                                        difference_per_cell,
                                        VectorTools::L2_norm);

    // momentum H1 error
    VectorTools::integrate_difference(dof_handler,
                                      solution,
                                      exact_solution,
                                      difference_per_cell,
                                      QGauss<dim>(fe_degree + 3),
                                      VectorTools::H1_seminorm,
                                      &momentum_mask);
    const double momentum_H1_error =
      VectorTools::compute_global_error(triangulation,
                                        difference_per_cell,
                                        VectorTools::H1_seminorm);

    // Variables to store previous errors for convergence rate calculation
    static double last_Area_L2_error     = 0;
    static double last_Area_H1_error     = 0;
    static double last_momentum_L2_error = 0;
    static double last_momentum_H1_error = 0;

    // Output results with convergence rates
    std::cout << std::scientific << std::setprecision(3);
    std::cout << "=== Error Analysis at Time t = " << time
              << " (Refinement Level " << k + 1 << ") ===" << std::endl;

    std::cout << " Area L2 error:      " << std::setw(12) << Area_L2_error
              << "   Conv_rate: " << std::setw(6)
              << (k == 0 ? 0.0 :
                           std::log(last_Area_L2_error / Area_L2_error) /
                             std::log(2.0))
              << std::endl;

    std::cout << " Area H1 error:      " << std::setw(12) << Area_H1_error
              << "   Conv_rate: " << std::setw(6)
              << (k == 0 ? 0.0 :
                           std::log(last_Area_H1_error / Area_H1_error) /
                             std::log(2.0))
              << std::endl;

    std::cout << " momentum L2 error:  " << std::setw(12) << momentum_L2_error
              << "   Conv_rate: " << std::setw(6)
              << (k == 0 ?
                    0.0 :
                    std::log(last_momentum_L2_error / momentum_L2_error) /
                      std::log(2.0))
              << std::endl;

    std::cout << " momentum H1 error:  " << std::setw(12) << momentum_H1_error
              << "   Conv_rate: " << std::setw(6)
              << (k == 0 ?
                    0.0 :
                    std::log(last_momentum_H1_error / momentum_H1_error) /
                      std::log(2.0))
              << std::endl;

    std::cout << " DoFs: " << dof_handler.n_dofs() << "   h ≈ "
              << 1.0 / triangulation.n_active_cells() << std::endl;
    std::cout << std::string(70, '=') << std::endl;

    // Update previous error values
    last_Area_L2_error     = Area_L2_error;
    last_Area_H1_error     = Area_H1_error;
    last_momentum_L2_error = momentum_L2_error;
    last_momentum_H1_error = momentum_H1_error;
  }


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
            GridGenerator::hyper_cube(triangulation);
            triangulation.refine_global(n_global_refinements);
          }
        else
          {
            triangulation.refine_global(1);
          }

        setup_system();

        ExactSolutionBloodFlow<spacedim> exact_solution;
        exact_solution.set_time(0.0);

        // Project initial conditions
        AffineConstraints<double> constraints;
        constraints.close();
        VectorTools::project(dof_handler,
                             constraints,
                             QGauss<dim>(fe_degree + 1),
                             exact_solution,
                             solution);

        solution_old = solution;

        // Run time stepping to final_time
        time = 0.0;
        assemble_mass_matrix();

        n_time_steps =
          static_cast<unsigned int>(std::round(final_time / time_step));
        system_matrix_time.reinit(sparsity_pattern);
        tmp_vector.reinit(dof_handler.n_dofs());

        for (unsigned int step = 1; step <= n_time_steps; ++step)
          {
            time += time_step;
            exact_solution.set_time(time);

            std::cout << "Step " << step << "  t=" << time << std::endl;

            // Assemble system matrix
            assemble_system();

            // Form time-stepping system matrix: M/dt + A
            system_matrix_time.copy_from(system_matrix);
            system_matrix_time.add(1.0 / time_step, mass_matrix);

            // Form right-hand side: M/dt * u_old + right_hand
            mass_matrix.vmult(tmp_vector, solution_old);
            tmp_vector *= (1.0 / time_step);
            tmp_vector += right_hand_side;

            // Solve time step
            if (use_direct_solver)
              {
                SparseDirectUMFPACK direct;
                direct.initialize(system_matrix_time);
                direct.vmult(solution, tmp_vector);
              }
            else
              {
                SolverControl               solver_control(1000, 1e-14);
                SolverGMRES<Vector<double>> gmres(solver_control);
                PreconditionSSOR<>          preconditioner;
                const double                omega =
                  1.4; // SSOR relaxation parameter (typical range: 1.0-2.0)
                preconditioner.initialize(system_matrix_time, omega);
                gmres.solve(system_matrix_time,
                            solution,
                            tmp_vector,
                            preconditioner);

                std::cout << "  GMRES converged in "
                          << solver_control.last_step() << " iterations."
                          << std::endl;
              }

            std::cout << "norm of vector " << solution.l2_norm() << std::endl;

            solution_old = solution;
            compute_pressure();
            output_results(step);
          }

        // Compute errors at final time
        compute_errors(cycle);
      }
  }

  // Explicit instantiation
  template class BloodFlowSystem<1, 3>;

} // namespace dealii