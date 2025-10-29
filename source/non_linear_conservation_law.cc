#include "../include/non_linear_conservation_law.h"

#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/parsed_convergence_table.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/utilities.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

namespace
{
  auto set_default_expression = [](auto              &function,
                                   const std::string &expression) {
    function.declare_parameters_call_back.connect(
      [&]() { function.prm.set("Function expression", expression); });
  };
} // namespace

namespace dealii
{
  // ---------------------
  // Scratch/Copy data
  // ---------------------

  template <int dim, int spacedim>
  ConservationLawScratchData<dim, spacedim>::ConservationLawScratchData(
    const FiniteElement<dim, spacedim> &fe,
    const Quadrature<dim>              &quadrature,
    const Quadrature<dim - 1>          &quadrature_face)
    : fe_values(fe,
                quadrature,
                update_values | update_gradients | update_JxW_values |
                  update_quadrature_points)
    , fe_interface_values(fe,
                          quadrature_face,
                          update_values | update_gradients | update_JxW_values |
                            update_quadrature_points | update_normal_vectors)
  {}

  template <int dim, int spacedim>
  ConservationLawScratchData<dim, spacedim>::ConservationLawScratchData(
    const ConservationLawScratchData<dim, spacedim> &scratch_data)
    : fe_values(scratch_data.fe_values.get_fe(),
                scratch_data.fe_values.get_quadrature(),
                scratch_data.fe_values.get_update_flags())
    , fe_interface_values(scratch_data.fe_interface_values.get_fe(),
                          scratch_data.fe_interface_values.get_quadrature(),
                          scratch_data.fe_interface_values.get_update_flags())
  {}

  template <class Iterator>
  void
  ConservationLawCopyData::reinit(const Iterator    &cell,
                                  const unsigned int dofs_per_cell)
  {
    local_dof_indices.resize(dofs_per_cell);
    cell->get_dof_indices(local_dof_indices);
    cell_matrix.reinit(dofs_per_cell, dofs_per_cell);
    cell_rhs.reinit(dofs_per_cell);
    face_data.clear();
  }


  // ---------------------
  // Utility: member implementations for flux computations
  // ---------------------

  template <int dim, int spacedim>
  Tensor<1, spacedim>
  NonLinearConservationLaw<dim, spacedim>::compute_directional_vector(
    const typename DoFHandler<dim, spacedim>::active_cell_iterator &cell) const
  {
    // Unit tangent along the embedded 1D edge
    return (cell->vertex(1) - cell->vertex(0)) /
           cell->vertex(1).distance(cell->vertex(0));
  }

  template <int dim, int spacedim>
  Tensor<1, spacedim>
  NonLinearConservationLaw<dim, spacedim>::compute_flux(
    const Tensor<1, spacedim> &b,
    const double              &implicit_u,
    const double              &explicit_u) const
  {
    // Physical one-sided fluxes
    return 0.5 * explicit_u * implicit_u * b;
  }

  template <int dim, int spacedim>
  Tensor<1, spacedim>
  NonLinearConservationLaw<dim, spacedim>::compute_flux_diff(
    const Tensor<1, spacedim> &b,
    const double              &explicit_u) const
  {
    // Physical one-sided fluxes
    return 0.5 * explicit_u * b;
  }


  // ---------------------
  // Main class
  // ---------------------

  template <int dim, int spacedim>
  NonLinearConservationLaw<dim, spacedim>::NonLinearConservationLaw()
    : ParameterAcceptor("Conservation law/")
    , triangulation()
    , dof_handler(triangulation)
    , fe(nullptr)
    , initial_condition("Initial condition")
    , rhs_function("RHS function")
    , exact_solution("Exact solution")
  {
    add_parameter("Finite element degree", fe_degree);
    add_parameter("Output filename", output_filename);
    add_parameter("Output directory", output_directory);
    add_parameter("Use direct solver", use_direct_solver);
    add_parameter("Number of refinement cycles", n_refinement_cycles);
    add_parameter("Number of global refinement", n_global_refinements);
    add_parameter("Time step", time_step);
    add_parameter("Final time", final_time);
    add_parameter("Theta (penalty parameter)", theta);
    add_parameter("Omega (relaxation parameter)", omega);
    add_parameter("Picard iterations", max_picard_iterations);
    add_parameter("Picard tolerance", picard_tolerance);

    enter_subsection("Error");
    enter_my_subsection(this->prm);
    convergence_table.add_parameters(this->prm);
    leave_my_subsection(this->prm);
    leave_subsection();

    set_default_expression(initial_condition, "sin(pi*x)");
    set_default_expression(exact_solution, "sin(pi*x)*exp(-t)");
    set_default_expression(
      rhs_function, "-exp(-t)*sin(pi*x) + exp(-2*t)*pi*sin(pi*x)*cos(pi*x)");
  }

  template <int dim, int spacedim>
  void
  NonLinearConservationLaw<dim, spacedim>::initialize_params(
    const std::string &filename)
  {
    ParameterAcceptor::initialize(
      filename,
      "last_used_parameters_non_linear_conservation_law.prm",
      ParameterHandler::Short);
  }

  template <int dim, int spacedim>
  void
  NonLinearConservationLaw<dim, spacedim>::setup_system()
  {
    if (!fe)
      fe = std::make_unique<FE_DGQ<dim, spacedim>>(fe_degree);

    dof_handler.distribute_dofs(*fe);

    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_flux_sparsity_pattern(dof_handler, dsp);
    sparsity_pattern.copy_from(dsp);

    system_matrix.reinit(sparsity_pattern);
    mass_matrix.reinit(sparsity_pattern);
    system_matrix_time.reinit(sparsity_pattern);

    solution.reinit(dof_handler.n_dofs());
    solution_old.reinit(dof_handler.n_dofs());
    right_hand_side.reinit(dof_handler.n_dofs());
    tmp_vector.reinit(dof_handler.n_dofs());
  }

  template <int dim, int spacedim>
  void
  NonLinearConservationLaw<dim, spacedim>::assemble_mass_matrix()
  {
    mass_matrix = 0;
    Vector<double> dummy_rhs(dof_handler.n_dofs());

    // Define extractor for scalar field u
    const FEValuesExtractors::Scalar u_extractor(0);

    auto cell_worker = [&](const auto                                &cell,
                           ConservationLawScratchData<dim, spacedim> &scratch,
                           ConservationLawCopyData                   &copy) {
      const unsigned int ndofs = scratch.fe_values.get_fe().n_dofs_per_cell();
      copy.reinit(cell, ndofs);
      scratch.fe_values.reinit(cell);

      const auto &fe_v = scratch.fe_values;
      const auto &JxW  = fe_v.get_JxW_values();

      for (unsigned int q = 0; q < fe_v.n_quadrature_points; ++q)
        {
          for (unsigned int i = 0; i < ndofs; ++i)
            {
              for (unsigned int j = 0; j < ndofs; ++j)
                {
                  copy.cell_matrix(i, j) += fe_v[u_extractor].value(i, q) *
                                            fe_v[u_extractor].value(j, q) *
                                            JxW[q];
                }
            }
        }
    };

    const QGauss<dim>                         q(fe->tensor_degree() + 1);
    const QGauss<dim - 1>                     qf(fe->tensor_degree() + 1);
    ConservationLawScratchData<dim, spacedim> scratch(*fe, q, qf);
    ConservationLawCopyData                   copy;

    const AffineConstraints<double> constraints;
    auto copier = [&](const ConservationLawCopyData &c) {
      constraints.distribute_local_to_global(
        c.cell_matrix, c.cell_rhs, c.local_dof_indices, mass_matrix, dummy_rhs);
    };

    MeshWorker::mesh_loop(dof_handler.begin_active(),
                          dof_handler.end(),
                          cell_worker,
                          copier,
                          scratch,
                          copy,
                          MeshWorker::assemble_own_cells);
  }
  template <int dim, int spacedim>
  void
  NonLinearConservationLaw<dim, spacedim>::assemble_system()
  {
    using Iterator = typename DoFHandler<dim, spacedim>::active_cell_iterator;

    system_matrix   = 0;
    right_hand_side = 0;

    const FEValuesExtractors::Scalar u_extractor(0);

    // Cell worker: implicit volume + source
    auto cell_worker = [&](const Iterator                            &cell,
                           ConservationLawScratchData<dim, spacedim> &scratch,
                           ConservationLawCopyData                   &copy) {
      const unsigned int ndofs = scratch.fe_values.get_fe().n_dofs_per_cell();
      copy.reinit(cell, ndofs);
      scratch.fe_values.reinit(cell);

      const auto &fe_v     = scratch.fe_values;
      const auto &JxW      = fe_v.get_JxW_values();
      const auto &q_points = fe_v.get_quadrature_points();

      // Evaluate u^n at quadrature points
      std::vector<double> explicit_u(fe_v.n_quadrature_points);
      fe_v[u_extractor].get_function_values(solution, explicit_u);

      // Unit tangent vector along the embedded 1D edge
      const auto b_vec = compute_directional_vector(cell);

      // Source term f(x,t^{n+1})
      rhs_function.set_time(time);

      for (unsigned int q = 0; q < fe_v.n_quadrature_points; ++q)
        {
          const double rhs_value = rhs_function.value(q_points[q]);
          for (unsigned int i = 0; i < ndofs; ++i)
            {
              for (unsigned int j = 0; j < ndofs; ++j)
                {
                  const auto implicit_u_q = fe_v[u_extractor].value(j, q);
                  const auto flux =
                    compute_flux(b_vec, implicit_u_q, explicit_u[q]);

                  copy.cell_matrix(i, j) -=
                    flux * fe_v[u_extractor].gradient(i, q) * JxW[q];
                }
              // Source contribution
              copy.cell_rhs(i) +=
                rhs_value * fe_v[u_extractor].value(i, q) * JxW[q];
            }
        }
    };

    // Face worker: interior fluxes on RHS
    auto face_worker = [&](const Iterator                            &cell,
                           unsigned int                               f,
                           unsigned int                               sf,
                           const Iterator                            &ncell,
                           unsigned int                               nf,
                           unsigned int                               nsf,
                           ConservationLawScratchData<dim, spacedim> &scratch,
                           ConservationLawCopyData                   &copy) {
      FEInterfaceValues<dim, spacedim> &fe_iv = scratch.fe_interface_values;
      fe_iv.reinit(cell, f, sf, ncell, nf, nsf);

      const auto &JxW        = fe_iv.get_JxW_values();
      const auto &normals    = fe_iv.get_fe_face_values(0).get_normal_vectors();
      const unsigned int n_q = fe_iv.get_fe_face_values(0).n_quadrature_points;

      const auto b_vec = compute_directional_vector(cell);

      std::vector<double> explicit_uL(n_q), explicit_uR(n_q);
      fe_iv.get_fe_face_values(0)[u_extractor].get_function_values(solution,
                                                                   explicit_uL);
      fe_iv.get_fe_face_values(1)[u_extractor].get_function_values(solution,
                                                                   explicit_uR);
      copy.face_data.emplace_back();
      auto              &face = copy.face_data.back();
      const unsigned int nd   = fe_iv.n_current_interface_dofs();
      face.joint_dof_indices  = fe_iv.get_interface_dof_indices();
      face.cell_matrix.reinit(nd, nd);
      face.cell_rhs.reinit(nd);

      for (unsigned int q = 0; q < n_q; ++q)
        {
          const double an =
            compute_flux_diff(b_vec, explicit_uL[q]) * normals[q];

          for (unsigned int i = 0; i < nd; ++i)
            {
              for (unsigned int j = 0; j < nd; ++j)
                {
                  const auto implicit_uL = fe_iv[u_extractor].value(0, i, q);
                  const auto implicit_uR = fe_iv[u_extractor].value(1, j, q);

                  const auto fluxL =
                    compute_flux(b_vec, implicit_uL, explicit_uL[q]);
                  const auto fluxR =
                    compute_flux(b_vec, implicit_uR, explicit_uR[q]);

                  auto F_hat =
                    0.5 * (fluxL + fluxR) * normals[q] +
                    theta * std::abs(an) * (implicit_uR - implicit_uL);

                  face.cell_matrix(i, j) +=
                    F_hat * fe_iv[u_extractor].jump_in_values(i, q) * JxW[q];
                }
            }
        }
    };


    auto boundary_worker = [&](
                             const Iterator                            &cell,
                             unsigned int                               face_no,
                             ConservationLawScratchData<dim, spacedim> &scratch,
                             ConservationLawCopyData                   &copy) {
      scratch.fe_interface_values.reinit(cell, face_no);
      const auto &fe_face = scratch.fe_interface_values.get_fe_face_values(0);

      const auto        &JxW     = fe_face.get_JxW_values();
      const auto        &normals = fe_face.get_normal_vectors();
      const unsigned int n_q     = fe_face.n_quadrature_points;

      std::vector<double> explicit_u(n_q);
      // fe_face[u_extractor].get_function_values(solution_old, u_in);
      fe_face[u_extractor].get_function_values(solution, explicit_u);


      const auto         &q_points = fe_face.get_quadrature_points();
      std::vector<double> g(n_q);
      std::vector<double> external_u(n_q);

      exact_solution.set_time(time);
      exact_solution.value_list(q_points, external_u);

      const auto b_vec = compute_directional_vector(cell);

      for (unsigned int q = 0; q < n_q; ++q)
        {
          const double an =
            compute_flux_diff(b_vec, explicit_u[q]) * normals[q];

          if (an > 0)
            {
              for (unsigned int i = 0; i < fe_face.get_fe().dofs_per_cell; ++i)
                for (unsigned int j = 0; j < fe_face.get_fe().dofs_per_cell;
                     ++j)
                  {
                    const auto implicit_u = fe_face[u_extractor].value(j, q);
                    const auto flux =
                      compute_flux(b_vec, implicit_u, explicit_u[q]);

                    const auto f_hat =
                      flux * normals[q] + theta * an * implicit_u;

                    copy.cell_matrix(i, j) +=
                      f_hat * fe_face[u_extractor].value(i, q) * JxW[q]; // dx
                  }
            }
          else
            {
              const auto flux =
                compute_flux(b_vec, external_u[q], external_u[q]);

              const auto f_hat =
                flux * normals[q] + theta * std::abs(an) * external_u[q];

              for (unsigned int i = 0; i < fe_face.get_fe().dofs_per_cell; ++i)
                {
                  copy.cell_rhs(i) +=
                    -f_hat * fe_face[u_extractor].value(i, q) * JxW[q]; // dx
                }
            }
        }
    };

    const QGauss<dim>     quadrature(2 * fe->tensor_degree() + 1);
    const QGauss<dim - 1> quadrature_face(2 * fe->tensor_degree() + 1);
    ConservationLawScratchData<dim, spacedim> scratch(*fe,
                                                      quadrature,
                                                      quadrature_face);
    ConservationLawCopyData                   copy;

    const AffineConstraints<double> constraints;
    auto copier = [&](const ConservationLawCopyData &c) {
      // Volume
      constraints.distribute_local_to_global(c.cell_matrix,
                                             c.cell_rhs,
                                             c.local_dof_indices,
                                             system_matrix,
                                             right_hand_side);
      // Faces - matrix contributions (if any)
      for (auto &fd : c.face_data)
        constraints.distribute_local_to_global(fd.cell_matrix,
                                               fd.joint_dof_indices,
                                               system_matrix);
      // Face RHS contributions
      for (auto &fd : c.face_data)
        constraints.distribute_local_to_global(fd.cell_rhs,
                                               fd.joint_dof_indices,
                                               right_hand_side);
    };

    MeshWorker::mesh_loop(dof_handler.begin_active(),
                          dof_handler.end(),
                          cell_worker,
                          copier,
                          scratch,
                          copy,
                          MeshWorker::assemble_own_cells |
                            MeshWorker::assemble_boundary_faces |
                            MeshWorker::assemble_own_interior_faces_once,
                          boundary_worker,
                          face_worker);
  }

  template <int dim, int spacedim>
  void
  NonLinearConservationLaw<dim, spacedim>::solve()
  {
    if (use_direct_solver)
      {
        SparseDirectUMFPACK inverse;
        inverse.initialize(system_matrix_time);
        inverse.vmult(solution, tmp_vector);
      }
    else
      {
        SolverControl               ctrl(1000, 1e-14);
        SolverGMRES<Vector<double>> solver(ctrl);
        PreconditionSSOR<>          prec;
        prec.initialize(system_matrix_time, 1.4);
        solver.solve(system_matrix_time, solution, tmp_vector, prec);
        std::cout << "  GMRES iterations: " << ctrl.last_step() << std::endl;
      }
  }

  template <int dim, int spacedim>
  void
  NonLinearConservationLaw<dim, spacedim>::output_results(
    const unsigned int cycle,
    const unsigned int timestep_number) const
  {
    const unsigned int max_steps =
      static_cast<unsigned int>(std::round(final_time / time_step));
    const unsigned int n_pad = Utilities::needed_digits(max_steps);

    const std::string filename =
      output_filename + "-" + std::to_string(cycle) + "-" +
      Utilities::int_to_string(timestep_number, n_pad) + ".vtu";
    std::ofstream output(output_directory + "/" + filename);

    DataOut<dim, spacedim> data_out;
    data_out.attach_dof_handler(dof_handler);

    std::vector<std::string> names(1, "u");
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      interp(1, DataComponentInterpretation::component_is_scalar);

    data_out.add_data_vector(solution,
                             names,
                             DataOut<dim, spacedim>::type_dof_data,
                             interp);
    data_out.build_patches();
    data_out.write_vtu(output);
    static std::vector<std::pair<double, std::string>> pvd_output_records;
    if (timestep_number == 0)
      pvd_output_records.clear();
    pvd_output_records.push_back(std::make_pair(time, filename));
    std::ofstream pvd_output(output_directory + "/" + output_filename + "-" +
                             std::to_string(cycle) + ".pvd");
    DataOutBase::write_pvd_record(pvd_output, pvd_output_records);
  }

  template <int dim, int spacedim>
  void
  NonLinearConservationLaw<dim, spacedim>::run_convergence_study()
  {
    std::cout << "=== CONVERGENCE STUDY for DG" << fe_degree
              << " (Burgers) ===\n";

    for (unsigned int cycle = 0; cycle < n_refinement_cycles; ++cycle)
      {
        std::cout << "\n--- Refinement Cycle " << cycle << " ---\n";
        if (cycle == 0)
          {
            GridGenerator::hyper_cube(triangulation);
            triangulation.refine_global(n_global_refinements);
          }
        else
          triangulation.refine_global(1);

        setup_system();

        // Project initial condition u(x,0) = sin(pi x)
        AffineConstraints<double> cstr;
        cstr.close();
        initial_condition.set_time(0.0);
        VectorTools::project(dof_handler,
                             cstr,
                             QGauss<dim>(fe_degree + 2),
                             initial_condition,
                             solution);
        solution_old = solution;

        // Time loop
        time = 0.0;
        assemble_mass_matrix();

        n_time_steps =
          static_cast<unsigned int>(std::round(final_time / time_step));
        for (unsigned int step = 1; step <= n_time_steps; ++step)
          {
            time += time_step;
            std::cout << "Step " << step << "  t=" << time << std::endl;

            // Initialize Picard iteration
            solution = solution_old; // u^(0) := u^n
            Vector<double> solution_picard_old(solution.size());
            unsigned int   picard_iter = 0;
            bool           converged   = false;

            // Picard loop
            do
              {
                solution_picard_old = solution; // save previous iterate

                // Assemble using current Picard guess
                assemble_system(); // uses solution as u_old

                // Build time-step system: (M/dt + A) u = M/dt u^n + RHS
                system_matrix_time.copy_from(mass_matrix);
                system_matrix_time *= (1.0 / time_step);
                system_matrix_time.add(1.0, system_matrix);

                mass_matrix.vmult(tmp_vector, solution_old);
                tmp_vector *= (1.0 / time_step);
                tmp_vector += right_hand_side;

                // Solve linear system for new iterate
                solve(); // result in 'solution'

                // solution = omega * solution_new + (1 - omega) * old
                solution.sadd(omega, 1.0 - omega, solution_picard_old);

                // Compute Picard residual
                Vector<double> diff = solution;
                diff.add(-1.0, solution_picard_old);
                const double picard_residual = diff.l2_norm();

                std::cout << "  Picard iter " << picard_iter
                          << "   residual = " << picard_residual << std::endl;

                ++picard_iter;

                if (picard_residual < picard_tolerance ||
                    picard_iter >= max_picard_iterations)
                  converged = true;
              }
            while (!converged && picard_iter < max_picard_iterations);
            std::cout << "  Picard converged in " << picard_iter
                      << " iterations.\n";


            // Advance in time
            solution_old = solution;
            output_results(cycle, step);
          }
        {
          Vector<double> tmp(dof_handler.n_dofs());
          const double   l2 =
            mass_matrix.matrix_scalar_product(solution, solution);

          // For manufactured solution u = e^{-t} sin(πx), L2 norm is
          // ||u||_{L2} = sqrt(int_{0}^}{1} sin^2 (pi x) e^{-2t})= sqrt(1/2) *
          // e^{-t}
          const double analytic_L2 = std::sqrt(0.5) * std::exp(-time);

          std::cout << std::scientific << std::setprecision(6);
          std::cout << "  ||u||_{L2} (discrete) = " << l2
                    << "   analytic = " << analytic_L2 << "   rel_err = "
                    << std::fabs(l2 - analytic_L2) / analytic_L2 << std::endl;
        }

        convergence_table.error_from_exact(dof_handler,
                                           solution,
                                           exact_solution);
      }
    convergence_table.output_table(std::cout);
  }
  // Explicit instantiations: 1D embedded in 3D
  template class NonLinearConservationLaw<1, 3>;

} // namespace dealii
