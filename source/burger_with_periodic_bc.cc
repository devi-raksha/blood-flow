#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/utilities.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/grid/grid_tools.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>

#include "../include/burger_equation.h"

namespace dealii
{
  // ---------------------
  // Exact and RHS classes
  // ---------------------

  template <int spacedim>
  double
  ExactSolutionBurger<spacedim>::value(const Point<spacedim> &p,
                                       const unsigned int) const
  {
    const double x = p[0]; // assumes parametric x on embedded 1D
    const double t = this->get_time();
    return std::exp(-t) * std::sin(numbers::PI * x);
  }

  template <int spacedim>
  Tensor<1, spacedim>
  ExactSolutionBurger<spacedim>::gradient(const Point<spacedim> &p,
                                          const unsigned int) const
  {
    // Only tangential derivative matters; return component-wise gradient
    Tensor<1, spacedim> g;
    const double        x = p[0];
    const double        t = this->get_time();
    // ∂u/∂x (in x-direction; for embedded case, will be projected along
    // tangent)
    const double du_dx = std::exp(-t) * numbers::PI * std::cos(numbers::PI * x);
    g[0]               = du_dx;
    for (unsigned int d = 1; d < spacedim; ++d)
      g[d] = 0.0;
    return g;
  }

  template <int spacedim>
  double
  RHS_Burger<spacedim>::value(const Point<spacedim> &p,
                              const unsigned int) const
  {
    // Return f(x,t) - the source function
    // For manufactured solution u = e^{-t} sin(πx), compute f such that u_t +
    // ∇·(u²/2) = f
    const double x = p[0];
    const double t = this->get_time();

    // u_exact = e^{-t} sin(πx)
    // u_t = -e^{-t} sin(πx)
    const double u_t = -std::exp(-t) * std::sin(numbers::PI * x);
    // ∇·(u²/2) = d/dx(u²/2) = u * du/dx = e^{-t} sin(πx) * e^{-t} π cos(πx) =
    // e^{-2t} π sin(πx) cos(πx)
    const double div_flux = std::exp(-2.0 * t) * numbers::PI *
                            std::sin(numbers::PI * x) *
                            std::cos(numbers::PI * x);

    // f = u_t + ∇·(u²/2)
    return u_t + div_flux;
  }

  // ---------------------
  // Scratch/Copy data
  // ---------------------

  template <int dim, int spacedim>
  BurgerScratchData<dim, spacedim>::BurgerScratchData(
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
  BurgerScratchData<dim, spacedim>::BurgerScratchData(
    const BurgerScratchData<dim, spacedim> &scratch_data)
    : fe_values(scratch_data.fe_values.get_fe(),
                scratch_data.fe_values.get_quadrature(),
                scratch_data.fe_values.get_update_flags())
    , fe_interface_values(scratch_data.fe_interface_values.get_fe(),
                          scratch_data.fe_interface_values.get_quadrature(),
                          scratch_data.fe_interface_values.get_update_flags())
  {}

  template <class Iterator>
  void
  BurgerCopyData::reinit(const Iterator &cell, const unsigned int dofs_per_cell)
  {
    local_dof_indices.resize(dofs_per_cell);
    cell->get_dof_indices(local_dof_indices);
    cell_matrix.reinit(dofs_per_cell, dofs_per_cell);
    cell_rhs.reinit(dofs_per_cell);
    face_data.clear();
  }


  // ---------------------
  // Utility functions
  // ---------------------

  template <int dim, int spacedim>
  double
  compute_tangent_normal_product_burger(
    const typename DoFHandler<dim, spacedim>::active_cell_iterator &cell,
    const Tensor<1, spacedim>                                      &normal)
  {
    // Unit tangent along the embedded 1D edge
    const Tensor<1, spacedim> b = (cell->vertex(1) - cell->vertex(0)) /
                                  cell->vertex(1).distance(cell->vertex(0));
    return b * normal;
  }


  // ---------------------
  // Main class
  // ---------------------

  template <int dim, int spacedim>
  BurgerEquation<dim, spacedim>::BurgerEquation()
    : triangulation()
    , dof_handler(triangulation)
    , fe(nullptr)
    , rhs_function(std::make_unique<RHS_Burger<spacedim>>())
  {
    add_parameter("Finite element degree", fe_degree);
    add_parameter("Output filename", output_filename);
    add_parameter("Use direct solver", use_direct_solver);
    add_parameter("Number of refinement cycles", n_refinement_cycles);
    add_parameter("Number of global refinement", n_global_refinements);
    add_parameter("Time step", time_step);
    add_parameter("Final time", final_time);
    add_parameter("Theta (penalty parameter)", theta);
    add_parameter("Omega (relaxation parameter)", omega);
  }

  template <int dim, int spacedim>
  void
  BurgerEquation<dim, spacedim>::initialize_params(const std::string &filename)
  {
    ParameterAcceptor::initialize(filename,
                                  "last_used_parameters_burger.prm",
                                  ParameterHandler::Short);
  }

  template <int dim, int spacedim>
  void
  BurgerEquation<dim, spacedim>::setup_system()
  {
    if (!fe)
      {
        fe = std::make_unique<FE_DGQ<dim, spacedim>>(fe_degree);
        // initial condition parser
        std::string vars =
          spacedim == 1 ? "x" : (spacedim == 2 ? "x,y" : "x,y,z");
        std::map<std::string, double> consts;
        initial_condition.initialize(vars, initial_expression, consts);
      }

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
  BurgerEquation<dim, spacedim>::assemble_mass_matrix()
  {
    mass_matrix = 0;
    Vector<double> dummy_rhs(dof_handler.n_dofs());

    // Define extractor for scalar field u
    const FEValuesExtractors::Scalar u_extractor(0);

    auto cell_worker = [&](const auto                       &cell,
                           BurgerScratchData<dim, spacedim> &scratch,
                           BurgerCopyData                   &copy) {
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

    const QGauss<dim>                q(fe->tensor_degree() + 1);
    const QGauss<dim - 1>            qf(fe->tensor_degree() + 1);
    BurgerScratchData<dim, spacedim> scratch(*fe, q, qf);
    BurgerCopyData                   copy;

    const AffineConstraints<double> constraints;
    auto                            copier = [&](const BurgerCopyData &c) {
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
  BurgerEquation<dim, spacedim>::assemble_system()
  {
    using Iterator = typename DoFHandler<dim, spacedim>::active_cell_iterator;

    system_matrix   = 0;
    right_hand_side = 0;

    const FEValuesExtractors::Scalar u_extractor(0);

    // Cell worker: implicit volume + source
    auto cell_worker = [&](const Iterator                   &cell,
                           BurgerScratchData<dim, spacedim> &scratch,
                           BurgerCopyData                   &copy) {
      const unsigned int ndofs = scratch.fe_values.get_fe().n_dofs_per_cell();
      copy.reinit(cell, ndofs);
      scratch.fe_values.reinit(cell);

      const auto &fe_v     = scratch.fe_values;
      const auto &JxW      = fe_v.get_JxW_values();
      const auto &q_points = fe_v.get_quadrature_points();

      // Evaluate u^n at quadrature points
      std::vector<double> u_old(fe_v.n_quadrature_points);
      // fe_v[u_extractor].get_function_values(solution_old, u_old);
      fe_v[u_extractor].get_function_values(solution, u_old);

      std::vector<Tensor<1, spacedim>> grad_uold(fe_v.n_quadrature_points);
      // fe_v[u_extractor].get_function_gradients(solution_old, grad_uold);
      fe_v[u_extractor].get_function_gradients(solution, grad_uold);


      // Unit tangent vector along the embedded 1D edge
      const auto b_vec = (cell->vertex(1) - cell->vertex(0)) /
                         cell->vertex(1).distance(cell->vertex(0));

      // Source term f(x,t^{n+1})
      rhs_function->set_time(time);

      for (unsigned int q = 0; q < fe_v.n_quadrature_points; ++q)
        {
          const double rhs_value = rhs_function->value(q_points[q]);
          const double u_old_q   = u_old[q];

          for (unsigned int i = 0; i < ndofs; ++i)
            {
              for (unsigned int j = 0; j < ndofs; ++j)
                {
                  copy.cell_matrix(i, j) -=
                    0.5 * u_old_q * b_vec * fe_v[u_extractor].value(j, q) *
                    fe_v[u_extractor].gradient(i, q) * JxW[q];
                }
              // Source contribution
              copy.cell_rhs(i) +=
                rhs_value * fe_v[u_extractor].value(i, q) * JxW[q];
            }
        }
    };

    // Face worker: interior fluxes on RHS
    auto face_worker = [&](const Iterator                   &cell,
                           unsigned int                      f,
                           unsigned int                      sf,
                           const Iterator                   &ncell,
                           unsigned int                      nf,
                           unsigned int                      nsf,
                           BurgerScratchData<dim, spacedim> &scratch,
                           BurgerCopyData                   &copy) {
      FEInterfaceValues<dim, spacedim> &fe_iv = scratch.fe_interface_values;
      fe_iv.reinit(cell, f, sf, ncell, nf, nsf);

      const auto &JxW        = fe_iv.get_JxW_values();
      const auto &normals    = fe_iv.get_fe_face_values(0).get_normal_vectors();
      const unsigned int n_q = fe_iv.get_fe_face_values(0).n_quadrature_points;

      std::vector<double> uL_old(n_q), uR_old(n_q);
      // fe_iv.get_fe_face_values(0)[u_extractor].get_function_values(solution_old,
      //                                                              uL_old);
      // fe_iv.get_fe_face_values(1)[u_extractor].get_function_values(solution_old,
      //                                                              uR_old);
      fe_iv.get_fe_face_values(0)[u_extractor].get_function_values(solution,
                                                                   uL_old);
      fe_iv.get_fe_face_values(1)[u_extractor].get_function_values(solution,
                                                                   uR_old);
      copy.face_data.emplace_back();
      auto              &face = copy.face_data.back();
      const unsigned int nd   = fe_iv.n_current_interface_dofs();
      face.joint_dof_indices  = fe_iv.get_interface_dof_indices();
      face.cell_matrix.reinit(nd, nd);
      face.cell_rhs.reinit(nd);

      for (unsigned int q = 0; q < n_q; ++q)
        {
          const double bn =
            compute_tangent_normal_product_burger<dim, spacedim>(cell,
                                                                 normals[q]);
          // Semi-implicit numerical flux: F̂ = {{u^n * u^{n+1}}} * bn + penalty
          for (unsigned int j = 0; j < nd; ++j)
            {
              // const double ul = fe_iv[u_extractor].value(0, j, q);
              // const double ur = fe_iv[u_extractor].value(1, j, q);

              // // Average flux: {{u^n * u^{n+1}}}
              // const double F_avg =
              //   0.25 * bn * (uL_old[q] * ul + uR_old[q] * ur);

              // const double h = cell->diameter();

              // // const double alpha = theta * h / (2 * time_step);

              for (unsigned int i = 0; i < nd; ++i)
                {
                  face.cell_matrix(i, j) +=
                    (bn * (uL_old[q] + uR_old[q]) / 4 *
                       fe_iv[u_extractor].jump_in_values(i, q) *
                       fe_iv[u_extractor].average_of_values(j, q) +
                     theta * std::abs(bn * (uL_old[q] + uR_old[q]) / 4) *
                       fe_iv[u_extractor].jump_in_values(i, q) *
                       fe_iv[u_extractor].jump_in_values(j, q)) *
                    JxW[q];
                }
            }
        }
    };


    auto boundary_worker = [&](const Iterator                   &cell,
                               unsigned int                      face_no,
                               BurgerScratchData<dim, spacedim> &scratch,
                               BurgerCopyData                   &copy) {
      scratch.fe_interface_values.reinit(cell, face_no);
      const auto &fe_face = scratch.fe_interface_values.get_fe_face_values(0);

      const auto        &JxW     = fe_face.get_JxW_values();
      const auto        &normals = fe_face.get_normal_vectors();
      const unsigned int n_q     = fe_face.n_quadrature_points;

      std::vector<double> u_in(n_q);
      // fe_face[u_extractor].get_function_values(solution_old, u_in);
      fe_face[u_extractor].get_function_values(solution, u_in);


      std::vector<Point<spacedim>> q_points(n_q);
      for (unsigned int q = 0; q < n_q; ++q)
        {
          const double bn =
            compute_tangent_normal_product_burger<dim, spacedim>(cell,
                                                                 normals[q]);


          q_points[q] =
            fe_face.quadrature_point(q); // or get_quadrature_points()
          std::vector<double> g(n_q);
          // std::vector<double> g1(n_q);

          ExactSolutionBurger<spacedim> exact;
          exact.set_time(time);
          exact.value_list(q_points, g);

          // Boundary data at t^n and t^{n+1}
          exact.set_time(time - time_step); // t^n
          const double u_ext_old = exact.value(fe_face.quadrature_point(q));

          if (bn * (u_ext_old / 2) > 0)
            {
              for (unsigned int i = 0; i < fe_face.get_fe().dofs_per_cell; ++i)
                for (unsigned int j = 0; j < fe_face.get_fe().dofs_per_cell;
                     ++j)
                  copy.cell_matrix(i, j) +=
                    fe_face[u_extractor].value(i, q)   // \phi_i
                    * fe_face[u_extractor].value(j, q) // \phi_j
                    * bn * u_ext_old / 2               // \beta . n = bn u_old/2
                    * JxW[q];                          // dx
            }
          else
            for (unsigned int i = 0; i < fe_face.get_fe().dofs_per_cell; ++i)
              copy.cell_rhs(i) += -fe_face[u_extractor].value(i, q) // \phi_i
                                  * g[q]                            // g*/
                                  * bn * u_ext_old / 2 // g1[q]/2 // \beta . n
                                  * JxW[q];            // dx
        }
    };

    (void)boundary_worker;

    const QGauss<dim>     quadrature(2 * fe->tensor_degree() + 1);
    const QGauss<dim - 1> quadrature_face(2 * fe->tensor_degree() + 1);
    BurgerScratchData<dim, spacedim> scratch(*fe, quadrature, quadrature_face);
    BurgerCopyData                   copy;

    const AffineConstraints<double> constraints;
    auto                            copier = [&](const BurgerCopyData &c) {
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


    auto null_boundary = [](const auto &, const unsigned int, auto &, auto &) {
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
                          null_boundary,
                          face_worker);
  }

  template <int dim, int spacedim>
  void
  BurgerEquation<dim, spacedim>::solve()
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
  BurgerEquation<dim, spacedim>::output_results(const unsigned int cycle) const
  {
    const std::string filename =
      output_filename + "-" + std::to_string(cycle) + ".vtu";
    std::ofstream output(filename);

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
    pvd_output_records.push_back(std::make_pair(time, filename));
    std::ofstream pvd_output(output_filename + ".pvd");
    DataOutBase::write_pvd_record(pvd_output, pvd_output_records);
  }

  template <int dim, int spacedim>
  void
  BurgerEquation<dim, spacedim>::compute_errors(unsigned int k)
  {
    Vector<float> diff(triangulation.n_active_cells());

    ExactSolutionBurger<spacedim> exact;
    exact.set_time(time);

    // L2 error
    VectorTools::integrate_difference(dof_handler,
                                      solution,
                                      exact,
                                      diff,
                                      QGauss<dim>(fe_degree + 2),
                                      VectorTools::L2_norm);
    const double L2 = VectorTools::compute_global_error(triangulation,
                                                        diff,
                                                        VectorTools::L2_norm);

    // H1 seminorm
    VectorTools::integrate_difference(dof_handler,
                                      solution,
                                      exact,
                                      diff,
                                      QGauss<dim>(fe_degree + 2),
                                      VectorTools::H1_seminorm);
    const double H1 =
      VectorTools::compute_global_error(triangulation,
                                        diff,
                                        VectorTools::H1_seminorm);

    static double lastL2 = 0.0, lastH1 = 0.0;

    std::cout << std::scientific << std::setprecision(3);
    std::cout << "=== Errors at t=" << time << " (cycle " << (k + 1)
              << ") ===\n";
    std::cout << " L2:  " << L2 << "  rate: "
              << (k == 0 ? 0.0 : std::log(lastL2 / L2) / std::log(2.0)) << "\n";
    std::cout << " H1s: " << H1 << "  rate: "
              << (k == 0 ? 0.0 : std::log(lastH1 / H1) / std::log(2.0)) << "\n";
    std::cout << " DoFs: " << dof_handler.n_dofs() << "   h≈"
              << 1.0 / triangulation.n_active_cells() << "\n";
    std::cout << std::string(60, '=') << std::endl;

    lastL2 = L2;
    lastH1 = H1;
  }

  template <int dim, int spacedim>
  void
  BurgerEquation<dim, spacedim>::run_convergence_study()
  {
    std::cout << "=== CONVERGENCE STUDY for DG" << fe_degree
              << " (Burgers) ===\n";

    for (unsigned int cycle = 0; cycle < n_refinement_cycles; ++cycle)
      {
        std::cout << "\n--- Refinement Cycle " << cycle << " ---\n";
        if (cycle == 0)
          {
            GridGenerator::hyper_cube(triangulation, -1.0, 1.0);

            // Mark boundary faces for periodic pairing
            for (auto &face : triangulation.active_face_iterators())
              if (face->at_boundary())
                {
                  const Point<spacedim> &center = face->center();
                  if (std::abs(center[0] - (-1)) < 1e-12)
                    face->set_boundary_id(0); // left boundary
                  else if (std::abs(center[0] - 1.0) < 1e-12)
                    face->set_boundary_id(1); // right boundary
                }

            triangulation.refine_global(n_global_refinements);

            // Collect and apply periodic face pairs
            std::vector<GridTools::PeriodicFacePair<
              typename Triangulation<dim, spacedim>::cell_iterator>>
              periodicity_vector;

            GridTools::collect_periodic_faces(triangulation,
                                              0, // boundary_id left
                                              1, // boundary_id right
                                              0, // direction (x-axis)
                                              periodicity_vector);

            triangulation.add_periodicity(periodicity_vector);

            std::cout << "  Periodic boundary conditions applied.\n";
          }
        else
          triangulation.refine_global(1);

        setup_system();

        // Project initial condition u(x,0) = sin(pi x)
        AffineConstraints<double> cstr;
        cstr.close();
        ExactSolutionBurger<spacedim> init;
        init.set_time(0.0);
        VectorTools::project(
          dof_handler, cstr, QGauss<dim>(fe_degree + 2), init, solution);
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
            output_results(step);
          }
        {
          Vector<double> tmp(dof_handler.n_dofs());
          mass_matrix.vmult(tmp, solution);    // tmp = M * solution
          const double l2_sq = solution * tmp; // u^T M u
          const double l2    = std::sqrt(std::max(0.0, l2_sq));

          // For manufactured solution u = e^{-t} sin(πx), L2 norm is
          // ||u||_{L2} = sqrt(int_{0}^}{1} sin^2 (pi x) e^{-2t})= sqrt(1/2) *
          // e^{-t}
          const double analytic_L2 = std::sqrt(0.5) * std::exp(-time);

          std::cout << std::scientific << std::setprecision(6);
          std::cout << "  ||u||_{L2} (discrete) = " << l2
                    << "   analytic = " << analytic_L2 << "   rel_err = "
                    << std::fabs(l2 - analytic_L2) / analytic_L2 << std::endl;
        }

        compute_errors(cycle);
      }
  }
  // Explicit instantiations: 1D embedded in 3D
  template class BurgerEquation<1, 3>;
  template class ExactSolutionBurger<3>;
  template class RHS_Burger<3>;

} // namespace dealii
