/* -----------------------------------------------------------------------------
 *
 *  SPDX-License-Identifier: LGPL-2.1-or-later
 *  Copyright:
 *  2024-2025 by the deal.II authors
 *
 *  This file is part of the blood-flow example built on the deal.II library.
 *  It provides the top-level executable that steers the templated
 *  BloodFlowSystem<1,3> class.  The structure mirrors main_embedded.cc so that
 *  build rules, CMake targets, and user habits remain consistent across
 *  multiple applications in the same repository.
 *
 *  --------------------------------------------------------------------------
 */

#include <deal.II/base/logstream.h> // deallog control
#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_handler.h>

#include <deal.II/lac/petsc_solver.h>

#include <metric_flow_x/blood_flow_system.h> // header exposing BloodFlowSystem

#include <cstdlib>
#include <iostream>
#include <stdexcept> // for std::invalid_argument

using namespace dealii;
using namespace MetricFlowX;

namespace MetricFlowX
{
  /** The standalone executable's IDA/time-stepping policy.
   *
   * BloodFlowSystem remains the mathematical Problem: it owns the mesh,
   * physics, residual, Jacobians, state, and observables.  This runner owns
   * IDA, linear-solver policy, time stepping, and output orchestration.
   */
  template <int dim, int spacedim>
  class BloodFlowIDARunner
  {
  public:
    using Problem    = BloodFlowSystem<dim, spacedim>;
    using VectorType = MetricFlowX::VectorType;

    void
    run(Problem &problem)
    {
      problem.pcout << "=== Blood Flow HDG, polynomial degree p = "
                    << problem.fe_degree << ", running on "
                    << problem.n_mpi_processes << " MPI rank(s) ===\n";

      for (unsigned int cycle = 0; cycle < problem.n_refinement_cycles; ++cycle)
        {
          if (problem.verbosity > 0)
            problem.pcout << "\n--- Refinement cycle " << cycle << " ---\n";

          if (cycle == 0)
            problem.setup();
          else
            problem.triangulation_.refine_global(1);

          problem.open_csv_files();
          if (cycle > 0)
            {
              problem.setup_system();
              problem.initialize_terminal_capacitors();
              problem.build_per_cell_mass_inv();
            }
          problem.initialize_state(problem.solution,
                                   problem.ida_parameters.initial_time);
          problem.initialize_state_derivative(
            problem.solution_dot, problem.ida_parameters.initial_time);
          problem.time = problem.ida_parameters.initial_time;

          auto check_jacobian_fd = [&problem](const VectorType &y0) {
            const double t     = problem.time;
            const double eps   = 1e-7;
            const double alpha = 0.0;

            VectorType ydot0(problem.locally_owned_dofs_,
                             problem.mpi_communicator_);
            ydot0 = 0.0;

            problem.assemble_jacobian(t, y0, ydot0, alpha);

            VectorType v(problem.locally_owned_dofs_,
                         problem.mpi_communicator_);
            std::srand(problem.this_mpi_process + 1);
            for (const auto i : problem.locally_owned_dofs_)
              v(i) = 2.0 * std::rand() / double(RAND_MAX) - 1.0;
            v.compress(VectorOperation::insert);

            VectorType yp(y0), ym(y0);
            yp.add(eps, v);
            ym.add(-eps, v);

            VectorType Fp(problem.locally_owned_dofs_,
                          problem.mpi_communicator_);
            VectorType Fm(problem.locally_owned_dofs_,
                          problem.mpi_communicator_);
            problem.assemble_residual(t, yp, ydot0, Fp);
            problem.assemble_residual(t, ym, ydot0, Fm);

            VectorType fd(Fp);
            fd -= Fm;
            fd /= (2.0 * eps);

            VectorType Jv(problem.locally_owned_dofs_,
                          problem.mpi_communicator_);
            problem.jacobian_matrix.vmult(Jv, v);

            VectorType d(Jv);
            d -= fd;

            const double abs_err = d.l2_norm();
            const double rel_err = abs_err / std::max(1.0, fd.l2_norm());

            double                  worst_local = 0.0;
            types::global_dof_index worst_i     = numbers::invalid_dof_index;
            for (const auto i : problem.locally_owned_dofs_)
              if (std::abs(Jv(i) - fd(i)) > worst_local)
                {
                  worst_local = std::abs(Jv(i) - fd(i));
                  worst_i     = i;
                }

            const double worst =
              Utilities::MPI::max(worst_local, problem.mpi_communicator_);
            const unsigned int reporter =
              Utilities::MPI::min(worst_local >= worst && worst > 0.0 ?
                                    problem.this_mpi_process :
                                    numbers::invalid_unsigned_int,
                                  problem.mpi_communicator_);

            if (reporter == problem.this_mpi_process &&
                worst_i != numbers::invalid_dof_index)
              std::cout << "  worst row " << worst_i << " on rank "
                        << problem.this_mpi_process << ": "
                        << (problem.cell_dofs_owned.is_element(worst_i) ?
                              "CELL" :
                            problem.trace_dofs_owned.is_element(worst_i) ?
                              "TRACE" :
                              "RCR")
                        << "  |Jv-fd| = " << worst << std::endl;

            problem.pcout << "\n=====================================\n"
                          << "Central FD Jacobian check\n"
                          << "eps        = " << eps << "\n"
                          << "||Jv-FD||  = " << abs_err << "\n"
                          << "||FD||     = " << fd.l2_norm() << "\n"
                          << "relative   = " << rel_err << "\n"
                          << "=====================================\n";
          };

          if (problem.verbosity > 0)
            check_jacobian_fd(problem.solution);

          problem.ida_parameters.ic_type =
            SUNDIALS::IDA<VectorType>::AdditionalData::use_y_diff;
          SUNDIALS::IDA<VectorType> ida(problem.ida_parameters,
                                        problem.mpi_communicator_);

          ida.reinit_vector = [&problem](VectorType &v) {
            v.reinit(problem.locally_owned_dofs_, problem.mpi_communicator_);
          };

          ida.differential_components = [&problem]() -> IndexSet {
            IndexSet is(problem.n_total_dofs);
            is.add_indices(problem.cell_dofs_owned);
            is.add_indices(problem.rcr_dofs_owned);
            is.compress();
            return is;
          };

          ida.residual = [&problem](const double      t,
                                    const VectorType &y,
                                    const VectorType &ydot,
                                    VectorType       &res) -> int {
            problem.assemble_residual(t, y, ydot, res);
            return 0;
          };

          ida.setup_jacobian = [&problem](const double      t,
                                          const VectorType &y,
                                          const VectorType &ydot,
                                          const double      alpha) -> int {
            TimerOutput::Scope ts(problem.computing_timer, "setup_jacobian");
            problem.assemble_jacobian(t, y, ydot, alpha);
            problem.linear_system_matrix.copy_from(problem.jacobian_matrix);

            if (problem.use_direct_solver)
              {
#ifdef USE_PETSC_LA
                problem.direct_solver =
                  std::make_unique<PETScWrappers::SparseDirectMUMPS>(
                    problem.direct_solver_control);
#else
                problem.direct_solver =
                  std::make_unique<TrilinosWrappers::SolverDirect>(
                    problem.direct_solver_control);
                problem.direct_solver->initialize(problem.linear_system_matrix);
#endif
              }
            else
              {
                problem.ilu_preconditioner =
                  std::make_unique<LA::MPI::PreconditionILU>();
                problem.ilu_preconditioner->initialize(
                  problem.linear_system_matrix);
              }
            return 0;
          };

          ida.solve_with_jacobian = [&problem](const VectorType &r,
                                               VectorType       &z,
                                               const double /*tol*/) -> int {
            TimerOutput::Scope ts(problem.computing_timer,
                                  "solve_with_jacobian");

            if (problem.use_direct_solver)
              {
#ifdef USE_PETSC_LA
                problem.direct_solver->solve(problem.linear_system_matrix,
                                             z,
                                             r);
#else
                problem.direct_solver->solve(z, r);
#endif
              }
            else
              {
                SolverControl   solver_control(1000, 1e-10 * r.l2_norm());
                LA::SolverGMRES solver(solver_control);
                solver.solve(problem.linear_system_matrix,
                             z,
                             r,
                             *problem.ilu_preconditioner);
              }
            return 0;
          };

          ida.output_step = [&problem](const double      t,
                                       const VectorType &sol,
                                       const VectorType & /*ydot*/,
                                       const unsigned int step_number) {
            problem.time = t;
            problem.compute_pressure(sol, problem.pressure_);
            problem.output_results(sol, problem.pressure_, step_number);
            problem.write_csv_row(t, sol);
          };

          problem.solution_dot = 0.0;
          problem.time         = problem.ida_parameters.initial_time;
          ida.solve_dae(problem.solution, problem.solution_dot);

          problem.close_csv_files();
          problem.compute_pressure(problem.solution, problem.pressure_);
          problem.compute_errors(cycle);
        }
    }
  };
} // namespace MetricFlowX

int
main(int argc, char **argv)
{
  try
    {
      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
      /* --------------------------- 1. Locate parameter file -----------------
       */
      std::string par_name;
      if (argc > 1)
        par_name = argv[1]; // first CLI argument
      else
        par_name = "parameters.prm"; // fallback default

      /* ---------------------- 2. Initialise deal.II logging -----------------
       */
      dealii::deallog.depth_console(1);

      /* ------------------------- 3. Set up the problem ----------------------
       */
      BloodFlowSystem<1, 3> problem; // 1-dim geometry embedded in \mathbb{R}^3
      problem.initialize_params(par_name);
      BloodFlowIDARunner<1, 3> runner;
      runner.run(problem);

      /* ------------------------ 4. Normal program exit ----------------------
       */
      return 0;
    }

  /* --------------------- 5. Dedicated exception catchers ------------------ */
  catch (const std::invalid_argument &theta_range) // parameter-specific errors
    {
      std::cerr << '\n'
                << "----------------------------------------------------\n"
                << "Invalid parameter: \n"
                << theta_range.what() << '\n'
                << "Aborting!\n"
                << "----------------------------------------------------\n";
      return 1;
    }
  catch (const std::exception &exc) // general deal.II/runtime errors
    {
      std::cerr << '\n'
                << "----------------------------------------------------\n"
                << "Exception on processing: \n"
                << exc.what() << '\n'
                << "Aborting!\n"
                << "----------------------------------------------------\n";
      return 1;
    }
  catch (...) // fallback
    {
      std::cerr << '\n'
                << "----------------------------------------------------\n"
                << "Unknown exception!\n"
                << "Aborting!\n"
                << "----------------------------------------------------\n";
      return 1;
    }
}
