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
#include <deal.II/base/timer.h>

#include <deal.II/lac/petsc_solver.h>

#include <metric_flow_x/blood_flow_system.h> // header exposing BloodFlowSystem

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
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

          if (cycle > 0)
            {
              problem.setup_system();
              problem.initialize_terminal_capacitors();
              problem.build_per_cell_mass_inv();
            }

          // After setup_system(), so the vessel probe cells and DoF indices
          // belong to the current mesh.  Opens HDG_IDA_Vessel_*.csv,
          // mass_conservation.csv and ida_step_history.csv, and resets the
          // IDA counters for this cycle.
          problem.open_output_files();
          problem.initialize_state(problem.solution,
                                   problem.ida_parameters.initial_time);
          problem.initialize_state_derivative(
            problem.solution_dot, problem.ida_parameters.initial_time);
          problem.time = problem.ida_parameters.initial_time;

          auto check_jacobian_fd = [&problem](const VectorType &y0) {
            using Comp          = typename Problem::Component;
            const MPI_Comm comm = problem.mpi_communicator_;
            const double   t    = problem.time;
            const double   eps  = 1e-6; // relative perturbation

            VectorType ydot0(problem.locally_owned_dofs_, comm);
            ydot0 = 0.0;
            problem.assemble_jacobian(t, y0, ydot0, /*alpha=*/0.0);

            // Global max |y| over a block: used as a floor for unknowns that
            // are (close to) zero, e.g. the velocities at rest at t = 0.
            auto block_max = [&](const IndexSet &set) {
              double m = 0.0;
              for (const auto i : set)
                m = std::max(m, std::abs(y0(i)));
              return Utilities::MPI::max(m, comm);
            };

            struct ColBlock
            {
              std::string     name;
              const IndexSet *set;
              double          floor;
            };
            const IndexSet &sA  = problem.component_dofs(Comp::area);
            const IndexSet &sU  = problem.component_dofs(Comp::velocity);
            const IndexSet &sAh = problem.component_dofs(Comp::area_trace);
            const IndexSet &sUh = problem.component_dofs(Comp::velocity_trace);

            const std::vector<ColBlock> col_blocks = {
              {"A", &sA, 1e-3 * block_max(sA)},
              {"U", &sU, std::max(1e-3, 1e-3 * block_max(sU))},
              {"A_hat", &sAh, 1e-3 * block_max(sAh)},
              {"U_hat", &sUh, std::max(1e-3, 1e-3 * block_max(sUh))},
              {"Pc", &problem.rcr_dofs_owned, 1.0}};

            problem.pcout
              << "\n========== FD Jacobian check (scaled, eps = " << eps
              << ") ==========\n"
              << "  column   | rel.err CELL rows | TRACE rows | RCR rows\n";

            std::srand(problem.this_mpi_process + 1);

            for (const auto &cb : col_blocks)
              {
                // v perturbs only this column block, each entry scaled to the
                // size of the unknown it perturbs.
                VectorType v(problem.locally_owned_dofs_, comm);
                v = 0.0;
                for (const auto i : *cb.set)
                  {
                    const double r = 2.0 * std::rand() / double(RAND_MAX) - 1.0;
                    v(i)           = r * std::max(std::abs(y0(i)), cb.floor);
                  }
                v.compress(VectorOperation::insert);
                if (v.l2_norm() == 0.0)
                  continue; // e.g. no capacitor unknowns

                VectorType yp(y0), ym(y0);
                yp.add(eps, v);
                ym.add(-eps, v);

                VectorType Fp(problem.locally_owned_dofs_, comm);
                VectorType Fm(problem.locally_owned_dofs_, comm);
                problem.assemble_residual(t, yp, ydot0, Fp);
                problem.assemble_residual(t, ym, ydot0, Fm);

                VectorType fd(Fp);
                fd -= Fm;
                fd /= (2.0 * eps);

                VectorType Jv(problem.locally_owned_dofs_, comm);
                problem.jacobian_matrix.vmult(Jv, v);

                // Squared error and squared reference, per row type:
                // [0] cell, [1] trace, [2] rcr.
                double local[6] = {0, 0, 0, 0, 0, 0};
                for (const auto i : problem.locally_owned_dofs_)
                  {
                    const unsigned int k =
                      problem.cell_dofs_owned.is_element(i)  ? 0 :
                      problem.trace_dofs_owned.is_element(i) ? 1 :
                                                               2;
                    const double di = Jv(i) - fd(i);
                    local[k] += di * di;
                    local[3 + k] += fd(i) * fd(i);
                  }
                double global[6];
                MPI_Allreduce(local, global, 6, MPI_DOUBLE, MPI_SUM, comm);

                auto rel = [&](const unsigned int k) -> std::string {
                  const double       e = std::sqrt(global[k]);
                  const double       r = std::sqrt(global[3 + k]);
                  std::ostringstream s;
                  s << std::scientific << std::setprecision(2);
                  if (e == 0.0 && r == 0.0)
                    s << "      -     ";
                  else if (r == 0.0)
                    s << "abs " << e; // Jacobian entry where residual is flat
                  else
                    s << e / r;
                  return s.str();
                };

                problem.pcout << "  " << std::setw(8) << std::left << cb.name
                              << " | " << std::setw(17) << rel(0) << " | "
                              << std::setw(10) << rel(1) << " | " << rel(2)
                              << "\n";
              }
            problem.pcout << "==========================================="
                             "==========\n";
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

          ida.residual = [&problem, &ida](const double      t,
                                    const VectorType &y,
                                    const VectorType &ydot,
                                    VectorType       &res) -> int {
            problem.poll_ida_internal_step(ida);
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

          const double initial_output_time =
            problem.ida_parameters.initial_time;
          const double final_output_time = problem.ida_parameters.final_time;
          double       next_output_time  = initial_output_time;
          unsigned int output_number     = 0;
          ida.output_step =
            [&problem,
             &ida,
             initial_output_time,
             final_output_time,
             &next_output_time,
             &output_number](const double      t,
                             const VectorType &sol,
                             const VectorType & /*ydot*/,
                             const unsigned int /*step_number*/) {
              problem.time = t;

              const double tolerance =
                16.0 * std::numeric_limits<double>::epsilon() *
                std::max(std::abs(t), std::abs(next_output_time));
              const double final_time_tolerance =
                16.0 * std::numeric_limits<double>::epsilon() *
                std::max(std::abs(t), std::abs(final_output_time));
                problem.poll_ida_internal_step(ida, /*resync_time=*/true);
              const bool output_due =
                problem.time_interval == 0.0 || t >= next_output_time ||
                next_output_time - t <= tolerance || t >= final_output_time ||
                final_output_time - t <= final_time_tolerance;
              if (!output_due)
                return;

              // IDA internal steps / nonlinear iterations taken since the
              // previous *written* output -> ida_step_history.csv.  Recorded
              // after the output_due check so each row spans one output
              // interval set by "Output time interval".
              problem.record_ida_step(ida, t);

              problem.compute_pressure(sol, problem.pressure_);
              problem.output_results(sol, problem.pressure_, output_number++);
              // HDG_IDA_Vessel_*.csv + mass_conservation.csv
              problem.write_output_step(t, sol);

              if (problem.time_interval > 0.0)
                {
                  const long double elapsed =
                    (static_cast<long double>(t + tolerance) -
                     initial_output_time) /
                    problem.time_interval;
                  const long double intervals = std::floor(elapsed) + 1.0L;
                  next_output_time            = static_cast<double>(
                    static_cast<long double>(initial_output_time) +
                    intervals * problem.time_interval);
                  if (next_output_time <= t)
                    next_output_time =
                      std::nextafter(t,
                                     std::numeric_limits<double>::infinity());
                }
            };

          problem.solution_dot = 0.0;
          problem.time         = problem.ida_parameters.initial_time;

          Timer solve_timer;
          solve_timer.start();
          ida.solve_dae(problem.solution, problem.solution_dot);
          solve_timer.stop();

          // Closes the vessel / mass / step-history CSVs and appends one row
          // to ida_statistics.csv.  Must run while `ida` is still alive.
          problem.finalize_output(ida, solve_timer.wall_time());

          problem.compute_pressure(problem.solution, problem.pressure_);
          problem.compute_errors(cycle);
        }

      // One row per TimerOutput section -> scaling_timings.csv
      problem.write_scaling_timings();
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