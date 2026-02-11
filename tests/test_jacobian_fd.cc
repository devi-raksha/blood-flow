#include <deal.II/base/logstream.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/numerics/vector_tools.h>

#include "../include/blood_flow_system_updated_1d3d.h"
#include "tests.h"

#include <random>
#include <iomanip>

using namespace dealii;

void
test()
{
  BloodFlowSystem<1, 3> problem;
  problem.initialize_params(PRM_DIR "jacobian_fd.prm");

  GridGenerator::hyper_cube(problem.triangulation, 0, 1);
  problem.triangulation.refine_global(problem.n_global_refinements);

  problem.setup_system();

  AffineConstraints<double> constraints;
  constraints.close();

  VectorTools::project(problem.dof_handler,
                       constraints,
                       QGauss<1>(problem.fe_degree + 1),
                       problem.initial_condition,
                       problem.solution);

  problem.solution_old = problem.solution;

  // ---------------------------
  // Assemble at base state
  // ---------------------------
  problem.assemble_system();

  Vector<double> R0 = problem.residual_vector;

  // ---------------------------
  //  Create random vector for finite difference check
  // ---------------------------
  Vector<double> delta(problem.solution.size());
  for (unsigned int i = 0; i < delta.size(); ++i)
    delta[i] = 2.0 * (double(rand()) / RAND_MAX - 0.5);

  delta /= delta.l2_norm(); // normalize direction

  const double eps = 1e-7;

// Store original solution
Vector<double> U0 = problem.solution;

// ---------------------------
// Compute J * delta
// ---------------------------
Vector<double> Jdelta(problem.solution.size());
problem.jacobian_matrix.vmult(Jdelta, delta);

// ---------------------------
// Compute R(u + eps*delta)
// ---------------------------
problem.solution = U0;
problem.solution.add(eps, delta);
problem.assemble_system();
Vector<double> R_plus = problem.residual_vector;

// ---------------------------
// Compute R(u - eps*delta)
// ---------------------------
problem.solution = U0;
problem.solution.add(-eps, delta);
problem.assemble_system();
Vector<double> R_minus = problem.residual_vector;

// ---------------------------
// Central difference
// ---------------------------
Vector<double> FD = R_plus;
FD -= R_minus;
FD /= (2.0 * eps);

// Restore original solution
problem.solution = U0;
problem.assemble_system();

// ---------------------------
// Compare
// ---------------------------
Vector<double> error = Jdelta;
error += FD;

deallog << std::scientific << std::setprecision(6);

deallog <<"delta : " << delta.l1_norm() << std::endl;

deallog << "Relative error: "
        << error.l2_norm() / FD.l2_norm()
        << std::endl;

  
}

int
main()
{
  initlog();
  test();
}
