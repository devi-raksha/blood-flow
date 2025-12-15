// ---------------------------------------------------------------------
//
// Jacobian applied to linear manufactured state.
// Manufactured solution (linear.prm): A = x + 1, U = 2 - x, with RHS set so
// that the exact solution remains linear in space/time.
//
// ---------------------------------------------------------------------

#include <deal.II/grid/grid_generator.h>

#include "blood_flow_system.h"
#include "tests.h"

using namespace dealii;

void
test()
{
  BloodFlowSystem<1, 3> problem;
  problem.initialize_params(PRM_DIR "linear.prm");
  GridGenerator::hyper_cube(problem.triangulation, 0, 1);
  problem.triangulation.refine_global(problem.n_global_refinements);
  problem.setup_system();

  problem.compute_initial_solution(problem.solution, problem.time);

  const double   t   = problem.time;
  Vector<double> rhs(problem.solution.size());
  problem.assemble_implicit_function(t, problem.solution, rhs);
  problem.assemble_jacobian(
    t, problem.solution, Vector<double>(problem.solution.size()));

  Vector<double> v(problem.solution.size());
  v = 1.0;
  Vector<double> Jv(problem.solution.size());
  problem.jacobian_matrix.vmult(Jv, v);
  Vector<double> ones(problem.solution.size());
  ones = 1.0;

  deallog << "rhs_l2 " << rhs.l2_norm() << std::endl;
  deallog << "Jv_l2 " << Jv.l2_norm() << std::endl;
  deallog << "Jv_sum " << Jv * ones << std::endl;
}

int
main()
{
  initlog();
  test();
}
