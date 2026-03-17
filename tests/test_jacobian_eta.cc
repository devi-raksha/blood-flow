// ---------------------------------------------------------------------
//
// Jacobian applied to constant perturbation for a trivial exact solution.
// Exact state: A = 1, U = 0, RHS = 0. For dW = (1,1), J*dW should be (0,
// -mu).
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
  problem.initialize_params(PRM_DIR "jacobian_const.prm");
  GridGenerator::hyper_cube(problem.triangulation, 0, 1);
  problem.triangulation.refine_global(problem.n_global_refinements);
  problem.setup_system();

  const double t = 0.0;

  problem.compute_initial_solution(problem.solution, t);

  Vector<double> residual(problem.solution.size());
  problem.assemble_implicit_function(t, problem.solution, residual);
  problem.assemble_jacobian(t, problem.solution, residual);

  deallog << "Residual = " << residual.l2_norm() << std::endl;

  Vector<double> ones(problem.solution.size());
  ones = 1.0;
  Vector<double> Jdw(problem.solution.size());
  problem.jacobian_matrix.vmult(Jdw, ones);

  // Sum over all dofs; expected momentum component contributes -mu times
  // measure
  const double sum_Jdw = Jdw * ones; // (J dw, ones)_L2 = int J&dW dx = -mu
                                     // * measure
  deallog << "J*ones = " << sum_Jdw << " (it should be: -integral(mu, 0,1))"
          << std::endl;
  deallog << "mu = " << problem.par["mu"] << std::endl;
}

int
main()
{
  initlog();
  test();
}
