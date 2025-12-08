#include "../include/blood_flow_system_updated_1d3d.h"
#include <deal.II/grid/grid_generator.h>
#include <deal.II/base/logstream.h>

using namespace dealii;

template<int dim, int spacedim>
void test()
{
    BloodFlowSystem<dim, spacedim> problem;

    problem.initialize_params("parameters.prm");

    // Build mesh & system
    GridGenerator::hyper_cube(problem.triangulation, 0.0, 1.0);
    problem.triangulation.refine_global(problem.n_global_refinements);

    problem.setup_system();

    // Set initial solution
    problem.solution = 0;
    problem.solution_old = 0;
    problem.time = 0.0;

    // Must compute residual once to initialize old state if used
    problem.assemble_system();

    const unsigned int n = problem.solution.size();
    const double eps = 1e-8;

    Vector<double> Jcol(n), FDcol(n);

    double max_err = 0.0;

    for (unsigned int j = 0; j < n; ++j)
    {
        // --- Build y+ and y− ---
        Vector<double> y_plus = problem.solution;
        Vector<double> y_minus = problem.solution;

        y_plus[j]  += eps;
        y_minus[j] -= eps;

        // --- Compute F(y+) ---
        problem.solution = y_plus;
        problem.assemble_system();           // fills residual_vector
        Vector<double> F_plus = problem.residual_vector;

        // --- Compute F(y−) ---
        problem.solution = y_minus;
        problem.assemble_system();
        Vector<double> F_minus = problem.residual_vector;

        // --- Central finite difference column ---
        FDcol = F_plus;
        FDcol.add(-1.0, F_minus);
        FDcol /= (2.0 * eps);

        // --- Extract Jacobian column: J e_j ---
        Vector<double> ej(n), J_ej(n);
        ej = 0.0;
        ej[j] = 1.0;
        problem.jacobian_matrix.vmult(J_ej, ej);

        // --- Compare ---
        Vector<double> diff = FDcol;
        diff.add(-1.0, J_ej);
        max_err = std::max(max_err, diff.linfty_norm());
    }

    std::cout << "\n========================================\n";
    std::cout << " FINITE DIFFERENCE JACOBIAN CHECK\n";
    std::cout << " max |FD - J|_inf = " << max_err << std::endl;
    std::cout << "========================================\n";
}

int main()
{
    deallog.depth_console(0);

    test();

    return 0;
}
