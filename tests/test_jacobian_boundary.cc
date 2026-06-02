// ---------------------------------------------------------------------
//
// Test boundary residual vs boundary Jacobian block.
//
// Assembles only assemble_boundary_residuals and assemble_jacobian_boundary_block,
// then checks every entry of the (cell,cell) and (cell,trace) blocks
// with central finite differences.
//
// Output:
//   L2_error       — ||J_an - J_fd||_F  over all (cell-row, any-col) entries
//   worst_row      — row index with the largest per-row L2 error
//   row_L2_error   — ||J_an[worst_row,:] - J_fd[worst_row,:]||_2
// ---------------------------------------------------------------------

#include <deal.II/grid/grid_in.h>

#include <deal.II/lac/vector.h>

#include <cmath>
#include <iomanip>

#include "blood_flow_system.h"
#include "tests.h"
#include "vtk_utils.h"

using namespace dealii;

void
test()
{
  BloodFlowSystem<1, 3> problem;
  problem.initialize_params(PRM_DIR "constant.prm");
  dealii::GridIn<1, 3> grid_in;
  grid_in.attach_triangulation(problem.triangulation);
  std::ifstream mesh_file(problem.vtk_file_path);
  grid_in.read_vtk(mesh_file);

  VTKUtils::read_cell_data(problem.vtk_file_path,
                           "vessel_id",
                           problem.cell_vessel_ids);
  VTKUtils::read_cell_data(problem.vtk_file_path, "a0", problem.cell_a0);
  VTKUtils::read_cell_data(problem.vtk_file_path, "a_d", problem.cell_a_d);
  VTKUtils::read_cell_data(problem.vtk_file_path, "E", problem.cell_E);
  VTKUtils::read_cell_data(problem.vtk_file_path,
                           "h_wall",
                           problem.cell_h_wall);
  VTKUtils::read_cell_data(problem.vtk_file_path, "p_d", problem.cell_p_d);
  VTKUtils::read_cell_data(problem.vtk_file_path, "p0", problem.cell_p0);
  VTKUtils::read_cell_data(problem.vtk_file_path, "L", problem.cell_L);
  VTKUtils::read_cell_data(problem.vtk_file_path, "r_d", problem.cell_r_d);

  VTKUtils::read_vertex_data(problem.vtk_file_path,
                             "boundary_id",
                             problem.point_boundary_id);
  VTKUtils::read_vertex_data(problem.vtk_file_path, "R1", problem.point_R1);
  VTKUtils::read_vertex_data(problem.vtk_file_path, "R2", problem.point_R2);
  VTKUtils::read_vertex_data(problem.vtk_file_path, "C", problem.point_C);
  VTKUtils::read_vertex_data(problem.vtk_file_path,
                             "P_out",
                             problem.point_P_out);

  {
    unsigned int idx = 0;
    for (auto &cell : problem.triangulation.active_cell_iterators())
      {
        cell->set_material_id(
          static_cast<unsigned int>(problem.cell_vessel_ids[idx]));
        for (unsigned int f = 0; f < GeometryInfo<1>::faces_per_cell; ++f)
          if (cell->face(f)->at_boundary())
            {
              const unsigned int v = cell->face(f)->vertex_index(0);
              cell->face(f)->set_boundary_id(
                static_cast<types::boundary_id>(problem.point_boundary_id[v]));
            }
        ++idx;
      }
  }

  for (const auto &cell : problem.triangulation.active_cell_iterators())
    for (unsigned int f = 0; f < GeometryInfo<1>::faces_per_cell; ++f)
      if (cell->face(f)->at_boundary())
        {
          const types::boundary_id          bid = cell->face(f)->boundary_id();
          const unsigned int                v = cell->face(f)->vertex_index(0);
          BloodFlowSystem<1, 3>::RCRPhysics rcr;
          rcr.R1    = problem.point_R1[v];
          rcr.R2    = problem.point_R2[v];
          rcr.C     = problem.point_C[v];
          rcr.P_out = problem.point_P_out[v];
          if (rcr.R2 > 0.0)
            problem.rcr_map[bid] = rcr;
        }

  problem.triangulation.refine_global(problem.n_global_refinements);
  problem.setup_system();
  problem.initialize_terminal_capacitors();
  problem.build_per_cell_mass_inv();
  problem.compute_initial_solution(problem.solution, problem.time);

  // ── assemble only the boundary Jacobian block ────────────────────────────────
  const double       t       = problem.time;
  const unsigned int n_cell  = problem.n_cell_dofs;
  const unsigned int n_total = problem.n_total_dofs;

  deallog << "n_cell=" << n_cell << "  n_total=" << n_total << std::endl;

  problem.jacobian_matrix = 0.0;
  problem.assemble_jacobian_trace_boundary_block(t, problem.solution);

  // ── central FD check ─────────────────────────────────────────────────────
  const double eps = 1e-8;

  Vector<double> yp(n_total), ym(n_total), Fp(n_total), Fm(n_total);
  Vector<double> ej(n_total), Jcol(n_total);

  // row_sq[i] accumulates sum_j (J_an[i,j] - J_fd[i,j])^2
  Vector<double> row_sq(n_total);
  double         l2_sq = 0.0;

  for (unsigned int j = 0; j < n_total; ++j)
    {
      const double h = eps; //* (1.0 + std::abs(problem.solution[j]));

      yp = problem.solution;
      yp[j] += h;
      ym = problem.solution;
      ym[j] -= h;

      Fp.reinit(n_total);
      problem.assemble_trace_boundary_equations(t,yp, Fp);
      Fm.reinit(n_total);
      problem.assemble_trace_boundary_equations(t, ym, Fm);

      ej    = 0.0;
      ej[j] = 1.0;
      problem.jacobian_matrix.vmult(Jcol, ej);

      for (unsigned int i = 0; i < n_total; ++i)
        {
          const double fd  = (Fp[i] - Fm[i]) / (2.0 * h);
          const double err = Jcol[i] - fd;
          row_sq[i] += err * err;
          l2_sq += err * err;
        }
    }

  // Worst row: the one with the largest row-wise L2 error
  unsigned int worst_row = 0;
  for (unsigned int i = 1; i < n_total; ++i)
    if (row_sq[i] > row_sq[worst_row])
      worst_row = i;

  const double l2_err       = std::sqrt(l2_sq);
  const double worst_row_l2 = std::sqrt(row_sq[worst_row]);

  deallog << "L2_error=" << std::scientific << std::setprecision(4) << l2_err
          << "  worst_row=" << worst_row << "  row_L2_error=" << worst_row_l2
          << std::endl;
}

int
main()
{
  initlog();
  test();
}