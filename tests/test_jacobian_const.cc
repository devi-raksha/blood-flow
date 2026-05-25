// ---------------------------------------------------------------------
// Minimal Jacobian consistency test via finite differences.
// ---------------------------------------------------------------------

#include <deal.II/grid/grid_in.h>

#include <deal.II/lac/vector.h>

#include "blood_flow_system.h"
#include "tests.h"
#include "vtk_utils.h"

using namespace dealii;

void
test()
{
  BloodFlowSystem<1, 3> problem;
  problem.initialize_params(PRM_DIR "constant.prm");

  // ── Mesh ──────────────────────────────────────────────────────────────
  dealii::GridIn<1, 3> grid_in;
  grid_in.attach_triangulation(problem.triangulation);
  std::ifstream mesh_file(problem.vtk_file_path);
  AssertThrow(mesh_file.is_open(),
              ExcMessage("Cannot open VTK file: " + problem.vtk_file_path));
  grid_in.read_vtk(mesh_file);

  // ── Cell / point data ─────────────────────────────────────────────────
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

  // ── Material / boundary IDs ───────────────────────────────────────────
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

  // ── RCR map ───────────────────────────────────────────────────────────
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
          if (rcr.R1 > 0.0)
            problem.rcr_map[bid] = rcr;
        }

  // ── Setup ─────────────────────────────────────────────────────────────
  problem.triangulation.refine_global(problem.n_global_refinements);

  problem.setup_system();
  problem.initialize_terminal_capacitors();
  problem.build_per_cell_mass_inv();
  problem.compute_initial_solution(problem.solution, problem.time);

  const unsigned int n = problem.solution.size();
  const double       t = problem.time;

  // ── Base residual F(y_eq) ─────────────────────────────────────────────
  Vector<double> F0(n);
  problem.assemble_implicit_function(t, problem.solution, F0);
  deallog << "F0 norm = " << F0.l2_norm() << std::endl;

  // ── Analytic Jacobian ─────────────────────────────────────────────────
  Vector<double> dummy(n);
  problem.assemble_jacobian(t, problem.solution, dummy);

  // ── Finite-difference check ───────────────────────────────────────────
  const double       eps    = 1e-8;
  const double       tol    = 1e-4;
  const unsigned int stride = std::max(1u, n / 20u);

  Vector<double> y_pert(n), F_pert(n), e_j(n), Jv(n);
  double         max_err = 0.0;
  unsigned int   max_j   = 0;

  for (unsigned int j = 0; j < n; j += stride)
    {
      // FD column
      y_pert = problem.solution;
      y_pert[j] += eps;
      problem.assemble_implicit_function(t, y_pert, F_pert);

      // Analytic column
      e_j    = 0.0;
      e_j[j] = 1.0;
      problem.jacobian_matrix.vmult(Jv, e_j);

      double col_err = 0.0;
      for (unsigned int i = 0; i < n; ++i)
        {
          const double fd_ij  = (F_pert[i] - F0[i]) / eps;
          const double err_ij = std::abs(Jv[i] - fd_ij);
          col_err             = std::max(col_err, err_ij);
        }

      if (col_err > max_err)
        {
          max_err = col_err;
          max_j   = j;
        }
    }

  deallog << "max fd vs jacobian error =" << max_err << " at j=" << max_j << std::endl;

}

int
main()
{
  initlog();
  test();
}