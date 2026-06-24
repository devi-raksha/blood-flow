// ---------------------------------------------------------------------
//
// Jacobian applied to a constant perturbation at the trivial equilibrium.
//
// At the diastolic equilibrium produced by compute_initial_solution(),
// area = a_d (uniform), velocity = 0, and the residual is zero.
//
// For dW = ones, J·dW probes how a uniform perturbation in A and U is
// "fed back" through the residual. The only term that contributes a
// purely uniform (non-derivative) sensitivity is the friction source
//   R_U += -eta * U/A * phi_u,   eta = 2 * (xi+2) * pi * mu / rho
//
// Since assemble_jacobian returns J_IDA = ∂F/∂y + α·M, with
//   F_cell = M·ẏ − R_cell  ⇒  ∂F/∂y = −∂R/∂y,
// the dot-product (J·ones, ones)_{R^N} on the U-block evaluates to
//
//     sum_Jdw_U  =  +eta / a_d · |Ω|
//
// (the sign is positive because R has −eta·U/A, and ∂F/∂y has a
//  second sign flip). With μ = 0 this should be ~0 up to FD/roundoff.
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
  problem.initialize_params(PRM_DIR "aortic.prm");

  // ── same setup as the equilibrium test ───────────────────────────────────
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

  // ── Evaluate residual and Jacobian at the equilibrium ────────────────────
  const double t     = 0.0;
  const double alpha = 0.0; // 0 → matrix equals ∂F/∂y (no α·M term)

  Vector<double> ydot(problem.solution.size());
  ydot = 0.0;

  Vector<double> residual(problem.solution.size());
  problem.assemble_residual(t, problem.solution, ydot, residual);
  problem.assemble_jacobian(t, problem.solution, ydot, alpha);

  double pc_sq = 0, rest_sq = 0;
  for (unsigned int i = 0; i < residual.size(); ++i)
    (i >= problem.n_trace_end ? pc_sq : rest_sq) += residual[i] * residual[i];
  deallog << "rest residual=" << std::sqrt(rest_sq)
          << "  pc=" << std::sqrt(pc_sq) << std::endl;
  deallog << "||residual with Pc|| = " << residual.l2_norm()
          << "  (should be ~0 at equilibrium when terminal pressure is Pc = P_out )" << std::endl;

  // ── J · ones, dot with ones ──────────────────────────────────────────────
  Vector<double> ones(problem.solution.size());
  ones = 1.0;
  Vector<double> Jdw(problem.solution.size());
  problem.jacobian_matrix.vmult(Jdw, ones);

  const double sum_Jdw = Jdw * ones;

  // ── Expected value ───────────────────────────────────────────────────────
  // Only the friction term contributes a "uniform sensitivity":
  //   ∂R_U/∂U  contains  -eta/A · ∫ φ_u φ_u dx   (eta = 2(ξ+2)πμ/ρ)
  // Through ∂F/∂y = -∂R/∂y and the partition-of-unity sum over shape
  // functions, this yields +eta/A · |Ω| in (J·ones, ones).
  // a_d is per-vessel (via vessel_map), preserved across refinement since
  // it's keyed by cell->material_id().
  double expected = 0.0;
  {
    const double rho = problem.par["rho"];
    const double mu  = problem.par["mu"];
    const double xi  = problem.par["xi"];
    const double eta = 2.0 * (xi + 2.0) * numbers::PI * mu / rho;

    for (const auto &cell : problem.triangulation.active_cell_iterators())
      {
        const unsigned int vid      = cell->material_id();
        const double       a_d_cell = problem.vessel_map.at(vid).a_d;
        const double       length   = cell->measure();
        if (a_d_cell > 0.0)
          expected += (eta / a_d_cell) * length;
      }
  }

  deallog << "J·ones (sum)  = " << sum_Jdw << std::endl;
  deallog << "expected      = " << expected << "  (= Σ_cells (eta/a_d) · |K|)"
          << std::endl;
  deallog << "mu            = " << problem.par["mu"] << std::endl;
  deallog << "xi            = " << problem.par["xi"] << std::endl;
  deallog << "rho           = " << problem.par["rho"] << std::endl;
}

int
main()
{
  initlog();
  test();
}