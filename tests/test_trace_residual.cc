// ---------------------------------------------------------------------
//
// Copyright (C) 2024 by Luca Heltai
//
// This file is part of the bare-dealii-app application, based on the
// deal.II library.
//
// The bare-dealii-app application is free software; you can use it,
// redistribute it, and/or modify it under the terms of the Apache-2.0 License
// WITH LLVM-exception as published by the Free Software Foundation; either
// version 3.0 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE.md
// at the top level of the bare-dealii-app distribution.
//
// ---------------------------------------------------------------------
// Test: after initialize_trace_unknowns(), the trace residual
// F_trace(y_cell^0, y_hat) must be zero to machine precision.
// This verifies that the Newton solve converged correctly and that
// the initial condition is consistent before IDA even starts.

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
  problem.initialize_params(PRM_DIR "multi_vessel.prm");

  // --- Load mesh and physics ---
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
    unsigned int cell_idx = 0;
    for (auto &cell : problem.triangulation.active_cell_iterators())
      {
        cell->set_material_id(
          static_cast<unsigned int>(problem.cell_vessel_ids[cell_idx]));
        for (unsigned int f = 0; f < GeometryInfo<1>::faces_per_cell; ++f)
          if (cell->face(f)->at_boundary())
            {
              const unsigned int v = cell->face(f)->vertex_index(0);
              cell->face(f)->set_boundary_id(
                static_cast<types::boundary_id>(problem.point_boundary_id[v]));
            }
        ++cell_idx;
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
          if (rcr.R1 > 0.0)
            problem.rcr_map[bid] = rcr;
        }

  problem.triangulation.refine_global(problem.n_global_refinements);
  problem.setup_system();
  problem.initialize_terminal_capacitors();
  problem.build_per_cell_mass_inv();

  // Step 1: set cell DOFs to equilibrium (A=a_d, U=0); trace DOFs to a_d/0
  problem.compute_initial_solution(problem.solution, problem.time);

  // Step 2: check trace residual BEFORE initialize_trace_unknowns
  {
    Vector<double> F_before(problem.solution.size());
    problem.assemble_trace_interior_equations(problem.solution, F_before);
    problem.assemble_trace_boundary_equations(problem.time,
                                              problem.solution,
                                              F_before);
    problem.assemble_trace_junction_equations(problem.solution, F_before);

    double norm_before = 0.0;
    for (types::global_dof_index i = problem.n_cell_dofs;
         i < problem.n_total_dofs;
         ++i)
      norm_before += F_before[i] * F_before[i];
    norm_before = std::sqrt(norm_before);

    deallog << "Trace residual BEFORE initialize_trace_unknowns: "
            << norm_before << std::endl;
  }

  // Step 3: run Newton to find consistent trace unknowns
  problem.initialize_trace_unknowns(problem.solution, problem.time);

  // Step 4: check trace residual AFTER initialize_trace_unknowns
  {
    Vector<double> F_after(problem.solution.size());
    problem.assemble_trace_interior_equations(problem.solution, F_after);
    problem.assemble_trace_boundary_equations(problem.time,
                                              problem.solution,
                                              F_after);
    problem.assemble_trace_junction_equations(problem.solution, F_after);

    double norm_after = 0.0;
    for (types::global_dof_index i = problem.n_cell_dofs;
         i < problem.n_total_dofs;
         ++i)
      norm_after += F_after[i] * F_after[i];
    norm_after = std::sqrt(norm_after);

    deallog << "Trace residual AFTER initialize_trace_unknowns:  " << norm_after
            << std::endl;

    AssertThrow(norm_after < 1e-10,
                ExcMessage("initialize_trace_unknowns did not converge: "
                           "trace residual is not zero."));
  }

  deallog << "initialize_trace_unknowns PASSED." << std::endl;
}

int
main()
{
  initlog();
  test();
}