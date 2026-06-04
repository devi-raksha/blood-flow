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
// Test length of domain using per-cell mass matrix blocks.

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

  // --- Load mesh and physics (same as residual test) ---
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

  // --- Set material and boundary IDs ---
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

  // --- Populate rcr_map ---
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
  problem
    .build_per_cell_mass_inv(); // fills per_cell_mass_ and per_cell_mass_inv

  // Build v = 1 on area (component 0) cell DOFs, 0 elsewhere.
  // With this choice:  sum_K v_K^T M_K v_K = integral_Omega 1 dx = total
  // length.
  Vector<double> v(problem.solution.size());
  v = 0.0;

  const unsigned int n_dofs = problem.fe->n_dofs_per_cell();
  for (const auto &cell : problem.dof_handler.active_cell_iterators())
    {
      std::vector<types::global_dof_index> ldofs(n_dofs);
      cell->get_dof_indices(ldofs);
      for (unsigned int i = 0; i < n_dofs; ++i)
        if (problem.fe->system_to_component_index(i).first ==
            0) // area component
          v[ldofs[i]] = 1.0;
    }

  // Compute L = sum_K  v_K^T M_K v_K  (cell DOFs only; trace block untouched)
  double         L = 0.0;
  Vector<double> local_v(n_dofs), local_Mv(n_dofs);

  for (const auto &cell : problem.dof_handler.active_cell_iterators())
    {
      std::vector<types::global_dof_index> ldofs(n_dofs);
      cell->get_dof_indices(ldofs);

      for (unsigned int i = 0; i < n_dofs; ++i)
        local_v(i) = v[ldofs[i]];

      problem.per_cell_mass_.at(cell->id()).vmult(local_Mv, local_v);

      for (unsigned int i = 0; i < n_dofs; ++i)
        L += local_v(i) * local_Mv(i);
    }

  // Independent check: sum physical cell measures from the triangulation
  double L_geom = 0.0;
  for (const auto &cell : problem.triangulation.active_cell_iterators())
    L_geom += cell->measure();

  deallog << "L (mass matrix):   " << L << std::endl;
  deallog << "L (cell measures): " << L_geom << std::endl;

  AssertThrow(std::abs(L - L_geom) / L_geom < 1e-10,
              ExcMessage("Mass matrix length integration FAILED."));

  deallog << "Mass matrix length integration PASSED." << std::endl;
}

int
main()
{
  initlog();
  test();
}