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

// Finite-difference check of the Jacobian against assemble_jacobian

#include <deal.II/base/parsed_function.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/sparse_direct.h>

#include <deal.II/numerics/vector_tools.h>

#include <cstdlib>

#include "blood_flow_system.h"
#include "tests.h"

using namespace dealii;

void
test()
{
  BloodFlowSystem<1, 3> problem; // 1-dim geometry embedded in \mathbb{R}^3
  problem.initialize_params(PRM_DIR "constant.prm");

  deallog.depth_console(1);

  GridGenerator::hyper_cube(problem.triangulation, 0, 1);
  problem.triangulation.refine_global(problem.n_global_refinements);

  problem.setup_system();

  problem.compute_initial_solution(problem.solution, problem.time);

  const double       t = problem.time;
  Vector<double>     M_y_dot(problem.solution.size());
  const unsigned int n_dofs = problem.solution.size();

  problem.assemble_jacobian(t, problem.solution, M_y_dot);

  Vector<double> dW(n_dofs);
  for (unsigned int i = 0; i < n_dofs; ++i)
    dW[i] = random_value();

  const double eps = 1e-8;

  auto newp = problem.solution;
  newp.add(eps, dW);
  Vector<double> r_plus(n_dofs);
  problem.assemble_implicit_function(t, newp, r_plus);

  r_plus.add(-1.0, M_y_dot);
  r_plus *= 1 / eps;

  // Now compute J dW
  Vector<double> JdW(n_dofs);
  problem.jacobian_matrix.vmult(JdW, dW);

  double max_diff = r_plus.linfty_norm();
  double max_J    = JdW.linfty_norm();

  r_plus.add(-1.0, JdW);
  const double error = r_plus.linfty_norm();

  deallog << "Max FD: " << max_diff << std::endl;
  deallog << "Max Jacobian: " << max_J << std::endl;
  deallog << "Error : " << error << std::endl;
}


int
main()
{
  initlog();
  test();
}
