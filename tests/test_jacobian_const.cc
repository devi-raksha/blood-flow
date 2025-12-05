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

// Test constant functions

#include <deal.II/base/parsed_function.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/sparse_direct.h>

#include <deal.II/numerics/vector_tools.h>

#include "blood_flow_system.h"
#include "tests.h"

using namespace dealii;

void
test()
{
  BloodFlowSystem<1, 3> problem; // 1-dim geometry embedded in \mathbb{R}^3
  problem.initialize_params(PRM_DIR "jacobian_const.prm");
  GridGenerator::hyper_cube(problem.triangulation, 0, 1);
  problem.triangulation.refine_global(problem.n_global_refinements);

  problem.setup_system();

  problem.compute_initial_solution(problem.solution, problem.time);

  const double   t = problem.time;
  Vector<double> residual(problem.solution.size());
  problem.assemble_implicit_function(t, problem.solution, residual);
  problem.assemble_jacobian(t,
                            problem.solution,
                            Vector<double>(problem.solution.size()));

  // Finite-difference check: J*v ≈ (f(y+eps v)-f(y))/eps
  std::mt19937                     gen(42);
  std::uniform_real_distribution<> dist(-1.0, 1.0);
  Vector<double>                   v(problem.solution.size());
  for (unsigned int i = 0; i < v.size(); ++i)
    v[i] = dist(gen);

  const double eps = 1e-7 * std::max(1.0, v.linfty_norm()) /
                     std::max(1.0, problem.solution.linfty_norm());

  Vector<double> y_plus = problem.solution;
  y_plus.add(eps, v);

  Vector<double> res_plus(problem.solution.size());
  problem.assemble_implicit_function(t, y_plus, res_plus);

  Vector<double> Jv(problem.solution.size());
  problem.jacobian_matrix.vmult(Jv, v);

  // fd = (f(y+eps v)-f(y))/eps
  res_plus.add(-1.0, residual);
  res_plus /= eps;

  Vector<double> diff = res_plus;
  diff.add(-1.0, Jv);

  const double max_diff = diff.linfty_norm();
  const double ref      = std::max(1.0, Jv.linfty_norm());
  deallog << "Jacobian FD check: max_diff=" << max_diff << " ref=" << ref
          << std::endl;
}


int
main()
{
  initlog();
  test();
}
