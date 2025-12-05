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
  problem.initialize_params(PRM_DIR "quadratic.prm");
  GridGenerator::hyper_cube(problem.triangulation, 0, 1);
  problem.triangulation.refine_global(problem.n_global_refinements);

  problem.setup_system();

  problem.compute_initial_solution(problem.solution, problem.time);


  Vector<double> residual(problem.solution.size());
  problem.assemble_implicit_function(problem.time, problem.solution, residual);

  // The residual should be zero.
  deallog << "Residual norm: " << residual.l2_norm() << std::endl;
}


int
main()
{
  initlog();
  test();
}
