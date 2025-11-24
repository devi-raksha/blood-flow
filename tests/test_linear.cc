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

// Test linear functions

#include <deal.II/base/parsed_function.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/sparse_direct.h>

#include <deal.II/numerics/vector_tools.h>

#include "blood_flow_system_updated_1d3d.h"
#include "tests.h"

using namespace dealii;

void
test()
{
  BloodFlowSystem<1, 3> problem;
  GridGenerator::hyper_cube(problem.triangulation);
  problem.triangulation.refine_global(1);
  problem.setup_system();

  // Analytical manufactured field: A = 1 + x, U = 1
  FunctionParser<3> manufactured(2);
  manufactured.initialize("x,y,z", "1+x; 1", {});

  // Project this function into the FE space
  AffineConstraints<double> constraints;
  constraints.close();
  VectorTools::project(problem.dof_handler,
                       constraints,
                       QGauss<1>(problem.fe_degree + 1),
                       manufactured,
                       problem.solution);
  problem.solution_old = problem.solution;

  // Assemble Jacobian and residual at this state
  problem.assemble_system();

  // Compute J * W
  Vector<double> JW(problem.solution.size());
  problem.jacobian_matrix.vmult(JW, problem.solution);

  // Compute difference (J*W - residual)
  JW.add(-1.0, problem.residual_vector);

  std::cout << "‖J*W - R‖_L2 = " << JW.l2_norm() << std::endl;
  std::cout << "‖R‖_L2       = " << problem.residual_vector.l2_norm()
            << std::endl;
}

int
main()
{
  initlog();
  test();
}