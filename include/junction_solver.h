#ifndef JUNCTION_SOLVER_H
#define JUNCTION_SOLVER_H

#include <deal.II/base/point.h>

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/vector.h>


using namespace dealii;

// ======================================================
// Generalized Junction State
// ======================================================

struct JunctionState
{
  Vector<double> X;

  unsigned int
  n_vessels() const
  {
    Assert(X.size() % 2 == 0, ExcInternalError());
    return X.size() / 2;
  }

  // const-correct version of operator[] for A and U
  // Allow to change A and U values using A(i) and U(i) syntax
  double &
  A(const unsigned int i)
  {
    return X[2 * i];
  }

  double &
  U(const unsigned int i)
  {
    return X[2 * i + 1];
  }

  // const version for read-only access to A and U
  double
  A(const unsigned int i) const
  {
    return X[2 * i];
  }

  double
  U(const unsigned int i) const
  {
    return X[2 * i + 1];
  }
};

// For orientation always think like this if vessel ends at junction then
// for that it is + 1 and if it starts at junction then it is -1.
// So for face 0 it is -1 and for face 1 it is +1. This way we can easily write
// the mass conservation as sum(s_i * A_i * U_i) = 0 where s_i is the
// orientation.
template <int dim, int spacedim>
struct JunctionFace
{
  typename DoFHandler<dim, spacedim>::active_cell_iterator cell;
  unsigned int                                             face_no;
  int orientation; // +1 if outflow (face 1), -1 if inflow (face 0)
};

template <int dim, int spacedim>
struct JunctionInfo
{
  Point<spacedim>                          point;
  std::vector<JunctionFace<dim, spacedim>> faces;
  unsigned int
  n_vessels() const
  {
    return faces.size();
  }
};

// ======================================================
// Generic N-junction Newton Solver
// ======================================================
template <typename Physics>
class JunctionSolver
{
public:
  // Add dim and spacedim here, or template the function itself:
  template <int dim, int spacedim>
  JunctionState
  solve(const JunctionState               &X0,
        const std::vector<double>         &W_in,
        const JunctionInfo<dim, spacedim> &junction,
        const Physics                     &phys) const
  {
    const unsigned int K = junction.faces.size();
    AssertDimension(X0.X.size(), 2 * K);
    AssertDimension(W_in.size(), K);

    JunctionState X = X0;

    Vector<double>     R(2 * K);
    Vector<double>     dX(2 * K);
    FullMatrix<double> J(2 * K, 2 * K);

    constexpr unsigned int max_iter = 25;
    constexpr double       tol      = 1e-8;
    constexpr double       Amin     = 1e-10;

    for (unsigned int it = 0; it < max_iter; ++it)
      {
        // --------------------------------------------------
        // Residual: R_J(X, W_in) = 0
        // --------------------------------------------------
        phys.compute_junction_residual(X, W_in, junction, R);

        if (R.l2_norm() < tol)
          return X;

        // --------------------------------------------------
        // Jacobian: ∂R_J / ∂X
        // --------------------------------------------------
        phys.compute_junction_jacobian(X, junction, J);

        // --------------------------------------------------
        // Solve linear system J * dX = -R
        // --------------------------------------------------
        R *= -1.0;
        J.gauss_jordan();
        J.vmult(dX, R);

        // --------------------------------------------------
        // Line-search Newton update (positivity preserving)
        // --------------------------------------------------
        double        alpha = 1.0;
        JunctionState X_trial;
        X_trial.X.reinit(2 * K);

        while (alpha > 1e-6)
          {
            X_trial.X = X.X;

            for (unsigned int i = 0; i < K; ++i)
              {
                X_trial.A(i) += alpha * dX[2 * i];
                X_trial.U(i) += alpha * dX[2 * i + 1];
              }

            bool admissible = true;
            for (unsigned int i = 0; i < K; ++i)
              if (X_trial.A(i) <= Amin)
                {
                  admissible = false;
                  break;
                }

            if (admissible)
              break;

            alpha *= 0.5;
          }

        X = X_trial;
      }

    // If Newton does not converge, return last iterate
    return X;
  }
};

#endif
