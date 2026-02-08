#ifndef JUNCTION_SOLVER_H
#define JUNCTION_SOLVER_H

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/vector.h>

using namespace dealii;

// ======================================================
// Junction state (6 unknowns)
// ======================================================
struct JunctionState
{
  double Ap, Up;
  double Ad1, Ud1;
  double Ad2, Ud2;
};

struct JunctionInfo
{
  Point<3> point; // physical junction location

  struct FaceData
  {
    typename DoFHandler<1, 3>::active_cell_iterator cell;
    unsigned int                                    face_no;
  };

  FaceData              parent;
  std::vector<FaceData> daughters; // size = 2 for Y-junction
  std::set<std::pair<CellId, unsigned int>> junction_faces;
};


// ======================================================
// Generic junction solver (ROBUST VERSION)
// ======================================================
template <typename Physics>
class JunctionSolver
{
public:
  JunctionState
  solve(const JunctionState &X0,
        const double         Wp_minus,
        const double         Wd1_plus,
        const double         Wd2_plus,
        const Physics       &phys)
  {
    Vector<double>     R(6), dX(6);
    FullMatrix<double> J(6, 6);

    JunctionState X = X0;

    constexpr unsigned int max_iter = 25;
    constexpr double       tol      = 1e-8;
    constexpr double       Amin     = 1e-8;

    for (unsigned int it = 0; it < max_iter; ++it)
    {
      // --------------------------------------------------
      // Compute residual of junction equations (29–34)
      // --------------------------------------------------
      phys.compute_junction_residual(X, Wp_minus, Wd1_plus, Wd2_plus, R);

      if (R.l2_norm() < tol)
        return X;

      // --------------------------------------------------
      // Compute Jacobian of junction system (6x6)
      // --------------------------------------------------
      phys.compute_junction_jacobian(X, J);

      // --------------------------------------------------
      // Solve J * dX = -R
      // --------------------------------------------------
      R *= -1.0;
      J.gauss_jordan();
      J.vmult(dX, R);

      // --------------------------------------------------
      // Line-search Newton update (positivity preserving)
      // --------------------------------------------------
      double alpha = 1.0;
      JunctionState X_trial;

      while (alpha > 1e-6)
      {
        X_trial = X;

        X_trial.Ap  += alpha * dX[0];
        X_trial.Up  += alpha * dX[1];
        X_trial.Ad1 += alpha * dX[2];
        X_trial.Ud1 += alpha * dX[3];
        X_trial.Ad2 += alpha * dX[4];
        X_trial.Ud2 += alpha * dX[5];

        // Enforce physical admissibility
        if (X_trial.Ap  > Amin &&
            X_trial.Ad1 > Amin &&
            X_trial.Ad2 > Amin)
          break;

        alpha *= 0.5;
      }

      // Update iterate
      X = X_trial;
    }

    // If Newton does not converge, return last iterate
    return X;
  }
};

#endif
