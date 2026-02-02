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
// Generic junction solver
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

    constexpr unsigned int max_iter = 20;
    constexpr double       tol      = 1e-6;

    for (unsigned int it = 0; it < max_iter; ++it)
      {
        phys.compute_junction_residual(X, Wp_minus, Wd1_plus, Wd2_plus, R);
        if (R.l2_norm() < tol)
          break;

        phys.compute_junction_jacobian(X, J);
        J.gauss_jordan();
        J.vmult(dX, R);
       
        const double omega1 = 0.5;
        X.Ap -= omega1 *dX[0];
        X.Up -= omega1 * dX[1];
        X.Ad1 -= omega1* dX[2];
        X.Ud1 -=  omega1 *dX[3];
        X.Ad2 -=  omega1 *dX[4];
        X.Ud2 -= omega1 * dX[5];
      }

    return X;
  }
};

#endif
