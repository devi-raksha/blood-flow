#ifndef BLOOD_FLOW_SYSTEM_UPDATED_1D3D_H
#define BLOOD_FLOW_SYSTEM_UPDATED_1D3D_H

#include <deal.II/base/function.h>
#include <deal.II/base/function_parser.h>
#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/parsed_function.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_interface_values.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_values_extractors.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <memory>

#include "constants.h"
#include "function.h"
#include "junction_solver.h"

using namespace dealii;


// // ========================================================================
// // HELPER FUNCTION: Detect Non-Manifold Faces (Y-Junctions)
// // ========================================================================

// template <typename CellContainer>
// std::map<typename CellContainer::active_face_iterator,
//          std::vector<typename CellContainer::active_cell_iterator>>
// get_non_manifold_faces(const CellContainer &cell_container)
// {
//   // Map from face to list of cells touching that face
//   std::map<typename CellContainer::active_face_iterator,
//            std::vector<typename CellContainer::active_cell_iterator>>
//     face_to_cells;

//   // Loop over all cells and all faces
//   for (const auto &cell : cell_container.active_cell_iterators())
//     for (const auto f : cell->face_indices())
//       face_to_cells[cell->face(f)].push_back(cell);

//   // Remove all faces with only 2 neighbors (manifold faces)
//   // Keep only faces with > 2 neighbors (non-manifold = junctions!)
//   for (auto it = face_to_cells.begin(); it != face_to_cells.end();)
//     {
//       if (it->second.size() <= 2)
//         it = face_to_cells.erase(it);
//       else
//         ++it;
//     }

//   return face_to_cells;
// }


//====================================================================
// PHYSICAL CONSTANTS AND PARAMETERS
//====================================================================
using BloodFlowParameters = ParsedTools::Constants;

//====================================================================
// SCRATCH AND COPY DATA STRUCTURES
//====================================================================

/**
 * Scratch data for MeshWorker
 */
template <int dim, int spacedim>
struct BloodFlowScratchData
{
  BloodFlowScratchData(
    const FiniteElement<dim, spacedim> &fe,
    const Quadrature<dim>              &quadrature,
    const Quadrature<dim - 1>          &quadrature_face,
    const UpdateFlags update_flags = update_values | update_gradients |
                                     update_quadrature_points |
                                     update_JxW_values,
    const UpdateFlags interface_update_flags = update_values |
                                               update_gradients |
                                               update_quadrature_points |
                                               update_JxW_values |
                                               update_normal_vectors)
    : fe_values(fe, quadrature, update_flags)
    , fe_interface_values(fe, quadrature_face, interface_update_flags)
  {}

  BloodFlowScratchData(const BloodFlowScratchData<dim, spacedim> &scratch_data)
    : fe_values(scratch_data.fe_values.get_fe(),
                scratch_data.fe_values.get_quadrature(),
                scratch_data.fe_values.get_update_flags())
    , fe_interface_values(scratch_data.fe_interface_values.get_fe(),
                          scratch_data.fe_interface_values.get_quadrature(),
                          scratch_data.fe_interface_values.get_update_flags())
  {}

  FEValues<dim, spacedim>          fe_values;
  FEInterfaceValues<dim, spacedim> fe_interface_values;
};

/**
 * Copy data structures for MeshWorker
 */
struct BloodFlowCopyDataFace
{
  FullMatrix<double>                   cell_matrix;
  Vector<double>                       cell_rhs;
  std::vector<types::global_dof_index> joint_dof_indices;
};

struct BloodFlowCopyData
{
  FullMatrix<double>                   cell_matrix;
  Vector<double>                       cell_rhs;
  std::vector<types::global_dof_index> local_dof_indices;
  std::vector<BloodFlowCopyDataFace>   face_data;

  template <class Iterator>
  void
  reinit(const Iterator &cell, unsigned int dofs_per_cell)
  {
    cell_matrix.reinit(dofs_per_cell, dofs_per_cell);
    cell_rhs.reinit(dofs_per_cell);
    local_dof_indices.resize(dofs_per_cell);
    cell->get_dof_indices(local_dof_indices);
  }
};

//====================================================================
// MAIN CLASS DECLARATION
//====================================================================

template <int dim, int spacedim = dim>
class BloodFlowSystem : public ParameterAcceptor
{
public:
  BloodFlowSystem();

  void
  initialize_params(const std::string &filename = "");
  void
  setup_system();
  void
  assemble_system();
  void
  assemble_junction_terms();
  void
  assemble_mass_matrix();
  void
  solve();

  void
  output_results(const unsigned int cycle) const;
  void
  compute_pressure();
  void
  compute_errors(unsigned int k);
  void
  check_singular_rows() const;
  void
  run_convergence_study();
  void
  detect_bifurcation_junctions();

  // Junction physics (called by JunctionSolver) ---
  void
  compute_junction_residual(const JunctionState &X,
                            double               Wp_minus,
                            double               Wd1_plus,
                            double               Wd2_plus,
                            Vector<double>      &R) const;

  void
  compute_junction_jacobian(const JunctionState &X,
                            FullMatrix<double>  &J) const;

private:
  ParsedTools::Constants    par;
  AffineConstraints<double> constraints;

  using CellIterator = typename DoFHandler<dim, spacedim>::active_cell_iterator;

  struct JunctionCell
  {
    CellIterator cell;
    unsigned int vertex;
  };

  std::vector<JunctionInfo>                      junctions;
  std::set<std::pair<CellId, unsigned int>>      all_junction_faces;
  JunctionSolver<BloodFlowSystem<dim, spacedim>> junction_solver;

  // to know junction faces
  inline bool
  is_junction_face(const CellId &cell_id, const unsigned int face_no) const
  {
    return all_junction_faces.count({cell_id, face_no}) > 0;
  }


  // Embedded 1D-3D network coupling
  struct EmbeddedVessel
  {
    std::vector<Point<3>> centerline;      // 1D vessel path
    std::vector<double>   radius;          // Vessel radius at each point
    unsigned int          material_id = 0; // Vessel properties
  };

  std::vector<EmbeddedVessel> embedded_vessels;
  std::vector<Point<3>>       hypersingular_points; // Key coupling points

  // --------------------------------------------------
  // ===  Physical and Constitutive Relations  ===
  // --------------------------------------------------

  /**
   * Compute wave speed using the tube law
   */
  double
  compute_wave_speed(const double area) const
  {
    const double eps    = 0;
    const double A_safe = std::max(area, eps);
    const double dpda   = compute_pressure_derivative(area);
    return std::sqrt(A_safe / par["rho"] * dpda);
  }

  /**
   * Inverse function to compute area from wave speed (Newton method)
   */

  double
  inverse_compute_area_from_c(double c_target) const
  {
    double A = par["a0"]; // initial guess

    for (unsigned int iter = 0; iter < 20; ++iter)
      {
        double dpdA = compute_pressure_derivative(A);
        double cA   = std::sqrt(A / par["rho"] * dpdA);

        double f = cA - c_target;
        if (std::abs(f) < 1e-12)
          return A;

        // approximate derivative: dc/dA ≈ dpdA / (2*cA*rho)
        double df_dA = dpdA / (2.0 * cA * par["rho"]);

        // Newton update
        A = A - f / df_dA;

        if (A < 1e-12)
          A = 1e-12;
      }

    return A;
  }

  /**
   * compute area from pressure using tube law formula
   */

  double
  compute_area_from_pressure(const double P) const
  {
    const double m  = par["m"];
    const double mu = par["mu"];
    const double p0 = par["p0"];
    const double A0 = par["a0"];

    const double ratio = std::max((P - p0) / mu + 1.0, 1e-12);
    return A0 * std::pow(ratio, 1.0 / m);
  }

  /**
   * Compute pressure using the shifted tube law
   */
  double
  compute_pressure_value(const double area) const
  {
    // const double A_safe = std::max(area,  1e-2);
    const double eps   = 0; // to avoid zero pressure at zero area
    const double ratio = area / par["a0"] + eps;
    const double m     = par["m"];

    return par["mu"] * (std::pow(ratio + eps, m) - 1.0) + par["p0"];
  }

  /**
   * Compute pressure derivative dP/dA for shifted tube law
   */
  double
  compute_pressure_derivative(const double area) const
  {
    const double eps   = 0; // to avoid zero division
    const double ratio = std::max(area / par["a0"], eps);
    const double m     = par["m"];
    return par["mu"] * m * std::pow(ratio, m - 1.0) / par["a0"];
  }

  double
  compute_LF_penalty(const double area_L,
                     const double area_R,
                     const double U_L,
                     const double U_R,
                     const double bn,
                     const double bn_neighbor) const
  {
    const double cL = compute_wave_speed(area_L);
    const double cR = compute_wave_speed(area_R);

    // Multiply by bn (directional scaling)
    const double lambda1_L = (U_L - cL) * bn;
    const double lambda2_L = (U_L + cL) * bn;
    const double lambda1_R = (U_R - cR) * bn_neighbor;
    const double lambda2_R = (U_R + cR) * bn_neighbor;

    return std::max({std::abs(lambda1_L),
                     std::abs(lambda2_L),
                     std::abs(lambda1_R),
                     std::abs(lambda2_R)});
  }


  // --------------------------------------------------
  // ===  Geometry and Flux helper functions  ===
  // --------------------------------------------------

  /**
   * Compute the unit tangent vector b along the 1D vessel axis.
   */
  Tensor<1, spacedim>
  compute_directional_vector(
    const typename DoFHandler<dim, spacedim>::active_cell_iterator &cell) const
  {
    return (cell->vertex(1) - cell->vertex(0)) /
           cell->vertex(1).distance(cell->vertex(0));
  }

  /**
   * Compute the physical flux for the area equation:
   *   F_A = A * U * b
   */
  Tensor<1, spacedim>
  compute_physical_area_flux(
    const typename DoFHandler<dim, spacedim>::active_cell_iterator &cell,
    const double implicit_area,
    const double explicit_velocity) const
  {
    const Tensor<1, spacedim> b = compute_directional_vector(cell);
    return implicit_area * explicit_velocity * b;
  }

  /**
   * Compute the physical flux for the jacobian area equation:
   *   F_A =  (current_velocity[point] * trial_A +
                  current_area[point] * trial_U) * b;
   */
  Tensor<1, spacedim>
  compute_physical_area_jacobian_flux(
    const typename DoFHandler<dim, spacedim>::active_cell_iterator &cell,
    const double current_area,
    const double trial_velocity,
    const double current_velocity,
    const double trial_area) const
  {
    const Tensor<1, spacedim> b = compute_directional_vector(cell);
    return (current_area * trial_velocity + current_velocity * trial_area) * b;
  }

  /**
   * Compute the physical flux for the momentum equation:
   *   F_U = 0.5 * U * U * b + P/rho * b
   * (extend later with +P(A)*b if needed)
   */
  Tensor<1, spacedim>
  compute_physical_momentum_flux(
    const typename DoFHandler<dim, spacedim>::active_cell_iterator &cell,
    const double implicit_velocity,
    const double explicit_velocity,
    const double pressure) const
  {
    const Tensor<1, spacedim> b = compute_directional_vector(cell);
    return (0.5 * implicit_velocity * explicit_velocity +
            pressure / par["rho"]) *
           b;
  }

  /**
   * Compute the physical flux for the jacobian area equation:
   *   F_u =  ((c^2/A)*trial_A + U*trial_U)*b;
   */
  Tensor<1, spacedim>
  compute_physical_momentum_jacobian_flux(
    const typename DoFHandler<dim, spacedim>::active_cell_iterator &cell,
    const double                                                    c_squared,
    const double current_area,
    const double trial_area,
    const double current_velocity,
    const double trial_velocity) const
  {
    AssertThrow(current_area > 0,
                ExcMessage("Area must be positive to compute flux."));
    const Tensor<1, spacedim> b = compute_directional_vector(cell);
    return (c_squared / current_area * trial_area +
            current_velocity * trial_velocity) *
           b;
  }

  /**
   * Compute tangent-normal projection: (b · n)
   */
  double
  compute_tangent_normal_product(
    const typename DoFHandler<dim, spacedim>::active_cell_iterator &cell,
    const Tensor<1, spacedim> &normal) const
  {
    const Tensor<1, spacedim> b = compute_directional_vector(cell);
    return b * normal;
  }

  /**
   * Compute HLL flux at interface for residual computation
   */
  std::array<double, 2>
  HLL_residual_flux(
    const typename DoFHandler<dim, spacedim>::active_cell_iterator &cell,
    const typename DoFHandler<dim, spacedim>::active_cell_iterator &ncell,
    double                                                          A_L,
    double                                                          U_L,
    double                                                          A_R,
    double                                                          U_R,
    const Tensor<1, spacedim> &normal) const
  {
    // wave speeds
    double c_L = compute_wave_speed(A_L);
    double c_R = compute_wave_speed(A_R);

    double s_L = std::min(U_L - c_L, U_R - c_R);
    double s_R = std::max(U_L + c_L, U_R + c_R);

    // physical fluxes projected on normal
    double FAL = compute_physical_area_flux(cell, A_L, U_L) * normal;
    double FUL = compute_physical_momentum_flux(cell,
                                                U_L,
                                                U_L,
                                                compute_pressure_value(A_L)) *
                 normal;

    double FAR = compute_physical_area_flux(ncell, A_R, U_R) * normal;
    double FUR = compute_physical_momentum_flux(ncell,
                                                U_R,
                                                U_R,
                                                compute_pressure_value(A_R)) *
                 normal;

    // Case 1: all waves go right
    if (s_L >= 0.0)
      return {{FAL, FUL}};

    // Case 2: all waves go left
    if (s_R <= 0.0)
      return {{FAR, FUR}};

    // Case 3: mixed — HLL formula
    double inv = 1.0 / (s_R - s_L);

    double FHLL_A = (s_R * FAL - s_L * FAR + s_R * s_L * (A_R - A_L)) * inv;
    double FHLL_U = (s_R * FUL - s_L * FUR + s_R * s_L * (U_R - U_L)) * inv;

    return {{FHLL_A, FHLL_U}};
  }


  /**
   * Compute HLL flux at interface for JACOBIAN computation
   * For jacobian: linearized fluxes using current state
   */
  std::array<double, 2>
  HLL_jacobian_flux(
    const typename DoFHandler<dim, spacedim>::active_cell_iterator &cell,
    const typename DoFHandler<dim, spacedim>::active_cell_iterator &ncell,
    double                                                          A_L,
    double                                                          U_L,
    double                                                          A_R,
    double                                                          U_R,
    double                                                          trial_A_L,
    double                                                          trial_U_L,
    double                                                          trial_A_R,
    double                                                          trial_U_R,
    const Tensor<1, spacedim> &normal) const
  {
    // wave speeds at current state
    double c_L = compute_wave_speed(A_L);
    double c_R = compute_wave_speed(A_R);

    double s_L = std::min(U_L - c_L, U_R - c_R);
    double s_R = std::max(U_L + c_L, U_R + c_R);

    // Jacobian of area flux: ∂(A*U*b)/∂trial = (trial_U*A + trial_A*U)*b
    double FAL_jac = compute_physical_area_jacobian_flux(
                       cell, A_L, trial_U_L, U_L, trial_A_L) *
                     normal;

    // Jacobian of momentum flux: ∂(0.5*U²+P/ρ)*b)/∂trial
    double c_L_sq  = c_L * c_L;
    double FUL_jac = compute_physical_momentum_jacobian_flux(
                       cell, c_L_sq, A_L, trial_A_L, U_L, trial_U_L) *
                     normal;

    double FAR_jac = compute_physical_area_jacobian_flux(
                       ncell, A_R, trial_U_R, U_R, trial_A_R) *
                     normal;

    double c_R_sq  = c_R * c_R;
    double FUR_jac = compute_physical_momentum_jacobian_flux(
                       ncell, c_R_sq, A_R, trial_A_R, U_R, trial_U_R) *
                     normal;

    // Case 1: all waves go right
    if (s_L >= 0.0)
      return {{FAL_jac, FUL_jac}};

    // Case 2: all waves go left
    if (s_R <= 0.0)
      return {{FAR_jac, FUR_jac}};

    // Case 3: mixed — HLL formula with jacobian
    double inv = 1.0 / (s_R - s_L);

    double FHLL_A_jac =
      (s_R * FAL_jac - s_L * FAR_jac + s_R * s_L * (trial_A_R - trial_A_L)) *
      inv;
    double FHLL_U_jac =
      (s_R * FUL_jac - s_L * FUR_jac + s_R * s_L * (trial_U_R - trial_U_L)) *
      inv;

    return {{FHLL_A_jac, FHLL_U_jac}};
  }


  //   private:
  // Mesh and finite elements
  Triangulation<dim, spacedim>                  triangulation;
  DoFHandler<dim, spacedim>                     dof_handler;
  std::unique_ptr<FiniteElement<dim, spacedim>> fe;

  // Linear system
  SparsityPattern      sparsity_pattern;
  SparseMatrix<double> jacobian_matrix;
  SparseMatrix<double> mass_matrix;
  Vector<double>       solution;
  Vector<double>       solution_old;
  Vector<double>       pressure;
  Vector<double>       tmp_vector;
  std::vector<double>  mass_residual_at_junction;

  Vector<double> residual_vector;
  Vector<double> newton_update;
  // Parameters
  unsigned int fe_degree              = 1;
  std::string  constants              = "1.0";
  std::string  output_filename        = "solution";
  bool         use_direct_solver      = true;
  bool         use_riemann_invariants = false;
  bool         use_junction_mesh      = false;
  unsigned int n_refinement_cycles    = 1;
  unsigned int n_global_refinements   = 5;

  // Time stepping parameters
  double       time_step    = 0.01;
  double       final_time   = 1.0;
  double       time         = 0.0;
  unsigned int n_time_steps = 0;

  // Numerical parameters
  double omega    = 1;
  double theta    = 0.5;
  double theta_bd = 0.5;


  // Newton iteration parameters
  unsigned int max_newton_iterations = 20;
  double       newton_tolerance      = 1e-8;


  // Allow access to private members for all free functions whose name
  // contains test
  friend void
  test();


  ParsedTools::Function<spacedim> initial_condition;
  ParsedTools::Function<spacedim> rhs_function;
  ParsedTools::Function<spacedim> exact_solution;
};

#endif // BLOOD_FLOW_SYSTEM_UPDATED_1D3D_H