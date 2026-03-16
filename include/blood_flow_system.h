#ifndef BLOOD_FLOW_SYSTEM_H
#define BLOOD_FLOW_SYSTEM_H

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

#include <deal.II/sundials/arkode.h>

#include <iosfwd>
#include <memory>

#include "constants.h"
#include "function.h"
#include "junction_solver.h"

using namespace dealii;

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
  detect_junctions();
  void
  initialize_terminal_capacitors();

  void
  create_vascular_network();
  void
  setup_system();
  void
  assemble_jacobian(const double          t,
                    const Vector<double> &y,
                    const Vector<double> &Mydot);
  void
  assemble_junction_jacobian_blocks(const Vector<double> &y);
  void
  assemble_implicit_function(const double          t,
                             const Vector<double> &y,
                             Vector<double>       &Mydot);
  void
  assemble_junction_residual(const Vector<double> &y, Vector<double> &Mydot);
  void
  assemble_mass_matrix();
  void
  compute_initial_solution(Vector<double> &dst, const double t);

  void
  output_results(const Vector<double> &y,
                 const Vector<double> &pressure_vec,
                 const unsigned int    cycle) const;


  void
  compute_pressure(const Vector<double> &y, Vector<double> &pressure_vec) const;
  void
  compute_errors(unsigned int k);
  void
  run();

  enum class NumericalFluxType
  {
    HLL,
    LAX_FRIEDRICHS,
    HLL_SYMPY
  };
  void
  set_numerical_flux(NumericalFluxType flux_type)
  {
    numerical_flux_type = flux_type;
  }

  NumericalFluxType
  get_numerical_flux() const
  {
    return numerical_flux_type;
  }

  void
  compute_junction_residual(const JunctionState               &X,
                            const std::vector<double>         &W_in,
                            const JunctionInfo<dim, spacedim> &junction,
                            Vector<double>                    &R) const;

  void
  compute_junction_jacobian(const JunctionState               &X,
                            const JunctionInfo<dim, spacedim> &junction,
                            FullMatrix<double>                &J) const;


private:
  ParsedTools::Constants    par;
  AffineConstraints<double> constraints;

  // Key: Boundary ID, Value: Pressure at the previous time step
  // One capacitor pressure per terminal boundary
  std::map<types::boundary_id, double> terminal_Pc_storage;
  // std::map<std::pair<CellId, unsigned int>, double> terminal_Pc_storage;
  std::set<dealii::types::boundary_id> terminal_boundary_ids;
  using CellIterator = typename DoFHandler<dim, spacedim>::active_cell_iterator;

  std::vector<JunctionInfo<dim, spacedim>>       junctions;
  std::set<std::pair<CellId, unsigned int>>      all_junction_faces;
  JunctionSolver<BloodFlowSystem<dim, spacedim>> junction_solver;


  inline bool
  is_junction_face(const CellId &cell_id, const unsigned int face_no) const
  {
    // Explicitly using std::make_pair resolves initializer_list ambiguity
    return all_junction_faces.count(std::make_pair(cell_id, face_no)) > 0;
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


  SUNDIALS::ARKode<Vector<double>>::AdditionalData arkode_parameters;
  NumericalFluxType numerical_flux_type     = NumericalFluxType::HLL;
  std::string       numerical_flux_type_str = "HLL";
  /**
   * Wrapper for residual flux - selects HLL or Lax-Friedrichs
   */
  std::array<double, 2>
  numerical_flux(double bn_L,
                 double bn_R,
                 double A_L,
                 double U_L,
                 double A_R,
                 double U_R) const
  {
    if (numerical_flux_type == NumericalFluxType::HLL)
      return hll_numerical_flux(bn_L, bn_R, A_L, U_L, A_R, U_R);
    else if (numerical_flux_type == NumericalFluxType::HLL_SYMPY)
      return hll_sympy_numerical_flux(bn_L, bn_R, A_L, U_L, A_R, U_R);
    else
      return lf_numerical_flux(bn_L, bn_R, A_L, U_L, A_R, U_R);
  }

  /**
   * Wrapper for Jacobian flux - selects HLL or Lax-Friedrichs
   */
  std::array<double, 2>
  numerical_flux_jacobian(double bn_L,
                          double bn_R,
                          double A_L,
                          double U_L,
                          double A_R,
                          double U_R,
                          double trial_A_L,
                          double trial_U_L,
                          double trial_A_R,
                          double trial_U_R) const
  {
    if (numerical_flux_type == NumericalFluxType::HLL)
      return hll_numerical_flux_jacobian(bn_L,
                                         bn_R,
                                         A_L,
                                         U_L,
                                         A_R,
                                         U_R,
                                         trial_A_L,
                                         trial_U_L,
                                         trial_A_R,
                                         trial_U_R);
    else if (numerical_flux_type == NumericalFluxType::HLL_SYMPY)
      return hll_sympy_numerical_flux_jacobian(bn_L,
                                               bn_R,
                                               A_L,
                                               U_L,
                                               A_R,
                                               U_R,
                                               trial_A_L,
                                               trial_U_L,
                                               trial_A_R,
                                               trial_U_R);
    else
      return lf_numerical_flux_jacobian(bn_L,
                                        bn_R,
                                        A_L,
                                        U_L,
                                        A_R,
                                        U_R,
                                        trial_A_L,
                                        trial_U_L,
                                        trial_A_R,
                                        trial_U_R);
  }

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
   * Compute wave speed derivative using the tube law
   */
  double
  compute_wave_speed_derivative(const double area) const
  {
    const double eps    = 1e-10;
    const double A_safe = std::max(area, eps);
    return compute_wave_speed(A_safe) * par["m"] / (2.0 * A_safe);
  }

  /**
   * Compute pressure using the shifted tube law
   */
  // double
  // compute_pressure_value(const double area) const
  // {
  //   // const double A_safe = std::max(area,  1e-2);
  //   const double eps   = 0; // to avoid zero pressure at zero area
  //   const double ratio = area / par["a0"] + eps;
  //   const double m     = par["m"];

  //   return par["mu"] * (std::pow(ratio + eps, m) - 1.0) + par["p0"];
  // }


  /**
   * Compute beta_p which reflects the material properties for shifted tube law
   */
  double
  compute_beta_p(const double E, const double h_wall) const
  {
    return (4.0 * std::sqrt(3.14159) / 3.0) * E * h_wall;
  }

  double
  compute_pressure_value(double A) const
  {
    const double A0   = par["a0"];
    const double beta = compute_beta_p(par["E"], par["h_wall"]);

    return par["p0"] + beta / A0 * (std::sqrt(A) - std::sqrt(A0)) + par["p_d"];
  }

  /**
   * Compute pressure derivative dP/dA for shifted tube law
   */

  double
  compute_pressure_derivative(const double area) const
  {
    const double beta_p =
      compute_beta_p(par["E"], par["h_wall"]); // Example wall thickness
    return beta_p / (2.0 * std::sqrt(area) * par["a0"]);
  }



  // /**
  //  * Compute pressure derivative dP/dA for shiftyed tube law
  //  */
  // double
  // compute_pressure_derivative(const double area) const
  // {
  //   const double eps   = 0; // to avoid zero division
  //   const double ratio = std::max(area / par["a0"], eps);
  //   const double m     = par["m"];
  //   return par["E"] * m * std::pow(ratio, m - 1.0) / par["a0"];
  // }

  double
  compute_LF_penalty(const double area_L,
                     const double area_R,
                     const double U_L,
                     const double U_R,
                     const double bn_L,
                     const double bn_R) const
  {
    const double cL = compute_wave_speed(area_L);
    const double cR = compute_wave_speed(area_R);

    // Multiply by bn (directional scaling)
    const double lambda1_L = (U_L - cL) * bn_L;
    const double lambda2_L = (U_L + cL) * bn_L;
    const double lambda1_R = (U_R - cR) * bn_R;
    const double lambda2_R = (U_R + cR) * bn_R;

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

  // For scalar projection of physcial fluxes on normals we create another
  // functions

  /**
   * Compute scalar physical flux for the area equation:
   *   F_A = A * U * bn
   */
  double
  compute_scalar_physical_area_flux(double       bn,
                                    const double implicit_area,
                                    const double explicit_velocity) const
  {
    return implicit_area * explicit_velocity * bn;
  }

  /**
   * Compute the scalar physical flux for the jacobian area equation:
   *   F_A =  (current_velocity[point] * trial_A +
                  current_area[point] * trial_U) * bn;
   */
  double
  compute_scalar_physical_area_jacobian_flux(double       bn,
                                             const double current_area,
                                             const double trial_velocity,
                                             const double current_velocity,
                                             const double trial_area) const
  {
    return (current_area * trial_velocity + current_velocity * trial_area) * bn;
  }

  /**
   * Compute the physical flux for the momentum equation:
   *   F_U = 0.5 * U * U * b + P/rho * b
   * (extend later with +P(A)*b if needed)
   */
  double
  compute_scalar_physical_momentum_flux(double       bn,
                                        const double implicit_velocity,
                                        const double explicit_velocity,
                                        const double pressure) const
  {
    return (0.5 * implicit_velocity * explicit_velocity +
            pressure / par["rho"]) *
           bn;
  }

  /**
   * Compute the scalar physical flux for the jacobian area equation:
   *   F_u =  ((c^2/A)*trial_A + U*trial_U)*bn;
   */
  double
  compute_scalar_physical_momentum_jacobian_flux(
    double       bn,
    const double c_squared,
    const double current_area,
    const double trial_area,
    const double current_velocity,
    const double trial_velocity) const
  {
    AssertThrow(current_area > 0,
                ExcMessage("Area must be positive to compute flux."));

    return (c_squared / current_area * trial_area +
            current_velocity * trial_velocity) *
           bn;
  }


  /**
   * compute the Lax-Friedrich flux for residual computation
   */

  std::array<double, 2>
  lf_numerical_flux(double bn_L,
                    double bn_R,
                    double A_L,
                    double U_L,
                    double A_R,
                    double U_R) const
  {
    // physical fluxes projected on normal
    double FAL = compute_scalar_physical_area_flux(bn_L, A_L, U_L);
    double FUL = compute_scalar_physical_momentum_flux(
      bn_L, U_L, U_L, compute_pressure_value(A_L));


    double FAR = compute_scalar_physical_area_flux(bn_R, A_R, U_R);
    double FUR = compute_scalar_physical_momentum_flux(
      bn_R, U_R, U_R, compute_pressure_value(A_R));

    double beta  = compute_LF_penalty(A_L, A_R, U_L, U_R, bn_L, bn_R);
    double alpha = theta * beta;
    // LF flux
    double FLF_A = 0.5 * (FAL + FAR) - 0.5 * alpha * (A_R - A_L);
    double FLF_U = 0.5 * (FUL + FUR) - 0.5 * alpha * (U_R - U_L);

    return {{FLF_A, FLF_U}};
  }

  /**
   * compute the Lax-Friedrich flux for JACOBIAN computation
   * For jacobian: linearized fluxes using current state
   */
  std::array<double, 2>
  lf_numerical_flux_jacobian(double bn_L,
                             double bn_R,
                             double A_L,
                             double U_L,
                             double A_R,
                             double U_R,
                             double trial_A_L,
                             double trial_U_L,
                             double trial_A_R,
                             double trial_U_R) const
  {
    // Jacobian of physical fluxes projected on normal
    double FAL_jac = compute_scalar_physical_area_jacobian_flux(
      bn_L, A_L, trial_U_L, U_L, trial_A_L);
    double c_L     = compute_wave_speed(A_L);
    double c_L_sq  = c_L * c_L;
    double FUL_jac = compute_scalar_physical_momentum_jacobian_flux(
      bn_L, c_L_sq, A_L, trial_A_L, U_L, trial_U_L);
    double FAr_jac = compute_scalar_physical_area_jacobian_flux(
      bn_R, A_R, trial_U_R, U_R, trial_A_R);
    double c_R     = compute_wave_speed(A_R);
    double c_R_sq  = c_R * c_R;
    double FUR_jac = compute_scalar_physical_momentum_jacobian_flux(
      bn_R, c_R_sq, A_R, trial_A_R, U_R, trial_U_R);
    double beta = compute_LF_penalty(A_L, A_R, U_L, U_R, bn_L, bn_R);

    double alpha = theta * beta;
    // LF flux jacobian
    double FLF_A_jac =
      0.5 * (FAL_jac + FAr_jac) - 0.5 * alpha * (trial_A_R - trial_A_L);
    double FLF_U_jac =
      0.5 * (FUL_jac + FUR_jac) - 0.5 * alpha * (trial_U_R - trial_U_L);
    return {{FLF_A_jac, FLF_U_jac}};
  }

  std::array<double, 2>
  hll_sympy_numerical_flux(double bn_L,
                           double bn_R,
                           double A_L,
                           double U_L,
                           double A_R,
                           double U_R) const
  {
    double FHLL_A, FHLL_U;
    // FHLL_A
    {
      const double t_0  = A_L * U_L * bn_L;
      const double t_1  = 1.0 / par["a0"];
      const double t_2  = par["m"] * par["mu"] / par["rho"];
      const double t_3  = std::sqrt(t_2 * std::pow(A_L * t_1, par["m"]));
      const double t_4  = (1.0 / 2.0) * t_3;
      const double t_5  = std::sqrt(t_2 * std::pow(A_R * t_1, par["m"]));
      const double t_6  = (1.0 / 2.0) * t_5;
      const double t_7  = (1.0 / 2.0) * U_L + (1.0 / 2.0) * U_R;
      const double t_8  = -t_4 - t_6 + t_7;
      const double t_9  = A_R * U_R * bn_R;
      const double t_10 = t_4 + t_6 + t_7;
      FHLL_A =
        ((t_8 >= 0) ? (t_0) :
                      ((t_10 <= 0) ?
                         (t_9) :
                         ((t_0 * t_10 + t_10 * t_8 * (-A_L + A_R) - t_8 * t_9) /
                          (t_3 + t_5))));
    }

    // FHLL_U
    {
      const double t_0  = 1.0 / par["rho"];
      const double t_1  = 1.0 / par["a0"];
      const double t_2  = std::pow(A_L * t_1, par["m"]);
      const double t_3  = bn_L * (0.5 * std::pow(U_L, 2) +
                                 t_0 * (par["mu"] * (t_2 - 1) + par["p0"]));
      const double t_4  = par["m"] * par["mu"] * t_0;
      const double t_5  = std::sqrt(t_2 * t_4);
      const double t_6  = (1.0 / 2.0) * t_5;
      const double t_7  = std::pow(A_R * t_1, par["m"]);
      const double t_8  = std::sqrt(t_4 * t_7);
      const double t_9  = (1.0 / 2.0) * t_8;
      const double t_10 = (1.0 / 2.0) * U_L + (1.0 / 2.0) * U_R;
      const double t_11 = t_10 - t_6 - t_9;
      const double t_12 = bn_R * (0.5 * std::pow(U_R, 2) +
                                  t_0 * (par["mu"] * (t_7 - 1) + par["p0"]));
      const double t_13 = t_10 + t_6 + t_9;
      FHLL_U            = ((t_11 >= 0) ?
                             (t_3) :
                             ((t_13 <= 0) ?
                                (t_12) :
                                ((-t_11 * t_12 + t_11 * t_13 * (-U_L + U_R) + t_13 * t_3) /
                      (t_5 + t_8))));
    }
    return {{FHLL_A, FHLL_U}};
  }

  std::array<double, 2>
  hll_sympy_numerical_flux_jacobian(double bn_L,
                                    double bn_R,
                                    double A_L,
                                    double U_L,
                                    double A_R,
                                    double U_R,
                                    double trial_A_L,
                                    double trial_U_L,
                                    double trial_A_R,
                                    double trial_U_R) const
  {
    double dFHLL_A, dFHLL_U;
    // dFHLL_A
    {
      const double t_0  = A_L * bn_L;
      const double t_1  = t_0 * trial_U_L;
      const double t_2  = U_L * bn_L * trial_A_L;
      const double t_3  = 1.0 / par["a0"];
      const double t_4  = par["m"] * par["mu"] / par["rho"];
      const double t_5  = std::sqrt(t_4 * std::pow(A_L * t_3, par["m"]));
      const double t_6  = (1.0 / 2.0) * t_5;
      const double t_7  = std::sqrt(t_4 * std::pow(A_R * t_3, par["m"]));
      const double t_8  = (1.0 / 2.0) * t_7;
      const double t_9  = (1.0 / 2.0) * U_L + (1.0 / 2.0) * U_R;
      const double t_10 = -t_6 - t_8 + t_9;
      const double t_11 = A_R * bn_R;
      const double t_12 = t_11 * trial_U_R;
      const double t_13 = U_R * bn_R * trial_A_R;
      const double t_14 = t_6 + t_8 + t_9;
      const double t_15 = t_5 + t_7;
      const double t_16 = trial_A_L / A_L;
      const double t_17 = trial_A_R / A_R;
      const double t_18 = U_L * t_0;
      const double t_19 = U_R * t_11;
      const double t_20 = -A_L + A_R;
      const double t_21 = t_14 * t_20;
      const double t_22 = (1.0 / 4.0) * par["m"];
      const double t_23 = t_16 * t_22 * t_5;
      const double t_24 = t_17 * t_22 * t_7;
      const double t_25 = (1.0 / 2.0) * trial_U_L + (1.0 / 2.0) * trial_U_R;
      const double t_26 = t_23 + t_24 + t_25;
      const double t_27 = -t_23 - t_24 + t_25;
      dFHLL_A =
        ((t_10 >= 0) ?
           (t_1 + t_2) :
           ((t_14 <= 0) ?
              (t_12 + t_13) :
              ((t_1 * t_14 - t_10 * t_12 - t_10 * t_13 +
                t_10 * t_14 * (-trial_A_L + trial_A_R) + t_10 * t_20 * t_26 +
                t_14 * t_2 + t_18 * t_26 - t_19 * t_27 + t_21 * t_27) /
                 t_15 +
               (-par["m"] * t_16 * t_6 - par["m"] * t_17 * t_8) *
                 (-t_10 * t_19 + t_10 * t_21 + t_14 * t_18) /
                 std::pow(t_15, 2))));
    }

    // dFHLL_U
    {
      const double t_0  = 1.0 / par["a0"];
      const double t_1  = std::pow(A_L * t_0, par["m"]);
      const double t_2  = 1.0 / par["rho"];
      const double t_3  = par["m"] * par["mu"] * t_2;
      const double t_4  = t_1 * t_3;
      const double t_5  = trial_A_L / A_L;
      const double t_6  = bn_L * (1.0 * U_L * trial_U_L + t_4 * t_5);
      const double t_7  = std::sqrt(t_4);
      const double t_8  = (1.0 / 2.0) * t_7;
      const double t_9  = std::pow(A_R * t_0, par["m"]);
      const double t_10 = t_3 * t_9;
      const double t_11 = std::sqrt(t_10);
      const double t_12 = (1.0 / 2.0) * t_11;
      const double t_13 = (1.0 / 2.0) * U_L + (1.0 / 2.0) * U_R;
      const double t_14 = -t_12 + t_13 - t_8;
      const double t_15 = trial_A_R / A_R;
      const double t_16 = bn_R * (1.0 * U_R * trial_U_R + t_10 * t_15);
      const double t_17 = t_12 + t_13 + t_8;
      const double t_18 = t_11 + t_7;
      const double t_19 = bn_L * (0.5 * std::pow(U_L, 2) +
                                  t_2 * (par["mu"] * (t_1 - 1) + par["p0"]));
      const double t_20 = bn_R * (0.5 * std::pow(U_R, 2) +
                                  t_2 * (par["mu"] * (t_9 - 1) + par["p0"]));
      const double t_21 = -U_L + U_R;
      const double t_22 = t_17 * t_21;
      const double t_23 = (1.0 / 4.0) * par["m"];
      const double t_24 = t_23 * t_5 * t_7;
      const double t_25 = t_11 * t_15 * t_23;
      const double t_26 = (1.0 / 2.0) * trial_U_L + (1.0 / 2.0) * trial_U_R;
      const double t_27 = t_24 + t_25 + t_26;
      const double t_28 = -t_24 - t_25 + t_26;
      dFHLL_U           = ((t_14 >= 0) ?
                             (t_6) :
                             ((t_17 <= 0) ?
                                (t_16) :
                                ((-t_14 * t_16 + t_14 * t_17 * (-trial_U_L + trial_U_R) +
                        t_14 * t_21 * t_27 + t_17 * t_6 + t_19 * t_27 -
                        t_20 * t_28 + t_22 * t_28) /
                         t_18 +
                       (-par["m"] * t_12 * t_15 - par["m"] * t_5 * t_8) *
                         (-t_14 * t_20 + t_14 * t_22 + t_17 * t_19) /
                         std::pow(t_18, 2))));
    }
    return {{dFHLL_A, dFHLL_U}};
  }

  /**
   * Compute HLL flux at interface for residual computation
   */
  std::array<double, 2>
  hll_numerical_flux(double bn_L,
                     double bn_R,
                     double A_L,
                     double U_L,
                     double A_R,
                     double U_R) const
  {
    // wave speeds
    double c_L = compute_wave_speed(A_L);
    double c_R = compute_wave_speed(A_R);

    // For Roe averages, avg eigenvalues could be used instead of min/max
    // See sec 10.5 in "Riemann Solvers and Numerical Methods for Fluids
    // Dynamics" by E.F. Toro

    double U_bar = 0.5 * (U_L + U_R);
    double c_bar = 0.5 * (c_L + c_R);
    double s_L   = U_bar - c_bar; // std::min(U_L - c_L, U_R - c_R);
    double s_R   = U_bar + c_bar; // std::max(U_L + c_L, U_R + c_R);

    // physical fluxes projected on normal
    double FAL = compute_scalar_physical_area_flux(bn_L, A_L, U_L);
    double FUL = compute_scalar_physical_momentum_flux(
      bn_L, U_L, U_L, compute_pressure_value(A_L));


    double FAR = compute_scalar_physical_area_flux(bn_R, A_R, U_R);
    double FUR = compute_scalar_physical_momentum_flux(
      bn_R, U_R, U_R, compute_pressure_value(A_R));


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
  hll_numerical_flux_jacobian(double bn_L,
                              double bn_R,
                              double A_L,
                              double U_L,
                              double A_R,
                              double U_R,
                              double trial_A_L,
                              double trial_U_L,
                              double trial_A_R,
                              double trial_U_R) const
  {
    // wave speeds at current state
    double c_L = compute_wave_speed(A_L);
    double c_R = compute_wave_speed(A_R);

    double U_bar = 0.5 * (U_L + U_R);
    double c_bar = 0.5 * (c_L + c_R);
    double s_L   = U_bar - c_bar; // std::min(U_L - c_L, U_R - c_R);
    double s_R   = U_bar + c_bar; // std::max(U_L + c_L, U_R + c_R);

    // Jacobian of area flux: ∂(A*U*b)/∂trial = (trial_U*A + trial_A*U)*b
    double FAL_jac = compute_scalar_physical_area_jacobian_flux(
      bn_L, A_L, trial_U_L, U_L, trial_A_L);


    // Jacobian of momentum flux: ∂(0.5*U²+P/ρ)*b)/∂trial
    double c_L_sq  = c_L * c_L;
    double FUL_jac = compute_scalar_physical_momentum_jacobian_flux(
      bn_L, c_L_sq, A_L, trial_A_L, U_L, trial_U_L);

    double FAR_jac = compute_scalar_physical_area_jacobian_flux(
      bn_R, A_R, trial_U_R, U_R, trial_A_R);

    double c_R_sq  = c_R * c_R;
    double FUR_jac = compute_scalar_physical_momentum_jacobian_flux(
      bn_R, c_R_sq, A_R, trial_A_R, U_R, trial_U_R);

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


  void
  get_trace(
    const Vector<double>                                           &vec,
    const typename DoFHandler<dim, spacedim>::active_cell_iterator &cell,
    unsigned int                                                    face_no,
    double                                                         &A,
    double                                                         &U) const;



  //   private:
  // Mesh and finite elements
  Triangulation<dim, spacedim>                  triangulation;
  DoFHandler<dim, spacedim>                     dof_handler;
  std::unique_ptr<FiniteElement<dim, spacedim>> fe;

  std::string outlet_type;
  bool
  is_terminal_boundary(dealii::types::boundary_id bid) const;

  // Linear system
  SparsityPattern      sparsity_pattern;
  SparseMatrix<double> jacobian_matrix;
  SparseMatrix<double> mass_matrix;
  SparseMatrix<double> linear_system_matrix;
  SparseDirectUMFPACK  linear_solver;
  SparseDirectUMFPACK  mass_solver;
  unsigned int         verbosity        = 0;
  std::string          output_directory = "";
  Vector<double>       solution;
  Vector<double>       pressure;
  Vector<double>       theoretical_peak;


  // Parameters
  unsigned int fe_degree              = 1;
  std::string  constants              = "1.0";
  std::string  output_filename        = "solution";
  bool         use_direct_solver      = true;
  bool         use_riemann_invariants = false;
  bool         use_junction_mesh      = false;
  unsigned int n_refinement_cycles    = 1;
  unsigned int n_global_refinements   = 5;
  double       time                   = 0.0;

  // Numerical parameters
  double theta    = 0.5;
  double theta_bd = 0.5;


  // Allow access to private members for all free functions whose name
  // is test
  friend void
  test();


  ParsedTools::Function<spacedim> initial_condition;
  ParsedTools::Function<spacedim> rhs_function;
  ParsedTools::Function<spacedim> exact_solution;
  ParsedTools::Function<1> inflow_function; // inflow depends on time only

  mutable TimerOutput computing_timer;
};

#endif // BLOOD_FLOW_SYSTEM_H
