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
#include <deal.II/numerics/vector_tools.h>

#include <memory>

#include "constants.h"
#include "function.h"

namespace dealii
{
  using namespace dealii;

  //====================================================================
  // PHYSICAL CONSTANTS AND PARAMETERS
  //====================================================================
  using BloodFlowParameters = ParsedTools::Constants;

  //====================================================================
  // EXACT SOLUTION AND MANUFACTURED SOLUTION
  //====================================================================

  /**
   * Exact solution class for manufactured solution
   */
  // template <int spacedim>
  // class ExactSolutionBloodFlow : public Function<spacedim>
  // {
  // private:
  //   // Parameters for manufactured solution
  //   double r0, a0, L, T0, atilde, qtilde;

  // public:
  //   ExactSolutionBloodFlow()
  //     : Function<spacedim>(2) // 2 components: area and velocity
  //     , r0(BloodFlowParameters::R0)
  //     , a0(numbers::PI * r0 * r0)
  //     , L(BloodFlowParameters::L)
  //     , T0(BloodFlowParameters::T0)
  //     , atilde(0.1 * a0)
  //     , qtilde(0.0)
  //   {}

  //   virtual double
  //   value(const Point<spacedim> &p, const unsigned int component) const
  //   override
  //   {
  //     const double x = p[0];
  //     const double t = this->get_time();

  //     if (component == 0) // area A
  //       return a0 + atilde * std::sin(2.0 * numbers::PI * x / L) *
  //                     std::cos(2.0 * numbers::PI * t / T0);
  //     else // velocity U
  //       return qtilde - (atilde * L / T0) *
  //                         std::cos(2.0 * numbers::PI * x / L) *
  //                         std::sin(2.0 * numbers::PI * t / T0);
  //   }

  //   virtual void
  //   vector_value(const Point<spacedim> &p,
  //                Vector<double>        &values) const override
  //   {
  //     Assert(values.size() == 2, ExcDimensionMismatch(values.size(), 2));
  //     values[0] = value(p, 0);
  //     values[1] = value(p, 1);
  //   }

  //   virtual Tensor<1, spacedim>
  //   gradient(const Point<spacedim> &p,
  //            const unsigned int     component) const override
  //   {
  //     const double x = p[0];
  //     const double t = this->get_time();

  //     Tensor<1, spacedim> grad;

  //     if (component == 0) // area A gradient
  //       {
  //         grad[0] = atilde * (2.0 * numbers::PI / L) *
  //                   std::cos(2.0 * numbers::PI * x / L) *
  //                   std::cos(2.0 * numbers::PI * t / T0);
  //       }
  //     else if (component == 1) // velocity U gradient
  //       {
  //         grad[0] = (atilde * L / T0) * (2.0 * numbers::PI / L) *
  //                   std::sin(2.0 * numbers::PI * x / L) *
  //                   std::sin(2.0 * numbers::PI * t / T0);
  //       }

  //     return grad;
  //   }

  //   virtual void
  //   vector_gradient(const Point<spacedim>            &p,
  //                   std::vector<Tensor<1, spacedim>> &gradients) const
  //                   override
  //   {
  //     Assert(gradients.size() == 2, ExcDimensionMismatch(gradients.size(),
  //     2)); gradients[0] = gradient(p, 0); gradients[1] = gradient(p, 1);
  //   }

  //   virtual void
  //   vector_value_list(const std::vector<Point<spacedim>> &points,
  //                     std::vector<Vector<double>> &value_list) const override
  //   {
  //     const unsigned int n = points.size();
  //     Assert(value_list.size() == n,
  //            ExcDimensionMismatch(value_list.size(), n));
  //     for (unsigned int i = 0; i < n; ++i)
  //       vector_value(points[i], value_list[i]);
  //   }

  //   // Getters for parameters
  //   double
  //   get_reference_area() const
  //   {
  //     return a0;
  //   }
  //   double
  //   get_amplitude() const
  //   {
  //     return atilde;
  //   }
  //   double
  //   get_length() const
  //   {
  //     return L;
  //   }
  //   double
  //   get_period() const
  //   {
  //     return T0;
  //   }
  // };

  //====================================================================
  // RIGHT-HAND SIDE FUNCTIONS (MANUFACTURED SOLUTION)
  //====================================================================

  /**
   * RHS A-forcing term f_a(x,t) for manufactured solution
   */
  // template <int spacedim>
  // class RHS_A_BloodFlow : public Function<spacedim>
  // {
  // private:
  //   double r0, a0, L, T0, atilde, qtilde;

  // public:
  //   RHS_A_BloodFlow()
  //     : Function<spacedim>(1)
  //     , r0(BloodFlowParameters::R0)
  //     , a0(numbers::PI * r0 * r0)
  //     , L(BloodFlowParameters::L)
  //     , T0(BloodFlowParameters::T0)
  //     , atilde(0.1 * a0)
  //     , qtilde(0.0)
  //   {}

  //   virtual double
  //   value(const Point<spacedim> &p,
  //         const unsigned int /*component*/ = 0) const override
  //   {
  //     const double x = p[0];
  //     const double t = this->get_time();

  //     // Forcing term from manufactured solution
  //     return std::sin(2.0 * numbers::PI * x / L) *
  //              std::sin(2.0 * numbers::PI * t / T0) *
  //              (-2.0 * numbers::PI / T0 * atilde +
  //               (a0 + atilde * std::sin(2.0 * numbers::PI * x / L) *
  //                       std::cos(2.0 * numbers::PI * t / T0)) *
  //                 2.0 * numbers::PI / T0 * atilde) +
  //            atilde * std::cos(2.0 * numbers::PI * x / L) *
  //              std::cos(2.0 * numbers::PI * t / T0) * (2.0 * numbers::PI / L)
  //              * (qtilde - (atilde * L / T0) *
  //                          std::cos(2.0 * numbers::PI * x / L) *
  //                          std::sin(2.0 * numbers::PI * t / T0));
  //   }
  // };

  // /**
  //  * RHS U-forcing term f_u(x,t) for manufactured solution
  //  */
  // template <int spacedim>
  // class RHS_U_BloodFlow : public Function<spacedim>
  // {
  // private:
  //   double r0, a0, L, T0, atilde, rho, elastic_modulus, viscosity_c, m;

  // public:
  //   RHS_U_BloodFlow()
  //     : Function<spacedim>(1)
  //     , r0(BloodFlowParameters::R0)
  //     , a0(numbers::PI * r0 * r0)
  //     , L(BloodFlowParameters::L)
  //     , T0(BloodFlowParameters::T0)
  //     , atilde(0.1 * a0)
  //     , rho(BloodFlowParameters::RHO)
  //     , elastic_modulus(BloodFlowParameters::ELASTIC_MODULUS)
  //     , viscosity_c(BloodFlowParameters::VISCOSITY_C)
  //     , m(BloodFlowParameters::TUBE_LAW_EXPONENT)
  //   {}

  //   virtual double
  //   value(const Point<spacedim> &p,
  //         const unsigned int /*component*/ = 0) const override
  //   {
  //     const double x = p[0];
  //     const double t = this->get_time();
  //     const double A = a0 + atilde * std::sin(2.0 * numbers::PI * x / L) *
  //                             std::cos(2.0 * numbers::PI * t / T0);

  //     return std::cos(2.0 * numbers::PI * x / L) *
  //              std::cos(2.0 * numbers::PI * t / T0) *
  //              (-L * L / (T0 * T0) + elastic_modulus / (rho * std::pow(a0,
  //              m)) *
  //                                      std::pow(A, m - 1)) +
  //            (atilde - (atilde * L / T0) * std::cos(2.0 * numbers::PI * x /
  //            L) *
  //                        std::sin(2.0 * numbers::PI * t / T0)) *
  //              ((2.0 * numbers::PI / T0) * atilde *
  //                 std::sin(2.0 * numbers::PI * x / L) *
  //                 std::sin(2.0 * numbers::PI * t / T0) +
  //               viscosity_c);
  //   }

  //   // Setters for runtime parameter updates
  //   void
  //   set_rho(double new_rho)
  //   {
  //     rho = new_rho;
  //   }
  //   void
  //   set_elastic_modulus(double new_E)
  //   {
  //     elastic_modulus = new_E;
  //   }
  //   void
  //   set_viscosity_c(double new_c)
  //   {
  //     viscosity_c = new_c;
  //   }
  // };

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

    BloodFlowScratchData(
      const BloodFlowScratchData<dim, spacedim> &scratch_data)
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
    solve();

    double compute_max_wave_speed(const Vector<double> &solution) const;

    void
    output_results(const unsigned int cycle) const;
    void
    compute_pressure();
    void
    compute_errors(unsigned int k);
    void
    run_convergence_study();

  private:
    ParsedTools::Constants par;
    // --------------------------------------------------
    // ===  Physical and Constitutive Relations  ===
    // --------------------------------------------------

    /**
     * Compute wave speed using the tube law
     */
    double
    compute_wave_speed(const double area) const
    {
     // const double A_safe = std::max(area,  1e-2);
     const double eps = 0; // to avoid zero wave speed at zero area
      const double ratio = area / par["a0"] + eps;
      const double m     = par["m"];
      const double dpda  = par["mu"] * m * std::pow(ratio, m - 1.0) / par["a0"]; // while using shifted tube law
     // const double dpda = par["mu"] * m * std::exp(m*(ratio- 1.0)) / par["a0"]; // while using exponential tube law
    // const double dpda = par["mu"] * m / par["a0"]*(1 + ratio); // while using logarithmic tube law
      return std::sqrt(area / par["rho"] * dpda);
    }

    // /**
    //  * Compute pressure using the exponential  tube law
    //  */
    // double
    // compute_pressure_value(const double area) const
    // {
    //  // const double A_safe = std::max(area,  1e-2);
    //   const double ratio = area / par["a0"];
    //   const double m     = par["m"];
    //   return par["mu"] * (std::exp(m*(ratio - 1.0)) -1) + par["p0"];
    // }

    // /**
    //  * Compute pressure derivative dP/dA for exponential tube law
    //  */
    // double
    // compute_pressure_derivative(const double area) const
    // {
    //   const double ratio = area / par["a0"];
    //   const double m     = par["m"];
    //   return par["mu"] * m * std::exp(m*(ratio- 1.0)) / par["a0"];
    // }

    // /**
    //  * Compute pressure using the loagarithmic tube law
    //  */
    // double
    // compute_pressure_value(const double area) const
    // {
    //   //const double A_safe = std::max(area,  1e-2);
    //   const double ratio = area / par["a0"];
    //   const double m     = par["m"];
    //   return par["mu"] * m / par["a0"]*(area - par["a0"]+ std::log(ratio)) + par["p0"];
    // }

    // /**
    //  * Compute pressure derivative dP/dA for logarithmic tube law
    //  */
    // double
    // compute_pressure_derivative(const double area) const
    // {
    //   //const double A_safe = std::max(area,  1e-2);
    //   const double ratio = area / par["a0"];
    //   const double m     = par["m"];
    //   return par["mu"] * m / par["a0"]*(1 + ratio);
    // }
  


    /**
     * Compute pressure using the shifted tube law
     */
    double
    compute_pressure_value(const double area) const
    {
     // const double A_safe = std::max(area,  1e-2);
      const double eps   = 0; // to avoid zero pressure at zero area
      const double ratio = area / par["a0"]  + eps ;
      const double m     = par["m"];
      
      return par["mu"] * (std::pow(ratio + eps, m) - 1.0) + par["p0"];
    }

    /**
     * Compute pressure derivative dP/dA for shiftyed tube law
     */
    double
    compute_pressure_derivative(const double area) const
    { 
      const double eps = 0; // to avoid zero division
      const double ratio = area / par["a0"] + eps ;
      const double m     = par["m"];
      return par["mu"] * m * std::pow(ratio , m - 1.0) / par["a0"];
    }


    /**
     * Compute Lax–Friedrichs penalty parameter alpha
     * alpha = max(|U_L \pm c_L|, |U_R \pm c_R|)
     */
    double
    compute_LF_penalty(const double area_L,
                       const double area_R,
                       const double U_L,
                       const double U_R) const
    {
      // Compute left and right wave speeds
      const double cL = compute_wave_speed(area_L);
      const double cR = compute_wave_speed(area_R);

      // Compute four characteristic speeds and return maximum magnitude
      const double alpha = std::max(std::abs(U_L) + cL,
                                     std::abs(U_R) + cR);
                                    
      return alpha;
    }

    // --------------------------------------------------
    // ===  Geometry and Flux helper functions  ===
    // --------------------------------------------------

    /**
     * Compute the unit tangent vector b along the 1D vessel axis.
     */
    Tensor<1, spacedim>
    compute_directional_vector(
      const typename DoFHandler<dim, spacedim>::active_cell_iterator &cell)
      const
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
      return (current_area * trial_velocity + current_velocity * trial_area) *
             b;
    }

    /**
     * Compute the physical flux for the momentum equation:
     *   F_U = 0.5 * U⁻ * U⁺ * b
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


    //   private:
    // Mesh and finite elements
    Triangulation<dim, spacedim>                  triangulation;
    DoFHandler<dim, spacedim>                     dof_handler;
    std::unique_ptr<FiniteElement<dim, spacedim>> fe;

    // Linear system
    SparsityPattern      sparsity_pattern;
    SparseMatrix<double> jacobian_matrix;
    Vector<double>       solution;
    Vector<double>       solution_old;
    Vector<double>       pressure;
    Vector<double>       tmp_vector;

    Vector<double> residual_vector;
    Vector<double> newton_update;
    // Parameters
    unsigned int fe_degree            = 1;
    std::string  constants            = "1.0";
    std::string  output_filename      = "solution";
    bool         use_direct_solver    = true;
    unsigned int n_refinement_cycles  = 1;
    unsigned int n_global_refinements = 5;

    // Time stepping parameters
    double       time_step    = 0.01;
    double       final_time   = 1.0;
    double       time         = 0.0;
    unsigned int n_time_steps = 0;

    // Numerical parameters
    double omega = 1;
    double theta = 0.5;
    double eta   = 1.0;

    // Newton iteration parameters
    unsigned int max_newton_iterations = 20;
    double       newton_tolerance      = 1e-8;

    // // Function parsers
    // FunctionParser<spacedim> initial_A;
    // FunctionParser<spacedim> initial_U;
    // FunctionParser<spacedim> pressure_bc;

    // RHS functions

    // Allow access to private members for any function whose name contains
    // "test"
    template <typename T>
    friend auto
    test_access(T *) -> decltype(void(&T::test), int());

    // Allow access to private members for all free functions whose name
    // contains test
    friend auto
    test(...) -> void;


    ParsedTools::Function<spacedim> initial_condition;
    ParsedTools::Function<spacedim> rhs_function;
    ParsedTools::Function<spacedim> exact_solution;
  };

} // namespace dealii

#endif // BLOOD_FLOW_SYSTEM_UPDATED_1D3D_H