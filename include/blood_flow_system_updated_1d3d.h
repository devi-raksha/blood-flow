#ifndef BLOOD_FLOW_SYSTEM_UPDATED_1D3D_H
#define BLOOD_FLOW_SYSTEM_UPDATED_1D3D_H

#include <deal.II/base/function.h>
#include <deal.II/base/function_parser.h>
#include <deal.II/base/parameter_acceptor.h>
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

namespace dealii
{
  using namespace dealii;

  //====================================================================
  // PHYSICAL CONSTANTS AND PARAMETERS
  //====================================================================
  namespace BloodFlowParameters
  {
    // Default physical parameters for blood flow
    constexpr double DEFAULT_RHO = 1.06;                 // Density (g/cm³)
    constexpr double DEFAULT_REFERENCE_AREA = 3.1416e-4; // Reference area (cm²)
    constexpr double DEFAULT_ELASTIC_MODULUS = 80;       // Elastic modulus (Pa)
    constexpr double DEFAULT_REFERENCE_PRESSURE =
      0.0;                                      // Reference pressure (Pa)
    constexpr double DEFAULT_VISCOSITY_C = 0.0; // Viscosity coefficient
    constexpr double DEFAULT_THETA       = 1.0; // Penalty parameter
    constexpr double DEFAULT_ETA         = 1.0; // Stability parameter

    // Manufactured solution parameters
    constexpr double DEFAULT_R0 = 9.99e-3; // Reference radius (m)
    constexpr double DEFAULT_L  = 1.0;     // Domain length (m)
    constexpr double DEFAULT_T0 = 0.1;     // Time period (s)

    // Tube law exponent
    constexpr double TUBE_LAW_EXPONENT = 0.5;
  } // namespace BloodFlowParameters

  //====================================================================
  // PHYSICAL AND CONSTITUTIVE RELATIONS
  //====================================================================

  /**
   * Compute wave speed using the tube law
   */
  template <int dim, int spacedim>
  double
  compute_wave_speed(const double area,
                     const double reference_area,
                     const double elastic_modulus,
                     const double density)
  {
    const double ratio = area / reference_area;
    const double m     = BloodFlowParameters::TUBE_LAW_EXPONENT;
    const double dpda =
      elastic_modulus * m * std::pow(ratio, m - 1.0) / reference_area;
    return std::sqrt(area / density * dpda);
  }

  /**
   * Compute pressure using the tube law
   */
  template <int dim, int spacedim>
  double
  compute_pressure_value(const double area,
                         const double reference_area,
                         const double elastic_modulus,
                         const double reference_pressure)
  {
    const double ratio = area / reference_area;
    const double m     = BloodFlowParameters::TUBE_LAW_EXPONENT;
    return elastic_modulus * (std::pow(ratio, m) - 1.0) + reference_pressure;
  }

  /**
   * Compute pressure derivative dP/dA
   */
  template <int dim, int spacedim>
  double
  compute_pressure_derivative(const double area,
                              const double reference_area,
                              const double elastic_modulus)
  {
    const double ratio = area / reference_area;
    const double m     = BloodFlowParameters::TUBE_LAW_EXPONENT;
    return elastic_modulus * m * std::pow(ratio, m - 1.0) / reference_area;
  }

  //====================================================================
  // FLUX COMPUTATION FUNCTIONS
  //====================================================================

  /**
   * Compute Lax-Friedrichs penalty parameter for area equation: A^{n+1}P(A^n)/ρ
   */
  template <int dim, int spacedim>
  double
  compute_rusanov_penalty_area(const double area_left_old,
                               const double area_right_old,
                               const double reference_area,
                               const double elastic_modulus,
                               const double rho)
  {
    // For flux F_A = A^{n+1} * P(A^n) / ρ
    // dF_A/dA = P(A^n) / ρ (derivative w.r.t. A^{n+1})
    const double P_left  = compute_pressure_value<dim, spacedim>(area_left_old,
                                                                reference_area,
                                                                elastic_modulus,
                                                                0.0);
    const double P_right = compute_pressure_value<dim, spacedim>(
      area_right_old, reference_area, elastic_modulus, 0.0);

    return std::max(std::abs(P_left), std::abs(P_right)) / rho;
  }

  /**
   * Compute Lax-Friedrichs penalty parameter for momentum equation:
   * (Q^n)^2/(ρ*A^n)
   */
  template <int dim, int spacedim>
  double
  compute_rusanov_penalty_momentum(const double momentum_left_old,
                                   const double momentum_right_old,
                                   const double area_left_old,
                                   const double area_right_old,
                                   const double rho)
  {
    // For flux F_Q = (Q^n)^2 / (ρ * A^n)
    // dF_Q/dQ = 2*Q^n / (ρ * A^n) (derivative w.r.t. Q)
    const double dFQ_left =
      2.0 * std::abs(momentum_left_old) / (rho * area_left_old);
    const double dFQ_right =
      2.0 * std::abs(momentum_right_old) / (rho * area_right_old);

    return std::max(dFQ_left, dFQ_right);
  }

  /**
   * Compute area flux: A^{n+1} * P(A^n) / ρ
   */
  template <int dim, int spacedim>
  double
  compute_area_pressure_flux(const double area_new,
                             const double area_old,
                             const double reference_area,
                             const double elastic_modulus,
                             const double rho)
  {
    const double pressure_old = compute_pressure_value<dim, spacedim>(
      area_old, reference_area, elastic_modulus, 0.0);
    return area_new * pressure_old / rho;
  }

  /**
   * Compute momentum convection flux: (Q^n)^2 / (ρ * A^n)
   */
  template <int dim, int spacedim>
  double
  compute_momentum_convection_flux(const double momentum_old,
                                   const double area_old,
                                   const double rho)
  {
    return (momentum_old * momentum_old) / (rho * area_old);
  }

  /**
   * Updated numerical Rusanov flux computation
   */
  /**
   * Compute Rusanov (Lax–Friedrichs) flux for the area equation:
   * ̂F_A = 0.5(F_A⁻ + F_A⁺) – 0.5 α (A⁺ – A⁻)
   */
  template <int dim, int spacedim>
  double
  compute_area_rusanov_flux(const double area_left_new,
                            const double area_right_new,
                            const double area_left_old,
                            const double area_right_old,
                            const double reference_area,
                            const double elastic_modulus,
                            const double rho,
                            const double b_dot_n)
  {
    // Physical one‐sided fluxes
    const double FA_L =
      compute_area_pressure_flux<dim, spacedim>(
        area_left_new, area_left_old, reference_area, elastic_modulus, rho) *
      b_dot_n;
    const double FA_R =
      compute_area_pressure_flux<dim, spacedim>(
        area_right_new, area_right_old, reference_area, elastic_modulus, rho) *
      b_dot_n;

    // Penalty parameter
    const double alpha = compute_rusanov_penalty_area<dim, spacedim>(
      area_left_old, area_right_old, reference_area, elastic_modulus, rho);

    // Rusanov flux
    return 0.5 * (FA_L + FA_R) - 0.5 * alpha * (area_right_new - area_left_new);
  }


  /**
   * Compute Rusanov (Lax–Friedrichs) flux for the momentum equation:
   * ̂F_Q = 0.5(F_Q⁻ + F_Q⁺) – 0.5 beta (Q⁺ – Q⁻)
   */
  template <int dim, int spacedim>
  double
  compute_momentum_rusanov_flux(const double momentum_left_old,
                                const double momentum_right_old,
                                const double area_left_old,
                                const double area_right_old,
                                const double rho,
                                const double b_dot_n)
  {
    // Physical one‐sided fluxes
    const double FQ_L =
      compute_momentum_convection_flux<dim, spacedim>(momentum_left_old,
                                                      area_left_old,
                                                      rho) *
      b_dot_n;
    const double FQ_R =
      compute_momentum_convection_flux<dim, spacedim>(momentum_right_old,
                                                      area_right_old,
                                                      rho) *
      b_dot_n;

    // Penalty parameter
    const double beta =
      compute_rusanov_penalty_momentum<dim, spacedim>(momentum_left_old,
                                                      momentum_right_old,
                                                      area_left_old,
                                                      area_right_old,
                                                      rho);

    // Rusanov flux
    return 0.5 * (FQ_L + FQ_R) -
           0.5 * beta * (momentum_right_old - momentum_left_old);
  }


  /**
   * Compute tangent-normal product b·n
   */
  template <int dim, int spacedim, typename CellIterator>
  double
  compute_tangent_normal_product(const CellIterator        &cell,
                                 const Tensor<1, spacedim> &normal)
  {
    const auto b_vec = (cell->vertex(1) - cell->vertex(0)) /
                       cell->vertex(1).distance(cell->vertex(0));
    return b_vec * normal;
  }

  //====================================================================
  // EXACT SOLUTION AND MANUFACTURED SOLUTION
  //====================================================================

  /**
   * Exact solution class for manufactured solution
   */
  template <int spacedim>
  class ExactSolutionBloodFlow : public Function<spacedim>
  {
  private:
    // Parameters for manufactured solution
    double r0, a0, L, T0, atilde, qtilde;

  public:
    ExactSolutionBloodFlow()
      : Function<spacedim>(2) // 2 components: area A and momentum q
      , L(BloodFlowParameters::DEFAULT_L)
    {}

    virtual double
    value(const Point<spacedim> &p, const unsigned int component) const override
    {
      const double x = p[0];
      // const double t = this->get_time();
      const double u_c = 4.0; // baseline velocity
      if (component == 0)     // area A
        return std::pow(std::sin(2.0 * numbers::PI * x) + u_c, -1);
      else // momentum q
        return 1.0;
    }

    virtual void
    vector_value(const Point<spacedim> &p,
                 Vector<double>        &values) const override
    {
      Assert(values.size() == 2, ExcDimensionMismatch(values.size(), 2));
      values[0] = value(p, 0);
      values[1] = value(p, 1);
    }

    virtual Tensor<1, spacedim>
    gradient(const Point<spacedim> &p,
             const unsigned int     component) const override
    {
      const double x = p[0];
      // const double t = this->get_time();

      Tensor<1, spacedim> grad;

      if (component == 0) // area A gradient
        {
          const double u_c = 4.0; // baseline velocity
          grad[0] = -2.0 * numbers::PI * std::cos(2.0 * numbers::PI * x) /
                    std::pow(std::sin(2.0 * numbers::PI * x) + u_c, 2);
        }
      else if (component == 1) // momentum Q gradient
        {
          grad[0] = 0.0;
        }

      return grad;
    }

    virtual void
    vector_gradient(const Point<spacedim>            &p,
                    std::vector<Tensor<1, spacedim>> &gradients) const override
    {
      Assert(gradients.size() == 2, ExcDimensionMismatch(gradients.size(), 2));
      gradients[0] = gradient(p, 0);
      gradients[1] = gradient(p, 1);
    }

    virtual void
    vector_value_list(const std::vector<Point<spacedim>> &points,
                      std::vector<Vector<double>> &value_list) const override
    {
      const unsigned int n = points.size();
      Assert(value_list.size() == n,
             ExcDimensionMismatch(value_list.size(), n));
      for (unsigned int i = 0; i < n; ++i)
        vector_value(points[i], value_list[i]);
    }

    // Getters for parameters
    double
    get_reference_area() const
    {
      return a0;
    }
    double
    get_amplitude() const
    {
      return atilde;
    }
    double
    get_length() const
    {
      return L;
    }
    double
    get_period() const
    {
      return T0;
    }
  };

  //====================================================================
  // RIGHT-HAND SIDE FUNCTIONS (MANUFACTURED SOLUTION)
  //====================================================================

  /**
   * RHS A-forcing term f_a(x,t) for manufactured solution
   */
  template <int spacedim>
  class RHS_A_BloodFlow : public Function<spacedim>
  {
  private:
    double r0, a0, L, T0, atilde, qtilde;

  public:
    RHS_A_BloodFlow()
      : Function<spacedim>(1)
      , r0(BloodFlowParameters::DEFAULT_R0)
      , a0(numbers::PI * r0 * r0)
      , L(BloodFlowParameters::DEFAULT_L)
      , T0(BloodFlowParameters::DEFAULT_T0)
      , atilde(0.1 * a0)
      , qtilde(0.0)
    {}

    virtual double
    value(const Point<spacedim> &,
          const unsigned int /*component*/ = 0) const override
    {
      // const double x = p[0];
      // const double t = this->get_time();

      // Forcing term from manufactured solution
      return 0;
    }
  };

  /**
   * RHS U-forcing term f_u(x,t) for manufactured solution
   */
  template <int spacedim>
  class RHS_Q_BloodFlow : public Function<spacedim>
  {
  private:
    double r0, a0, L, T0, atilde, rho, elastic_modulus, viscosity_c, m;

  public:
    RHS_Q_BloodFlow()
      : Function<spacedim>(1)
      , r0(BloodFlowParameters::DEFAULT_R0)
      , a0(numbers::PI * r0 * r0)
      , L(BloodFlowParameters::DEFAULT_L)
      , T0(BloodFlowParameters::DEFAULT_T0)
      , atilde(0.1 * a0)
      , rho(BloodFlowParameters::DEFAULT_RHO)
      , elastic_modulus(BloodFlowParameters::DEFAULT_ELASTIC_MODULUS)
      , viscosity_c(BloodFlowParameters::DEFAULT_VISCOSITY_C)
      , m(BloodFlowParameters::TUBE_LAW_EXPONENT)
    {}

    virtual double
    value(const Point<spacedim> &p,
          const unsigned int /*component*/ = 0) const override
    {
      const double x = p[0];
      // const double t = this->get_time();
      const double u_c = 4.0; // baseline velocity
      const double A   = std::pow(std::sin(2.0 * numbers::PI * x) + u_c, -1);

      return std::cos(2.0 * numbers::PI * x) +
             elastic_modulus / (rho * std::pow(a0, m)) * std::pow(A, m - 1);
    }

    // Setters for runtime parameter updates
    void
    set_rho(double new_rho)
    {
      rho = new_rho;
    }
    void
    set_elastic_modulus(double new_E)
    {
      elastic_modulus = new_E;
    }
    void
    set_viscosity_c(double new_c)
    {
      viscosity_c = new_c;
    }
  };

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
    assemble_mass_matrix();
    void
    assemble_system();
    void
    solve();
    void
    output_results(const unsigned int cycle) const;
    void
    compute_pressure();
    void
    compute_errors(unsigned int k);
    void
    run_convergence_study();

    // Exact solution typedef for easy access
    using ExactSolution = ExactSolutionBloodFlow<spacedim>;

    //   private:
    // Mesh and finite elements
    Triangulation<dim, spacedim>                  triangulation;
    DoFHandler<dim, spacedim>                     dof_handler;
    std::unique_ptr<FiniteElement<dim, spacedim>> fe;

    // Linear system
    SparsityPattern      sparsity_pattern;
    SparseMatrix<double> system_matrix;
    SparseMatrix<double> mass_matrix;
    SparseMatrix<double> system_matrix_time;
    Vector<double>       solution;
    Vector<double>       solution_old;
    Vector<double>       right_hand_side;
    Vector<double>       pressure;
    Vector<double>       tmp_vector;

    // Parameters
    unsigned int fe_degree            = 1;
    std::string  constants            = "1.0";
    std::string  output_filename      = "solution";
    bool         use_direct_solver    = true;
    unsigned int n_refinement_cycles  = 4;
    unsigned int n_global_refinements = 4;

    // Physical parameters
    double       time_step       = 0.01;
    double       final_time      = 0.1;
    double       time            = 0.0;
    unsigned int n_time_steps    = 0;
    double       rho             = BloodFlowParameters::DEFAULT_RHO;
    double       viscosity_c     = BloodFlowParameters::DEFAULT_VISCOSITY_C;
    double       reference_area  = BloodFlowParameters::DEFAULT_REFERENCE_AREA;
    double       elastic_modulus = BloodFlowParameters::DEFAULT_ELASTIC_MODULUS;
    double reference_pressure = BloodFlowParameters::DEFAULT_REFERENCE_PRESSURE;
    double theta              = BloodFlowParameters::DEFAULT_THETA;
    double eta                = BloodFlowParameters::DEFAULT_ETA;

    // Function expressions for initial and boundary conditions
    std::string initial_A_expression =
      "1/ (sin(2*pi*x)+4)"; // Manufactured solution
    std::string initial_Q_expression = "1";
    // std::string pressure_bc_expression = "0.0";

    // Function parsers
    FunctionParser<spacedim> initial_A;
    FunctionParser<spacedim> initial_Q;
    FunctionParser<spacedim> pressure_bc;

    // RHS functions
    std::unique_ptr<RHS_A_BloodFlow<spacedim>> rhs_A_function;
    std::unique_ptr<RHS_Q_BloodFlow<spacedim>> rhs_Q_function;

    // Allow access to private members for any function whose name contains
    // "test"
    template <typename T>
    friend auto
    test_access(T *) -> decltype(void(&T::test), int());

    // Allow access to private members for all free functions whose name
    // contains test
    friend auto
    test(...) -> void;
  };

} // namespace dealii

#endif // BLOOD_FLOW_SYSTEM_UPDATED_1D3D_H