#ifndef NON_LINEAR_CONSERVATION_LAW_H
#define NON_LINEAR_CONSERVATION_LAW_H

#include <deal.II/base/function_parser.h>
#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/parsed_convergence_table.h>
#include <deal.II/base/parsed_function.h>
#include <deal.II/base/tensor_function.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_interface_values.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>

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

#include <deal.II/meshworker/mesh_loop.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <memory>

namespace dealii
{
  // Forward declarations
  template <int dim, int spacedim>
  class NonLinearConservationLaw;

  // Scratch data for assembly
  template <int dim, int spacedim>
  struct ConservationLawScratchData
  {
    ConservationLawScratchData(const FiniteElement<dim, spacedim> &fe,
                               const Quadrature<dim>              &quadrature,
                               const Quadrature<dim - 1> &quadrature_face);

    ConservationLawScratchData(
      const ConservationLawScratchData<dim, spacedim> &scratch_data);

    FEValues<dim, spacedim>          fe_values;
    FEInterfaceValues<dim, spacedim> fe_interface_values;
  };

  // Copy data for assembly
  struct ConservationLawCopyData
  {
    FullMatrix<double>                   cell_matrix;
    Vector<double>                       cell_rhs;
    std::vector<types::global_dof_index> local_dof_indices;

    struct FaceData
    {
      FullMatrix<double>                   cell_matrix;
      Vector<double>                       cell_rhs;
      std::vector<types::global_dof_index> joint_dof_indices;
    };
    std::vector<FaceData> face_data;

    template <class Iterator>
    void
    reinit(const Iterator &cell, const unsigned int dofs_per_cell);
  };

  // .........Flux computation functions................
  template <int dim, int spacedim>
  double
  compute_burger_lax_friedrichs_flux(const double u_left,
                                     const double u_right,
                                     const double b_dot_n);

  template <int dim, int spacedim>
  double
  compute_tangent_normal_product_burger(
    const typename DoFHandler<dim, spacedim>::active_cell_iterator &cell,
    const Tensor<1, spacedim>                                      &normal);

  // Main class
  template <int dim, int spacedim>
  class NonLinearConservationLaw : public ParameterAcceptor
  {
  public:
    NonLinearConservationLaw();

    void
    initialize_params(const std::string &filename);
    void
    run_convergence_study();

  private:
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
    compute_errors(unsigned int k);

    // Mesh and DOF management
    Triangulation<dim, spacedim>                  triangulation;
    DoFHandler<dim, spacedim>                     dof_handler;
    std::unique_ptr<FiniteElement<dim, spacedim>> fe;

    // Linear algebra objects
    SparsityPattern      sparsity_pattern;
    SparseMatrix<double> system_matrix;
    SparseMatrix<double> mass_matrix;
    SparseMatrix<double> system_matrix_time;
    Vector<double>       solution;
    Vector<double>       solution_old;
    Vector<double>       right_hand_side;
    Vector<double>       tmp_vector;

    // Parameters
    unsigned int fe_degree            = 1;
    std::string  output_filename      = "solution";
    std::string  output_directory     = ".";
    bool         use_direct_solver    = true;
    unsigned int n_refinement_cycles  = 4;
    unsigned int n_global_refinements = 4;
    double       time_step            = 0.01;
    double       final_time           = 1.0;
    double       theta                = 1.0; // penalty parameter
    double       omega                = 1.0; // relaxation parameter
    double       time                 = 0.0;
    unsigned int n_time_steps         = 0;

    // Picard iteration parameters
    const unsigned int max_picard_iterations = 10;
    const double       picard_tolerance      = 1e-8;

    // Function parsers
    ParameterAcceptorProxy<Functions::ParsedFunction<spacedim>>
      initial_condition;

    ParameterAcceptorProxy<Functions::ParsedFunction<spacedim>> rhs_function;

    ParameterAcceptorProxy<Functions::ParsedFunction<spacedim>> exact_solution;

    ParsedConvergenceTable convergence_table;
  };

} // namespace dealii

#endif // NON_LINEAR_CONSERVATION_LAW_H_