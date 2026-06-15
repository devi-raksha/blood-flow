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
#include <deal.II/lac/dynamic_sparsity_pattern.h>
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

// #include <deal.II/sundials/arkode.h>
#include <deal.II/sundials/ida.h>

#include <array>
#include <iosfwd>
#include <map>
#include <memory>
#include <set>
#include <utility>

#include "constants.h"
#include "function.h"

using namespace dealii;

using BloodFlowParameters = ParsedTools::Constants;

// ---------------------------------------------------------------------------
// FaceTraceDof
//
// One (A_hat, U_hat) trace pair lives on every unique face of the mesh.
// In 1-D (dim=1, spacedim=3) a "face" is a single vertex / 0-D object,
// so there is exactly one quadrature point on it and one pair of scalar
// unknowns per face.
//
// face_key  = (CellId, local_face_number)   — canonical: the *lower* global
//             cell-index side when the face is interior.
// a_hat_dof = global DOF index of the area  trace unknown
// u_hat_dof = global DOF index of the velocity trace unknown
// ---------------------------------------------------------------------------
struct FaceTraceDof
{
  types::global_dof_index a_hat_dof;
  types::global_dof_index u_hat_dof;
};

// ---------------------------------------------------------------------------
// Scratch / copy-data structures for cell and face integrals.  
// ---------------------------------------------------------------------------
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

  BloodFlowScratchData(const BloodFlowScratchData<dim, spacedim> &src)
    : fe_values(src.fe_values.get_fe(),
                src.fe_values.get_quadrature(),
                src.fe_values.get_update_flags())
    , fe_interface_values(src.fe_interface_values.get_fe(),
                          src.fe_interface_values.get_quadrature(),
                          src.fe_interface_values.get_update_flags())
  {}

  FEValues<dim, spacedim>          fe_values;
  FEInterfaceValues<dim, spacedim> fe_interface_values;
};

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
  reinit(const Iterator &cell, const unsigned int dofs_per_cell)
  {
    cell_matrix.reinit(dofs_per_cell, dofs_per_cell);
    cell_rhs.reinit(dofs_per_cell);
    local_dof_indices.resize(dofs_per_cell);
    cell->get_dof_indices(local_dof_indices);
  }
};

// ===========================================================================
// Main class
// ===========================================================================
template <int dim, int spacedim = dim>
class BloodFlowSystem : public ParameterAcceptor
{
public:
  BloodFlowSystem();

  void
  initialize_params(const std::string &filename = "");
  void
  setup_system();

  // Build the face-trace DOF map and extend the solution vector.
  // Must be called after dof_handler.distribute_dofs() inside setup_system.
  void
  build_face_dof_map();

  void
  detect_junctions();
  void
  initialize_terminal_capacitors();
  void
  update_terminal_pressures(const double          dt,
                            const Vector<double> &evaluation_point);

  // Residual F(t,y) for IDA(mplicit Differential-Algebraic solver) assembles
  // cell + trace equations.
  void
  assemble_residual(const double          t,
                             const Vector<double> &y,
                             const Vector<double> &ydot,
                             Vector<double>       &residual);

  // Jacobian ∂F/∂y — assembles all four blocks
  // (cell–cell, cell–trace, trace–cell, trace–trace).
  void
  assemble_jacobian(const double          t,
                    const Vector<double> &y,
                    const Vector<double> &ydot,
                    double alpha);

  // Builds per_cell_mass_inv: the local M_K^-1 for every cell K and also stores mass matrix.
  void
  build_per_cell_mass_inv();
  void
  compute_initial_solution(Vector<double> &dst, const double t);
  void
  initialize_trace_unknowns(Vector<double> &sol, const double t);

  void
  output_results(const Vector<double> &y,
                 const Vector<double> &pressure_vec,
                 const Vector<double> &theoretical_peak,
                 unsigned int          cycle) const;

  void
  compute_pressure(const Vector<double> &y, Vector<double> &pressure_vec) const;

  void
  compute_theoretical_peak(Vector<double> &theoretical_peak) const;
  void
  compute_errors(unsigned int k);
  void
  run();

  // Numerical flux type selector
  enum class NumericalFluxType
  {
    HLL,
    HLL_HDG,
    LAX_FRIEDRICHS
  };

  void
  set_numerical_flux(const NumericalFluxType flux_type)
  {
    numerical_flux_type = flux_type;
  }

  NumericalFluxType
  get_numerical_flux() const
  {
    return numerical_flux_type;
  }

private:
  // -----------------------------------------------------------------------
  // Physical parameters
  // -----------------------------------------------------------------------
  ParsedTools::Constants    par;
  AffineConstraints<double> constraints;
  double                    current_dt  = 0.0;
  double                    last_rcr_dt = 0.0;

  // -----------------------------------------------------------------------
  // Vessel / RCR physics (read from VTK)
  // -----------------------------------------------------------------------
  struct VesselPhysicalProperties
  {
    double a0, r_d, a_d, E, h_wall, p_d, p0, L;
    // define inlet and outlet radii for tapered vessels.
    //if r_in = r_out (or both zero), the vessel is treated as uniform.
    // and a_d is used as the default area for the pressure law.
    double r_in = 0.0, r_out = 0.0;
  };
  std::map<unsigned int, VesselPhysicalProperties> vessel_map;

  struct RCRPhysics
  {
    double R1, R2, C, P_out;
  };
  std::map<unsigned int, RCRPhysics>   rcr_map;
  std::map<types::boundary_id, double> terminal_Pc_storage;
  std::set<types::boundary_id>         terminal_boundary_ids;

  // Raw VTK cell/point data arrays
  Vector<double> cell_vessel_ids, cell_a0, cell_r_d, cell_a_d, cell_E;
  Vector<double> cell_h_wall, cell_p_d, cell_p0, cell_L;
  Vector<double> cell_r_in, cell_r_out; 
  Vector<double> point_boundary_id, point_R1, point_R2, point_C, point_P_out;

  // -----------------------------------------------------------------------
  // Mesh and FE spaces (cell unknowns only — DGQ)
  // -----------------------------------------------------------------------
  Triangulation<dim, spacedim>                  triangulation;
  DoFHandler<dim, spacedim>                     dof_handler;
  std::unique_ptr<FiniteElement<dim, spacedim>> fe;

  // -----------------------------------------------------------------------
  // Face-trace DOF management
  //
  // face_dof_map_:  (CellId, local_face_no) -> FaceTraceDof
  //
  //   * For interior faces the key uses the cell whose CellId compares
  //     *less* among the two neighbours — canonical ordering guarantees
  //     uniqueness without needing a separate "processed" set.
  //   * For boundary faces and junction faces, the unique cell owning
  //     that face edge is used directly.
  //
  // n_cell_dofs_  : dof_handler.n_dofs()   (A,U per DGQ cell)
  // n_trace_dofs_ : 2 * (number of unique faces)
  // n_total_dofs_ : n_cell_dofs_ + n_trace_dofs_
  //
  // Solution vector layout:  [ cell block | trace block ]
  //                           0 …n_cell-1   n_cell … n_total-1
  // -----------------------------------------------------------------------
  std::map<std::pair<CellId, unsigned int>, FaceTraceDof> face_dof_map;

  types::global_dof_index n_cell_dofs  = 0;
  types::global_dof_index n_trace_dofs = 0;
  types::global_dof_index n_total_dofs = 0;

  // Capacitor pressures as differential DAE unknowns(one per RCR terminal with
  // C > 0)

  std::map<types::boundary_id, types::global_dof_index> rcr_pc_dof;
  types::global_dof_index n_rcr_dofs  = 0;
  types::global_dof_index n_trace_end = 0; // = n_cell_dofs + n_trace_dofs
  void
  build_rcr_dof_map();
  void
  assemble_rcr_capacitor_equations(const Vector<double> &y,
                                   const Vector<double> &ydot,
                                   Vector<double>       &F);
  void
  assemble_jacobian_rcr_capacitor_block(const Vector<double> &y);

  // -----------------------------------------------------------------------
  // Junction detection
  //
  // A junction is a mesh vertex touched by more than or equal to two cells having different vessel IDs.
  // all_junction_faces holds (CellId, local_face_no) for every half-face
  // that ends at a junction vertex — used to skip those faces in the
  // ordinary boundary-condition assembly so they are handled exclusively
  // by assemble_trace_junction_equations().
  // -----------------------------------------------------------------------
  struct JunctionHalfFace
  {
    typename DoFHandler<dim, spacedim>::active_cell_iterator cell;
    unsigned int                                             face_no;
    int                                                      orientation; // ±1
  };

  struct JunctionInfo
  {
    Point<spacedim>               location;
    std::vector<JunctionHalfFace> half_faces; // one entry per incident vessel
    unsigned int
    n_vessels() const
    {
      return static_cast<unsigned int>(half_faces.size());
    }
  };

  std::vector<JunctionInfo>                 junctions;
  std::set<std::pair<CellId, unsigned int>> all_junction_faces;

  // -----------------------------------------------------------------------
  // Linear algebra  (sized to n_total_dofs)
  // -----------------------------------------------------------------------
  SparsityPattern      sparsity_pattern;
  SparseMatrix<double> jacobian_matrix;
  SparseMatrix<double> linear_system_matrix;
  SparseDirectUMFPACK  linear_solver;

  // Per-cell inverse mass matrices.
  // The system seen by ARKode is the pure ODE
  //   dy_cell/dt = M_K^-1 * F_cell(y)     (cell block)
  //   0           = F_trace(y)             (trace block, algebraic)

  std::map<CellId, FullMatrix<double>> per_cell_mass_inv;
  std::map<CellId, FullMatrix<double>> per_cell_mass_;

  Vector<double> solution;
  Vector<double> solution_dot;
  Vector<double> pressure;
  Vector<double> theoretical_peak;

  // -----------------------------------------------------------------------
  // User parameters
  // -----------------------------------------------------------------------
  unsigned int fe_degree            = 1;
  std::string  constants            = "1.0";
  std::string  output_filename      = "solution";
  bool         use_direct_solver    = true;
  bool         use_junction_mesh     = true;
  bool         use_riemann_invariants = true;
  unsigned int n_refinement_cycles  = 1;
  unsigned int n_global_refinements = 5;
  std::string  vtk_file_path        = "mesh.vtk";
  std::string  output_directory     = "";
  unsigned int verbosity            = 0;
  std::string  outlet_type;
  double       theta    = 0.5;
  double       theta_bd = 0.5;
  double       time     = 0.0;

  NumericalFluxType numerical_flux_type     = NumericalFluxType::HLL;
  std::string       numerical_flux_type_str = "HLL";

  SUNDIALS::IDA<Vector<double>>::AdditionalData ida_parameters;

  ParsedTools::Function<spacedim> rhs_function;
  ParsedTools::Function<spacedim> exact_solution;
  ParsedTools::Function<1>        inflow_function;

  mutable TimerOutput computing_timer;


  // CSV timeseries output
  std::ofstream                                         csv_P_, csv_Q_;
  std::vector<std::pair<unsigned int, Point<spacedim>>> probe_targets_;

  void
  open_csv_files();
  void
  write_csv_row(const double t, const Vector<double> &sol);
  void
  close_csv_files();
  // -----------------------------------------------------------------------
  // Internal helpers — geometry
  // -----------------------------------------------------------------------

  bool
  is_terminal_boundary(types::boundary_id bid) const
  {
    return terminal_boundary_ids.count(bid) > 0;
  }

  bool
  is_junction_face(const CellId &cid, const unsigned int f) const
  {
    return all_junction_faces.count(std::make_pair(cid, f)) > 0;
  }

  Tensor<1, spacedim>
  compute_directional_vector(
    const typename DoFHandler<dim, spacedim>::active_cell_iterator &cell) const
  {
    return (cell->vertex(1) - cell->vertex(0)) /
           cell->vertex(1).distance(cell->vertex(0));
  }

  double
  compute_tangent_normal_product(
    const typename DoFHandler<dim, spacedim>::active_cell_iterator &cell,
    const Tensor<1, spacedim> &normal) const
  {
    return compute_directional_vector(cell) * normal;
  }

  // -----------------------------------------------------------------------
  // Internal helpers — physics / constitutive law
  // -----------------------------------------------------------------------

  // Compute the local diastolic cross-sectional area at a cell's centroid
  // by linearly interpolating in arc-length between r_in (inlet) and
  // r_out (outlet) of the parent vessel.
  //
  // If r_in == r_out == 0 (no taper data), fall back to the vessel-level a_d.

  double
  compute_a_d_local(
    const typename DoFHandler<dim, spacedim>::active_cell_iterator &cell) const
  {
    const unsigned int vid = cell->material_id();
    const auto        &vpp = vessel_map.at(vid);

    if (vpp.r_in <= 0.0 || vpp.r_out <= 0.0)
      return vpp.a_d;

    const auto &[s_min, s_max] = vessel_s_bounds.at(vid);

    const Tensor<1, spacedim> d_hat = compute_directional_vector(cell);

    const double s = cell->center() * d_hat;

    const double xi_geometry =
      (s_max - s_min > 1e-14) ? (s - s_min) / (s_max - s_min) : 0.5;

    const double r = vpp.r_in + xi_geometry * (vpp.r_out - vpp.r_in);

    return numbers::PI * r * r; 
  }

  double
  compute_a_d_at_face(
    const typename DoFHandler<dim, spacedim>::active_cell_iterator &cell,
    const unsigned int face_no) const
  {
    const unsigned int vid = cell->material_id();
    const auto        &vpp = vessel_map.at(vid);
    if (vpp.r_in <= 0.0 || vpp.r_out <= 0.0)
      return vpp.a_d;

    const auto &[s_min, s_max]      = vessel_s_bounds.at(vid);
    const Tensor<1, spacedim> d_hat = compute_directional_vector(cell);
    const double              s     = cell->vertex(face_no) * d_hat;
    const double              xi_geometry =
      (s_max - s_min > 1e-14) ? (s - s_min) / (s_max - s_min) : 0.5;
    const double r = vpp.r_in + xi_geometry * (vpp.r_out - vpp.r_in);
    return numbers::PI * r * r;
  }

  double
  compute_beta_p(const double E, const double h_wall) const
  {
    return (4.0 * std::sqrt(3.14159265358979323846) / 3.0) * E * h_wall;
  }


  // ---- Physics helpers taking explicit a_d_local (preferred call-sites) ----

  double
  compute_pressure_value(const double       A,
                         const unsigned int vid,
                         const double       a_d_local) const
  {
    const auto  &vpp  = vessel_map.at(vid);
    const double beta = compute_beta_p(vpp.E, vpp.h_wall);
    return vpp.p0 + beta / a_d_local * (std::sqrt(A) - std::sqrt(a_d_local)) +
           vpp.p_d;
  }

  double
  compute_pressure_derivative(const double       A,
                              const unsigned int vid,
                              const double       a_d_local) const
  {
    const auto  &vpp    = vessel_map.at(vid);
    const double beta_p = compute_beta_p(vpp.E, vpp.h_wall);
    return beta_p / (2.0 * a_d_local * std::sqrt(std::max(A, 1e-30)));
  }

  double
  compute_wave_speed(const double       A,
                     const unsigned int vid,
                     const double       a_d_local) const
  {
    const double A_safe = std::max(A, 1e-10);
    return std::sqrt(A_safe / par["rho"] *
                     compute_pressure_derivative(A_safe, vid, a_d_local));
  }

  double
  compute_wave_speed_derivative(const double       A,
                                const unsigned int vid,
                                const double       a_d_local) const
  {
    const double A_safe = std::max(A, 1e-10);
    return compute_wave_speed(A_safe, vid, a_d_local) * par["m"] /
           (2.0 * A_safe);
  }

  // ---- Cell-iterator convenience wrappers (compute a_d_local internally) ---

  double
  compute_pressure_value(
    const double                                                    A,
    const typename DoFHandler<dim, spacedim>::active_cell_iterator &cell) const
  {
    return compute_pressure_value(A,
                                  cell->material_id(),
                                  compute_a_d_local(cell));
  }

  double
  compute_pressure_derivative(
    const double                                                    A,
    const typename DoFHandler<dim, spacedim>::active_cell_iterator &cell) const
  {
    return compute_pressure_derivative(A,
                                       cell->material_id(),
                                       compute_a_d_local(cell));
  }

  double
  compute_wave_speed(
    const double                                                    A,
    const typename DoFHandler<dim, spacedim>::active_cell_iterator &cell) const
  {
    return compute_wave_speed(A, cell->material_id(), compute_a_d_local(cell));
  }

  double
  compute_wave_speed_derivative(
    const double                                                    A,
    const typename DoFHandler<dim, spacedim>::active_cell_iterator &cell) const
  {
    return compute_wave_speed_derivative(A,
                                         cell->material_id(),
                                         compute_a_d_local(cell));
  }

  // Convenience wrappers for face-trace access (
  //      compatibility with junctions and boundary conditions where we only
  //      have a vid, not a cell iterator) ------------------------------------

  double
  compute_pressure_value(const double A, const unsigned int vid) const
  {
    return compute_pressure_value(A, vid, vessel_map.at(vid).a_d);
  }

  double
  compute_pressure_derivative(const double A, const unsigned int vid) const
  {
    return compute_pressure_derivative(A, vid, vessel_map.at(vid).a_d);
  }

  double
  compute_wave_speed(const double A, const unsigned int vid) const
  {
    return compute_wave_speed(A, vid, vessel_map.at(vid).a_d);
  }

  double
  compute_wave_speed_derivative(const double A, const unsigned int vid) const
  {
    return compute_wave_speed_derivative(A, vid, vessel_map.at(vid).a_d);
  }


    // -----------------------------------------------------------------------
    // Internal helpers — face-trace access
    // -----------------------------------------------------------------------

    // Return the canonical key for a face.  For interior faces the cell
    // with the lexicographically smaller CellId is always the key owner.
    std::pair<CellId, unsigned int> canonical_face_key(
      const typename DoFHandler<dim, spacedim>::active_cell_iterator &cell,
      const unsigned int face_no) const;
    std::map<unsigned int, std::pair<double, double>> vessel_s_bounds;
    // Read (A_hat, U_hat) from the trace block of y.
    void get_face_trace(
      const Vector<double>                                           &y,
      const typename DoFHandler<dim, spacedim>::active_cell_iterator &cell,
      unsigned int                                                    face_no,
      double                                                         &A_hat,
      double &U_hat) const;

    // -----------------------------------------------------------------------
    // Internal helpers — physical fluxes (scalar projections)
    // -----------------------------------------------------------------------

    double scalar_area_flux(const double bn, const double A, const double U)
      const
    {
      return A * U * bn;
    }

    double scalar_momentum_flux(const double bn,
                                const double U,
                                const double pressure,
                                const double rho) const
    {
      return (0.5 * U * U + pressure / rho) * bn;
    }

    // Linearised (Jacobian) versions
    double scalar_area_flux_jac(const double bn,
                                const double A,
                                const double U,
                                const double dA,
                                const double dU) const
    {
      return (A * dU + U * dA) * bn;
    }

    double scalar_momentum_flux_jac(
      const double bn,
      const double c2_over_A, // 1/rho dp/da = c^2/A
      const double U,
      const double dA,
      const double dU) const
    {
      return (c2_over_A * dA + U * dU) * bn;
    }

    double compute_LF_penalty(const double       A_L,
                              const double       A_R,
                              const double       U_L,
                              const double       U_R,
                              const double       bn_L,
                              const double       bn_R,
                              const unsigned int vid_L,
                              const unsigned int vid_R,
                              const double       a_d_L,
                              const double       a_d_R) const
    {
      const double cL = compute_wave_speed(A_L, vid_L, a_d_L);
      const double cR = compute_wave_speed(A_R, vid_R, a_d_R);
      return std::max({std::abs((U_L - cL) * bn_L),
                       std::abs((U_L + cL) * bn_L),
                       std::abs((U_R - cR) * bn_R),
                       std::abs((U_R + cR) * bn_R)});
    }

    // -----------------------------------------------------------------------
    // Internal helpers — numerical fluxes
    // -----------------------------------------------------------------------

    std::array<double, 2> hll_flux(double       bn_L,
                                   double       bn_R,
                                   double       A_L,
                                   double       U_L,
                                   double       A_R,
                                   double       U_R,
                                   unsigned int vid_L,
                                   unsigned int vid_R,
                                   double       a_d_L,
                                   double       a_d_R) const;

    std::array<double, 2>
    hll_hdg_flux(double       bn_L,
             double      /*bn_R*/,
             double       A_L,
             double       U_L,
             double       A_R,
             double       U_R,
             unsigned int /*vid_L*/,
             unsigned int vid_R,
             double       /*a_d_L*/,
             double       a_d_R) const;


    std::array<double, 2> lf_flux(double       bn_L,
                                  double       bn_R,
                                  double       A_L,
                                  double       U_L,
                                  double       A_R,
                                  double       U_R,
                                  unsigned int vid_L,
                                  unsigned int vid_R,
                                  double       a_d_L,
                                  double       a_d_R) const;

    std::array<double, 2> numerical_flux(double       bn_L,
                                         double       bn_R,
                                         double       A_L,
                                         double       U_L,
                                         double       A_R,
                                         double       U_R,
                                         unsigned int vid_L,
                                         unsigned int vid_R,
                                         double       a_d_L,
                                         double       a_d_R) const
    {
      if (numerical_flux_type == NumericalFluxType::HLL)
        {
          return hll_flux(
            bn_L, bn_R, A_L, U_L, A_R, U_R, vid_L, vid_R, a_d_L, a_d_R);
        }
      else if (numerical_flux_type == NumericalFluxType::HLL_HDG)
        {
          return hll_hdg_flux(
            bn_L, bn_R, A_L, U_L, A_R, U_R, vid_L, vid_R, a_d_L, a_d_R);
        }
      else
        {
          return lf_flux(
            bn_L, bn_R, A_L, U_L, A_R, U_R, vid_L, vid_R, a_d_L, a_d_R);
        }
    }

    std::array<double, 2> hll_flux_jac(double       bn_L,
                                       double       bn_R,
                                       double       A_L,
                                       double       U_L,
                                       double       A_R,
                                       double       U_R,
                                       double       dA_L,
                                       double       dU_L,
                                       double       dA_R,
                                       double       dU_R,
                                       unsigned int vid_L,
                                       unsigned int vid_R,
                                      const double a_d_L,
                                      const double a_d_R) const;

    std::array<double, 2>
    hll_hdg_flux_jac(double       bn_L,
                 double       /*bn_R*/,
                 double       A_L,
                 double       U_L,
                 double       A_R,
                 double       U_R,
                 double       dA_L,
                 double       dU_L,
                 double       dA_R,
                 double       dU_R,
                 unsigned int /*vid_L*/,
                 unsigned int vid_R,
                 const double /*a_d_L*/,
                 const double a_d_R) const;



    std::array<double, 2> lf_flux_jac(double       bn_L,
                                      double       bn_R,
                                      double       A_L,
                                      double       U_L,
                                      double       A_R,
                                      double       U_R,
                                      double       dA_L,
                                      double       dU_L,
                                      double       dA_R,
                                      double       dU_R,
                                      unsigned int vid_L,
                                      unsigned int vid_R,
                                      const double a_d_L,
                                      const double a_d_R) const;

    std::array<double, 2> numerical_flux_jac(double       bn_L,
                                             double       bn_R,
                                             double       A_L,
                                             double       U_L,
                                             double       A_R,
                                             double       U_R,
                                             double       dA_L,
                                             double       dU_L,
                                             double       dA_R,
                                             double       dU_R,
                                             unsigned int vid_L,
                                             unsigned int vid_R,
                                             const double a_d_L,
                                             const double a_d_R) const
    {
      if (numerical_flux_type == NumericalFluxType::HLL)
      {
        return hll_flux_jac(
          bn_L, bn_R, A_L, U_L, A_R, U_R, dA_L, dU_L, dA_R, dU_R, vid_L, vid_R, a_d_L, a_d_R);
        }
     else if (numerical_flux_type == NumericalFluxType::HLL_HDG)
       { return hll_hdg_flux_jac(
          bn_L, bn_R, A_L, U_L, A_R, U_R, dA_L, dU_L, dA_R, dU_R, vid_L, vid_R, a_d_L, a_d_R);
       }
       else{
      return lf_flux_jac(
        bn_L, bn_R, A_L, U_L, A_R, U_R, dA_L, dU_L, dA_R, dU_R, vid_L, vid_R, a_d_L, a_d_R);
      }
    }

    // -----------------------------------------------------------------------
    // Assembly sub-routines
    // -----------------------------------------------------------------------

    // Cell residuals: volume integrals + flux through face traces.
    void assemble_cell_residuals(const double          t,
                                 const Vector<double> &y,
                                 Vector<double>       &F);

    // Trace equations for interior faces (Riemann-invariant continuity).
    void assemble_trace_interior_equations(const Vector<double> &y,
                                           Vector<double>       &F);

    // Trace equations for boundary faces (inflow Q / RCR / reflection).
    void assemble_trace_boundary_equations(const double          t,
                                           const Vector<double> &y,
                                           Vector<double>       &F);

    // Trace equations for junction faces (mass conservation +
    // tottotalal-head continuity + Riemann compatibility per vessel).
    void assemble_trace_junction_equations(const Vector<double> &y,
                                           Vector<double>       &F);

    // Jacobian blocks
    void assemble_jacobian_cell_block(const double t, const Vector<double> &y);

    void assemble_jacobian_trace_interior_block(const Vector<double> &y);

    void assemble_jacobian_trace_boundary_block(const double          t,
                                                const Vector<double> &y);

    void assemble_jacobian_trace_junction_block(const Vector<double> &y);

    // Mass matrix - only acts on the cell block; trace block rows/cols = 0.
    void build_extended_sparsity_pattern();

    friend void test();
  };

#endif // BLOOD_FLOW_SYSTEM_H