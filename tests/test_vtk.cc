// -----------------------------------------------------------------------------
// Test: Read 1D vascular network from VTK and detect topology
// -----------------------------------------------------------------------------

#include <deal.II/base/geometry_info.h>
#include <deal.II/base/logstream.h>

#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/tria.h>

#include <filesystem>
#include <fstream>
#include <map>

#include "blood_flow_system.h"
#include "tests.h"
#include "vtk_utils.h"

using namespace dealii;

template <int dim, int spacedim>
void
test(const std::string &filename)
{
  Triangulation<dim, spacedim> triangulation;
  DoFHandler<dim, spacedim>    dof_handler(triangulation);
  Vector<double>               output_vector;
  std::vector<std::string>     data_names;

  // -----------------------------
  // Read VTK
  // -----------------------------
  // VTKWrappers::read_tria(filename, triangulation);

  VTKUtils::read_vtk(filename, dof_handler, output_vector, data_names);

  // -----------------------------
  // Basic checks
  // -----------------------------
  deallog << "Cells:    " << triangulation.n_active_cells() << std::endl;
  deallog << "Vertices: " << triangulation.n_vertices() << std::endl;

  // -----------------------------
  // Compute vertex degree
  // -----------------------------
  std::map<unsigned int, unsigned int> vertex_degree;

  for (const auto &cell : triangulation.active_cell_iterators())
    for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v)
      vertex_degree[cell->vertex_index(v)]++;

  // -----------------------------
  // Detect topology
  // -----------------------------
  unsigned int n_junctions = 0;
  unsigned int n_terminals = 0;

  deallog << "Vertex classification:" << std::endl;

  for (const auto &it : vertex_degree)
    {
      const unsigned int vid = it.first;
      const unsigned int deg = it.second;

      // Get the actual physical location of the vertex
      Point<spacedim> pos = triangulation.get_vertices()[vid];

      if (deg == 1)
        {
          // Point 0 is at x=0.0, Points 2 & 3 are at x=2.0
          if (pos[0] < 0.5)
            {
              deallog << "  Vertex " << vid << " @ (" << pos[0]
                      << ") is the INFLOW" << std::endl;
            }
          else
            {
              deallog << "  Vertex " << vid << " @ (" << pos[0]
                      << ") is an OUTLET" << std::endl;
              n_terminals++; // Keep this for your AssertThrow count
            }
        }
      else if (deg > 2)
        {
          deallog << "  Vertex " << vid << " is the JUNCTION" << std::endl;
          n_junctions++;
        }
    }

  // -----------------------------
  // Summary
  // -----------------------------
  deallog << "Total terminals: " << n_terminals << std::endl;
  deallog << "Total junctions: " << n_junctions << std::endl;
}


// -----------------------------------------------------------------------------
// MAIN
// -----------------------------------------------------------------------------

int
main()
{
  deallog.depth_console(10);
  try
    {
      test<1, 3>("../../../../notebooks/bifurcation_physics.vtk");
    }
  catch (std::exception &e)
    {
      std::cerr << "Exception: " << e.what() << std::endl;
      return 1;
    }

  return 0;
}