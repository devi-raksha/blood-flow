#include <deal.II/base/logstream.h> // deallog control
#include <deal.II/base/parameter_handler.h>

#include <iostream>
#include <stdexcept> // for std::invalid_argument

#include "include/non_linear_conservation_law.h"


using namespace dealii;

int
main(int argc, char **argv)
{
  try
    {
      /* --------------------------- 1. Locate parameter file -----------------
       */
      std::string par_name;
      if (argc > 1)
        par_name = argv[1]; // first CLI argument
      else
        par_name =
          "parameters_non_linear_conservation_law.prm"; // fallback default

      /* ---------------------- 2. Initialise deal.II logging -----------------
       */
      dealii::deallog.depth_console(2);

      /* ------------------------- 3. Set up the problem ----------------------
       */
      NonLinearConservationLaw<1, 3>
        problem; // 1-dim geometry embedded in \mathbb{R}^3
      problem.initialize_params(par_name);
      // problem.run(); // perform mesh gen, time loop, I/O
      problem.run_convergence_study();

      /* ------------------------ 4. Normal program exit ----------------------
       */
      return 0;
    }

  /* --------------------- 5. Dedicated exception catchers ------------------ */
  catch (const std::invalid_argument &theta_range) // parameter-specific errors
    {
      std::cerr << '\n'
                << "----------------------------------------------------\n"
                << "Invalid parameter: \n"
                << theta_range.what() << '\n'
                << "Aborting!\n"
                << "----------------------------------------------------\n";
      return 1;
    }
  catch (const std::exception &exc) // general deal.II/runtime errors
    {
      std::cerr << '\n'
                << "----------------------------------------------------\n"
                << "Exception on processing: \n"
                << exc.what() << '\n'
                << "Aborting!\n"
                << "----------------------------------------------------\n";
      return 1;
    }
  catch (...) // fallback
    {
      std::cerr << '\n'
                << "----------------------------------------------------\n"
                << "Unknown exception!\n"
                << "Aborting!\n"
                << "----------------------------------------------------\n";
      return 1;
    }
}
