# # Set the name of the project and target:
# SET(TARGET "DG_advection_reaction_estimator")

# # Declare all source files the target consists of:
# SET(TARGET_SRC
#   source/DG_advection_reaction.cc
#   main.cc
# )

# PROJECT(${TARGET} CXX)


# CMAKE_MINIMUM_REQUIRED(VERSION 3.13.4)

# FIND_PACKAGE(deal.II 9.5.0
#   HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
#   )
# IF(NOT ${deal.II_FOUND})
#   MESSAGE(FATAL_ERROR "\n"
#     "*** Could not locate a (sufficiently recent) version of deal.II. ***\n\n"
#     "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
#     "or set an environment variable \"DEAL_II_DIR\" that contains this path."
#     )
# ENDIF()

# DEAL_II_INITIALIZE_CACHED_VARIABLES()
# PROJECT(${TARGET})
# DEAL_II_INVOKE_AUTOPILOT()

# Set the names of the projects and targets:
SET(TARGET1 "DG_advection_reaction_estimator")
SET(TARGET2 "blood_flow_system_updated_1d3d")
SET(TARGET3 "burger_equation")
SET(TARGET4 "burger_with_periodic_bc")
SET(TARGET5 "non_linear_conservation_law")

# Declare all source files for each target:
SET(TARGET1_SRC
  source/DG_advection_reaction.cc
  main.cc
)

set(TARGET2_SRC
  source/blood_flow_system_updated_1d3d.cc
  main_updated_1d3d.cc
)

set(TARGET3_SRC
  source/burger_equation.cc
  main_burger.cc
)

set(TARGET4_SRC
  source/burger_with_periodic_bc.cc
  main_burger.cc
)

set(TARGET5_SRC
  source/non_linear_conservation_law.cc
  main_non_linear_conservation_law.cc
)

CMAKE_MINIMUM_REQUIRED(VERSION 3.13.4)

FIND_PACKAGE(deal.II 9.5.0
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
  )
IF(NOT ${deal.II_FOUND})
  MESSAGE(FATAL_ERROR "\n"
    "*** Could not locate a (sufficiently recent) version of deal.II. ***\n\n"
    "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
    "or set an environment variable \"DEAL_II_DIR\" that contains this path."
    )
ENDIF()

DEAL_II_INITIALIZE_CACHED_VARIABLES()
PROJECT(DG_advection_reaction_multi CXX)

# Create first executable target
ADD_EXECUTABLE(${TARGET1} ${TARGET1_SRC})
DEAL_II_SETUP_TARGET(${TARGET1})

# Create second executable target  
ADD_EXECUTABLE(${TARGET2} ${TARGET2_SRC})
DEAL_II_SETUP_TARGET(${TARGET2})

ADD_EXECUTABLE(${TARGET3} ${TARGET3_SRC})
DEAL_II_SETUP_TARGET(${TARGET3})

ADD_EXECUTABLE(${TARGET4} ${TARGET4_SRC})
DEAL_II_SETUP_TARGET(${TARGET4})

ADD_EXECUTABLE(${TARGET5} ${TARGET5_SRC})
DEAL_II_SETUP_TARGET(${TARGET5})

# Add tests
ADD_SUBDIRECTORY(tests)
ENABLE_TESTING()