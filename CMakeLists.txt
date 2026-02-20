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
SET(TARGET "blood_flow_system")

# Declare all source files for each target:
SET(TARGET_SRC
  source/blood_flow_system.cc
  source/main.cc
  source/constants.cc
  source/function.cc
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

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Create first executable target
ADD_EXECUTABLE(${TARGET} ${TARGET_SRC})
DEAL_II_SETUP_TARGET(${TARGET})

file(GLOB LIB_SRC source/*.cc)
list(REMOVE_ITEM LIB_SRC ${CMAKE_CURRENT_SOURCE_DIR}/source/main.cc)
ADD_LIBRARY(test_library SHARED ${LIB_SRC})
set(TEST_LIBRARIES test_library)
DEAL_II_SETUP_TARGET(test_library)

ADD_DEFINITIONS(-DPRM_DIR="${CMAKE_CURRENT_SOURCE_DIR}/parameters/")

# Add tests
ADD_SUBDIRECTORY(tests)
ENABLE_TESTING()