CMAKE_MINIMUM_REQUIRED(VERSION 3.23)

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

file(GLOB LIB_SRC source/*.cc)
ADD_LIBRARY(test_library SHARED ${LIB_SRC})
set(TEST_LIBRARIES test_library)
DEAL_II_SETUP_TARGET(test_library)

INCLUDE_DIRECTORIES(include)

# Mirror parameters/ into the build tree so tests and examples always use
# configured files.  Template files end in .prm.in and are expanded with the
# source tree location baked in.
set(SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
file(GLOB_RECURSE PARAMETER_FILES CONFIGURE_DEPENDS
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}/parameters"
  "${CMAKE_CURRENT_SOURCE_DIR}/parameters/*")
foreach(parameter_file IN LISTS PARAMETER_FILES)
  set(parameter_source "${CMAKE_CURRENT_SOURCE_DIR}/parameters/${parameter_file}")
  set(parameter_destination "${CMAKE_CURRENT_BINARY_DIR}/parameters/${parameter_file}")
  get_filename_component(parameter_destination_dir "${parameter_destination}" DIRECTORY)
  file(MAKE_DIRECTORY "${parameter_destination_dir}")

  if(parameter_file MATCHES "\\.prm\\.in$")
    string(REGEX REPLACE "\\.in$" "" parameter_destination "${parameter_destination}")
    configure_file("${parameter_source}" "${parameter_destination}" @ONLY)
  else()
    configure_file("${parameter_source}" "${parameter_destination}" COPYONLY)
  endif()
endforeach()

# Create one executable per file in apps/
file(GLOB APP_SOURCES apps/*.cc)
foreach(app_src ${APP_SOURCES})
  get_filename_component(app_name ${app_src} NAME_WE)
  add_executable(${app_name} ${app_src})
  target_link_libraries(${app_name} test_library)
  DEAL_II_SETUP_TARGET(${app_name})
endforeach()

ADD_DEFINITIONS(-DPRM_DIR="${CMAKE_CURRENT_BINARY_DIR}/parameters/")

# Add tests
ADD_SUBDIRECTORY(tests)
ENABLE_TESTING()
