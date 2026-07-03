# -------------------------------------------------------------------
# CLI configuration:
#   - Native release  : cmake -DCMAKE_BUILD_TYPE=Release ..
#   - Native debug    : cmake -DCMAKE_BUILD_TYPE=Debug ..
#
# Then: make -j12
# -------------------------------------------------------------------
cmake_minimum_required(VERSION 3.5)
project(arch VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

# ============================================================================
# FetchContent for dependencies
# ============================================================================
include(FetchContent)

# ----------------------------------------------------------------------------
# EIGEN Library
# ----------------------------------------------------------------------------
message(STATUS "Fetching Arch library...")
FetchContent_Declare(
    arch
    GIT_REPOSITORY https://github.com/xaliphostes/arch3.git
    GIT_TAG        main
    GIT_SHALLOW    TRUE
    SOURCE_DIR ${CMAKE_SOURCE_DIR}/extern/arch
    BINARY_DIR ${CMAKE_BINARY_DIR}/extern/arch-build
)
FetchContent_MakeAvailable(arch)
FetchContent_GetProperties(arch SOURCE_DIR ARCH_INCLUDE_DIR)

# ----------------------------------------------------------------------------
# ROSETTA Library
# ----------------------------------------------------------------------------
message(STATUS "Fetching Rosetta library...")
FetchContent_Declare(
    Rosetta
    GIT_REPOSITORY https://github.com/xaliphostes/rosetta.git
    GIT_TAG        main
    GIT_SHALLOW    TRUE
    SOURCE_DIR ${CMAKE_SOURCE_DIR}/extern/rosetta
    BINARY_DIR ${CMAKE_BINARY_DIR}/extern/rosetta-build
)
FetchContent_MakeAvailable(rosetta)
FetchContent_GetProperties(rosetta SOURCE_DIR ROSETTA_SOURCE_DIR)
set(ROSETTA_INCLUDE_DIR "${ROSETTA_SOURCE_DIR}/include")

# The rosetta_gen code generator lives in tools/ and has its
# own standalone CMakeLists.txt — build it as a subdirectory.
# NOTE: rosetta_gen requires CMake >= 3.28.
#
# The fetched arch3 dependency (extern/arch) already builds rosetta_gen
# into this same binary dir when its BUILD_ROSETTA_GEN option is ON, so
# only add it here if that hasn't already defined the target — otherwise
# CMake errors on the reused binary directory / duplicate target.
if(NOT TARGET rosetta_gen)
    message(STATUS "Building rosetta_gen tool...")
    add_subdirectory(
        ${ROSETTA_SOURCE_DIR}/tools/rosetta_gen
        ${CMAKE_BINARY_DIR}/extern/rosetta_gen-build
    )
endif()
