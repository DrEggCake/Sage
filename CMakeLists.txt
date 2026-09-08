cmake_minimum_required(VERSION 3.20)

project(sage LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(FetchContent)

FetchContent_Declare(
    nlohmann_json
    URL https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz
)

FetchContent_MakeAvailable(nlohmann_json)

# MuJoCo
set(MUJOCO_DIR "$ENV{HOME}/.mujoco" CACHE PATH "MuJoCo installation directory")
find_library(MUJOCO_LIB mujoco HINTS "${MUJOCO_DIR}/lib")
find_path(MUJOCO_INCLUDE_DIR mujoco/mujoco.h HINTS "${MUJOCO_DIR}/include")

# Sage core library (brain engine)
add_library(sage_core STATIC
    brain/Brain.cpp
    brain/Neuron.cpp
    brain/Synapse.cpp
    utils/BrainSave.cpp
)

target_link_libraries(sage_core
    PRIVATE
        nlohmann_json::nlohmann_json
)

target_include_directories(sage_core
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

# Main sage binary
add_executable(sage main.cpp)
target_link_libraries(sage PRIVATE sage_core)

# Brain engine tests
add_executable(brain_test tests/brain_test.cpp)
target_link_libraries(brain_test PRIVATE sage_core nlohmann_json::nlohmann_json)
target_include_directories(brain_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Cross-simulator test harness
if(MUJOCO_LIB AND MUJOCO_INCLUDE_DIR)
    message(STATUS "MuJoCo found: ${MUJOCO_LIB}")

    add_executable(cross_sim_test
        tests/cross_sim_harness.cpp
    )

    target_link_libraries(cross_sim_test
        PRIVATE
            sage_core
            ${MUJOCO_LIB}
            nlohmann_json::nlohmann_json
    )

    target_include_directories(cross_sim_test
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}
            ${MUJOCO_INCLUDE_DIR}
    )

    target_link_directories(cross_sim_test
        PRIVATE
            "${MUJOCO_DIR}/lib"
    )
else()
    message(WARNING "MuJoCo not found — skipping cross_sim_test build")
endif()
