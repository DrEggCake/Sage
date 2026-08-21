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

add_executable(sage
    main.cpp

    brain/Brain.cpp
    brain/Neuron.cpp
    brain/Synapse.cpp

    utils/BrainSave.cpp
)

target_link_libraries(sage
    PRIVATE
        nlohmann_json::nlohmann_json
)

target_include_directories(sage
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)
