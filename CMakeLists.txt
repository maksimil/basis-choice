cmake_minimum_required(VERSION 3.10)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(BasisChoice LANGUAGES CXX)

file(GLOB EXE_SRC ./src/*.cpp)
file(GLOB LIB_SRC ./lib/*.cpp)

add_executable(basis-choice ${LIB_SRC} ${EXE_SRC})

if (${USE_EIGEN})
    find_package(Eigen3 3.3 REQUIRED NO_MODULE)
    target_link_libraries(basis-choice Eigen3::Eigen)
    target_compile_definitions(basis-choice PRIVATE USE_EIGEN)
endif()

if (${PROFILE_BUILD})
    target_compile_options(basis-choice PRIVATE -fno-omit-frame-pointer -fno-inline)
endif()

target_compile_features(basis-choice PRIVATE cxx_std_11)
target_compile_options(basis-choice PRIVATE -Wall -Wextra -Wpedantic -Werror)
target_include_directories(basis-choice PRIVATE ./src ./lib)
