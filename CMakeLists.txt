cmake_minimum_required(VERSION 3.10)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(BasisChoice LANGUAGES CXX)

add_executable(basis-choice ./src/main.cpp)

target_include_directories(basis-choice PUBLIC "./src")
