cmake_minimum_required(VERSION 3.16)

project(GroupHW2 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CTRACK_DISABLE_EXECUTION_POLICY TRUE)

include(FetchContent)

FetchContent_Declare(
    ctrack
    GIT_REPOSITORY https://github.com/Compaile/ctrack.git
    GIT_TAG main
)

FetchContent_MakeAvailable(ctrack)

find_package(OpenMP REQUIRED)

add_executable(MyAssignment
    main.cpp
    matrix.cpp
)

target_include_directories(MyAssignment PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(MyAssignment
    PRIVATE
        OpenMP::OpenMP_CXX
        ctrack::ctrack
)