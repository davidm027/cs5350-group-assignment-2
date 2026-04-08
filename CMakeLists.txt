cmake_minimum_required(VERSION 3.16)

project(GroupHW2 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CTRACK_DISABLE_EXECUTION_POLICY TRUE)

if(POLICY CMP0167)
  cmake_policy(SET CMP0167 NEW)
endif()


include(FetchContent)

FetchContent_Declare(
    ctrack
    GIT_REPOSITORY https://github.com/Compaile/ctrack.git
    GIT_TAG main
)

FetchContent_MakeAvailable(ctrack)

find_package(OpenMP REQUIRED)
find_package(ctrack REQUIRED)
find_package(Boost REQUIRED COMPONENTS program_options)

add_executable(MyAssignment
    main.cpp
    matrix.cpp
)

include_directories(${Boost_INCLUDE_DIRS})

target_include_directories(MyAssignment PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(MyAssignment
    PRIVATE
        OpenMP::OpenMP_CXX
        ctrack::ctrack
        ${Boost_PROGRAM_OPTIONS_LIBRARY}
)