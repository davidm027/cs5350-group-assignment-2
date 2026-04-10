cmake_minimum_required(VERSION 3.16)

project(GroupHW2 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(POLICY CMP0167)
  cmake_policy(SET CMP0167 NEW)
endif()

set(CMAKE_TRY_COMPILE_TARGET_TYPE "STATIC_LIBRARY")

find_package(OpenMP REQUIRED)
find_package(Boost CONFIG REQUIRED COMPONENTS program_options)

add_executable(MyAssignment
    main.cpp
    matrix.cpp
)

target_include_directories(MyAssignment PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/ctrack/include)

target_compile_options(MyAssignment PRIVATE ${OpenMP_CXX_FLAGS})

target_link_libraries(MyAssignment
    PRIVATE
        Boost::program_options
        ${OpenMP_CXX_FLAGS}
        ${OpenMP_CXX_LIBRARIES}
)
