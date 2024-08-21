# Minimum required version of CMake
cmake_minimum_required(VERSION 3.10)

# Project name and version
project(openCV_Tesseract_test VERSION 1.0)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find OpenCV package
find_package(OpenCV REQUIRED)

# Manually set Tesseract paths (adjust the paths to your installation)
set(Tesseract_INCLUDE_DIRS "/usr/include")
set(Tesseract_LIBRARIES "/usr/lib/x86_64-linux-gnu/libtesseract.so")

# Add the source files to the project
add_executable(OpenCV_Tesseract_test main.cpp LabelOCR.cpp DetectLabel.cpp)
add_executable(OpenCV_Tesseract_train trainSVM.cpp DetectLabel.cpp)

# Include directories for OpenCV
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${Tesseract_INCLUDE_DIRS})

# Link libraries
target_link_libraries(OpenCV_Tesseract_test ${OpenCV_LIBS} ${Tesseract_LIBRARIES})
target_link_libraries(OpenCV_Tesseract_train ${OpenCV_LIBS} ${Tesseract_LIBRARIES})

# Compiler flags
target_compile_options(OpenCV_Tesseract_test PRIVATE -Wall -Wextra -O2)
target_compile_options(OpenCV_Tesseract_train PRIVATE -Wall -Wextra -O2)