# cmake_minimum_required(VERSION 2.8)

# project(ImageEditing)
# set(SRC_DIR ${PROJECT_SOURCE_DIR}/src/)
# set(INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include/)
# set(LIB_DIR ${PROJECT_SOURCE_DIR}/lib/)

# include_directories(${INCLUDE_DIR})
# include_directories(${LIB_DIR})

# # add C/C++ > preprocess: "XKEYCHECK_H"
# add_Definitions("-D_XKEYCHECK_H")
# add_definitions(-DPROJECT_DIR="${PROJECT_SOURCE_DIR}")

# add_executable(ImageEditing 
#     ${SRC_DIR}Main.cpp
#     ${SRC_DIR}Globals.h
#     ${SRC_DIR}Globals.inl
#     ${SRC_DIR}ImageWidget.h
#     ${SRC_DIR}ImageWidget.cpp
#     ${SRC_DIR}ScriptHandler.h
#     ${SRC_DIR}ScriptHandler.cpp
#     ${SRC_DIR}TargaImage.h
#     ${SRC_DIR}TargaImage.cpp)

# add_library(libtarga ${SRC_DIR}libtarga.h ${SRC_DIR}libtarga.c)

# target_link_libraries(ImageEditing 
# debug ${LIB_DIR}Debug/fltk_formsd.lib      optimized ${LIB_DIR}Release/fltk_forms.lib
# debug ${LIB_DIR}Debug/fltk_gld.lib         optimized ${LIB_DIR}Release/fltk_gl.lib
# debug ${LIB_DIR}Debug/fltk_imagesd.lib     optimized ${LIB_DIR}Release/fltk_images.lib
# debug ${LIB_DIR}Debug/fltk_jpegd.lib       optimized ${LIB_DIR}Release/fltk_jpeg.lib
# debug ${LIB_DIR}Debug/fltk_pngd.lib        optimized ${LIB_DIR}Release/fltk_png.lib
# debug ${LIB_DIR}Debug/fltk_zd.lib          optimized ${LIB_DIR}Release/fltk_z.lib
# debug ${LIB_DIR}Debug/fltkd.lib            optimized ${LIB_DIR}Release/fltk.lib)

# target_link_libraries(ImageEditing libtarga)

cmake_minimum_required(VERSION 3.5)

project(ImageEditing)
set(SRC_DIR ${PROJECT_SOURCE_DIR}/src/)

# Set the C++ standard to C++11
set(CMAKE_CXX_STANDARD 11)

# Find the FLTK library
find_package(FLTK REQUIRED)

# Add the program executable
add_executable(ImageEditing 
    ${SRC_DIR}Main.cpp
    ${SRC_DIR}Globals.h
    ${SRC_DIR}Globals.inl
    ${SRC_DIR}ImageWidget.h
    ${SRC_DIR}ImageWidget.cpp
    ${SRC_DIR}ScriptHandler.h
    ${SRC_DIR}ScriptHandler.cpp
    ${SRC_DIR}TargaImage.h
    ${SRC_DIR}TargaImage.cpp)

add_library(libtarga ${SRC_DIR}libtarga.h ${SRC_DIR}libtarga.c)
target_link_libraries(ImageEditing libtarga)

# Link the FLTK library
target_link_libraries(ImageEditing ${FLTK_LIBRARIES})