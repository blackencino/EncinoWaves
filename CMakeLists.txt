##-*****************************************************************************
## Copyright 2015 Christopher Jon Horvath
##
## Licensed under the Apache License, Version 2.0 (the "License");
## you may not use this file except in compliance with the License.
## You may obtain a copy of the License at
##
##     http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS,
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
## See the License for the specific language governing permissions and
## limitations under the License.
##-*****************************************************************************

# This is the top level of the EncinoWaves project. Bins and then libs!
# Also, this top level should include any "global" build configurations
CMAKE_MINIMUM_REQUIRED( VERSION 3.0 )
cmake_policy(SET CMP0048 NEW)

# Declare an EncinoWaves Project
PROJECT( EWAV VERSION 1.0.0 )

IF (NOT DEFINED QUIET)
  SET(CMAKE_VERBOSE_MAKEFILE TRUE)
ENDIF()

#-******************************************************************************
# RPATH STUFF
#-******************************************************************************

# use, i.e. don't skip the full RPATH for the build tree
SET(CMAKE_SKIP_BUILD_RPATH  FALSE)

# when building, don't use the install RPATH already
# (but later on when installing)
SET(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)

SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

# add the automatically determined parts of the RPATH
# which point to directories outside the build tree to the install RPATH
SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)


# the RPATH to be used when installing, but only if it's not a system directory
LIST(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES "${CMAKE_INSTALL_PREFIX}/lib" isSystemDir)
IF("${isSystemDir}" STREQUAL "-1")
   SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
ENDIF("${isSystemDir}" STREQUAL "-1")

#-******************************************************************************
#-******************************************************************************
# EMLD
#-******************************************************************************
#-******************************************************************************

# Declare the version
#SET( PROJECT_VERSION_MAJOR 1 )
#SET( PROJECT_VERSION_MINOR 0 )
#SET( PROJECT_VERSION_PATCH 0 )
#SET( PROJECT_VERSION
#     ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH} )
#SET( VERSION ${PROJECT_VERSION} )

MESSAGE(STATUS "CMAKE SYSTEM NAME = ${CMAKE_SYSTEM_NAME}" )

# Set some debug vs opt flags
IF ( "${CMAKE_BUILD_TYPE}" MATCHES "DebugHard" )
  ADD_DEFINITIONS( -DDEBUG=1 -DDEBUG_HARD=1 -UNDEBUG -O0 -g )
ELSEIF ( "${CMAKE_BUILD_TYPE}" MATCHES "Debug" )
  ADD_DEFINITIONS( -DDEBUG=1 -UDEBUG_HARD -UNDEBUG -O0 -g )
ELSEIF ( "${CMAKE_BUILD_TYPE}" MATCHES "Release" )
  #ADD_DEFINITIONS( -DNDEBUG=1 -UDEBUG_HARD -UDEBUG -O3 -g )
  ADD_DEFINITIONS( -DNDEBUG=1 -UDEBUG_HARD -UDEBUG -O3 )
ENDIF()

# FAST MATH
ADD_DEFINITIONS( -ffast-math )

# Windows flag
SET( WINDOWS FALSE )
IF( "${CMAKE_SYSTEM_NAME}" MATCHES "Windows" )
  MESSAGE( FATAL_ERROR "Windows not supported." )
ENDIF()

# Darwin flag
SET( DARWIN FALSE )
IF( "${CMAKE_SYSTEM_NAME}" MATCHES "Darwin" )
  SET( DARWIN TRUE )
ENDIF()

# Linux flag
SET( LINUX FALSE )
IF( "${CMAKE_SYSTEM_NAME}" MATCHES "Linux" )
  SET( LINUX TRUE )
ENDIF()

#-******************************************************************************
# SPECIFY C++ 14 - BECAUSE IT IS AWESOME
#-******************************************************************************
IF( ${DARWIN} )
  #ADD_DEFINITIONS( "-std=c++14" )
  ADD_DEFINITIONS( "-std=c++14 -Wno-parentheses" )
ELSE()
  #ADD_DEFINITIONS( "-std=c++0x" )
ENDIF()

#-******************************************************************************
# SPECIFY PLATFORM DEFINES
#-******************************************************************************

IF( ${DARWIN} )
  ADD_DEFINITIONS( -DPLATFORM_DARWIN -DPLATFORM=DARWIN )
ELSE()
  ADD_DEFINITIONS( -DPLATFORM_LINUX -DPLATFORM=LINUX )
ENDIF()

#-******************************************************************************
# CMAKE MODULE STUFF
#-******************************************************************************

# check in the source directory
set( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/build )

#-******************************************************************************
# TESTING STUFF
#-******************************************************************************
# Globally enable testing.
INCLUDE(CTest)
ENABLE_TESTING()
IF(BUILD_TESTING)
  SET(BUILDNAME "${BUILDNAME}" CACHE STRING "Name of build on the dashboard")
  MARK_AS_ADVANCED(BUILDNAME)
ENDIF(BUILD_TESTING)
MESSAGE( STATUS "Enabled Testing" )

#-******************************************************************************
# FPIC STUFF
#-******************************************************************************

ADD_DEFINITIONS( -fPIC )

#-******************************************************************************
# WARNINGS MANAGEMENT
#-******************************************************************************

# Imath and Boost make fairly heavy use of the register keyword, which makes
# for quite a bit of warnings.
ADD_DEFINITIONS( "-Wno-deprecated-register" )


#-******************************************************************************
# THREAD BUILDING BLOCKS
# We handle these separately from the 'needs', because it's hard to
# use more than one of them on a system at any given time.
#-******************************************************************************

SET( TBB_ROOT $ENV{TBB_ROOT} )
SET( TBB_INCLUDE_PATH "${TBB_ROOT}/include" )
SET( TBB_LINK_PATH "${TBB_ROOT}/lib" )
FIND_LIBRARY( EWAV_TBB_LIB tbb
	      PATHS
	      ${TBB_LINK_PATH}
        NO_DEFAULT_PATH
        NO_CMAKE_ENVIRONMENT_PATH
        NO_SYSTEM_ENVIRONMENT_PATH
        NO_CMAKE_SYSTEM_PATH )
FIND_LIBRARY( EWAV_TBBMALLOC_LIB tbbmalloc
	      PATHS
	      ${TBB_LINK_PATH}
        NO_DEFAULT_PATH
        NO_CMAKE_ENVIRONMENT_PATH
        NO_SYSTEM_ENVIRONMENT_PATH
        NO_CMAKE_SYSTEM_PATH  )

FIND_LIBRARY( EWAV_TBB_DEBUG_LIB tbb_debug
	      PATHS
	      ${TBB_LINK_PATH}
        NO_DEFAULT_PATH
        NO_CMAKE_ENVIRONMENT_PATH
        NO_SYSTEM_ENVIRONMENT_PATH
        NO_CMAKE_SYSTEM_PATH  )
FIND_LIBRARY( EWAV_TBBMALLOC_DEBUG_LIB tbbmalloc_debug
	      PATHS
	      ${TBB_LINK_PATH}
        NO_DEFAULT_PATH
        NO_CMAKE_ENVIRONMENT_PATH
        NO_SYSTEM_ENVIRONMENT_PATH
        NO_CMAKE_SYSTEM_PATH  )

SET( EWAV_TBB_OPT_LIBS
     ${EWAV_TBBMALLOC_LIB}
     ${EWAV_TBB_LIB} )

SET( EWAV_TBB_DEBUG_LIBS
     ${EWAV_TBBMALLOC_DEBUG_LIB}
     ${EWAV_TBB_DEBUG_LIB} )

IF ( "${CMAKE_BUILD_TYPE}" MATCHES "Debug" )
   SET( EWAV_TBB_LIBS ${EWAV_TBB_DEBUG_LIBS} )
ELSE()
   SET( EWAV_TBB_LIBS ${EWAV_TBB_OPT_LIBS} )
ENDIF()

MESSAGE( STATUS "TBB libs: ${EWAV_TBB_LIBS}" )

INCLUDE_DIRECTORIES( ${TBB_INCLUDE_PATH} )

#-******************************************************************************
#-******************************************************************************
# We handle our dependent libraries by having a 'needs' installation where
# everything is installed together. This makes it possible (at all) to
# have the libraries all cross compiled, and to have a single include
# path and link path.  It takes quite some time to set up, but is worth it.
#-******************************************************************************
#-******************************************************************************

SET( EWAVNEEDS_ROOT $ENV{EWAVNEEDS_ROOT} )

SET( ZLIB_ROOT ${EWAVNEEDS_ROOT} )
SET( TIFF_ROOT ${EWAVNEEDS_ROOT} )
SET( JPEG_ROOT ${EWAVNEEDS_ROOT} )
SET( BOOST_ROOT ${EWAVNEEDS_ROOT} )
SET( ILMBASE_ROOT ${EWAVNEEDS_ROOT} )
SET( OPENEXR_ROOT ${EWAVNEEDS_ROOT} )
SET( GLFW_ROOT ${EWAVNEEDS_ROOT} )
SET( GLEW_ROOT ${EWAVNEEDS_ROOT} )
SET( HDF5_ROOT ${EWAVNEEDS_ROOT} )
SET( ALEMBIC_ROOT ${EWAVNEEDS_ROOT} )
SET( FFTW_ROOT ${EWAVNEEDS_ROOT} )

INCLUDE_DIRECTORIES( ${EWAVNEEDS_ROOT}/include )
INCLUDE_DIRECTORIES( ${EWAVNEEDS_ROOT}/include/OpenEXR )

#-******************************************************************************
#-******************************************************************************
# Z LIB
#-******************************************************************************
#-******************************************************************************

IF( ${DARWIN} )
  FIND_PACKAGE( ZLIB REQUIRED )
  SET( EWAV_Z_LIBS ${ZLIB_LIBRARIES} )
ELSE()
  SET( EWAV_Z_LIBS ${EWAVNEEDS_ROOT}/lib/libz.a )
ENDIF()

#-******************************************************************************
#-******************************************************************************
# THREADS
#-******************************************************************************
#-******************************************************************************

IF( ${DARWIN} )
  FIND_PACKAGE( Threads REQUIRED )
  SET( EWAV_THREAD_LIBS ${CMAKE_THREAD_LIBS_INIT} )
ELSE()
  SET( EWAV_THREAD_LIBS pthread )
ENDIF()

#-******************************************************************************
#-******************************************************************************
# X11 and Xrandr stuff (not sure if needed on darwin)
#-******************************************************************************
#-******************************************************************************


IF( ${DARWIN} )


ELSE()

#-******************************************************************************
# THIS IS A VERY INELEGANT WAY TO BUNDLE GLFW REQUIREMENTS.
# I COULDN'T FIND A SEXY FindGLFW.cmake on the net.
#-******************************************************************************

FIND_PACKAGE( Xrandr REQUIRED )
FIND_PACKAGE( X11 REQUIRED )

FIND_LIBRARY( rt_LIBRARIES rt )
FIND_LIBRARY( Xi_LIBRARIES Xi )
FIND_LIBRARY( Xxf86vm_LIBRARIES Xxf86vm )
FIND_LIBRARY( Xrender_LIBRARIES Xrender )
FIND_LIBRARY( Xext_LIBRARIES Xext )
FIND_LIBRARY( xcb_LIBRARIES xcb )
FIND_LIBRARY( Xau_LIBRARIES Xau )

SET( EWAV_XMISC_LIBS
    ${rt_LIBRARIES}
    ${Xrandr_LIBRARIES}
    ${Xi_LIBRARIES}
    ${Xxf86vm_LIBRARIES}
    ${Xrender_LIBRARIES}
    ${Xext_LIBRARIES}
    ${X11_LIBRARIES}
    ${xcb_LIBRARIES}
    ${Xau_LIBRARIES} )

ENDIF()

#-******************************************************************************
# OPEN GL STUFF
#-******************************************************************************
FIND_PACKAGE( OpenGL REQUIRED )
#FIND_PACKAGE( GLUT REQUIRED )

IF( ${DARWIN} )
  SET( EWAV_GLFW_LIBS
     ${EWAVNEEDS_ROOT}/lib/libglfw3.a
     ${OPENGL_LIBRARIES} )
ELSE()

SET( EWAV_GLFW_LIBS
     ${EWAVNEEDS_ROOT}/lib/libglfw3.a
     ${EWAVNEEDS_ROOT}/lib/libGLEW.a
     ${OPENGL_LIBRARIES} )
ENDIF()

IF( ${DARWIN} )
  # Do a darwin OpenGL thing.
  SET( EWAV_GL_LIBS
       ${EWAV_GLFW_LIBS}
       "-framework OpenGL"
       "-framework IOKit"
       "-framework Cocoa"
       "-framework Carbon"
       "-framework Foundation"
       "-framework CoreVideo"
       "-framework QuartzCore" )
ELSE()
  SET( EWAV_GL_LIBS
       ${EWAV_GLFW_LIBS}
       ${EWAVNEEDS_ROOT}/lib/libGLEW.a
       ${EWAV_XMISC_LIBS}
       ${OPENGL_LIBRARIES} )
ENDIF()

#-******************************************************************************
# BOOST STUFF
#-******************************************************************************

#SET( EWAV_BOOST_LIBS
#     ${EWAVNEEDS_ROOT}/lib/libboost_chrono.a
#     ${EWAVNEEDS_ROOT}/lib/libboost_python.a
#     ${EWAVNEEDS_ROOT}/lib/libboost_random.a
#     ${EWAVNEEDS_ROOT}/lib/libboost_regex.a
#     ${EWAVNEEDS_ROOT}/lib/libboost_program_options.a
#     ${EWAVNEEDS_ROOT}/lib/libboost_thread.a
#     ${EWAVNEEDS_ROOT}/lib/libboost_system.a )

SET( EWAV_BOOST_LIBS
     ${EWAVNEEDS_ROOT}/lib/libboost_chrono.a
     ${EWAVNEEDS_ROOT}/lib/libboost_random.a
     ${EWAVNEEDS_ROOT}/lib/libboost_regex.a
     ${EWAVNEEDS_ROOT}/lib/libboost_program_options.a
     ${EWAVNEEDS_ROOT}/lib/libboost_thread.a
     ${EWAVNEEDS_ROOT}/lib/libboost_system.a )

#-******************************************************************************
# ILM BASE
#-******************************************************************************

SET( EWAV_ILMBASE_LIBS
     ${EWAVNEEDS_ROOT}/lib/libImath.a
     ${EWAVNEEDS_ROOT}/lib/libIlmThread.a
     ${EWAVNEEDS_ROOT}/lib/libIex.a
     ${EWAVNEEDS_ROOT}/lib/libHalf.a )

#-******************************************************************************
# OPEN EXR
#-******************************************************************************

SET( EWAV_OPENEXR_LIBS
     ${EWAVNEEDS_ROOT}/lib/libIlmImf.a
     ${EWAVNEEDS_ROOT}/lib/libtiff.a
     ${EWAVNEEDS_ROOT}/lib/libjpeg.a )

#-******************************************************************************
# HDF5
#-******************************************************************************

SET( EWAV_HDF5_LIBS
     ${EWAVNEEDS_ROOT}/lib/libhdf5_hl.a
     ${EWAVNEEDS_ROOT}/lib/libhdf5.a )

#-******************************************************************************
# ALEMBIC
#-******************************************************************************

SET( EWAV_ALEMBIC_LIBS
     ${EWAVNEEDS_ROOT}/lib/libAlembicAbcOpenGL.a
     ${EWAVNEEDS_ROOT}/lib/libAlembicAbcCollection.a
     ${EWAVNEEDS_ROOT}/lib/libAlembicAbcMaterial.a
     ${EWAVNEEDS_ROOT}/lib/libAlembicAbcGeom.a
     ${EWAVNEEDS_ROOT}/lib/libAlembicAbc.a
     ${EWAVNEEDS_ROOT}/lib/libAlembicAbcCoreFactory.a
     ${EWAVNEEDS_ROOT}/lib/libAlembicAbcCoreOgawa.a
     ${EWAVNEEDS_ROOT}/lib/libAlembicAbcCoreHDF5.a
     ${EWAVNEEDS_ROOT}/lib/libAlembicAbcCoreAbstract.a
     ${EWAVNEEDS_ROOT}/lib/libAlembicOgawa.a
     ${EWAVNEEDS_ROOT}/lib/libAlembicUtil.a )

##-*****************************************************************************
# FFTW
##-*****************************************************************************

SET( EWAV_FFTW_LIBS
     ${EWAVNEEDS_ROOT}/lib/libfftw3_threads.a
     ${EWAVNEEDS_ROOT}/lib/libfftw3f_threads.a
     ${EWAVNEEDS_ROOT}/lib/libfftw3.a
     ${EWAVNEEDS_ROOT}/lib/libfftw3f.a )

#-******************************************************************************
# SUBDIRS
#-******************************************************************************

# Recurse into src
ADD_SUBDIRECTORY( src )


