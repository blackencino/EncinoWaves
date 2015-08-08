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

SET( H_FILES
     All.h
     Foundation.h
     GLCamera.h
     MeshDrawHelper.h
     PointsDrawHelper.h
     LinesDrawHelper.h
     Sim.h
     StdShaders.h
     ViewerGLFW.h )

SET( CXX_FILES
     GLCamera.cpp
     MeshDrawHelper.cpp
     PointsDrawHelper.cpp
     LinesDrawHelper.cpp
     Sim.cpp
     StdShaders.cpp
     ViewerGLFW.cpp )

SET( SOURCE_FILES ${CXX_FILES} ${H_FILES} )

# Create the library
ADD_LIBRARY( EncinoWavesSimpleSimViewer ${SOURCE_FILES} )

# Recurse into test subdirectory
ADD_SUBDIRECTORY( Tests )

#-******************************************************************************
#-******************************************************************************
# INSTALLATION
#-******************************************************************************
#-******************************************************************************

INSTALL( TARGETS EncinoWavesSimpleSimViewer
         LIBRARY DESTINATION lib
         ARCHIVE DESTINATION lib/static )

INSTALL( FILES ${H_FILES}
         DESTINATION include/EncinoWaves/SimpleSimViewer
         PERMISSIONS OWNER_READ GROUP_READ WORLD_READ )

