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

# C++ files for this project
SET( H_FILES
     Exception.h
     Foundation.h
     Functions.h
     Timer.h
     All.h )

SET( CXX_FILES
     Exception.cpp )

SET( SOURCE_FILES ${CXX_FILES} ${H_FILES} )

ADD_LIBRARY( EncinoWavesUtil ${SOURCE_FILES} )

INSTALL( TARGETS EncinoWavesUtil
         LIBRARY DESTINATION lib
         ARCHIVE DESTINATION lib/static )

INSTALL( FILES ${H_FILES}
         DESTINATION include/EncinoWaves/Util
         PERMISSIONS OWNER_READ GROUP_READ WORLD_READ )

ADD_SUBDIRECTORY( Tests )
