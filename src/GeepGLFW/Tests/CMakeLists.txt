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

SET( GLFW_TEST_LIBS
  EncinoWavesGeepGLFW
  EncinoWavesUtil
  ${EWAV_GL_LIBS}
  ${EWAV_BOOST_LIBS}
  ${EWAV_THREAD_LIBS}
  ${EWAV_Z_LIBS} m )

ADD_EXECUTABLE( glfw_helloWorld glfw_helloWorld.cpp )
TARGET_LINK_LIBRARIES( glfw_helloWorld ${GLFW_TEST_LIBS} )

ADD_EXECUTABLE( glfw_helloWorld2 glfw_helloWorld2.cpp )
TARGET_LINK_LIBRARIES( glfw_helloWorld2 ${GLFW_TEST_LIBS} )
