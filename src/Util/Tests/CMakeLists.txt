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

SET( THIS_LIBS
  EncinoWavesUtil
  ${EWAV_ALEMBIC_LIBS}
  ${EWAV_HDF5_LIBS}
  ${EWAV_ILMBASE_LIBS}
  ${EWAV_BOOST_LIBS}
  ${EWAV_TBB_LIBS}
  ${EWAV_GL_LIBS}
  ${EWAV_THREAD_LIBS}
  ${EWAV_Z_LIBS} m )

SET( CPP11_LIBS m )

#-******************************************************************************
# CPP11 Test
ADD_EXECUTABLE( test_util_Cpp11 test_Cpp11.cpp )
TARGET_LINK_LIBRARIES( test_util_Cpp11 ${THIS_LIBS} )
ADD_TEST( TEST_util_Cpp11 test_util_Cpp11 )

#-******************************************************************************
# Exception Test
ADD_EXECUTABLE( test_util_Exceptions test_Exceptions.cpp )
TARGET_LINK_LIBRARIES( test_util_Exceptions ${THIS_LIBS} )
ADD_TEST( TEST_util_Exceptions test_util_Exceptions )

#-******************************************************************************
# Functions Test
ADD_EXECUTABLE( test_util_Functions test_Functions.cpp )
TARGET_LINK_LIBRARIES( test_util_Functions ${THIS_LIBS} )
ADD_TEST( TEST_util_Functions test_util_Functions )

