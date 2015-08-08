# We shall worry about windowsification later.

#-******************************************************************************
#-******************************************************************************
# FIRST, ILMBASE STUFF
#-******************************************************************************
#-******************************************************************************

IF(NOT DEFINED ILMBASE_ROOT )
    MESSAGE( STATUS "SERIOUSLY WHY AM I HERE" )
    MESSAGE( STATUS "ILMBASE_ROOT: ${ILMBASE_ROOT}" )

    IF ( ${CMAKE_HOST_UNIX} )
        IF( ${DARWIN} )
          # TODO: set to default install path when shipping out
          SET( EWAV_ILMBASE_ROOT NOTFOUND )
        ELSE()
          # TODO: set to default install path when shipping out
          SET( EWAV_ILMBASE_ROOT "/usr/local/ilmbase-1.0.1/" )
        ENDIF()
    ELSE()
        IF ( ${WINDOWS} )
          # TODO: set to 32-bit or 64-bit path
          SET( EWAV_ILMBASE_ROOT "C:/Program Files (x86)/ilmbase-1.0.1/" )
        ELSE()
          SET( EWAV_ILMBASE_ROOT NOTFOUND )
        ENDIF()
    ENDIF()
ELSE()
  SET( EWAV_ILMBASE_ROOT ${ILMBASE_ROOT} )
ENDIF()

MESSAGE( STATUS "EWAV_ILMBASE_ROOT: ${EWAV_ILMBASE_ROOT}" )

SET(LIBRARY_PATHS
    ${EWAV_ILMBASE_ROOT}/lib
    ~/Library/Frameworks
    /Library/Frameworks
    /usr/local/lib
    /usr/lib
    /sw/lib
    /opt/local/lib
    /opt/csw/lib
    /opt/lib
    /usr/freeware/lib64
)

IF( DEFINED ILMBASE_LIBRARY_DIR )
  SET( LIBRARY_PATHS ${ILMBASE_LIBRARY_DIR} ${LIBRARY_PATHS} )
ENDIF()

SET(INCLUDE_PATHS
    ${EWAV_ILMBASE_ROOT}/include/OpenEXR/
    ~/Library/Frameworks
    /Library/Frameworks
    /usr/local/include/OpenEXR/
    /usr/local/include
    /usr/include
    /usr/include/OpenEXR
    /sw/include # Fink
    /opt/local/include # DarwinPorts
    /opt/csw/include # Blastwave
    /opt/include
    /usr/freeware/include
)

FIND_PATH( EWAV_ILMBASE_INCLUDE_DIRECTORY ImathMath.h
           PATHS
           ${INCLUDE_PATHS}
           NO_DEFAULT_PATH
           NO_CMAKE_ENVIRONMENT_PATH
           NO_CMAKE_PATH
           NO_SYSTEM_ENVIRONMENT_PATH
           NO_CMAKE_SYSTEM_PATH
           DOC "The directory where ImathMath.h resides" )

IF( NOT DEFINED EWAV_ILMBASE_HALF_LIB )
  FIND_LIBRARY( EWAV_ILMBASE_HALF_LIB Half
                PATHS
                ${LIBRARY_PATHS}
                NO_DEFAULT_PATH
                NO_CMAKE_ENVIRONMENT_PATH
                NO_CMAKE_PATH
                NO_SYSTEM_ENVIRONMENT_PATH
                NO_CMAKE_SYSTEM_PATH
                DOC "The Half library" )
ENDIF()

IF( NOT DEFINED EWAV_ILMBASE_IEX_LIB )
  FIND_LIBRARY( EWAV_ILMBASE_IEX_LIB Iex
                PATHS
                ${LIBRARY_PATHS}
                NO_DEFAULT_PATH
                NO_CMAKE_ENVIRONMENT_PATH
                NO_CMAKE_PATH
                NO_SYSTEM_ENVIRONMENT_PATH
                NO_CMAKE_SYSTEM_PATH
                DOC "The Iex library" )
ENDIF()

IF( NOT DEFINED EWAV_ILMBASE_ILMTHREAD_LIB )
  FIND_LIBRARY( EWAV_ILMBASE_ILMTHREAD_LIB IlmThread
                 PATHS
                 ${LIBRARY_PATHS}
                 NO_DEFAULT_PATH
                 NO_CMAKE_ENVIRONMENT_PATH
                 NO_CMAKE_PATH
                 NO_SYSTEM_ENVIRONMENT_PATH
                 NO_CMAKE_SYSTEM_PATH
                 DOC "The IlmThread library" )
ENDIF()

IF( NOT DEFINED EWAV_ILMBASE_IMATH_LIB )
  FIND_LIBRARY( EWAV_ILMBASE_IMATH_LIB Imath
                PATHS
                ${LIBRARY_PATHS}
                NO_DEFAULT_PATH
                NO_CMAKE_ENVIRONMENT_PATH
                NO_CMAKE_PATH
                NO_SYSTEM_ENVIRONMENT_PATH
                NO_CMAKE_SYSTEM_PATH
                DOC "The Imath library" )
ENDIF()


IF ( ${EWAV_ILMBASE_HALF_LIB} STREQUAL "EWAV_ILMBASE_HALF_LIB-NOTFOUND" )
  MESSAGE( FATAL_ERROR "ilmbase libraries (Half, Iex, IlmThread, Imath) not found, required 1" )
ENDIF()

IF ( ${EWAV_ILMBASE_IEX_LIB} STREQUAL "EWAV_ILMBASE_IEX_LIB-NOTFOUND" )
  MESSAGE( FATAL_ERROR "ilmbase libraries (Half, Iex, IlmThread, Imath) not found, required 2" )
ENDIF()

IF ( ${EWAV_ILMBASE_ILMTHREAD_LIB} STREQUAL "EWAV_ILMBASE_ILMTHREAD_LIB-NOTFOUND" )
  MESSAGE( FATAL_ERROR "ilmbase libraries (Half, Iex, IlmThread, Imath) not found, required 3" )
ENDIF()

IF ( ${EWAV_ILMBASE_IMATH_LIB} STREQUAL "EWAV_ILMBASE_IMATH_LIB-NOTFOUND" )
  MESSAGE( FATAL_ERROR "ilmbase libraries (Half, Iex, IlmThread, Imath) not found, required 4" )
ENDIF()

IF ( ${EWAV_ILMBASE_INCLUDE_DIRECTORY} STREQUAL "EWAV_ILMBASE_INCLUDE_DIRECTORY-NOTFOUND" )
  MESSAGE( FATAL_ERROR "ilmbase header files not found, required: EWAV_ILMBASE_ROOT: ${EWAV_ILMBASE_ROOT}" )
ENDIF()


MESSAGE( STATUS "ILMBASE INCLUDE PATH: ${EWAV_ILMBASE_INCLUDE_DIRECTORY}" )
MESSAGE( STATUS "HALF LIB: ${EWAV_ILMBASE_HALF_LIB}" )
MESSAGE( STATUS "IEX LIB: ${EWAV_ILMBASE_IEX_LIB}" )
MESSAGE( STATUS "ILMTHREAD LIB: ${EWAV_ILMBASE_ILMTHREAD_LIB}" )
MESSAGE( STATUS "IMATH LIB: ${EWAV_ILMBASE_IMATH_LIB}" )

SET( ILMBASE_FOUND TRUE )
SET( EWAV_ILMBASE_LIBS
       ${EWAV_ILMBASE_IMATH_LIB}
       ${EWAV_ILMBASE_HALF_LIB}
       ${EWAV_ILMBASE_IEX_LIB}
       ${EWAV_ILMBASE_ILMTHREAD_LIB} )


