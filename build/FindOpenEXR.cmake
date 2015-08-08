

# We shall worry about windowsification later.

#-******************************************************************************
#-******************************************************************************
# NOW, OPENEXR STUFF. EXR IS OPTIONAL, WHERASE ILMBASE IS NOT
#-******************************************************************************
#-******************************************************************************

IF(NOT DEFINED OPENEXR_ROOT)
    IF ( ${CMAKE_HOST_UNIX} )
        IF( ${DARWIN} )
          # TODO: set to default install path when shipping out
          SET( EWAV_OPENEXR_ROOT NOTFOUND )
        ELSE()
          # TODO: set to default install path when shipping out
          SET( EWAV_OPENEXR_ROOT "/usr/local/openexr-1.6.1/" )
        ENDIF()
    ELSE()
        IF ( ${WINDOWS} )
          # TODO: set to 32-bit or 64-bit path
          SET( EWAV_OPENEXR_ROOT NOTFOUND )
        ELSE()
          SET( EWAV_OPENEXR_ROOT NOTFOUND )
        ENDIF()
    ENDIF()
ELSE()
  SET( EWAV_OPENEXR_ROOT ${OPENEXR_ROOT} )
ENDIF()

IF(NOT $ENV{OPENEXR_ROOT}x STREQUAL "x")
  SET( EWAV_OPENEXR_ROOT $ENV{OPENEXR_ROOT})
ELSE()
  SET( ENV{OPENEXR_ROOT} ${OPENEXR_ROOT} )
ENDIF()


SET(LIBRARY_PATHS
    ${EWAV_OPENEXR_ROOT}/lib
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

SET(INCLUDE_PATHS
    ${EWAV_OPENEXR_ROOT}/include/OpenEXR/
    ~/Library/Frameworks
    /Library/Frameworks
    /usr/local/include/OpenEXR/
    /usr/local/include
    /usr/include
    /sw/include # Fink
    /opt/local/include # DarwinPorts
    /opt/csw/include # Blastwave
    /opt/include
    /usr/freeware/include
)

FIND_PATH( EWAV_OPENEXR_INCLUDE_PATH ImfRgba.h
           PATHS
           ${INCLUDE_PATHS}
           NO_DEFAULT_PATH
           NO_CMAKE_ENVIRONMENT_PATH
           NO_CMAKE_PATH
           NO_SYSTEM_ENVIRONMENT_PATH
           NO_CMAKE_SYSTEM_PATH
           DOC "The directory where ImfRgba.h resides" )

FIND_LIBRARY( EWAV_OPENEXR_ILMIMF_LIB IlmImf
              PATHS
              ${LIBRARY_PATHS}
              NO_DEFAULT_PATH
              NO_CMAKE_ENVIRONMENT_PATH
              NO_CMAKE_PATH
              NO_SYSTEM_ENVIRONMENT_PATH
              NO_CMAKE_SYSTEM_PATH
              DOC "The IlmImf library" )


SET( OPENEXR_FOUND TRUE )

IF ( ${EWAV_OPENEXR_INCLUDE_PATH} STREQUAL "EWAV_OPENEXR_INCLUDE_PATH-NOTFOUND" )
  MESSAGE( STATUS "OpenEXR include path not found, disabling" )
  SET( OPENEXR_FOUND FALSE )
ENDIF()

IF ( ${EWAV_OPENEXR_ILMIMF_LIB} STREQUAL "EWAV_OPENEXR_ILMIMF_LIB-NOTFOUND" )
  MESSAGE( STATUS "OpenEXR libraries not found, disabling" )
  SET( OPENEXR_FOUND FALSE )
  SET( EWAV_OPENEXR_LIBS NOTFOUND )
ENDIF()

IF (OPENEXR_FOUND)
  MESSAGE( STATUS "OPENEXR INCLUDE PATH: ${EWAV_OPENEXR_INCLUDE_PATH}" )
  SET( EWAV_OPENEXR_LIBS ${EWAV_OPENEXR_ILMIMF_LIB} )
ENDIF()


