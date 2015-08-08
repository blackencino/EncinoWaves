#-******************************************************************************

FIND_PACKAGE( IlmBase )

IF( ILMBASE_FOUND )
  SET( EWAV_ILMBASE_INCLUDE_DIRECTORY ${EWAV_ILMBASE_INCLUDE_DIRECTORY} )
  SET( EWAV_ILMBASE_LIBRARIES ${EWAV_ILMBASE_LIBS} )
  SET( EWAV_ILMBASE_FOUND 1 CACHE STRING "Set to 1 if IlmBase is found, 0 otherwise" )
ELSE()
  SET( EWAV_ILMBASE_FOUND 0 CACHE STRING "Set to 1 if IlmBase is found, 0 otherwise" )
ENDIF()

