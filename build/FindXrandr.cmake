# FROM: http://code.google.com/p/xceed/source/browse/trunk/cmake/FindXrandr.cmake?r=6
# - Find the Xrandr include file and library
#

SET(Xrandr_INC_SEARCH_PATH
    /usr/X11R6/include
    /usr/local/include
    /usr/include/X11
    /usr/openwin/include
    /usr/openwin/share/include
    /opt/graphics/OpenGL/include
    /usr/include)

SET(Xrandr_LIB_SEARCH_PATH
    /usr/X11R6/lib
    /usr/local/lib
    /usr/openwin/lib
    /usr/lib)


FIND_PATH(Xrandr_INCLUDE_DIR X11/extensions/Xrandr.h
          ${Xrandr_INC_SEARCH_PATH})

FIND_LIBRARY(Xrandr_LIBRARIES NAMES Xrandr PATH ${Xrandr_LIB_SEARCH_PATH})

IF (Xrandr_INCLUDE_DIR AND Xrandr_LIBRARIES)
    SET(Xrandr_FOUND TRUE)
ENDIF (Xrandr_INCLUDE_DIR AND Xrandr_LIBRARIES)

IF (Xrandr_FOUND)
    INCLUDE(CheckLibraryExists)

    CHECK_LIBRARY_EXISTS(${Xrandr_LIBRARIES}
                         "XRRConfigSizes"
                         ${Xrandr_LIBRARIES}
                         Xrandr_HAS_CONFIG)

    IF (NOT Xrandr_HAS_CONFIG AND Xrandr_FIND_REQUIRED)
        MESSAGE(FATAL_ERROR "Could NOT find Xrandr")
    ENDIF (NOT Xrandr_HAS_CONFIG AND Xrandr_FIND_REQUIRED)
ENDIF (Xrandr_FOUND)

MARK_AS_ADVANCED(
    Xrandr_INCLUDE_DIR
    Xrandr_LIBRARIES
    )

