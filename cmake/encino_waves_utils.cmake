function(encino_waves_define_library libname)
  set(options)
  set(one_value_args)
  set(multi_value_args SOURCES HEADERS DEPENDENCIES PRIVATE_DEPS)
  cmake_parse_arguments(ENCINO_WAVES_CURLIB 
                        "${options}"
                        "${one_value_args}"
                        "${multi_value_args}"
                        ${ARGN})
  add_library(${libname} STATIC
      ${ENCINO_WAVES_CURLIB_HEADERS}
      ${ENCINO_WAVES_CURLIB_SOURCES})

  target_compile_features(${libname} PUBLIC cxx_std_17)
  set_target_properties(${libname} PROPERTIES 
      CXX_STANDARD_REQUIRED ON 
      CXX_EXTENSIONS OFF
      POSITION_INDEPENDENT_CODE ON
  )
  set_property(TARGET ${libname} 
      PROPERTY PUBLIC_HEADER
      ${ENCINO_WAVES_CURLIB_HEADERS})
  if (ENCINO_WAVES_CURLIB_DEPENDENCIES)
      target_link_libraries(${libname} PUBLIC ${ENCINO_WAVES_CURLIB_DEPENDENCIES})
  endif()
  if (ENCINO_WAVES_CURLIB_PRIVATE_DEPS)
      target_link_libraries(${libname} PRIVATE ${ENCINO_WAVES_CURLIB_PRIVATE_DEPS})
  endif()
  target_include_directories(${libname} 
      PUBLIC
       $<BUILD_INTERFACE:${encino_waves_SOURCE_DIR}/src>
       $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/encino_waves>
  )

  add_library(encino_waves::${libname} ALIAS ${libname})

  install(TARGETS ${libname}
      EXPORT encino_waves
      RUNTIME DESTINATION bin
      LIBRARY DESTINATION lib
      ARCHIVE DESTINATION lib
      PUBLIC_HEADER DESTINATION include/encino_waves/${libname}
  )
endfunction()

function(encino_waves_define_executable binname)
  set(options)
  set(one_value_args)
  set(multi_value_args SOURCES DEPENDENCIES)
  cmake_parse_arguments(ENCINO_WAVES_CURLIB 
                        "${options}"
                        "${one_value_args}"
                        "${multi_value_args}"
                        ${ARGN})
  add_executable(${binname}
      ${ENCINO_WAVES_CURLIB_SOURCES})

  target_compile_features(${binname} PUBLIC cxx_std_17)
  set_target_properties(${binname} PROPERTIES 
      CXX_STANDARD_REQUIRED ON 
      CXX_EXTENSIONS OFF
      POSITION_INDEPENDENT_CODE ON
  )
  if (ENCINO_WAVES_CURLIB_DEPENDENCIES)
      target_link_libraries(${binname} PUBLIC ${ENCINO_WAVES_CURLIB_DEPENDENCIES})
  endif()
  target_include_directories(${binname} 
      PUBLIC
       $<BUILD_INTERFACE:${encino_waves_SOURCE_DIR}/src>
       $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/encino_waves>
  )

  add_executable(encino_waves::${binname} ALIAS ${binname})

  install(TARGETS ${binname}
      EXPORT encino_waves
      RUNTIME DESTINATION bin
      LIBRARY DESTINATION lib
      ARCHIVE DESTINATION lib
      PUBLIC_HEADER DESTINATION include/encino_waves/${binname}
  )
endfunction()


function(encino_waves_define_test testname)
    if (ENCINO_WAVES_DO_TESTS)
        message(STATUS "Creating test: ${testname}")
        set(options)
        set(one_value_args)
        set(multi_value_args SOURCES DEPENDENCIES)
        cmake_parse_arguments(ENCINO_WAVES_CURLIB 
                              "${options}"
                              "${one_value_args}"
                              "${multi_value_args}"
                              ${ARGN})
        add_executable(${testname} ${ENCINO_WAVES_CURLIB_SOURCES})
        target_compile_features(${testname} PUBLIC cxx_std_17)
        set_target_properties(${testname} PROPERTIES 
            CXX_STANDARD_REQUIRED ON 
            CXX_EXTENSIONS OFF
            POSITION_INDEPENDENT_CODE ON
        )
        if (ENCINO_WAVES_CURLIB_DEPENDENCIES)
            target_link_libraries(${testname} PUBLIC ${ENCINO_WAVES_CURLIB_DEPENDENCIES})
        endif()
        target_include_directories(${testname} 
            PUBLIC
             $<BUILD_INTERFACE:${encino_waves_SOURCE_DIR}>
             $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
        )

        add_test(NAME TEST_${testname} 
            COMMAND ${testname}
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    endif()
endfunction()