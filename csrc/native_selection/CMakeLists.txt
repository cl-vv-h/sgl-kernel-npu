cmake_minimum_required(VERSION 3.20)
project(sgl_native_selection LANGUAGES NONE)

include("${CMAKE_CURRENT_LIST_DIR}/../cmake/domains.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/../cmake/native_selection.cmake")
# Discovery does not select CANN IDs. Legacy dependency expansion still runs in
# the real CANN build, and its actual selection is passed to Torch and the wheel.
set(SGL_CANN_OPERATORS ALL)
sgl_add_operator_domains()
sgl_add_cpp_operator_domains()
set(type_args)
if(SGL_BUILD_TYPE)
    list(APPEND type_args BUILD_TYPE "${SGL_BUILD_TYPE}")
endif()
sgl_select_native_operators(cann cpp ARCH "${SGL_ARCH_VERSION}"
    ${type_args} OPERATORS ${SGL_NATIVE_OPERATORS})
# IDs contain only letters/digits/underscores, so JSON escaping is unnecessary.
foreach(kind IN ITEMS cann cpp)
    set(items)
    foreach(id IN LISTS ${kind})
        list(APPEND items "\"${id}\"")
    endforeach()
    list(JOIN items ", " ${kind}_json)
endforeach()
sgl_collect_cpp_sources(cpp_sources cpp_definitions ${cpp})
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/selection.json"
    "{\"cann\": [${cann_json}], \"cpp\": [${cpp_json}]}\n")
# Also expose the precise CPP compilation inputs for inspection and CPU tests.
string(REPLACE ";" "\n" cpp_source_text "${cpp_sources}")
string(REPLACE ";" "\n" cpp_definition_text "${cpp_definitions}")
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/cpp-sources.txt" "${cpp_source_text}")
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/cpp-definitions.txt" "${cpp_definition_text}")
message(STATUS "Selected CANN operators: ${cann}")
message(STATUS "Selected CPP operators: ${cpp}")
