cmake_minimum_required(VERSION 3.14)
project(LLMTextCore LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (MSVC)
    add_compile_options(/utf-8)
endif()

set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)

set(GGML_CUDA OFF CACHE BOOL "" FORCE)

add_subdirectory(llama.cpp)

set(LLMTEXTCORE_VERSION "0.4.0")

find_package(Git QUIET)
set(LLAMA_CPP_COMMIT "unknown")
if (GIT_FOUND AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/.git")
    execute_process(
        COMMAND ${GIT_EXECUTABLE} rev-parse --short HEAD
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp"
        OUTPUT_VARIABLE LLAMA_CPP_COMMIT
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
endif()

add_library(llmtextcore SHARED llmtextcore.cpp llmtextcore.def)
set_target_properties(llmtextcore PROPERTIES OUTPUT_NAME "LLMTextCore")

target_compile_definitions(llmtextcore PRIVATE
    LLMTEXTCORE_EXPORTS
    LLMTEXTCORE_VERSION="${LLMTEXTCORE_VERSION}"
    LLAMA_CPP_COMMIT="${LLAMA_CPP_COMMIT}"
)

target_include_directories(llmtextcore PRIVATE
    llama.cpp/include
    llama.cpp/ggml/include
)

target_link_libraries(llmtextcore PRIVATE llama)

add_executable(llmtextcore_cli llmtextcore_cli.c)
set_target_properties(llmtextcore_cli PROPERTIES OUTPUT_NAME "LLMTextCore")

target_link_libraries(llmtextcore_cli PRIVATE llmtextcore)
target_include_directories(llmtextcore_cli PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

if (WIN32)
    target_link_libraries(llmtextcore_cli PRIVATE Shell32)
endif()
