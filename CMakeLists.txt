cmake_minimum_required(VERSION 3.14)
project(LLMTextCore LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# MSVCが日本語コメント入りのUTF-8ソースを誤解釈しないようにする
if (MSVC)
    add_compile_options(/utf-8)
endif()

# llama.cppを静的ライブラリとしてビルドし、最終的に1つのDLLにまとめる
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)

# CUDA環境が無い場合のビルドエラーを避けるため、明示的にOFFにしておく
# (GPUを使いたい場合はこの行を消すか、コマンドラインで -DGGML_CUDA=ON を指定)
set(GGML_CUDA OFF CACHE BOOL "" FORCE)

# ----- llama.cppの取得とバージョン固定 -----
# 指定コミットに固定する。動作確認済みのハッシュ。
set(LLAMA_CPP_PIN_COMMIT "8e5a58d")

find_package(Git REQUIRED)

if (NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/.git")
    message(STATUS "llama.cpp が見つからないため clone します...")
    execute_process(
        COMMAND ${GIT_EXECUTABLE} clone https://github.com/ggml-org/llama.cpp llama.cpp
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        RESULT_VARIABLE LLAMA_CLONE_RESULT
    )
    if (NOT LLAMA_CLONE_RESULT EQUAL 0)
        message(FATAL_ERROR "llama.cpp の clone に失敗しました")
    endif()
endif()

# 現在のコミットを確認し、目的のコミットと違えばfetch+checkoutする
execute_process(
    COMMAND ${GIT_EXECUTABLE} rev-parse --short HEAD
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp"
    OUTPUT_VARIABLE LLAMA_CPP_CURRENT_COMMIT
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

if (NOT LLAMA_CPP_CURRENT_COMMIT STREQUAL LLAMA_CPP_PIN_COMMIT)
    message(STATUS "llama.cpp を ${LLAMA_CPP_PIN_COMMIT} に固定します (現在: ${LLAMA_CPP_CURRENT_COMMIT})")
    execute_process(
        COMMAND ${GIT_EXECUTABLE} fetch origin ${LLAMA_CPP_PIN_COMMIT}
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp"
        RESULT_VARIABLE LLAMA_FETCH_RESULT
    )
    execute_process(
        COMMAND ${GIT_EXECUTABLE} checkout ${LLAMA_CPP_PIN_COMMIT}
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp"
        RESULT_VARIABLE LLAMA_CHECKOUT_RESULT
    )
    if (NOT LLAMA_CHECKOUT_RESULT EQUAL 0)
        message(FATAL_ERROR "llama.cpp の checkout (${LLAMA_CPP_PIN_COMMIT}) に失敗しました")
    endif()
endif()

add_subdirectory(llama.cpp)

# ----- バージョン情報の埋め込み -----
# LLMTextCore自体のバージョン。リリースごとに更新する。
set(LLMTEXTCORE_VERSION "0.8.2")

# llama.cppサブモジュールのコミットハッシュを取得 (失敗時は "unknown")
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

# ----- LLMTextCore.dll -----
# .def でエクスポート名を装飾なしに固定する (32bit/64bitどちらでも同じ名前で呼べるように)
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

# ----- LLMTextCore.exe (CLI / HSPからpipeexecで使う想定) -----
add_executable(llmtextcore_cli llmtextcore_cli.c)
set_target_properties(llmtextcore_cli PROPERTIES OUTPUT_NAME "LLMTextCore")

target_link_libraries(llmtextcore_cli PRIVATE llmtextcore)
target_include_directories(llmtextcore_cli PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(llmtextcore_cli PRIVATE Shell32)
