#define LLMTEXTCORE_EXPORTS
#include "llmtextcore.h"
#include "llama.h"
#include "ggml.h"

#include <atomic>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#endif

#ifndef LLMTEXTCORE_VERSION
#define LLMTEXTCORE_VERSION "0.0.0-dev"
#endif
#ifndef LLAMA_CPP_COMMIT
#define LLAMA_CPP_COMMIT "unknown"
#endif

// ----- 内部状態 -----
static llama_model*   g_model = nullptr;
static llama_context* g_ctx   = nullptr;

static std::thread      g_worker_thread;
static std::atomic<int> g_status{LLMTC_STATUS_IDLE};
static std::mutex       g_result_mutex;
static std::string      g_async_result;   // 非同期生成の結果 (常にUTF-8)
static std::string      g_reply_buffer;   // do_generate 内での作業バッファ (常にUTF-8)

static int g_verbose  = 0;
static int g_encoding = LLMTC_ENCODING_UTF8;

// ----- エンコーディング変換ヘルパー (Windowsのみ) -----

#ifdef _WIN32
static std::string convert_encoding(const std::string& src, UINT from_cp, UINT to_cp) {
    if (src.empty()) return src;
    int wlen = MultiByteToWideChar(from_cp, 0, src.c_str(), -1, nullptr, 0);
    if (wlen <= 0) return src;
    std::wstring ws(wlen, L'\0');
    MultiByteToWideChar(from_cp, 0, src.c_str(), -1, &ws[0], wlen);
    int alen = WideCharToMultiByte(to_cp, 0, ws.c_str(), -1, nullptr, 0, nullptr, nullptr);
    if (alen <= 0) return src;
    std::string result(alen, '\0');
    WideCharToMultiByte(to_cp, 0, ws.c_str(), -1, &result[0], alen, nullptr, nullptr);
    if (!result.empty() && result.back() == '\0') result.pop_back();
    return result;
}
static std::string to_utf8(const std::string& s) {
    return (g_encoding == LLMTC_ENCODING_SJIS) ? convert_encoding(s, CP_ACP, CP_UTF8) : s;
}
static std::string to_output(const std::string& s) {
    return (g_encoding == LLMTC_ENCODING_SJIS) ? convert_encoding(s, CP_UTF8, CP_ACP) : s;
}
#else
static std::string to_utf8(const std::string& s)   { return s; }
static std::string to_output(const std::string& s) { return s; }
#endif

// ----- ヘルパー -----

// src(UTF-8 or 変換済み) を out_buffer にコピーする。変換は呼び出し前に済ませること。
static int copy_to_buffer(const char* src, char* out_buffer, int buffer_size) {
    if (!out_buffer || buffer_size <= 0) return -1;
    if (!src) { out_buffer[0] = '\0'; return -2; }
    size_t len      = strlen(src);
    size_t copy_len = (len < (size_t)(buffer_size - 1)) ? len : (size_t)(buffer_size - 1);
    memcpy(out_buffer, src, copy_len);
    out_buffer[copy_len] = '\0';
    return (int)copy_len;
}

// 内部UTF-8文字列をエンコーディング変換してバッファに書き込む。
static int copy_to_buffer_encoded(const std::string& utf8_src, char* out_buffer, int buffer_size) {
    std::string converted = to_output(utf8_src);
    return copy_to_buffer(converted.c_str(), out_buffer, buffer_size);
}

static void llmtc_log_callback(ggml_log_level level, const char* text, void* user_data) {
    (void)user_data;
    if (g_verbose || level == GGML_LOG_LEVEL_ERROR) {
        fputs(text, stderr);
    }
}

// ----- 生成処理本体 (内部は常にUTF-8) -----

static bool do_generate(const char* system_prompt_utf8, const char* user_message_utf8,
                         double temperature, int max_tokens) {
    g_reply_buffer.clear();
    if (!g_ctx || !g_model || !user_message_utf8) return false;

    float temp    = (temperature > 0.0) ? (float)temperature : 0.7f;
    int   max_tok = (max_tokens  > 0)   ? max_tokens         : 512;

    std::vector<llama_chat_message> msgs;
    if (system_prompt_utf8 && system_prompt_utf8[0] != '\0') {
        msgs.push_back({ "system", system_prompt_utf8 });
    }
    msgs.push_back({ "user", user_message_utf8 });

    const llama_vocab* vocab = llama_model_get_vocab(g_model);
    const char* tmpl = llama_model_chat_template(g_model, nullptr);

    std::vector<char> buf(8192);
    int32_t n = llama_chat_apply_template(
        tmpl, msgs.data(), msgs.size(), true, buf.data(), (int32_t)buf.size());
    if (n < 0) return false;
    if ((size_t)n > buf.size()) {
        buf.resize(n);
        n = llama_chat_apply_template(
            tmpl, msgs.data(), msgs.size(), true, buf.data(), (int32_t)buf.size());
    }
    std::string prompt(buf.data(), n);

    llama_memory_clear(llama_get_memory(g_ctx), true);

    int n_tokens = -llama_tokenize(vocab, prompt.c_str(), (int32_t)prompt.size(),
                                   nullptr, 0, true, true);
    std::vector<llama_token> tokens(n_tokens);
    llama_tokenize(vocab, prompt.c_str(), (int32_t)prompt.size(),
                   tokens.data(), (int32_t)tokens.size(), true, true);

    llama_sampler* sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(sampler, llama_sampler_init_temp(temp));
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    llama_batch batch = llama_batch_get_one(tokens.data(), (int32_t)tokens.size());
    int n_generated = 0;
    bool success = true;

    while (true) {
        if (llama_decode(g_ctx, batch) != 0) { success = false; break; }

        llama_token new_token = llama_sampler_sample(sampler, g_ctx, -1);
        if (llama_vocab_is_eog(vocab, new_token)) break;

        char piece[256];
        int len = llama_token_to_piece(vocab, new_token, piece, sizeof(piece), 0, true);
        if (len > 0) g_reply_buffer.append(piece, len);

        llama_token next_input = new_token;
        batch = llama_batch_get_one(&next_input, 1);

        if (++n_generated >= max_tok) break;
    }

    llama_sampler_free(sampler);
    return success;
}

// ----- 公開API -----

const char* LLMTC_CALL llmtc_version_info(void) {
    static std::string info =
        std::string("LLMTextCore ") + LLMTEXTCORE_VERSION +
        " (llama.cpp commit " + LLAMA_CPP_COMMIT + ")";
    return info.c_str();
}

int LLMTC_CALL llmtc_version_info_to_buffer(char* out_buffer, int buffer_size) {
    return copy_to_buffer(llmtc_version_info(), out_buffer, buffer_size);
}

void LLMTC_CALL llmtc_set_verbose(int enable) {
    g_verbose = enable ? 1 : 0;
}

void LLMTC_CALL llmtc_set_encoding(int mode) {
    g_encoding = (mode == LLMTC_ENCODING_SJIS) ? LLMTC_ENCODING_SJIS : LLMTC_ENCODING_UTF8;
}

int LLMTC_CALL llmtc_init(const char* model_path, int n_ctx, int n_gpu_layers) {
    llama_log_set(llmtc_log_callback, nullptr);
    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = n_gpu_layers;

    g_model = llama_model_load_from_file(model_path, mparams);
    if (!g_model) return -1;

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = n_ctx > 0 ? n_ctx : 4096;

    g_ctx = llama_init_from_model(g_model, cparams);
    if (!g_ctx) {
        llama_model_free(g_model);
        g_model = nullptr;
        return -2;
    }
    return 0;
}

// ---- 同期API ----

const char* LLMTC_CALL llmtc_generate(const char* system_prompt, const char* user_message,
                                       double temperature, int max_tokens) {
    // 入力は常にUTF-8として受け取る (変換しない)
    if (!do_generate(system_prompt ? system_prompt : "",
                     user_message  ? user_message  : "",
                     temperature, max_tokens)) return nullptr;
    return g_reply_buffer.c_str(); // 常にUTF-8
}

int LLMTC_CALL llmtc_generate_to_buffer(const char* system_prompt, const char* user_message,
                                         double temperature, int max_tokens,
                                         char* out_buffer, int buffer_size) {
    // 入力は常にUTF-8として受け取る (変換しない)
    bool ok = do_generate(system_prompt ? system_prompt : "",
                          user_message  ? user_message  : "",
                          temperature, max_tokens);
    if (!ok) return copy_to_buffer(nullptr, out_buffer, buffer_size);
    // 出力だけをエンコーディング変換する
    return copy_to_buffer_encoded(g_reply_buffer, out_buffer, buffer_size);
}

// ---- 非同期API ----

int LLMTC_CALL llmtc_generate_async(const char* system_prompt, const char* user_message,
                                     double temperature, int max_tokens) {
    if (g_status.load() == LLMTC_STATUS_BUSY) return -1;
    if (!g_ctx || !g_model || !user_message)   return -2;

    if (g_worker_thread.joinable()) g_worker_thread.join();

    g_status.store(LLMTC_STATUS_BUSY);

    // 入力は常にUTF-8として受け取る (変換しない)。スレッドに渡すためコピーだけする。
    std::string sp = system_prompt ? system_prompt : "";
    std::string um = user_message;

    g_worker_thread = std::thread([sp, um, temperature, max_tokens]() {
        bool ok = do_generate(sp.c_str(), um.c_str(), temperature, max_tokens);
        {
            std::lock_guard<std::mutex> lock(g_result_mutex);
            g_async_result = ok ? g_reply_buffer : ""; // 常にUTF-8で保持
        }
        g_status.store(ok ? LLMTC_STATUS_DONE : LLMTC_STATUS_ERROR);
    });

    return 0;
}

int LLMTC_CALL llmtc_get_status(void) {
    return g_status.load();
}

int LLMTC_CALL llmtc_get_result_to_buffer(char* out_buffer, int buffer_size) {
    std::lock_guard<std::mutex> lock(g_result_mutex);
    // 取り出し時に出力エンコーディングへ変換する
    int len = copy_to_buffer_encoded(g_async_result, out_buffer, buffer_size);
    g_async_result.clear();
    g_status.store(LLMTC_STATUS_IDLE);
    return len;
}

void LLMTC_CALL llmtc_free(void) {
    if (g_worker_thread.joinable()) g_worker_thread.join();

    if (g_ctx)   { llama_free(g_ctx);        g_ctx = nullptr; }
    if (g_model) { llama_model_free(g_model); g_model = nullptr; }
    llama_backend_free();

    g_reply_buffer.clear();
    g_async_result.clear();
    g_status.store(LLMTC_STATUS_IDLE);
}
