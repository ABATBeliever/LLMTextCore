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

static std::thread          g_worker_thread;
static std::atomic<int>     g_status{LLMTC_STATUS_IDLE};
static std::atomic<bool>    g_cancel_flag{false};   // true になると次トークン前にループを抜ける

static std::mutex  g_result_mutex;
static std::string g_async_result;    // 最終結果 (DONE/CANCELLED どちらでも有効)

static std::mutex  g_partial_mutex;
static std::string g_partial_buffer;  // 途中テキスト (累積)

static std::string g_reply_buffer;    // do_generate 内作業用

static int    g_verbose        = 0;
static int    g_encoding       = LLMTC_ENCODING_UTF8;
static float  g_top_p          = 0.95f;
static float  g_repeat_penalty = 1.0f;   // 1.0 = 無効
static int    g_penalty_last_n = 64;

// エラーメッセージ: llmtc_init のたびにクリアし、ERRORログや独自エラーを追記する
static std::mutex  g_error_mutex;
static std::string g_last_error;

// ----- エンコーディング変換 -----

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
static std::string to_output(const std::string& s) {
    return (g_encoding == LLMTC_ENCODING_SJIS) ? convert_encoding(s, CP_UTF8, CP_ACP) : s;
}
#else
static std::string to_output(const std::string& s) { return s; }
#endif

// ----- バッファヘルパー -----

static int copy_to_buffer(const char* src, char* out_buffer, int buffer_size) {
    if (!out_buffer || buffer_size <= 0) return -1;
    if (!src) { out_buffer[0] = '\0'; return -2; }
    size_t len      = strlen(src);
    size_t copy_len = (len < (size_t)(buffer_size - 1)) ? len : (size_t)(buffer_size - 1);
    memcpy(out_buffer, src, copy_len);
    out_buffer[copy_len] = '\0';
    return (int)copy_len;
}

static int copy_to_buffer_encoded(const std::string& utf8_src, char* out_buffer, int buffer_size) {
    return copy_to_buffer(to_output(utf8_src).c_str(), out_buffer, buffer_size);
}

// ----- ログ -----

static void llmtc_log_callback(ggml_log_level level, const char* text, void* user_data) {
    (void)user_data;
    if (g_verbose || level == GGML_LOG_LEVEL_ERROR) fputs(text, stderr);
    // ERRORレベルのログはg_last_errorにも蓄積する
    if (level == GGML_LOG_LEVEL_ERROR && text) {
        std::lock_guard<std::mutex> lk(g_error_mutex);
        g_last_error += text;
    }
}

// ----- 生成処理本体 -----

static bool do_generate(const char* system_prompt_utf8, const char* user_message_utf8,
                         double temperature, int max_tokens) {
    // 開始時にキャンセルフラグをリセット
    g_cancel_flag.store(false);
    g_reply_buffer.clear();
    {
        std::lock_guard<std::mutex> lk(g_partial_mutex);
        g_partial_buffer.clear();
    }

    if (!g_ctx || !g_model || !user_message_utf8) return false;

    float temp     = (temperature > 0.0) ? (float)temperature : 0.7f;
    int   max_tok  = (max_tokens  > 0)   ? max_tokens         : 512;

    std::vector<llama_chat_message> msgs;
    if (system_prompt_utf8 && system_prompt_utf8[0] != '\0')
        msgs.push_back({ "system", system_prompt_utf8 });
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

    // サンプラーチェーンを組み立てる
    // 順序: penalties → top_p → temperature → dist
    llama_sampler* sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(sampler,
        llama_sampler_init_penalties(g_penalty_last_n, g_repeat_penalty, 0.0f, 0.0f));
    llama_sampler_chain_add(sampler, llama_sampler_init_top_p(g_top_p, 1));
    llama_sampler_chain_add(sampler, llama_sampler_init_temp(temp));
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    llama_batch batch = llama_batch_get_one(tokens.data(), (int32_t)tokens.size());
    int  n_generated = 0;
    bool success     = true;

    while (true) {
        // キャンセルチェック: 各トークンの生成前に確認する
        if (g_cancel_flag.load()) break;

        if (llama_decode(g_ctx, batch) != 0) { success = false; break; }

        llama_token new_token = llama_sampler_sample(sampler, g_ctx, -1);
        if (llama_vocab_is_eog(vocab, new_token)) break;

        char piece[256];
        int len = llama_token_to_piece(vocab, new_token, piece, sizeof(piece), 0, true);
        if (len > 0) {
            g_reply_buffer.append(piece, len);
            std::lock_guard<std::mutex> lk(g_partial_mutex);
            g_partial_buffer.append(piece, len);
        }

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

void LLMTC_CALL llmtc_set_verbose(int enable)  { g_verbose  = enable ? 1 : 0; }
void LLMTC_CALL llmtc_set_encoding(int mode)   {
    g_encoding = (mode == LLMTC_ENCODING_SJIS) ? LLMTC_ENCODING_SJIS : LLMTC_ENCODING_UTF8;
}

int LLMTC_CALL llmtc_set_generation_params(double top_p, double repeat_penalty, int penalty_last_n) {
    if (top_p          > 0.0) g_top_p          = (float)top_p;
    if (repeat_penalty > 0.0) g_repeat_penalty  = (float)repeat_penalty;
    if (penalty_last_n > 0)   g_penalty_last_n  = penalty_last_n;
    else if (penalty_last_n == -1) g_penalty_last_n = -1; // コンテキスト全体
    return 0;
}

int LLMTC_CALL llmtc_init(const char* model_path, int n_ctx, int n_gpu_layers) {
    // 二重初期化ガード
    if (g_model != nullptr) {
        std::lock_guard<std::mutex> lk(g_error_mutex);
        g_last_error = "llmtc_init: already initialized, call llmtc_free first";
        return -3;
    }

    // エラーログをリセットしてから初期化開始
    {
        std::lock_guard<std::mutex> lk(g_error_mutex);
        g_last_error.clear();
    }

    llama_log_set(llmtc_log_callback, nullptr);
    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = n_gpu_layers;

    g_model = llama_model_load_from_file(model_path, mparams);
    if (!g_model) return -1;

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = n_ctx > 0 ? n_ctx : 4096;

    g_ctx = llama_init_from_model(g_model, cparams);
    if (!g_ctx) { llama_model_free(g_model); g_model = nullptr; return -2; }
    return 0;
}

void LLMTC_CALL llmtc_free(void) {
    if (g_worker_thread.joinable()) g_worker_thread.join();
    if (g_ctx)   { llama_free(g_ctx);        g_ctx = nullptr; }
    if (g_model) { llama_model_free(g_model); g_model = nullptr; }
    llama_backend_free();
    g_reply_buffer.clear();
    g_async_result.clear();
    { std::lock_guard<std::mutex> lk(g_partial_mutex); g_partial_buffer.clear(); }
    { std::lock_guard<std::mutex> lk(g_error_mutex);   g_last_error.clear(); }
    g_status.store(LLMTC_STATUS_IDLE);
    g_cancel_flag.store(false);
}

int LLMTC_CALL llmtc_get_last_error_to_buffer(char* out_buffer, int buffer_size) {
    std::lock_guard<std::mutex> lk(g_error_mutex);
    return copy_to_buffer(g_last_error.empty() ? "(no error)" : g_last_error.c_str(),
                          out_buffer, buffer_size);
}

int LLMTC_CALL llmtc_get_model_info_to_buffer(char* out_buffer, int buffer_size) {
    if (!g_model || !g_ctx) return copy_to_buffer("(not loaded)", out_buffer, buffer_size);
    char name_buf[256] = "(unknown)";
    llama_model_desc(g_model, name_buf, sizeof(name_buf));
    uint64_t params = llama_model_n_params(g_model);
    int      n_ctx  = (int)llama_n_ctx(g_ctx);
    char info[512];
    snprintf(info, sizeof(info),
        "name: %s\nparams: %llu\nn_ctx: %d",
        name_buf, (unsigned long long)params, n_ctx);
    return copy_to_buffer(info, out_buffer, buffer_size);
}

int LLMTC_CALL llmtc_get_n_ctx(void) {
    return g_ctx ? (int)llama_n_ctx(g_ctx) : 0;
}

// ---- 同期API ----

const char* LLMTC_CALL llmtc_generate(const char* system_prompt, const char* user_message,
                                       double temperature, int max_tokens) {
    if (!do_generate(system_prompt ? system_prompt : "",
                     user_message  ? user_message  : "",
                     temperature, max_tokens)) return nullptr;
    return g_reply_buffer.c_str();
}

int LLMTC_CALL llmtc_generate_to_buffer(const char* system_prompt, const char* user_message,
                                         double temperature, int max_tokens,
                                         char* out_buffer, int buffer_size) {
    bool ok = do_generate(system_prompt ? system_prompt : "",
                          user_message  ? user_message  : "",
                          temperature, max_tokens);
    if (!ok) return copy_to_buffer(nullptr, out_buffer, buffer_size);
    return copy_to_buffer_encoded(g_reply_buffer, out_buffer, buffer_size);
}

// ---- 非同期API ----

int LLMTC_CALL llmtc_generate_async(const char* system_prompt, const char* user_message,
                                     double temperature, int max_tokens) {
    if (g_status.load() == LLMTC_STATUS_BUSY) return -1;
    if (!g_ctx || !g_model || !user_message)   return -2;

    if (g_worker_thread.joinable()) g_worker_thread.join();

    g_status.store(LLMTC_STATUS_BUSY);

    std::string sp = system_prompt ? system_prompt : "";
    std::string um = user_message;

    g_worker_thread = std::thread([sp, um, temperature, max_tokens]() {
        bool ok = do_generate(sp.c_str(), um.c_str(), temperature, max_tokens);

        // キャンセルされた場合: g_reply_bufferに途中までの結果が入っている
        bool cancelled = g_cancel_flag.load();
        {
            std::lock_guard<std::mutex> lk(g_result_mutex);
            // ok=false(llama_decodeエラー)のときのみ空にする
            // cancel時はok=trueのまま(途中結果あり)
            g_async_result = ok ? g_reply_buffer : "";
        }

        if (cancelled) {
            g_status.store(LLMTC_STATUS_CANCELLED);
        } else {
            g_status.store(ok ? LLMTC_STATUS_DONE : LLMTC_STATUS_ERROR);
        }
    });

    return 0;
}

void LLMTC_CALL llmtc_cancel(void) {
    g_cancel_flag.store(true);
}

int LLMTC_CALL llmtc_get_status(void) {
    return g_status.load();
}

int LLMTC_CALL llmtc_get_partial_result_to_buffer(char* out_buffer, int buffer_size) {
    std::lock_guard<std::mutex> lk(g_partial_mutex);
    return copy_to_buffer_encoded(g_partial_buffer, out_buffer, buffer_size);
}

int LLMTC_CALL llmtc_get_result_to_buffer(char* out_buffer, int buffer_size) {
    // DONE と CANCELLED どちらでも結果を取り出せる
    std::lock_guard<std::mutex> lk(g_result_mutex);
    int len = copy_to_buffer_encoded(g_async_result, out_buffer, buffer_size);
    g_async_result.clear();
    g_status.store(LLMTC_STATUS_IDLE);
    return len;
}
