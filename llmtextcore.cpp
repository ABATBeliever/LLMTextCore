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
#define LLMTEXTCORE_VERSION "0.7.5"
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
static bool   g_backend_initialized = false; // llama_backend_init/free を対で呼ぶための管理フラグ

// 直近のエラーメッセージ。生成系APIやllmtc_initの呼び出しごとにクリアされる。
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

// ----- 改行コード変換 (Windows専用ソフトのため、出力は常にCRLFに統一する) -----

#ifdef _WIN32
// 単独の '\n' (直前が '\r' でないもの) だけを "\r\n" に変換する。
// 既に "\r\n" の箇所は変化しないので、二重変換 ("\r\r\n") にはならない。
static std::string lf_to_crlf(const char* src, size_t len) {
    std::string out;
    out.reserve(len + len / 8); // 改行が多少あっても再確保が起きにくいよう少し余分に確保
    for (size_t i = 0; i < len; ++i) {
        char c = src[i];
        if (c == '\n' && (i == 0 || src[i - 1] != '\r')) {
            out.push_back('\r');
        }
        out.push_back(c);
    }
    return out;
}
#endif

// ----- バッファヘルパー -----

static int copy_to_buffer(const char* src, char* out_buffer, int buffer_size) {
    if (!out_buffer || buffer_size <= 0) return -1;
    if (!src) { out_buffer[0] = '\0'; return -2; }
#ifdef _WIN32
    // Shift-JISに変換済みの文字列が渡ってきても、0x0A/0x0Dはどちらの1バイト目・2バイト目にも
    // 出現しない範囲のバイト値なので、ここでLF→CRLF変換をしても文字の区切りを壊さない。
    std::string converted = lf_to_crlf(src, strlen(src));
    const char* final_src  = converted.c_str();
    size_t      len        = converted.size();
#else
    const char* final_src = src;
    size_t      len        = strlen(src);
#endif
    size_t copy_len = (len < (size_t)(buffer_size - 1)) ? len : (size_t)(buffer_size - 1);
    memcpy(out_buffer, final_src, copy_len);
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
    // ERRORレベルのログはg_last_errorにも蓄積する(上限4KB)
    if (level == GGML_LOG_LEVEL_ERROR && text) {
        std::lock_guard<std::mutex> lk(g_error_mutex);
        if (g_last_error.size() < 4096) {
            g_last_error += text;
        }
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

    // 今回の生成用にエラーバッファをリセットする
    {
        std::lock_guard<std::mutex> lk(g_error_mutex);
        g_last_error.clear();
    }

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
        if (n < 0) return false; // 2回目も失敗した場合のガード
    }
    std::string prompt(buf.data(), n);

    llama_memory_clear(llama_get_memory(g_ctx), true);

    int n_tokens = -llama_tokenize(vocab, prompt.c_str(), (int32_t)prompt.size(),
                                   nullptr, 0, true, true);
    if (n_tokens <= 0) return false; // 空プロンプトや異常時のガード
    std::vector<llama_token> tokens(n_tokens);
    llama_tokenize(vocab, prompt.c_str(), (int32_t)prompt.size(),
                   tokens.data(), (int32_t)tokens.size(), true, true);

    // サンプラーチェーンを組み立てる
    // 順序: penalties → top_p → temperature → dist
    // llama_sampler_init_penalties(n_vocab, penalty_last_n, penalty_repeat, penalty_freq, penalty_present)
    const int32_t n_vocab = llama_vocab_n_tokens(vocab); // vocab は llama_model_get_vocab(g_model) 取得済み
    llama_sampler* sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(sampler,
        llama_sampler_init_penalties(n_vocab, g_penalty_last_n, g_repeat_penalty, 0.0f, 0.0f));
    llama_sampler_chain_add(sampler, llama_sampler_init_top_p(g_top_p, 1));
    llama_sampler_chain_add(sampler, llama_sampler_init_temp(temp));
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    llama_batch batch = llama_batch_get_one(tokens.data(), (int32_t)tokens.size());
    int  n_generated = 0;
    bool success     = true;
    // next_inputをループ外に宣言する。
    // ループ内で宣言するとイテレーション終了時に破棄され、
    // batch.tokenがダングリングポインタになるため。
    llama_token next_input = 0;

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

        next_input = new_token;
        batch = llama_batch_get_one(&next_input, 1);

        if (++n_generated >= max_tok) break;
    }

    llama_sampler_free(sampler);
    return success;
}

// ----- 生成の排他制御 -----
//
// 同期API・非同期APIの入口は必ずこの try_start_generation() を通す。
// 「開始してよいか判定」と「BUSYへの変更」を compare_exchange で1つのアトミック操作に
// まとめることで、複数スレッドが同時に生成を開始してしまう競合を防ぐ。
// 開始できるのは IDLE または ERROR のときのみ。BUSY / DONE / CANCELLED からは開始できない。
static bool try_start_generation() {
    int expected = g_status.load();
    for (;;) {
        if (expected == LLMTC_STATUS_BUSY ||
            expected == LLMTC_STATUS_DONE ||
            expected == LLMTC_STATUS_CANCELLED) {
            return false;
        }
        // expected は IDLE か ERROR。ここで他スレッドが割り込んでいなければ BUSY にできる。
        // 割り込まれていた場合は expected が最新値に更新されるので、ループして再判定する。
        if (g_status.compare_exchange_weak(expected, LLMTC_STATUS_BUSY)) {
            return true;
        }
    }
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
    if (penalty_last_n > 0)        g_penalty_last_n = penalty_last_n;
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

    // model_path NULLチェック
    if (!model_path) {
        std::lock_guard<std::mutex> lk(g_error_mutex);
        g_last_error = "llmtc_init: model_path is NULL";
        return -1;
    }

    // エラーログをリセットしてから初期化開始
    {
        std::lock_guard<std::mutex> lk(g_error_mutex);
        g_last_error.clear();
    }

    llama_log_set(llmtc_log_callback, nullptr);
    llama_backend_init();
    g_backend_initialized = true;

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = n_gpu_layers;

    g_model = llama_model_load_from_file(model_path, mparams);
    if (!g_model) {
        llama_backend_free();
        g_backend_initialized = false;
        return -1;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = n_ctx > 0 ? n_ctx : 4096;

    g_ctx = llama_init_from_model(g_model, cparams);
    if (!g_ctx) {
        llama_model_free(g_model);
        g_model = nullptr;
        llama_backend_free();
        g_backend_initialized = false;
        return -2;
    }
    return 0;
}

void LLMTC_CALL llmtc_free(void) {
    // 実行中の非同期生成があれば、完了を待たずにキャンセルしてから解放する
    // (生成が行われていない場合はキャンセルフラグを立てても無害)
    g_cancel_flag.store(true);
    if (g_worker_thread.joinable()) g_worker_thread.join();
    if (g_ctx)   { llama_free(g_ctx);        g_ctx = nullptr; }
    if (g_model) { llama_model_free(g_model); g_model = nullptr; }
    if (g_backend_initialized) {
        llama_backend_free();
        g_backend_initialized = false;
    }
    g_reply_buffer.clear();
    { std::lock_guard<std::mutex> lk(g_result_mutex);  g_async_result.clear(); }
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
    // 非同期生成が実行中、またはDONE/CANCELLEDの結果が未取得の場合は失敗する
    if (!try_start_generation()) {
        std::lock_guard<std::mutex> lk(g_error_mutex);
        g_last_error = "llmtc_generate: another generation is already running, "
                       "or an async result is pending (call llmtc_get_result_to_buffer first)";
        return nullptr;
    }

    bool ok = do_generate(system_prompt ? system_prompt : "",
                          user_message  ? user_message  : "",
                          temperature, max_tokens);
    g_status.store(LLMTC_STATUS_IDLE);

    if (!ok) return nullptr;
#ifdef _WIN32
    // この関数は copy_to_buffer を経由しないので、ここで個別にCRLF化する。
    // (呼び出し側への契約「次の生成呼び出しまで有効なポインタ」は変わらない)
    g_reply_buffer = lf_to_crlf(g_reply_buffer.c_str(), g_reply_buffer.size());
#endif
    return g_reply_buffer.c_str();
}

int LLMTC_CALL llmtc_generate_to_buffer(const char* system_prompt, const char* user_message,
                                         double temperature, int max_tokens,
                                         char* out_buffer, int buffer_size) {
    if (!try_start_generation()) {
        std::lock_guard<std::mutex> lk(g_error_mutex);
        g_last_error = "llmtc_generate_to_buffer: another generation is already running, "
                       "or an async result is pending (call llmtc_get_result_to_buffer first)";
        return copy_to_buffer(nullptr, out_buffer, buffer_size);
    }

    bool ok = do_generate(system_prompt ? system_prompt : "",
                          user_message  ? user_message  : "",
                          temperature, max_tokens);
    g_status.store(LLMTC_STATUS_IDLE);

    if (!ok) return copy_to_buffer(nullptr, out_buffer, buffer_size);
    return copy_to_buffer_encoded(g_reply_buffer, out_buffer, buffer_size);
}

// ---- 非同期API ----

int LLMTC_CALL llmtc_generate_async(const char* system_prompt, const char* user_message,
                                     double temperature, int max_tokens) {
    // 事前チェック: 分かりやすいエラーコード(-1 / -4)を返すための下見
    const int st = g_status.load();
    if (st == LLMTC_STATUS_BUSY) return -1;
    // DONE/CANCELLEDの結果をまだ取り出していない場合も開始できない
    if (st == LLMTC_STATUS_DONE || st == LLMTC_STATUS_CANCELLED) return -4;
    if (!g_ctx || !g_model || !user_message) return -2;

    // 実際の開始可否はここでアトミックに確定させる(他スレッドとの競合対策)
    if (!try_start_generation()) return -1;

    if (g_worker_thread.joinable()) g_worker_thread.join();

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
    // BUSY中は呼べない (状態を破壊しないためのガード)
    if (g_status.load() == LLMTC_STATUS_BUSY) {
        if (out_buffer && buffer_size > 0) out_buffer[0] = '\0';
        return -3;
    }
    // DONE と CANCELLED どちらでも結果を取り出せる
    std::lock_guard<std::mutex> lk(g_result_mutex);
    int len = copy_to_buffer_encoded(g_async_result, out_buffer, buffer_size);
    g_async_result.clear();
    g_status.store(LLMTC_STATUS_IDLE);
    return len;
}
