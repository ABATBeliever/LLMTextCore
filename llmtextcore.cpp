#define LLMTEXTCORE_EXPORTS
#include "llmtextcore.h"
#include "llama.h"
#include "ggml.h"

#include <cstdio>
#include <string>
#include <vector>
#include <cstring>

#ifndef LLMTEXTCORE_VERSION
#define LLMTEXTCORE_VERSION "0.4.0"
#endif
#ifndef LLAMA_CPP_COMMIT
#define LLAMA_CPP_COMMIT "unknown"
#endif

// ----- 内部状態 -----
// このDLLはステートレス設計。会話履歴は一切保持しない。
// 保持するのはモデル/コンテキストそのもののみ。
static llama_model*   g_model   = nullptr;
static llama_context* g_ctx     = nullptr;
static llama_sampler* g_sampler = nullptr;

static std::string g_reply_buffer;
static int g_verbose = 0; // 0: エラーのみ表示 / 1: 詳細ログを表示

// 文字列を呼び出し側のバッファにコピーする共通ヘルパー。
// 収まらない場合は切り詰める。書き込んだバイト数 (終端NUL抜き) を返す。
static int copy_to_buffer(const char* src, char* out_buffer, int buffer_size) {
    if (!out_buffer || buffer_size <= 0) return -1;
    if (!src) { out_buffer[0] = '\0'; return -2; }

    size_t len = strlen(src);
    size_t copy_len = (len < (size_t)(buffer_size - 1)) ? len : (size_t)(buffer_size - 1);
    memcpy(out_buffer, src, copy_len);
    out_buffer[copy_len] = '\0';
    return (int)copy_len;
}

// llama.cpp / ggml が出すログをここで一括フィルタする。
// g_verbose が 0 のときはエラーだけ標準エラー出力に出す。
static void llmtc_log_callback(ggml_log_level level, const char* text, void* user_data) {
    (void)user_data;
    if (g_verbose || level == GGML_LOG_LEVEL_ERROR) {
        fputs(text, stderr);
    }
}

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

int LLMTC_CALL llmtc_init(const char* model_path, int n_ctx, int n_gpu_layers) {
    // ログ出力先を最初に登録しておく (これより前のログは抑制できない)
    llama_log_set(llmtc_log_callback, nullptr);

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = n_gpu_layers;

    g_model = llama_model_load_from_file(model_path, mparams);
    if (!g_model) {
        return -1;
    }

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

// ステートレスな1回完結の生成処理本体。
// 呼び出すたびに直前までの文脈(KVキャッシュ)をクリアしてから、
// system_prompt + user_message の1ターンだけで応答を作る。
static bool do_generate(const char* system_prompt, const char* user_message,
                         double temperature, int max_tokens) {
    g_reply_buffer.clear();
    if (!g_ctx || !g_model || !user_message) return false;

    float temp    = (temperature > 0.0) ? (float)temperature : 0.7f;
    int   max_tok = (max_tokens   > 0)  ? max_tokens : 512;

    std::vector<llama_chat_message> msgs;
    if (system_prompt && system_prompt[0] != '\0') {
        msgs.push_back({ "system", system_prompt });
    }
    msgs.push_back({ "user", user_message });

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

    // (注意: ビルドエラーになる場合は llama.h 内で "kv_cache" や "memory_clear" を
    //  検索して正しい関数名に置き換えること)
    llama_memory_clear(llama_get_memory(g_ctx), true);

    int n_tokens = -llama_tokenize(vocab, prompt.c_str(), (int32_t)prompt.size(), nullptr, 0, true, true);
    std::vector<llama_token> tokens(n_tokens);
    llama_tokenize(vocab, prompt.c_str(), (int32_t)prompt.size(), tokens.data(), (int32_t)tokens.size(), true, true);

    // ステートレス設計のため、温度はここで毎回作り直す。
    if (g_sampler) { llama_sampler_free(g_sampler); g_sampler = nullptr; }
    g_sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(g_sampler, llama_sampler_init_temp(temp));
    llama_sampler_chain_add(g_sampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    llama_batch batch = llama_batch_get_one(tokens.data(), (int32_t)tokens.size());

    int n_generated = 0;
    while (true) {
        if (llama_decode(g_ctx, batch) != 0) break;

        llama_token new_token = llama_sampler_sample(g_sampler, g_ctx, -1);
        if (llama_vocab_is_eog(vocab, new_token)) break;

        char piece[256];
        int len = llama_token_to_piece(vocab, new_token, piece, sizeof(piece), 0, true);
        if (len > 0) g_reply_buffer.append(piece, len);

        llama_token next_input = new_token;
        batch = llama_batch_get_one(&next_input, 1);

        n_generated++;
        if (n_generated >= max_tok) break; // 最大トークン数で打ち切り
    }

    return true;
}

const char* LLMTC_CALL llmtc_generate(const char* system_prompt, const char* user_message,
                                       double temperature, int max_tokens) {
    if (!do_generate(system_prompt, user_message, temperature, max_tokens)) return nullptr;
    return g_reply_buffer.c_str();
}

int LLMTC_CALL llmtc_generate_to_buffer(const char* system_prompt, const char* user_message,
                                         double temperature, int max_tokens,
                                         char* out_buffer, int buffer_size) {
    bool ok = do_generate(system_prompt, user_message, temperature, max_tokens);
    return copy_to_buffer(ok ? g_reply_buffer.c_str() : nullptr, out_buffer, buffer_size);
}

void LLMTC_CALL llmtc_free(void) {
    if (g_sampler) { llama_sampler_free(g_sampler); g_sampler = nullptr; }
    if (g_ctx)     { llama_free(g_ctx);             g_ctx = nullptr; }
    if (g_model)   { llama_model_free(g_model);      g_model = nullptr; }
    llama_backend_free();

    g_reply_buffer.clear();
}
