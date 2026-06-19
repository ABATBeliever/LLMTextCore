#ifndef LLMTEXTCORE_H
#define LLMTEXTCORE_H

#ifdef _WIN32
  #ifdef LLMTEXTCORE_EXPORTS
    #define LLMTC_API __declspec(dllexport)
  #else
    #define LLMTC_API __declspec(dllimport)
  #endif
  #define LLMTC_CALL __stdcall
#else
  #define LLMTC_API
  #define LLMTC_CALL
#endif

#ifdef __cplusplus
extern "C" {
#endif

// LLMTextCore自身のバージョンと、ビルドに使ったllama.cppのコミットハッシュを返す。
// 戻り値は静的文字列なので free() しないこと。(C/C++向け)
LLMTC_API const char* LLMTC_CALL llmtc_version_info(void);

// HSPなど、ポインタを直接扱いにくい言語向けのバッファ版。
LLMTC_API int LLMTC_CALL llmtc_version_info_to_buffer(char* out_buffer, int buffer_size);

// デバッグログの表示有無を設定する (llmtc_init より前に呼ぶこと)
// enable=0: エラーのみ表示 (デフォルト) / enable=1: llama.cppの詳細ログを全部表示
LLMTC_API void LLMTC_CALL llmtc_set_verbose(int enable);

// モデルを読み込んで初期化する。成功で0、失敗で負値を返す。
// n_ctx: コンテキスト長 (0 ならデフォルト)
// n_gpu_layers: GPUにオフロードする層数 (0 ならCPUのみ)
LLMTC_API int LLMTC_CALL llmtc_init(const char* model_path, int n_ctx, int n_gpu_layers);

// ステートレスな1回完結の応答生成。
// system_prompt は NULL または空文字列でも可。
// temperature: 0以下でデフォルト値(0.7)を使用。
// max_tokens : 0以下でデフォルト値(512)を使用。
// 呼び出すたびに直前までの文脈はクリアされる(会話履歴は保持しない)。
// 複数ターンの会話を続けたい場合は、HSP側で過去のやり取りを
// user_message に連結して渡すこと。
// 戻り値は内部バッファを指しているため、呼び出し側で free() しないこと。
// 失敗時は NULL。(C/C++向け)
LLMTC_API const char* LLMTC_CALL llmtc_generate(
    const char* system_prompt,
    const char* user_message,
    double temperature,
    int max_tokens);

// HSPなど、ポインタを直接扱いにくい言語向けのバッファ版。
// out_buffer に応答をUTF-8で書き込み、書き込んだバイト数(終端のNULは含まない)を返す。
// buffer_sizeに収まらない場合は切り詰めて書き込む。失敗時は負値を返す。
LLMTC_API int LLMTC_CALL llmtc_generate_to_buffer(
    const char* system_prompt,
    const char* user_message,
    double temperature,
    int max_tokens,
    char* out_buffer,
    int buffer_size);

// モデル・コンテキストを解放する
LLMTC_API void LLMTC_CALL llmtc_free(void);

#ifdef __cplusplus
}
#endif

#endif // LLMTEXTCORE_H
