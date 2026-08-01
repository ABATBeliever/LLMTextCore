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

// llmtc_get_status の戻り値
#define LLMTC_STATUS_IDLE   0  // 待機中
#define LLMTC_STATUS_BUSY   1  // 生成中
#define LLMTC_STATUS_DONE   2  // 生成完了 (llmtc_get_result_to_buffer で結果を取得できる)
#define LLMTC_STATUS_ERROR -1  // 生成中にエラーが発生した

// llmtc_set_encoding の引数
#define LLMTC_ENCODING_UTF8 0  // UTF-8 のまま (デフォルト、C/C++等向け)
#define LLMTC_ENCODING_SJIS 1  // Shift-JIS (CP932) 自動変換 (HSP 3.x等向け)

#ifdef __cplusplus
extern "C" {
#endif

// LLMTextCore自身のバージョンと、ビルドに使ったllama.cppのコミットハッシュを返す。
// 戻り値は静的文字列なので free() しないこと。(C/C++向け)
LLMTC_API const char* LLMTC_CALL llmtc_version_info(void);

// バージョン文字列をバッファに書き込む (HSP等向け)
LLMTC_API int LLMTC_CALL llmtc_version_info_to_buffer(char* out_buffer, int buffer_size);

// デバッグログの表示有無を設定する (llmtc_init より前に呼ぶこと)
// enable=0: エラーのみ表示 (デフォルト) / enable=1: llama.cppの詳細ログを全部表示
LLMTC_API void LLMTC_CALL llmtc_set_verbose(int enable);

// 文字列入出力のエンコーディングを設定する。
// LLMTC_ENCODING_UTF8(0): UTF-8のまま (デフォルト、C/C++等向け)
// LLMTC_ENCODING_SJIS(1): Shift-JIS (CP932) 自動変換モード (HSP 3.x等向け)
//   入力 (system_prompt, user_message) を CP932→UTF-8 に変換してllama.cppに渡し、
//   *_to_buffer 系の出力は UTF-8→CP932 に変換してバッファに書き込む。
//   llmtc_generate() (ポインタ返し) は常にUTF-8のまま。
LLMTC_API void LLMTC_CALL llmtc_set_encoding(int mode);

// モデルを読み込んで初期化する。成功で0、失敗で負値を返す。
// n_ctx: コンテキスト長 (0 ならデフォルト)
// n_gpu_layers: GPUにオフロードする層数 (0 ならCPUのみ)
LLMTC_API int LLMTC_CALL llmtc_init(const char* model_path, int n_ctx, int n_gpu_layers);

// ---- 同期API (C/C++向け) ----

// ステートレスな1回完結の応答生成。完了まで呼び出し元をブロックする。
// 戻り値は内部バッファを指しているため free() しないこと。常にUTF-8。失敗時は NULL。
LLMTC_API const char* LLMTC_CALL llmtc_generate(
    const char* system_prompt,
    const char* user_message,
    double temperature,
    int max_tokens);

// 同上、バッファに書き込む版。llmtc_set_encoding の設定に従って変換する。
LLMTC_API int LLMTC_CALL llmtc_generate_to_buffer(
    const char* system_prompt,
    const char* user_message,
    double temperature,
    int max_tokens,
    char* out_buffer,
    int buffer_size);

// ---- 非同期API (HSP等のポーリング向け) ----

// 生成をバックグラウンドスレッドで開始し、即リターンする。
// 戻り値: 0=開始成功 / -1=前の生成がまだ実行中 / -2=初期化未完了
LLMTC_API int LLMTC_CALL llmtc_generate_async(
    const char* system_prompt,
    const char* user_message,
    double temperature,
    int max_tokens);

// 現在の非同期生成の状態を返す。
// 戻り値: LLMTC_STATUS_IDLE(0) / LLMTC_STATUS_BUSY(1) /
//         LLMTC_STATUS_DONE(2) / LLMTC_STATUS_ERROR(-1)
LLMTC_API int LLMTC_CALL llmtc_get_status(void);

// 非同期生成の結果をバッファに書き込む。LLMTC_STATUS_DONE の後にのみ呼ぶこと。
// llmtc_set_encoding の設定に従って変換する。
// 呼び出し後、状態は LLMTC_STATUS_IDLE に戻る。
LLMTC_API int LLMTC_CALL llmtc_get_result_to_buffer(char* out_buffer, int buffer_size);

// モデル・コンテキストを解放する。生成中の場合は完了を待ってから解放する。
LLMTC_API void LLMTC_CALL llmtc_free(void);

#ifdef __cplusplus
}
#endif

#endif // LLMTEXTCORE_H
