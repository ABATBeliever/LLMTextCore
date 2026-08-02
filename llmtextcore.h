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
#define LLMTC_STATUS_IDLE      0  // 待機中
#define LLMTC_STATUS_BUSY      1  // 生成中
#define LLMTC_STATUS_DONE      2  // 生成完了
#define LLMTC_STATUS_CANCELLED 3  // キャンセルされた (途中までの結果は取得可能)
#define LLMTC_STATUS_ERROR    -1  // エラー

// llmtc_set_encoding の引数
#define LLMTC_ENCODING_UTF8 0
#define LLMTC_ENCODING_SJIS 1

#ifdef __cplusplus
extern "C" {
#endif

// ---- バージョン ----
LLMTC_API const char* LLMTC_CALL llmtc_version_info(void);
LLMTC_API int         LLMTC_CALL llmtc_version_info_to_buffer(char* out_buffer, int buffer_size);

// ---- 設定 ----
LLMTC_API void LLMTC_CALL llmtc_set_verbose(int enable);
LLMTC_API void LLMTC_CALL llmtc_set_encoding(int mode);

// 生成パラメータのデフォルト値を設定する。llmtc_init の前後どちらでも呼べる。
// 0以下を渡した引数はデフォルト値のまま変更しない。
//
// top_p        : 0以下でデフォルト(0.95) / 1.0で無効化 / 0より大きく1.0以下で有効
//                nucleus sampling: 確率の高いトークンを合計がp以上になるまで絞り込む
// repeat_penalty: 0以下でデフォルト(1.0=無効) / 1.0より大きい値で有効 (推奨: 1.05〜1.2)
//                直近に出現したトークンのスコアを下げて繰り返しを抑制する
// penalty_last_n: ペナルティを適用する直近トークン数 / 0でデフォルト(64) / -1でコンテキスト全体
LLMTC_API int LLMTC_CALL llmtc_set_generation_params(double top_p, double repeat_penalty, int penalty_last_n);

// ---- 初期化 / 解放 ----
LLMTC_API int  LLMTC_CALL llmtc_init(const char* model_path, int n_ctx, int n_gpu_layers);
LLMTC_API void LLMTC_CALL llmtc_free(void);

// ---- モデル情報 ----
LLMTC_API int LLMTC_CALL llmtc_get_model_info_to_buffer(char* out_buffer, int buffer_size);
LLMTC_API int LLMTC_CALL llmtc_get_n_ctx(void);

// ---- 同期API ----
LLMTC_API const char* LLMTC_CALL llmtc_generate(
    const char* system_prompt,
    const char* user_message,
    double temperature,
    int max_tokens);

LLMTC_API int LLMTC_CALL llmtc_generate_to_buffer(
    const char* system_prompt,
    const char* user_message,
    double temperature,
    int max_tokens,
    char* out_buffer,
    int buffer_size);

// ---- 非同期API ----

// バックグラウンドで生成を開始して即リターン。
// 戻り値: 0=成功 / -1=前の生成が実行中 / -2=未初期化
LLMTC_API int  LLMTC_CALL llmtc_generate_async(
    const char* system_prompt,
    const char* user_message,
    double temperature,
    int max_tokens);

// 生成をキャンセルする。即リターン。
// 次のトークン生成の直前にキャンセルが反映される (数十ms以内)。
// キャンセル後は LLMTC_STATUS_CANCELLED になり、
// llmtc_get_result_to_buffer で途中までの結果を取得できる。
LLMTC_API void LLMTC_CALL llmtc_cancel(void);

// 現在の状態を返す。
LLMTC_API int LLMTC_CALL llmtc_get_status(void);

// 生成中の途中テキストを先頭からの全文で返す (BUSY中でも呼べる)。
LLMTC_API int LLMTC_CALL llmtc_get_partial_result_to_buffer(char* out_buffer, int buffer_size);

// 生成完了 (DONE / CANCELLED) 後に最終結果をバッファに書き込む。
// CANCELLED の場合は途中までの結果を返す。
// 呼び出し後、状態は IDLE に戻る。
LLMTC_API int LLMTC_CALL llmtc_get_result_to_buffer(char* out_buffer, int buffer_size);

#ifdef __cplusplus
}
#endif

#endif // LLMTEXTCORE_H
