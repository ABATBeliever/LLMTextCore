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
#define LLMTC_STATUS_IDLE   0
#define LLMTC_STATUS_BUSY   1
#define LLMTC_STATUS_DONE   2
#define LLMTC_STATUS_ERROR -1

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
// enable=0: エラーのみ表示 (デフォルト) / enable=1: 詳細ログ表示
LLMTC_API void LLMTC_CALL llmtc_set_verbose(int enable);

// LLMTC_ENCODING_UTF8(0): 入出力UTF-8のまま (デフォルト)
// LLMTC_ENCODING_SJIS(1): 出力バッファをUTF-8→CP932変換 (HSP 3.x向け)
// 注意: 入力 (system_prompt, user_message) は常にUTF-8を期待する
LLMTC_API void LLMTC_CALL llmtc_set_encoding(int mode);

// ---- 初期化 / 解放 ----
// 成功で0、失敗で負値を返す
// n_ctx=0 でデフォルト(4096)、n_gpu_layers=0 でCPUのみ
LLMTC_API int  LLMTC_CALL llmtc_init(const char* model_path, int n_ctx, int n_gpu_layers);
LLMTC_API void LLMTC_CALL llmtc_free(void);

// ---- モデル情報 (llmtc_init の後に呼ぶこと) ----

// ロード済みモデルの情報を1行ずつ改行区切りで返す。
// 書式例:
//   name: Qwen3 1.7B
//   params: 2030000000
//   n_ctx: 4096
LLMTC_API int LLMTC_CALL llmtc_get_model_info_to_buffer(char* out_buffer, int buffer_size);

// コンテキスト長をintで返す (llmtc_init 後のみ有効、未初期化時は0)
LLMTC_API int LLMTC_CALL llmtc_get_n_ctx(void);

// ---- 同期API (C/C++向け) ----

// 1回完結の応答生成。完了まで呼び出し元をブロックする。
// 戻り値は内部バッファへのポインタ (常にUTF-8)。解放不要。失敗時はNULL。
LLMTC_API const char* LLMTC_CALL llmtc_generate(
    const char* system_prompt,
    const char* user_message,
    double temperature,
    int max_tokens);

// 同上、バッファに書き込む版。llmtc_set_encodingの設定に従って変換する。
LLMTC_API int LLMTC_CALL llmtc_generate_to_buffer(
    const char* system_prompt,
    const char* user_message,
    double temperature,
    int max_tokens,
    char* out_buffer,
    int buffer_size);

// ---- 非同期API (HSP等のポーリング向け) ----

// バックグラウンドで生成を開始して即リターン。
// 戻り値: 0=成功 / -1=前の生成が実行中 / -2=未初期化
LLMTC_API int LLMTC_CALL llmtc_generate_async(
    const char* system_prompt,
    const char* user_message,
    double temperature,
    int max_tokens);

// 現在の状態を返す。
// LLMTC_STATUS_IDLE(0) / LLMTC_STATUS_BUSY(1) /
// LLMTC_STATUS_DONE(2) / LLMTC_STATUS_ERROR(-1)
LLMTC_API int LLMTC_CALL llmtc_get_status(void);

// 生成中の途中テキストを「先頭からの全文」で返す (BUSY中でも呼べる)。
// ストリーミング表示用。呼ぶたびに増えていく。
// 前回呼んだときとの差分を取るのは呼び出し側の責任。
// llmtc_set_encodingの設定に従って変換する。
LLMTC_API int LLMTC_CALL llmtc_get_partial_result_to_buffer(char* out_buffer, int buffer_size);

// 生成完了後、最終結果をバッファに書き込む (DONE後のみ有効)。
// 呼び出し後、状態はIDLEに戻る。
// llmtc_set_encodingの設定に従って変換する。
LLMTC_API int LLMTC_CALL llmtc_get_result_to_buffer(char* out_buffer, int buffer_size);

#ifdef __cplusplus
}
#endif

#endif // LLMTEXTCORE_H
