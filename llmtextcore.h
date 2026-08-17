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
// 成功で0、失敗で負値を返す。
// -1: モデルファイルの読み込み失敗
// -2: コンテキストの初期化失敗
// -3: すでに初期化済み (llmtc_free を呼んでから再度呼ぶこと)
// 失敗時は llmtc_get_last_error_to_buffer でエラー内容を取得できる。
LLMTC_API int  LLMTC_CALL llmtc_init(const char* model_path, int n_ctx, int n_gpu_layers);

// [v0.7.4] llmtc_generate_async 実行中に呼ばれた場合、内部で自動的にキャンセルしてから
// 解放処理を行う (完了まで無期限にブロックすることはない)。
// 注意: あくまで llmtc_generate_async のワーカースレッドに対する安全策であり、
// llmtc_generate / llmtc_generate_to_buffer (同期API) の実行中に「別スレッドから」
// llmtc_free を呼ぶケースは想定していない。同期APIは呼び出したスレッドをブロックする
// ので、free は同じスレッドで同期APIの呼び出しが終わった後に呼ぶこと。
LLMTC_API void LLMTC_CALL llmtc_free(void);

// 直近のエラーメッセージをバッファに書き込む。
// llmtc_init 失敗時に呼ぶことで原因を確認できる。
// llmtc_init を呼ぶたびにリセットされる。
// [v0.7.4] 加えて、生成系API (同期/非同期問わず) を呼ぶたびにもリセットされるため、
// 「今回の呼び出しで何が起きたか」がそれ以前の呼び出しの残骸と混ざらなくなった。
LLMTC_API int LLMTC_CALL llmtc_get_last_error_to_buffer(char* out_buffer, int buffer_size);

// ---- モデル情報 ----
LLMTC_API int LLMTC_CALL llmtc_get_model_info_to_buffer(char* out_buffer, int buffer_size);
LLMTC_API int LLMTC_CALL llmtc_get_n_ctx(void);

// ---- 同期API ----
//
// [v0.7.4] llmtc_generate_async によるバックグラウンド生成が実行中の場合、
// または DONE/CANCELLED の結果が未取得の場合、これらの同期APIは失敗する
// (llmtc_generate は NULL、llmtc_generate_to_buffer は負値を返す。
//  詳細は llmtc_get_last_error_to_buffer で確認できる)。
// 同期APIだけを使う通常の利用ではこの状態にはならないため、挙動に変化はない。
//
// llmtc_generate が返すポインタは、次に llmtc_generate / llmtc_generate_to_buffer /
// llmtc_generate_async のいずれかを呼ぶまでの間のみ有効。使い終わったら
// すぐに文字列としてコピーするか出力すること (strerror 等と同様の一時バッファ)。
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
