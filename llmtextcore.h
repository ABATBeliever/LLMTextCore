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

// llmtc_set_think_mode の mode 引数
#define LLMTC_THINK_MODE_NONE 0  // 分離しない (デフォルト。従来通り生の出力をそのまま返す)
#define LLMTC_THINK_MODE_TAG  1  // 開始/終了タグで挟まれた区間を思考として本文から分離する

#ifdef __cplusplus
extern "C" {
#endif

// ---- バージョン ----
LLMTC_API const char* LLMTC_CALL llmtc_version_info(void);
LLMTC_API int         LLMTC_CALL llmtc_version_info_to_buffer(char* out_buffer, int buffer_size);

// ---- APIレベル (呼び出し互換性のバージョン) ----
//
// LLMTEXTCORE_VERSION (0.8.0 等、ビルドごとの製品バージョン) とは別軸の値。
// <major>.<minor> の2つの整数で管理する。
//   major: 破壊的変更 (既存関数のシグネチャ変更・削除、既存の動作を変える変更) で上げる。
//          semverのメジャーバージョンと連動させる。
//   minor: 後方互換な追加 (新しい関数の追加など) で上げる。
// 内部リファクタリングや効率改善、バグ修正など、呼び出し側から見えるインターフェースに
// 変化が無い更新ではどちらも上がらないため、異なる LLMTEXTCORE_VERSION が同じAPIレベルを
// 共有することがある。
//
// 例: あるアプリがAPIレベル1.1で追加された関数を使いたい場合、
//     (major == 1 && minor >= 1) を満たすDLLを要求する、といった形でチェックする。
//     メジャーが上がった場合は既存関数が変更・削除されている可能性があるため、
//     「1.1以上2.0未満」のように上限も含めてチェックするのが安全。
//
// この値を変える際は、実際にAPI(このヘッダファイル)を変更するのと同じタイミングで
// 必ず一緒に更新すること (ソースにハードコードされた数値であり、ビルド設定などから
// 自動算出されるものではない)。
#define LLMTC_API_LEVEL_MAJOR 1
#define LLMTC_API_LEVEL_MINOR 0

LLMTC_API int LLMTC_CALL llmtc_get_api_level_major(void);
LLMTC_API int LLMTC_CALL llmtc_get_api_level_minor(void);

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

// ---- 思考(thinking/reasoning)分離 ----
//
// QwenやGPT-OSSなど、最近のモデルは最終回答の前に思考過程(reasoning)を出力するものが多いが、
// 出力形式はモデルによってバラバラ (例: Qwen3系は "<think>...</think>") なので、
// アプリ側で個別に対応するのではなく、DLL側でタグ指定による分離に対応する。
//
// mode = LLMTC_THINK_MODE_NONE:
//     分離しない。start_tag/end_tag は無視される。llmtc_generate系は生の出力をそのまま返す。
// mode = LLMTC_THINK_MODE_TAG:
//     start_tag ~ end_tag で挟まれた区間を「思考」として本文から取り除く。
//     生成結果(llmtc_generate / llmtc_generate_to_buffer / llmtc_get_result_to_buffer)は
//     思考区間を除いた本文のみを返すようになり、思考区間の内容は llmtc_get_thinking_to_buffer
//     で別途取得できる。
//     start_tag / end_tag に NULL または空文字列を渡した場合、その引数は変更されない
//     (未設定時のデフォルトは Qwen3系の "<think>" / "</think>")。
//     end_tag がまだ出力中に見つかっていない間は、start_tag以降の全文が「思考」として扱われる
//     (ストリーミング中に思考が全文思考として見えるのは正常な挙動)。
//
// llmtc_init の前後どちらでも呼べる。次の生成呼び出しから反映される。
LLMTC_API void LLMTC_CALL llmtc_set_think_mode(int mode, const char* start_tag, const char* end_tag);

// 直近の生成における「思考」部分を取得する。llmtc_get_partial_result_to_buffer と同様、
// BUSY中(ストリーミング中)でもその時点までの思考部分を取得できるし、生成完了後は
// 最終的な思考全文を取得できる (呼んでも内部状態はクリアされない。何度でも呼べる)。
// 思考モードがNONE、または今回の生成にタグが含まれていなかった場合は空文字列を返す。
LLMTC_API int LLMTC_CALL llmtc_get_thinking_to_buffer(char* out_buffer, int buffer_size);

// ---- 初期化 / 解放 ----
// 成功で0、失敗で負値を返す。
// -1: モデルファイルの読み込み失敗
// -2: コンテキストの初期化失敗
// -3: すでに初期化済み (llmtc_free を呼んでから再度呼ぶこと)
// 失敗時は llmtc_get_last_error_to_buffer でエラー内容を取得できる。
LLMTC_API int  LLMTC_CALL llmtc_init(const char* model_path, int n_ctx, int n_gpu_layers);

// llmtc_generate_async 実行中に呼ばれた場合は、内部で自動的にキャンセルしてから
// 解放処理を行う (完了まで無期限にブロックすることはない)。
// 注意: あくまで llmtc_generate_async のワーカースレッドに対する安全策であり、
// llmtc_generate / llmtc_generate_to_buffer (同期API) の実行中に「別スレッドから」
// llmtc_free を呼ぶケースは想定していない。同期APIは呼び出したスレッドをブロックする
// ので、free は同じスレッドで同期APIの呼び出しが終わった後に呼ぶこと。
LLMTC_API void LLMTC_CALL llmtc_free(void);

// 直近のエラーメッセージをバッファに書き込む。
// llmtc_init 失敗時に呼ぶことで原因を確認できる。
// llmtc_init を呼ぶたびにリセットされる。
// 生成系API (同期/非同期問わず) を呼ぶたびにもリセットされるため、
// 「今回の呼び出しで何が起きたか」が以前の呼び出しの内容と混ざることはない。
LLMTC_API int LLMTC_CALL llmtc_get_last_error_to_buffer(char* out_buffer, int buffer_size);

// ---- モデル情報 ----
LLMTC_API int LLMTC_CALL llmtc_get_model_info_to_buffer(char* out_buffer, int buffer_size);
LLMTC_API int LLMTC_CALL llmtc_get_n_ctx(void);

// ---- 同期API ----
//
// llmtc_generate_async によるバックグラウンド生成が実行中の場合、
// または DONE/CANCELLED の結果が未取得の場合、これらの同期APIは失敗する
// (llmtc_generate は NULL、llmtc_generate_to_buffer は負値を返す。
//  詳細は llmtc_get_last_error_to_buffer で確認できる)。
// 同期APIだけを使う通常の利用ではこの状態にはならない。
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
