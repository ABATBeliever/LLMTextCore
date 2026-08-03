# LLMTextCore

llama.cpp をラップして、ローカルLLMのテキスト生成機能をシンプルなC APIとCLIとして公開するライブラリです。

- **LLMTextCore.dll** — C/C++・HSP・C#・Python など、DLLを直接呼べる言語からプロセス内で使う
- **LLMTextCore.exe** — HSPのpipeexecなど、サブプロセス経由で標準出力を読む用途向けCLI

どちらも同じロジックを共有しており、CLIはDLLを薄くラップしているだけです。

## 設計方針

- **ステートレス**: DLL側は会話履歴を保持しません。複数ターンの会話は、呼び出し側がプロンプトを連結して渡してください。
- **シンプルなAPI**: `init` → `generate_async` → `get_result` の3ステップで動きます。
- **CPU推論専用**: GPUオフロードは現在サポートしていません。軽量モデルをCPUで動かすことを想定しています。
- **UTF-8内部処理**: DLL内部は常にUTF-8で処理します。HSP 3.x向けにShift-JIS変換オプションを提供しています。

## 必要なもの

- Windows 10/11 (x64)
- ビルド時: Visual Studio 2019以降 (C++17)、CMake 3.14以降、Git
- 実行時: .gguf形式のモデルファイル

## ビルド

```powershell
git clone https://github.com/ggml-org/llama.cpp
cmake -B build -A x64
cmake --build build --config Release
```

`build\Release\LLMTextCore.dll` と `build\Release\LLMTextCore.exe` が生成されます。  
配布物はこの2ファイルと .gguf モデルファイルだけで動作します。

## CLI の使い方

```powershell
# バージョン確認
LLMTextCore.exe -v

# ヘルプ
LLMTextCore.exe -h

# モデルのロード確認
LLMTextCore.exe -l model.gguf

# モデル情報表示
LLMTextCore.exe -i model.gguf

# 通常実行 (AIの応答のみ標準出力に出力)
LLMTextCore.exe model.gguf "こんにちは"

# 各引数をすべて指定する場合
LLMTextCore.exe model.gguf "質問文" temperature max_tokens verbose encoding stream top_p repeat_penalty
```

| 引数 | 省略時デフォルト | 説明 |
|---|---|---|
| temperature | 0.7 | 生成の温度。0以下で省略扱い |
| max_tokens | 512 | 1回の応答の最大トークン数。0以下で省略扱い |
| verbose | 0 | 1でllama.cppの詳細ログを表示 |
| encoding | 0 | 0=UTF-8出力 / 1=Shift-JIS出力 |
| stream | 0 | 1でトークンを生成しながら逐次出力 |
| top_p | 0.95 | Nucleus sampling。0以下で省略扱い、1.0で無効化 |
| repeat_penalty | 1.0 | 繰り返しペナルティ。0以下で省略扱い、1.0で無効 |

エラーメッセージは標準エラー出力に出ます。標準出力にはAIの応答のみが流れます。

## DLL API リファレンス

### 定数

```c
// llmtc_get_status の戻り値
#define LLMTC_STATUS_IDLE      0   // 待機中
#define LLMTC_STATUS_BUSY      1   // 生成中
#define LLMTC_STATUS_DONE      2   // 生成完了
#define LLMTC_STATUS_CANCELLED 3   // キャンセルされた
#define LLMTC_STATUS_ERROR    -1   // エラー

// llmtc_set_encoding の引数
#define LLMTC_ENCODING_UTF8    0   // UTF-8のまま (デフォルト)
#define LLMTC_ENCODING_SJIS    1   // Shift-JIS自動変換 (HSP 3.x向け)
```

### 関数一覧

| 関数 | 説明 |
|---|---|
| `llmtc_version_info()` | バージョン文字列を返す (ポインタ、C/C++向け) |
| `llmtc_version_info_to_buffer(buf, size)` | バージョン文字列をバッファに書き込む |
| `llmtc_set_verbose(enable)` | 詳細ログのON(1)/OFF(0)。`llmtc_init`より前に呼ぶ |
| `llmtc_set_encoding(mode)` | 出力エンコーディングを設定。`*_to_buffer`系の出力に反映される |
| `llmtc_set_generation_params(top_p, repeat_penalty, penalty_last_n)` | 生成パラメータのデフォルト値を設定 |
| `llmtc_init(model_path, n_ctx, n_gpu_layers)` | モデルを読み込む。成功で0、失敗で負値 |
| `llmtc_free()` | モデルとコンテキストを解放する |
| `llmtc_get_last_error_to_buffer(buf, size)` | 直近のエラーメッセージを取得。`llmtc_init`失敗時に原因を確認できる |
| `llmtc_get_model_info_to_buffer(buf, size)` | ロード済みモデルの情報(名前・パラメータ数・n_ctx)を取得 |
| `llmtc_get_n_ctx()` | コンテキスト長をintで返す |
| `llmtc_generate(system, user, temperature, max_tokens)` | 同期生成。完了までブロックする。応答ポインタを返す (C/C++向け) |
| `llmtc_generate_to_buffer(system, user, temperature, max_tokens, buf, size)` | 同期生成、バッファ書き込み版 |
| `llmtc_generate_async(system, user, temperature, max_tokens)` | 非同期生成を開始して即リターン |
| `llmtc_cancel()` | 生成をキャンセル。次のトークン前に反映される |
| `llmtc_get_status()` | 現在の非同期生成の状態を返す |
| `llmtc_get_partial_result_to_buffer(buf, size)` | 生成中の途中テキストを先頭から全文で返す (ストリーミング用) |
| `llmtc_get_result_to_buffer(buf, size)` | 生成完了後(DONEまたはCANCELLED)に結果を取得 |

### 入力エンコーディングについて

`system_prompt`・`user_message`の入力は**常にUTF-8として受け取ります**。  
`llmtc_set_encoding(LLMTC_ENCODING_SJIS)`を設定しても、入力の変換は行いません。変換されるのは`*_to_buffer`系の**出力バッファのみ**です。

HSP 3.xで日本語プロンプトを渡す場合は、ソースファイルをUTF-8で保存するか、`cnvatos()`等でUTF-8変換してから渡してください。

## HSP での使い方

64bit HSP (`#include "hsp3_64.as"`) を前提としています。

```hsp
#include "hsp3_64.as"

#uselib "LLMTextCore.dll"
#cfunc llmtc_version_info_to_buffer       "llmtc_version_info_to_buffer"       var, int
#func  llmtc_set_verbose                  "llmtc_set_verbose"                  int
#func  llmtc_set_encoding                 "llmtc_set_encoding"                 int
#cfunc llmtc_set_generation_params        "llmtc_set_generation_params"        double, double, int
#cfunc llmtc_init                         "llmtc_init"                         str, int, int
#func  llmtc_free                         "llmtc_free"
#cfunc llmtc_get_model_info_to_buffer     "llmtc_get_model_info_to_buffer"     var, int
#cfunc llmtc_get_n_ctx                    "llmtc_get_n_ctx"
#cfunc llmtc_generate_async               "llmtc_generate_async"               str, str, double, int
#func  llmtc_cancel                       "llmtc_cancel"
#cfunc llmtc_get_status                   "llmtc_get_status"
#cfunc llmtc_get_partial_result_to_buffer "llmtc_get_partial_result_to_buffer" var, int
#cfunc llmtc_get_result_to_buffer         "llmtc_get_result_to_buffer"         var, int
#cfunc llmtc_get_last_error_to_buffer     "llmtc_get_last_error_to_buffer"     var, int

#const LLMTC_STATUS_IDLE      0
#const LLMTC_STATUS_BUSY      1
#const LLMTC_STATUS_DONE      2
#const LLMTC_STATUS_CANCELLED 3
#const LLMTC_STATUS_ERROR    -1
#const LLMTC_ENCODING_UTF8    0
#const LLMTC_ENCODING_SJIS    1

model_path = "C:/path/to/model.gguf"

; 初期化
llmtc_set_verbose 0
llmtc_set_encoding LLMTC_ENCODING_UTF8
dummy = llmtc_set_generation_params(0.95, 1.1, 0)
rc = llmtc_init(model_path, 4096, 0)
if rc != 0 {
    sdim errbuf, 1024
    dummy = llmtc_get_last_error_to_buffer(errbuf, 1024)
    mes "init failed: " + errbuf
    stop
}

; 非同期生成
p1 = cnvatos("あなたは親切なアシスタントです")
p2 = cnvatos("こんにちは、自己紹介してください")
rc = llmtc_generate_async(p1, p2, 0.7, 256)

; ポーリング
sdim partial, 8192
prev = ""
*loop
    await 200
    dummy = llmtc_get_partial_result_to_buffer(partial, 8192)
    ; partial が前回より増えていれば差分を処理する
    if partial != prev : prev = partial
    if llmtc_get_status() = LLMTC_STATUS_BUSY : goto *loop

; 結果取得
sdim reply, 8192
dummy = llmtc_get_result_to_buffer(reply, 8192)
mes reply

llmtc_free
```

### キャンセルについて

`llmtc_cancel`を呼ぶと次のトークン生成前(数十ms以内)に生成が止まります。  
キャンセル後のステータスは`LLMTC_STATUS_CANCELLED`になり、`llmtc_get_result_to_buffer`で途中までの結果を取得できます。

### ストリーミング表示について

`llmtc_get_partial_result_to_buffer`は「先頭からの全テキスト」を返します。  
前回取得したテキストとの差分を取ることで、生成中に文字を逐次表示できます。  
`objprm`でテキストボックスの内容を更新する方法が実装しやすく見た目も自然です。

## モデルについて

.gguf形式のモデルファイルが必要です。Hugging Face などから入手できます。  
小さめのモデル(1〜3B, Q4〜Q6量子化)がCPU推論で扱いやすいサイズです。

モデルファイル自体はライセンスがそれぞれ異なります。利用前に各モデルのライセンスを確認してください。

## ライセンス

MIT License — 詳細は [LICENSE](LICENSE) を参照してください。

本ソフトウェアはサードパーティのライブラリを静的リンクしています。  
詳細は [CREDITS](CREDITS) を参照してください。
