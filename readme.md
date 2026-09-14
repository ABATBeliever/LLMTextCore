# LLMTextCore

llama.cpp をラップして、テキストチャット機能だけを**単一の**シンプルなC APIとCLIで公開するプロジェクトです。

HSPなどの言語からも呼び出せるよう設計しています。

- `LLMTextCore.dll` — C/C++/C#/Pythonなど、DLLを直接呼べる言語からin-processで使う用
- `LLMTextCore.exe` — CLI

両方とも中身は同じロジックで、CLIはDLLを薄くラップしているだけです。

## ビルド手順 (Windows / Visual Studio)

```powershell
cmake -B build -A x64
cmake --build build --config Release
```
`build\Release\LLMTextCore.dll` と `build\Release\LLMTextCore.exe` ができます。

## CLIの使い方 (v0.8.0〜)

v0.8.0でCLIの引数書式をフラグ形式に全面刷新しました (v0.7.x以前の位置引数形式とは
互換性がありません)。オプション一覧は `--help` を参照してください。

```powershell
.\LLMTextCore.exe --model model.gguf --prompt "こんにちは" --stream
```

主なオプション (詳細は `--help` を参照):

| オプション | 役割 |
|---|---|
| `-m, --model <path>` | モデルファイル(.gguf)のパス (必須) |
| `-p, --prompt <text>` | ユーザーメッセージ |
| `-s, --system <text>` | システムプロンプト (`""` で無効化) |
| `--temp <float>` | temperature |
| `--max-tokens <int>` | 最大トークン数 |
| `--top-p <float>` / `--repeat-penalty <float>` / `--penalty-last-n <int>` | サンプリングパラメータ |
| `--stream` | 逐次出力 |
| `--encoding utf8\|sjis` | 出力エンコーディング |
| `--think-cut` | 指定すると思考(thinking)を本文から分離する (既定: 分離しない) |
| `--think-start <tag>` / `--think-end <tag>` | 思考タグをカスタム指定 (既定: `<think>` / `</think>`) |
| `--show-thinking` | 分離した思考内容を標準エラー出力に表示 |
| `-l, --load-only` | モデルのロード確認のみ |
| `-i, --info` | モデル情報を表示して終了 |
| `--api-level` | APIレベル (`<major>.<minor>`) を表示して終了 |

標準出力には常にAIの応答本文のみが出力されます。エラーおよび`--show-thinking`指定時の
思考内容は標準エラー出力に出るため、`pipeexec`等でstdoutだけを読む使い方に影響しません。

なお、本ツールは互換性(動く環境の広さ)を優先し、GPUオフロードには対応していません。
常にCPU推論のみを行います。

## C API (llmtextcore.h)

| 関数 | 役割 |
|---|---|
| `llmtc_version_info()` | バージョン文字列を取得 (ポインタ返し、C/C++向け) |
| `llmtc_version_info_to_buffer(buf, size)` | バージョン文字列をバッファに書き込む (HSP等向け) |
| `llmtc_get_api_level_major()` / `llmtc_get_api_level_minor()` | APIレベルを取得 (下記参照) |
| `llmtc_set_verbose(int)` | ログ出力のON/OFF (`llmtc_init`より前に呼ぶ) |
| `llmtc_init(model_path, n_ctx, n_gpu_layers)` | モデル読み込み |
| `llmtc_generate(system_prompt, user_message, temperature, max_tokens)` | ステートレスな1回完結の応答生成。ポインタ返し (C/C++向け) |
| `llmtc_generate_to_buffer(system_prompt, user_message, temperature, max_tokens, buf, size)` | 同上、応答をバッファに書き込む (HSP等向け) |
| `llmtc_set_think_mode(mode, start_tag, end_tag)` | 思考(thinking)分離を設定 (下記参照) |
| `llmtc_get_thinking_to_buffer(buf, size)` | 直近の生成における思考部分を取得 |
| `llmtc_free()` | リソース解放 |

`llmtc_generate`系は呼び出すたびに直前までの文脈をクリアしてから生成するため、会話履歴は残りません。
`system_prompt`は`NULL`または空文字列でも構いません。

### 思考(thinking)分離 (v0.8.0〜)

Qwen3系のように、最終回答の前に`<think>...</think>`のような思考過程を出力するモデルに対応するため、
本文と思考を分離して取得できます。タグさえ分かれば、Qwen以外の形式(例えばGPT-OSS系のHarmony形式の
analysisチャンネルなど)にもある程度対応できます。

```c
// Qwen3系のデフォルトタグ(<think> / </think>)で分離を有効化
llmtc_set_think_mode(LLMTC_THINK_MODE_TAG, NULL, NULL);

// カスタムタグを指定する場合 (例: Harmony形式のanalysisチャンネルを思考とみなす場合の一例)
// ※ 実際のタグはモデルの出力を確認して調整すること
llmtc_set_think_mode(LLMTC_THINK_MODE_TAG, "<|channel|>analysis<|message|>", "<|end|>");

// 分離を無効化 (デフォルト。従来通り生の出力をそのまま返す)
llmtc_set_think_mode(LLMTC_THINK_MODE_NONE, NULL, NULL);
```

分離を有効にすると、`llmtc_generate`系や`llmtc_get_result_to_buffer`/`llmtc_get_partial_result_to_buffer`は
思考区間を除いた本文のみを返すようになります。思考部分は`llmtc_get_thinking_to_buffer`で別途取得できます
(ストリーミング中は`llmtc_get_partial_result_to_buffer`と同様、その時点までの途中結果を返します)。

### APIレベル (v0.8.0〜)

LLMTextCore自体のバージョン(`LLMTEXTCORE_VERSION`。0.8.0など)とは別に、呼び出し側の互換性判定用として
`<major>.<minor>`形式の「APIレベル」を用意しています。Androidの[API Level](https://developer.android.com/tools/releases/platforms)と同じ考え方です。

- **major**: 既存関数のシグネチャ変更・削除など、破壊的変更があった場合に上がる (semverのメジャーバージョンと連動)
- **minor**: 新しい関数の追加など、後方互換な変更があった場合に上がる

内部リファクタリングやバグ修正など、呼び出し側から見えるインターフェースに変化が無い更新ではどちらも
上がらないため、異なる`LLMTEXTCORE_VERSION`が同じAPIレベルを共有することがあります。

```c
// 例: APIレベル1.1で追加された関数を使いたいアプリ側のチェック
int major = llmtc_get_api_level_major();
int minor = llmtc_get_api_level_minor();
if (major == 1 && minor >= 1) {
    // 使える
}
// メジャーが上がっていると既存関数が変更/削除されている可能性があるため、
// 上限(2.0未満)も含めてチェックするのが安全
```

CLIからは `--api-level` で確認できます (`1.0` のように1行だけ出力)。

現在のAPIレベルは **1.0** です。

## 他言語に応用する

例えば、HSP3.7で動作させることができます。

最小サンプル(動作確認用):

```hsp
#include "hsp3_64.as" //64bitが前提です
#uselib "LLMTextCore.dll"
#cfunc llmtc_version_info_to_buffer  "llmtc_version_info_to_buffer" var, int
#cfunc llmtc_get_api_level_major     "llmtc_get_api_level_major"
#cfunc llmtc_get_api_level_minor     "llmtc_get_api_level_minor"
#func  llmtc_set_verbose             "llmtc_set_verbose" int
#cfunc llmtc_init                    "llmtc_init" str, int, int
#cfunc llmtc_generate_to_buffer      "llmtc_generate_to_buffer" str, str, double, int, var, int
#func  llmtc_free                    "llmtc_free"
model_path = "C:/Users/hoge/Documents/models/Qwen3-1.7B-Q6_K.gguf"

sdim verbuf, 256
dummy = llmtc_version_info_to_buffer(verbuf, 256)
mes "バージョン: " + verbuf

; --- APIレベルの確認 ---
; このスクリプトが使う関数はすべてAPIレベル1.0時点で存在するものなので、
; メジャーバージョンが1であることだけ確認すればよい
; (メジャーが変わると既存関数の意味が変わっている可能性があるため、必ずチェックする。
;  もし将来のマイナーバージョンで追加された関数を使う場合は、
;  ここに `if api_minor < 必要なマイナー番号 { ... }` も追加すること)
api_major = llmtc_get_api_level_major()
api_minor = llmtc_get_api_level_minor()
mes "APIレベル: " + api_major + "." + api_minor
if api_major != 1 {
	mes "このDLLのAPIレベルには対応していません (要求: メジャー1 / 実際: " + api_major + "." + api_minor + ")"
	stop
}

llmtc_set_verbose 0

mes "モデルを読み込み中... (重いモデルだとここで数秒~数十秒固まります)"
redraw 1 : await 10 ; 画面を更新してから重い処理に入る

rc = llmtc_init(model_path, 4096, 0)
if rc != 0 {
	mes "llmtc_init 失敗: rc=" + rc
	stop
}
mes "ロード成功"

sdim reply, 8192
len = llmtc_generate_to_buffer("あなたは親切なアシスタントです。", "Hello,how are you?", 0.7, 256, reply, 8192)
mes "----"
mes "AI: " + reply
dialog reply

sdim reply2, 8192
len2 = llmtc_generate_to_buffer("あなたは親切なアシスタントです。", "Please reply the word you said previous?", 0.7, 256, reply2, 8192)
mes "----"
mes "AI(独立した呼び出し): " + reply2
dialog reply2

; --- 解放 ---
llmtc_free
mes "----"
mes "完了"

```

## 注意点

- **モデルファイル(.gguf)は別配布が前提**です。サイズが大きく、モデル自体のライセンスも
  バイナリへの埋め込みを想定していない場合が多いため、実行時に外部ファイルとして読み込みます。
- llama.cppはMITライセンスです。配布する際は同梱モデルのライセンス条項も別途確認してください。
