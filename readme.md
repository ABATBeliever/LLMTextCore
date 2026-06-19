# LLMTextCore

llama.cpp をラップして、テキストチャット機能だけを**単一の**シンプルなC APIとCLIで公開するプロジェクトです。

HSPなどの言語からも呼び出せるよう設計しています。

- `LLMTextCore.dll` — C/C++/C#/Pythonなど、DLLを直接呼べる言語からin-processで使う用
- `LLMTextCore.exe` — CLI

両方とも中身は同じロジックで、CLIはDLLを薄くラップしているだけです。

**設計方針: ステートレス。** DLL側は会話履歴を一切保持しません。複数ターンの会話をしたい場合は、
呼び出し側で過去のやり取りをプロンプトとして連結し、`llmtc_generate`系の呼び出しのたびに
まるごと渡してください。

## ビルド手順 (Windows / Visual Studio)

```powershell
git clone https://github.com/ggml-org/llama.cpp
cmake -B build -A x64
cmake --build build --config Release
```

成功すると `build\Release\LLMTextCore.dll` と `build\Release\LLMTextCore.exe` ができます。
配布物は **この2ファイル + .ggufモデルファイル** だけで完結します。

## CLIの使い方

```powershell
# バージョン確認 (自身のバージョンとビルドに使ったllama.cppのコミットを表示)
.\LLMTextCore.exe -v

# ヘルプ
.\LLMTextCore.exe -h

# モデルのロード確認のみ (生成はしない)
.\LLMTextCore.exe -l model.gguf

# 通常実行
.\LLMTextCore.exe model.gguf "こんにちは" 0.7 256 0
```

引数は `<model.gguf> [質問文] [temperature] [max_tokens] [verbose(0/1)]`。
`質問文`以降は省略可能で、省略した場合はデフォルト値が使われます。

## C API (llmtextcore.h)

| 関数 | 役割 |
|---|---|
| `llmtc_version_info()` | バージョン文字列を取得 (ポインタ返し、C/C++向け) |
| `llmtc_version_info_to_buffer(buf, size)` | バージョン文字列をバッファに書き込む (HSP等向け) |
| `llmtc_set_verbose(int)` | ログ出力のON/OFF (`llmtc_init`より前に呼ぶ) |
| `llmtc_init(model_path, n_ctx, n_gpu_layers)` | モデル読み込み |
| `llmtc_generate(system_prompt, user_message, temperature, max_tokens)` | ステートレスな1回完結の応答生成。ポインタ返し (C/C++向け) |
| `llmtc_generate_to_buffer(system_prompt, user_message, temperature, max_tokens, buf, size)` | 同上、応答をバッファに書き込む (HSP等向け) |
| `llmtc_free()` | リソース解放 |

`llmtc_generate`系は呼び出すたびに直前までの文脈をクリアしてから生成するため、会話履歴は残りません。
`system_prompt`は`NULL`または空文字列でも構いません。

## 他言語に応用する

例えば、HSP3.7で動作させることができます。

最小サンプル(動作確認用):

```hsp
#include "hsp3_64.as" //64bitが前提です
#uselib "LLMTextCore.dll"
#cfunc llmtc_version_info_to_buffer "llmtc_version_info_to_buffer" var, int
#func  llmtc_set_verbose            "llmtc_set_verbose" int
#cfunc llmtc_init                   "llmtc_init" str, int, int
#cfunc llmtc_generate_to_buffer     "llmtc_generate_to_buffer" str, str, double, int, var, int
#func  llmtc_free                   "llmtc_free"
model_path = "C:/Users/hoge/Documents/models/Qwen3-1.7B-Q6_K.gguf"

sdim verbuf, 256
dummy = llmtc_version_info_to_buffer(verbuf, 256)
mes "バージョン: " + verbuf

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
