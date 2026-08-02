#include "llmtextcore.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#include <shellapi.h>

static char* wide_to_utf8(const wchar_t* w) {
    int len = WideCharToMultiByte(CP_UTF8, 0, w, -1, NULL, 0, NULL, NULL);
    char* buf = (char*)malloc((size_t)len);
    WideCharToMultiByte(CP_UTF8, 0, w, -1, buf, len, NULL, NULL);
    return buf;
}

static char* utf8_to_sjis(const char* utf8) {
    int wlen = MultiByteToWideChar(CP_UTF8, 0, utf8, -1, NULL, 0);
    wchar_t* ws = (wchar_t*)malloc(sizeof(wchar_t) * (size_t)wlen);
    MultiByteToWideChar(CP_UTF8, 0, utf8, -1, ws, wlen);
    int alen = WideCharToMultiByte(CP_ACP, 0, ws, -1, NULL, 0, NULL, NULL);
    char* buf = (char*)malloc((size_t)alen);
    WideCharToMultiByte(CP_ACP, 0, ws, -1, buf, alen, NULL, NULL);
    free(ws);
    return buf;
}
#endif

static void print_usage(const char* prog) {
    printf("LLMTextCore CLI\n\n");
    printf("使い方:\n");
    printf("  %s <model.gguf> [質問文] [temperature] [max_tokens] [verbose] [encoding] [stream]\n", prog);
    printf("  %s -l <model.gguf> [verbose]   モデルのロード確認のみ行う\n", prog);
    printf("  %s -i <model.gguf> [verbose]   モデル情報を表示して終了\n", prog);
    printf("  %s -v, --version               バージョン情報を表示して終了\n", prog);
    printf("  %s -h, --help                  このヘルプを表示して終了\n\n", prog);
    printf("引数:\n");
    printf("  temperature  生成の温度。0以下または省略でデフォルト値 (0.7)\n");
    printf("  max_tokens   最大トークン数。0以下または省略でデフォルト値 (512)\n");
    printf("  verbose      0=ログ抑制 (デフォルト) / 1=llama.cpp詳細ログを表示\n");
    printf("  encoding     0=UTF-8出力 (デフォルト) / 1=Shift-JIS出力\n");
    printf("  stream       0=完了後に一括出力 (デフォルト) / 1=生成しながら逐次出力\n\n");
    printf("通常実行時、標準出力にはAIの応答のみが出力されます。\n");
    printf("エラーメッセージは標準エラー出力に出ます。\n");
}

static int run_load_check(const char* model_path, int verbose) {
    llmtc_set_verbose(verbose);
    int rc = llmtc_init(model_path, 4096, 0);
    if (rc == 0) { printf("OK\n"); llmtc_free(); return 0; }
    const char* reason =
        (rc == -1) ? "モデルファイルの読み込みに失敗しました" :
        (rc == -2) ? "コンテキストの初期化に失敗しました" : "不明なエラー";
    printf("NG: %s (rc=%d)\n", reason, rc);
    return 1;
}

static int run_model_info(const char* model_path, int verbose) {
    llmtc_set_verbose(verbose);
    int rc = llmtc_init(model_path, 4096, 0);
    if (rc != 0) {
        fprintf(stderr, "llmtc_init 失敗 (rc=%d)\n", rc);
        return 1;
    }
    char info[512];
    llmtc_get_model_info_to_buffer(info, sizeof(info));
    printf("%s\n", info);
    llmtc_free();
    return 0;
}

// エンコーディングに応じてstrをstdoutに書き出す。改行はつけない。
static void print_encoded(const char* utf8_str, int encoding) {
#ifdef _WIN32
    if (encoding == LLMTC_ENCODING_SJIS) {
        char* sjis = utf8_to_sjis(utf8_str);
        SetConsoleOutputCP(CP_ACP);
        fputs(sjis, stdout);
        fflush(stdout);
        SetConsoleOutputCP(CP_UTF8);
        free(sjis);
        return;
    }
#endif
    fputs(utf8_str, stdout);
    fflush(stdout);
}

int main(int argc, char** argv) {
#ifdef _WIN32
    // コンソールはUTF-8に固定。Shift-JIS出力が必要な場合はソフトウェア変換で対応。
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);

    int wargc = 0;
    LPWSTR* wargv = CommandLineToArgvW(GetCommandLineW(), &wargc);
    char** uargv = (char**)malloc(sizeof(char*) * (size_t)wargc);
    for (int i = 0; i < wargc; i++) uargv[i] = wide_to_utf8(wargv[i]);
    LocalFree(wargv);
    argc = wargc;
    argv = uargv;
#endif

    if (argc < 2) { print_usage(argv[0]); return 1; }

    if (!strcmp(argv[1], "-v") || !strcmp(argv[1], "--version")) {
        printf("%s\n", llmtc_version_info()); return 0;
    }
    if (!strcmp(argv[1], "-h") || !strcmp(argv[1], "--help") || !strcmp(argv[1], "/?")) {
        print_usage(argv[0]); return 0;
    }
    if (!strcmp(argv[1], "-l") || !strcmp(argv[1], "--load-only")) {
        if (argc < 3) { fprintf(stderr, "エラー: -l にはモデルパスが必要です\n"); return 1; }
        return run_load_check(argv[2], (argc >= 4) ? atoi(argv[3]) : 0);
    }
    if (!strcmp(argv[1], "-i") || !strcmp(argv[1], "--info")) {
        if (argc < 3) { fprintf(stderr, "エラー: -i にはモデルパスが必要です\n"); return 1; }
        return run_model_info(argv[2], (argc >= 4) ? atoi(argv[3]) : 0);
    }

    const char* model_path = argv[1];
    const char* question   = (argc >= 3) ? argv[2] : "こんにちは、自己紹介してください。";
    double temperature     = (argc >= 4) ? atof(argv[3]) : -1.0;
    int    max_tokens      = (argc >= 5) ? atoi(argv[4]) : -1;
    int    verbose         = (argc >= 6) ? atoi(argv[5]) : 0;
    int    encoding        = (argc >= 7) ? atoi(argv[6]) : LLMTC_ENCODING_UTF8;
    int    stream          = (argc >= 8) ? atoi(argv[7]) : 0;

    llmtc_set_verbose(verbose);

    int rc = llmtc_init(model_path, 4096, 0);
    if (rc != 0) { fprintf(stderr, "llmtc_init 失敗 (rc=%d)\n", rc); return 1; }

    if (!stream) {
        // 通常モード: 完了してから一括出力
        const char* reply = llmtc_generate(
            "あなたは親切な日本語アシスタントです。",
            question, temperature, max_tokens);
        if (!reply) {
            fprintf(stderr, "応答の取得に失敗しました\n");
            llmtc_free();
            return 1;
        }
        print_encoded(reply, encoding);
        printf("\n");
    } else {
        // ストリーミングモード: 非同期生成 + ポーリングで差分を随時出力
        rc = llmtc_generate_async(
            "あなたは親切な日本語アシスタントです。",
            question, temperature, max_tokens);
        if (rc != 0) {
            fprintf(stderr, "generate_async 失敗 (rc=%d)\n", rc);
            llmtc_free();
            return 1;
        }

        char partial[65536]  = "";
        char prev[65536]     = "";
        int  prev_len        = 0;

        while (llmtc_get_status() == LLMTC_STATUS_BUSY) {
            llmtc_get_partial_result_to_buffer(partial, sizeof(partial));
            int cur_len = (int)strlen(partial);
            if (cur_len > prev_len) {
                // 増えた分だけ出力
                print_encoded(partial + prev_len, encoding);
                prev_len = cur_len;
                memcpy(prev, partial, (size_t)cur_len + 1);
            }
#ifdef _WIN32
            Sleep(10);
#endif
        }
        printf("\n"); // 生成終了後に改行

        // 結果を回収してステータスをIDLEに戻す
        char final_buf[65536];
        llmtc_get_result_to_buffer(final_buf, sizeof(final_buf));
    }

    llmtc_free();
    return 0;
}
