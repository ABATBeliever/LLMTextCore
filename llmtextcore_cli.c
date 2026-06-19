#include "llmtextcore.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#include <shellapi.h>

// UTF-16の文字列をUTF-8に変換する (呼び出し側でfreeすること)
static char* wide_to_utf8(const wchar_t* w) {
    int len = WideCharToMultiByte(CP_UTF8, 0, w, -1, NULL, 0, NULL, NULL);
    char* buf = (char*)malloc((size_t)len);
    WideCharToMultiByte(CP_UTF8, 0, w, -1, buf, len, NULL, NULL);
    return buf;
}
#endif

static void print_usage(const char* prog) {
    printf("LLMTextCore CLI\nCopyright (c) 2026 ABATBeliever. Under MIT License.\nBuild with llama-cpp(https://github.com/ggml-org/llama.cpp)\n\n");
    printf("使い方:\n");
    printf("  %s <model.gguf> [質問文] [temperature] [max_tokens] [verbose(0/1)]\n", prog);
    printf("  %s -l <model.gguf> [verbose(0/1)]   モデルのロード確認のみを行う\n", prog);
    printf("  %s -v, --version   バージョン情報を表示し終了\n", prog);
    printf("  %s -h, --help      このヘルプを表示し終了\n\n", prog);
    printf("引数:\n");
    printf("  model.gguf   読み込むモデルファイル (必須)\n");
    printf("  質問文       省略時は「こんにちは、自己紹介してください。」が利用される。\n");
    printf("  temperature  生成のtemperature。0以上1以下で指定する。省略時は0になる。\n");
    printf("  max_tokens   1回の応答の最大トークン数。省略時は-1(無制限)。\n");
    printf("  verbose      llama.cppの詳細ログを表示するか。省略時は0(表示しない)。\n\n");
    printf("通常実行時、標準出力にはAIの応答のみが出力されます。\n");
    printf("エラーメッセージは標準エラー出力に出力されます。\n");
}

// モデルのロードだけ試して結果を返す (-l モード)
static int run_load_check(const char* model_path, int verbose) {
    llmtc_set_verbose(verbose);
    int rc = llmtc_init(model_path, 4096, 0);
    if (rc == 0) {
        printf("OK\n");
        llmtc_free();
        return 0;
    }

    const char* reason =
        (rc == -1) ? "model_load_failed" :
        (rc == -2) ? "context_init_failed" : "unknown_error";
    printf("NG: %s (rc=%d)\n", reason, rc);
    return 1;
}

int main(int argc, char** argv) {
#ifdef _WIN32
    // コンソールの出力をUTF-8として解釈させる (表示の文字化け対策)
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);

    // argv をCRT経由(CP932で化ける)ではなく、Windows APIから
    // 直接UTF-16で取得し直してUTF-8に変換する (入力の文字化け対策)
    int wargc = 0;
    LPWSTR* wargv = CommandLineToArgvW(GetCommandLineW(), &wargc);
    char** uargv = (char**)malloc(sizeof(char*) * (size_t)wargc);
    for (int i = 0; i < wargc; i++) {
        uargv[i] = wide_to_utf8(wargv[i]);
    }
    LocalFree(wargv);
    argc = wargc;
    argv = uargv;
#endif

    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    if (strcmp(argv[1], "-v") == 0 || strcmp(argv[1], "--version") == 0) {
        printf("%s\n", llmtc_version_info());
        return 0;
    }
    if (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "/?") == 0) {
        print_usage(argv[0]);
        return 0;
    }
    if (strcmp(argv[1], "-l") == 0 || strcmp(argv[1], "--load-only") == 0) {
        if (argc < 3) {
            fprintf(stderr, "エラー: -l にはモデルパスが必要です\n");
            print_usage(argv[0]);
            return 1;
        }
        const char* model_path = argv[2];
        int verbose = (argc >= 4) ? atoi(argv[3]) : 0;
        return run_load_check(model_path, verbose);
    }

    const char* model_path = argv[1];
    const char* question   = (argc >= 3) ? argv[2] : "こんにちは、自己紹介してください。";
    double temperature     = (argc >= 4) ? atof(argv[3]) : 0; // 0 = デフォルト
    int    max_tokens      = (argc >= 5) ? atoi(argv[4]) : -1;   // -1 = デフォルト
    int    verbose          = (argc >= 6) ? atoi(argv[5]) : 0;

    llmtc_set_verbose(verbose); // llmtc_init より前に呼ぶ

    int rc = llmtc_init(model_path, /*n_ctx=*/4096, /*n_gpu_layers=*/0);
    if (rc != 0) {
        fprintf(stderr, "llmtc_init 失敗 (rc=%d)\n", rc);
        return 1;
    }

    const char* reply = llmtc_generate(
        "あなたは親切な、ローカルで動作する組み込みAIです。",
        question, temperature, max_tokens);
    if (!reply) {
        fprintf(stderr, "応答取得に失敗しました\n");
        llmtc_free();
        return 1;
    }

    printf("%s\n", reply);

    llmtc_free();
    return 0;
}
