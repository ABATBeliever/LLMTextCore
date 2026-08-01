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

// UTF-8文字列をShift-JIS(CP932)に変換して返す (呼び出し側でfreeすること)
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
    printf("  %s <model.gguf> [質問文] [temperature] [max_tokens] [verbose(0/1)] [encoding(0=utf8,1=sjis)]\n", prog);
    printf("  %s -l <model.gguf> [verbose(0/1)]   モデルのロード確認のみ行う\n", prog);
    printf("  %s -v, --version   バージョン情報を表示して終了\n", prog);
    printf("  %s -h, --help      このヘルプを表示して終了\n\n", prog);
    printf("引数:\n");
    printf("  model.gguf   読み込むモデルファイル (必須)\n");
    printf("  質問文       省略時は固定の挨拶文を使用\n");
    printf("  temperature  生成の温度。0以下、または省略でデフォルト値\n");
    printf("  max_tokens   1回の応答の最大トークン数。0以下、または省略でデフォルト値\n");
    printf("  verbose      1を指定するとllama.cppの詳細ログを表示。省略時は0\n");
    printf("  encoding     0=UTF-8(デフォルト) / 1=Shift-JIS出力\n\n");
    printf("通常実行時、標準出力にはAIの応答のみが出力されます。\n");
    printf("エラーメッセージは標準エラー出力に出ます。\n");
}

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

// UTF-8の文字列をencodingに応じて標準出力に書き出す。
// コンソールは常にUTF-8に設定済みなので、sjis指定時だけソフトウェア変換を挟む。
// (pipeexecなどでShift-JISのストリームが必要な場合に使用)
static void print_encoded(const char* utf8_str, int encoding) {
#ifdef _WIN32
    if (encoding == LLMTC_ENCODING_SJIS) {
        char* sjis = utf8_to_sjis(utf8_str);
        // sjis出力時はコードページをCP932に切り替えて出力し、戻す
        SetConsoleOutputCP(CP_ACP);
        fputs(sjis, stdout);
        fputc('\n', stdout);
        fflush(stdout);
        SetConsoleOutputCP(CP_UTF8);
        free(sjis);
        return;
    }
#endif
    printf("%s\n", utf8_str);
}

int main(int argc, char** argv) {
#ifdef _WIN32
    // 出力コードページを最初にUTF-8に固定する。
    // これより後で encoding=sjis を指定した場合はソフトウェア側で変換してから出力するので、
    // コンソール側は常にUTF-8のままでOK。
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);

    // 引数をUTF-8で取得し直す
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
            return 1;
        }
        int verbose = (argc >= 4) ? atoi(argv[3]) : 0;
        return run_load_check(argv[2], verbose);
    }

    const char* model_path = argv[1];
    const char* question   = (argc >= 3) ? argv[2] : "こんにちは、自己紹介してください。";
    double temperature     = (argc >= 4) ? atof(argv[3]) : -1.0;
    int    max_tokens      = (argc >= 5) ? atoi(argv[4]) : -1;
    int    verbose         = (argc >= 6) ? atoi(argv[5]) : 0;
    int    encoding        = (argc >= 7) ? atoi(argv[6]) : LLMTC_ENCODING_UTF8;

    llmtc_set_verbose(verbose);

    int rc = llmtc_init(model_path, 4096, 0);
    if (rc != 0) {
        fprintf(stderr, "llmtc_init 失敗 (rc=%d)\n", rc);
        return 1;
    }

    // EXEは自前で変換するのでDLL側のエンコーディングはUTF-8のまま
    const char* reply = llmtc_generate(
        "あなたは親切な日本語アシスタントです。",
        question, temperature, max_tokens);
    if (!reply) {
        fprintf(stderr, "応答取得に失敗しました\n");
        llmtc_free();
        return 1;
    }

    print_encoded(reply, encoding);

    llmtc_free();
    return 0;
}
