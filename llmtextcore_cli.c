#include "llmtextcore.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#include <windows.h>
#include <shellapi.h>

static char* wide_to_utf8(const wchar_t* w) {
    int len = WideCharToMultiByte(CP_UTF8, 0, w, -1, NULL, 0, NULL, NULL);
    if (len <= 0) return NULL;
    char* buf = (char*)malloc((size_t)len);
    if (!buf) return NULL;
    WideCharToMultiByte(CP_UTF8, 0, w, -1, buf, len, NULL, NULL);
    return buf;
}
static char* utf8_to_sjis(const char* utf8) {
    int wlen = MultiByteToWideChar(CP_UTF8, 0, utf8, -1, NULL, 0);
    if (wlen <= 0) return NULL;
    wchar_t* ws = (wchar_t*)malloc(sizeof(wchar_t) * (size_t)wlen);
    if (!ws) return NULL;
    MultiByteToWideChar(CP_UTF8, 0, utf8, -1, ws, wlen);
    int alen = WideCharToMultiByte(CP_ACP, 0, ws, -1, NULL, 0, NULL, NULL);
    if (alen <= 0) { free(ws); return NULL; }
    char* buf = (char*)malloc((size_t)alen);
    if (!buf) { free(ws); return NULL; }
    WideCharToMultiByte(CP_ACP, 0, ws, -1, buf, alen, NULL, NULL);
    free(ws);
    return buf;
}

// ---- 標準入力監視による中断 (--stop-key) ----
// --stop-key <key> 指定時、標準入力に指定した文字(キー)が書き込まれたら llmtc_cancel() を呼ぶ。
// HSPなど親プロセスからパイプ経由で中断要求を送る用途を想定している。
// 指定キー以外の入力は無視する。標準入力が閉じられただけ(EOF)の場合は中断しない。
// ReadFileでブロックする専用スレッドで待つのでポーリングは不要。
// 生成が先に終わった場合はスレッドがブロックしたまま残るが、joinせずプロセス終了で破棄される。

// キー名を、標準入力に届く文字(1バイト)へ変換する。
// 対応: 英数字1文字 (例: "q", "5"。英字は大文字小文字を区別しない)、およびESC/SPACE/TAB/ENTER。
// ENTERは '\n' ('\r'も同じ扱い)。認識できない場合は0を返す。
static int stop_key_name_to_char(const char* name) {
    if (!name || !name[0]) return 0;
    if (strlen(name) == 1) {
        unsigned char c = (unsigned char)name[0];
        if (isalnum(c)) return (int)c;
        return 0;
    }
    if (!_stricmp(name, "ESC") || !_stricmp(name, "ESCAPE")) return 0x1B;
    if (!_stricmp(name, "SPACE")) return ' ';
    if (!_stricmp(name, "TAB"))   return '\t';
    if (!_stricmp(name, "ENTER")) return '\n';
    return 0;
}

static int stop_key_matches(int stop_char, unsigned char c) {
    if (stop_char == '\n') return c == '\n' || c == '\r';
    return toupper(c) == toupper(stop_char);
}

static DWORD WINAPI stdin_watch_thread(LPVOID param) {
    int stop_char = (int)(INT_PTR)param;
    HANDLE h = GetStdHandle(STD_INPUT_HANDLE);
    char buf[64];
    DWORD n = 0;
    while (ReadFile(h, buf, sizeof(buf), &n, NULL) && n > 0) {
        for (DWORD i = 0; i < n; i++) {
            if (stop_key_matches(stop_char, (unsigned char)buf[i])) {
                llmtc_cancel();
                return 0;
            }
        }
    }
    return 0; // EOF/エラー: 中断せず終了
}

static void start_stdin_watch(int stop_char) {
    HANDLE h = GetStdHandle(STD_INPUT_HANDLE);
    if (h == NULL || h == INVALID_HANDLE_VALUE) return;
    HANDLE th = CreateThread(NULL, 0, stdin_watch_thread, (LPVOID)(INT_PTR)stop_char, 0, NULL);
    if (th) CloseHandle(th);
}

// ---- 出力ヘルパー ----
// out_stream には stdout (本文用) か stderr (thinking表示用) を渡す。
static void print_encoded(const char* utf8_str, int encoding, FILE* out_stream) {
    if (encoding == LLMTC_ENCODING_SJIS) {
        char* sjis = utf8_to_sjis(utf8_str);
        if (sjis) {
            SetConsoleOutputCP(CP_ACP);
            fputs(sjis, out_stream);
            fflush(out_stream);
            SetConsoleOutputCP(CP_UTF8);
            free(sjis);
        }
        return;
    }
    fputs(utf8_str, out_stream);
    fflush(out_stream);
}

// ---- 実行モード ----
enum {
    MODE_GENERATE = 0,
    MODE_LOAD_ONLY,
    MODE_INFO
};

// ---- オプション ----
typedef struct {
    int mode;

    const char* model_path;
    const char* prompt;
    const char* system_prompt;

    double temperature;
    int    max_tokens;
    int    n_ctx;
    double top_p;
    double repeat_penalty;
    int    penalty_last_n;

    int stream;
    int encoding;
    int verbose;

    int         think_cut;      // 0=false(既定, 分離しない) / 1=true(分離する)
    const char* think_start;    // NULLなら既定値 "<think>" を使用
    const char* think_end;      // NULLなら既定値 "</think>" を使用
    int         show_thinking;

    int         strict;         // 0=false(既定) / 1=true: 値の不自然な結合等をエラーにする
    int         stop_char;      // 0=未指定 / それ以外: 標準入力にこの文字が来たら生成を中断する
} Options;

static void options_init_defaults(Options* o) {
    o->mode            = MODE_GENERATE;
    o->model_path      = NULL;
    o->prompt          = "こんにちは、自己紹介してください。";
    o->system_prompt   = "あなたは親切な日本語アシスタントです。";
    o->temperature     = -1.0;
    o->max_tokens      = -1;
    o->n_ctx           = 4096;
    o->top_p           = -1.0;
    o->repeat_penalty  = -1.0;
    o->penalty_last_n  = 0;
    o->stream          = 0;
    o->encoding        = LLMTC_ENCODING_UTF8;
    o->verbose         = 0;
    o->think_cut       = 0;
    o->think_start     = NULL;
    o->think_end       = NULL;
    o->show_thinking   = 0;
    o->strict          = 0;
    o->stop_char       = 0;
}

static void print_usage(const char* prog) {
    (void)prog;
    printf("LLMTextCore CLI\n\n");
    printf("使い方:\n");
    printf(" LLMTextCore --model <model.gguf> [オプション]...\n");
    printf(" LLMTextCore -v | -h\n\n");
    printf("引数:\n");
    printf("  -m, --model <path>         モデルファイル(.gguf)のパス [必須]\n");
    printf("  -p, --prompt <text>        ユーザーメッセージ (既定: 汎用の挨拶文)\n");
    printf("  -s, --system <text>        システムプロンプト (\"\" で無効化)\n");
    printf("      --temp <float>         temperature (既定: 0.7)\n");
    printf("      --max-tokens <int>     最大トークン数 (既定: 512)\n");
    printf("      --n-ctx <int>          コンテキスト長 (既定: 4096)\n");
    printf("      --top-p <float>        nucleus sampling (既定: 0.95 / 1.0で無効)\n");
    printf("      --repeat-penalty <f>   繰り返し抑制 (既定: 1.0=無効)\n");
    printf("      --penalty-last-n <n>   repeat-penalty対象トークン数 (既定: 64 / -1で全体)\n");
    printf("      --stream               トークンを逐次出力する\n");
    printf("      --encoding <enc>       utf8 (既定) | sjis\n");
    printf("      --think-cut            思考(thinking)を本文から分離する (既定: 分離しない)\n");
    printf("      --think-start <tag>    思考の開始タグ (既定: <think> / --think-cut 指定時のみ使用)\n");
    printf("      --think-end <tag>      思考の終了タグ (既定: </think>)\n");
    printf("      --show-thinking        思考内容を標準エラー出力に表示する\n");
    printf("      --strict               値の不自然な結合など(例: スペース抜けで次の\n");
    printf("                             オプション名と連結した値)を検出しエラーにする\n");
    printf("      --stop-key <key>       標準入力に指定したキーが書き込まれると生成を中断する (--streamと併用時のみ)\n");
    printf("                             英数字1文字、またはESC/SPACE/TAB/ENTER\n");
    printf("      --verbose              llama.cppの詳細ログを表示する\n");
    printf("  -l, --load-only            モデルのロード確認のみ行う\n");
    printf("  -i, --info                 モデル情報を表示して終了\n");
    printf("  -v, --version              バージョン情報を表示して終了\n");
    printf("      --api-level            APIレベル (<major>.<minor>) を表示して終了\n");
    printf("  -h, --help                 このヘルプを表示して終了\n\n");
    printf("標準出力には応答本文のみ出力されます。エラーおよびthinking内容は標準エラー出力に出ます。\n");
}

// argv[i] がオプション名 name (または省略形 short_name) に一致するか
static int is_opt(const char* arg, const char* short_name, const char* long_name) {
    if (short_name && !strcmp(arg, short_name)) return 1;
    if (long_name  && !strcmp(arg, long_name))  return 1;
    return 0;
}

// --strict モードで「値がオプション名そのものに見える」場合の検出に使う既知フラグ一覧。
// 値が抜けて次のオプション名をそのまま食べてしまったケースを拾うためのもの。
// 新しいオプションを追加したら、ここにも追記すること。
static const char* KNOWN_FLAGS[] = {
    "-h", "--help", "-v", "--version", "--api-level",
    "-l", "--load-only", "-i", "--info",
    "-m", "--model", "-p", "--prompt", "-s", "--system",
    "--temp", "--max-tokens", "--n-ctx",
    "--top-p", "--repeat-penalty", "--penalty-last-n",
    "--stream", "--encoding",
    "--think-cut", "--think-start", "--think-end", "--show-thinking",
    "--verbose", "--strict", "--stop-key",
    NULL
};

static int looks_like_known_flag(const char* s) {
    for (int i = 0; KNOWN_FLAGS[i]; i++) {
        if (!strcmp(s, KNOWN_FLAGS[i])) return 1;
    }
    return 0;
}

// argv全体に指定のフラグ(値を取らない単純フラグ)が含まれるかどうかだけを調べる。
// --strict はどの位置に書かれても効くようにするため、本パースの前に先読みするのに使う。
static int argv_has_flag(int argc, char** argv, const char* flag) {
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], flag)) return 1;
    }
    return 0;
}

// 値を取る形式のオプション。値が無ければエラーメッセージを出して0を返す。
// strict時は、取得した値がそのまま既知のオプション名に一致する場合もエラーにする
// (スペースの入れ忘れで前のオプションの値と次のオプション名が連結してしまい、
//  結果として値が丸ごと欠落しているケースを拾うため)。
static int take_value(int argc, char** argv, int* i, const char** out_value, const char* opt_label, int strict) {
    if (*i + 1 >= argc) {
        fprintf(stderr, "エラー: %s には値の指定が必要です\n", opt_label);
        return 0;
    }
    const char* v = argv[*i + 1];
    if (strict && looks_like_known_flag(v)) {
        fprintf(stderr,
            "エラー(--strict): %s の値が別のオプション名 \"%s\" に見えます。"
            "値が指定されていない、またはスペースの入れ忘れによる意図しない連結の可能性があります\n",
            opt_label, v);
        return 0;
    }
    *out_value = argv[++(*i)];
    return 1;
}

// --strict時、数値オプションの値が末尾まで正しく数値として消費されたかを検証する。
// atoi/atoiは末尾にゴミがあっても黙って途中まで読んで成功したことにしてしまうため、
// (例: "64--stream" は atoi で 64 になってしまう) strict時はこれを検出してエラーにする。
static int parse_int_opt(const char* val, int strict, const char* opt_label, int* out) {
    if (!strict) { *out = atoi(val); return 1; }
    char* endptr = NULL;
    long v = strtol(val, &endptr, 10);
    if (endptr == val || *endptr != '\0') {
        fprintf(stderr,
            "エラー(--strict): %s の値 \"%s\" は不正な整数です"
            " (末尾に余分な文字がある、意図しない連結の可能性があります)\n", opt_label, val);
        return 0;
    }
    *out = (int)v;
    return 1;
}
static int parse_double_opt(const char* val, int strict, const char* opt_label, double* out) {
    if (!strict) { *out = atof(val); return 1; }
    char* endptr = NULL;
    double v = strtod(val, &endptr);
    if (endptr == val || *endptr != '\0') {
        fprintf(stderr,
            "エラー(--strict): %s の値 \"%s\" は不正な数値です"
            " (末尾に余分な文字がある、意図しない連結の可能性があります)\n", opt_label, val);
        return 0;
    }
    *out = v;
    return 1;
}

// argv を解析して Options に反映する。
// 戻り値: 0=成功 / 1=即終了してよい(-h/-v等を処理済み) / -1=エラー
static int parse_args(int argc, char** argv, Options* o) {
    // --strict はargv中のどこに書かれていても効くようにしたいので、
    // 本パースの前に先読みしておく (後段のtake_value/数値パースがこれを参照する)。
    o->strict = argv_has_flag(argc, argv, "--strict");

    for (int i = 1; i < argc; i++) {
        const char* a = argv[i];
        const char* val = NULL;

        if (is_opt(a, "-h", "--help"))    { print_usage(argv[0]); return 1; }
        else if (is_opt(a, "-v", "--version")) { printf("%s\n", llmtc_version_info()); return 1; }
        else if (is_opt(a, NULL, "--api-level")) {
            printf("%d.%d\n", llmtc_get_api_level_major(), llmtc_get_api_level_minor());
            return 1;
        }
        else if (is_opt(a, "-l", "--load-only")) { o->mode = MODE_LOAD_ONLY; }
        else if (is_opt(a, "-i", "--info"))       { o->mode = MODE_INFO; }

        else if (is_opt(a, "-m", "--model")) {
            if (!take_value(argc, argv, &i, &val, "--model", o->strict)) return -1;
            o->model_path = val;
        }
        else if (is_opt(a, "-p", "--prompt")) {
            if (!take_value(argc, argv, &i, &val, "--prompt", o->strict)) return -1;
            o->prompt = val;
        }
        else if (is_opt(a, "-s", "--system")) {
            if (!take_value(argc, argv, &i, &val, "--system", o->strict)) return -1;
            o->system_prompt = val;
        }
        else if (is_opt(a, NULL, "--temp")) {
            if (!take_value(argc, argv, &i, &val, "--temp", o->strict)) return -1;
            if (!parse_double_opt(val, o->strict, "--temp", &o->temperature)) return -1;
        }
        else if (is_opt(a, NULL, "--max-tokens")) {
            if (!take_value(argc, argv, &i, &val, "--max-tokens", o->strict)) return -1;
            if (!parse_int_opt(val, o->strict, "--max-tokens", &o->max_tokens)) return -1;
        }
        else if (is_opt(a, NULL, "--n-ctx")) {
            if (!take_value(argc, argv, &i, &val, "--n-ctx", o->strict)) return -1;
            if (!parse_int_opt(val, o->strict, "--n-ctx", &o->n_ctx)) return -1;
        }
        else if (is_opt(a, NULL, "--top-p")) {
            if (!take_value(argc, argv, &i, &val, "--top-p", o->strict)) return -1;
            if (!parse_double_opt(val, o->strict, "--top-p", &o->top_p)) return -1;
        }
        else if (is_opt(a, NULL, "--repeat-penalty")) {
            if (!take_value(argc, argv, &i, &val, "--repeat-penalty", o->strict)) return -1;
            if (!parse_double_opt(val, o->strict, "--repeat-penalty", &o->repeat_penalty)) return -1;
        }
        else if (is_opt(a, NULL, "--penalty-last-n")) {
            if (!take_value(argc, argv, &i, &val, "--penalty-last-n", o->strict)) return -1;
            if (!parse_int_opt(val, o->strict, "--penalty-last-n", &o->penalty_last_n)) return -1;
        }
        else if (is_opt(a, NULL, "--stream")) { o->stream = 1; }
        else if (is_opt(a, NULL, "--encoding")) {
            if (!take_value(argc, argv, &i, &val, "--encoding", o->strict)) return -1;
            if (!strcmp(val, "sjis")) o->encoding = LLMTC_ENCODING_SJIS;
            else if (!strcmp(val, "utf8")) o->encoding = LLMTC_ENCODING_UTF8;
            else { fprintf(stderr, "エラー: --encoding は utf8 か sjis を指定してください\n"); return -1; }
        }
        else if (is_opt(a, NULL, "--verbose")) { o->verbose = 1; }

        else if (is_opt(a, NULL, "--think-cut")) { o->think_cut = 1; }
        else if (is_opt(a, NULL, "--think-start")) {
            if (!take_value(argc, argv, &i, &val, "--think-start", o->strict)) return -1;
            o->think_start = val;
        }
        else if (is_opt(a, NULL, "--think-end")) {
            if (!take_value(argc, argv, &i, &val, "--think-end", o->strict)) return -1;
            o->think_end  = val;
        }
        else if (is_opt(a, NULL, "--show-thinking")) { o->show_thinking = 1; }

        else if (is_opt(a, NULL, "--strict")) { /* 冒頭で先読み済みなのでここでは何もしない */ }
        else if (is_opt(a, NULL, "--stop-key")) {
            if (!take_value(argc, argv, &i, &val, "--stop-key", o->strict)) return -1;
            o->stop_char = stop_key_name_to_char(val);
            if (o->stop_char == 0) {
                fprintf(stderr,
                    "エラー: --stop-key の値 \"%s\" を認識できません"
                    " (英数字1文字、またはESC/SPACE/TAB/ENTERを指定してください)\n", val);
                return -1;
            }
        }

        else {
            fprintf(stderr, "エラー: 不明なオプションです: %s\n", a);
            fprintf(stderr, "(%s --help でオプション一覧を確認できます)\n", argv[0]);
            return -1;
        }
    }
    return 0;
}

static int run_load_check(const Options* o) {
    llmtc_set_verbose(o->verbose);
    int rc = llmtc_init(o->model_path, o->n_ctx, 0); // 常にCPU推論 (n_gpu_layers=0固定)
    if (rc == 0) { printf("OK\n"); llmtc_free(); return 0; }
    char errbuf[1024];
    llmtc_get_last_error_to_buffer(errbuf, sizeof(errbuf));
    printf("NG: rc=%d: %s\n", rc, errbuf);
    return 1;
}

static int run_model_info(const Options* o) {
    llmtc_set_verbose(o->verbose);
    int rc = llmtc_init(o->model_path, o->n_ctx, 0); // 常にCPU推論 (n_gpu_layers=0固定)
    if (rc != 0) {
        char errbuf[1024];
        llmtc_get_last_error_to_buffer(errbuf, sizeof(errbuf));
        fprintf(stderr, "llmtc_init 失敗 (rc=%d): %s\n", rc, errbuf);
        return 1;
    }
    char info[512];
    llmtc_get_model_info_to_buffer(info, sizeof(info));
    printf("%s\n", info);
    llmtc_free();
    return 0;
}

// 思考内容の差分を stderr に出す。ヘッダー("---- thinking ----")は初回だけ出す。
static void flush_thinking_diff(int encoding, char* buf, int buf_size, int* prev_len, int* header_shown) {
    llmtc_get_thinking_to_buffer(buf, buf_size);
    int cur_len = (int)strlen(buf);
    if (cur_len > *prev_len) {
        if (!*header_shown) {
            fprintf(stderr, "---- thinking ----\n");
            *header_shown = 1;
        }
        print_encoded(buf + *prev_len, encoding, stderr);
        *prev_len = cur_len;
    }
}

static int run_generate(const Options* o) {
    llmtc_set_verbose(o->verbose);
    llmtc_set_generation_params(o->top_p, o->repeat_penalty, o->penalty_last_n);
    llmtc_set_think_mode(o->think_cut ? LLMTC_THINK_MODE_TAG : LLMTC_THINK_MODE_NONE, o->think_start, o->think_end);

    int rc = llmtc_init(o->model_path, o->n_ctx, 0); // 常にCPU推論 (n_gpu_layers=0固定)
    if (rc != 0) {
        char errbuf[1024];
        llmtc_get_last_error_to_buffer(errbuf, sizeof(errbuf));
        fprintf(stderr, "llmtc_init 失敗 (rc=%d): %s\n", rc, errbuf);
        return 1;
    }

    int exit_code = 0;

    if (!o->stream) {
        const char* reply = llmtc_generate(o->system_prompt, o->prompt, o->temperature, o->max_tokens);
        if (!reply) {
            char errbuf[1024];
            llmtc_get_last_error_to_buffer(errbuf, sizeof(errbuf));
            fprintf(stderr, "応答の取得に失敗しました: %s\n", errbuf);
            llmtc_free();
            return 1;
        }
        if (o->show_thinking) {
            char think_buf[65536];
            llmtc_get_thinking_to_buffer(think_buf, sizeof(think_buf));
            if (think_buf[0] != '\0') {
                fprintf(stderr, "---- thinking ----\n");
                print_encoded(think_buf, o->encoding, stderr);
                fprintf(stderr, "\n-------------------\n");
            }
        }
        print_encoded(reply, o->encoding, stdout);
        printf("\n");
    } else {
        rc = llmtc_generate_async(o->system_prompt, o->prompt, o->temperature, o->max_tokens);
        if (rc != 0) {
            fprintf(stderr, "generate_async 失敗 (rc=%d)\n", rc);
            llmtc_free();
            return 1;
        }

        // 非同期生成の開始後に標準入力の監視を始める (do_generate冒頭のフラグリセットと競合しないため)
        if (o->stop_char != 0) start_stdin_watch(o->stop_char);

        char partial[65536]       = "";
        char partial_think[65536] = "";
        int  prev_len       = 0;
        int  prev_think_len = 0;
        int  think_header_shown = 0;
        int  think_footer_shown = 0;

        while (llmtc_get_status() == LLMTC_STATUS_BUSY) {
            if (o->show_thinking) {
                flush_thinking_diff(o->encoding, partial_think, sizeof(partial_think), &prev_think_len, &think_header_shown);
            }
            llmtc_get_partial_result_to_buffer(partial, sizeof(partial));
            int cur_len = (int)strlen(partial);
            if (cur_len > prev_len) {
                if (o->show_thinking && think_header_shown && !think_footer_shown) {
                    fprintf(stderr, "\n-------------------\n");
                    think_footer_shown = 1;
                }
                print_encoded(partial + prev_len, o->encoding, stdout);
                prev_len = cur_len;
            }
            Sleep(10);
        }

        // ループを抜けた直後(最後のポーリングとBUSY→DONE遷移の間)に生成された
        // 取りこぼし分を確実に反映させるため、もう一度だけ差分を取る。
        int final_status = llmtc_get_status();
        if (o->show_thinking) {
            flush_thinking_diff(o->encoding, partial_think, sizeof(partial_think), &prev_think_len, &think_header_shown);
        }
        llmtc_get_partial_result_to_buffer(partial, sizeof(partial));
        int cur_len = (int)strlen(partial);
        if (cur_len > prev_len) {
            if (o->show_thinking && think_header_shown && !think_footer_shown) {
                fprintf(stderr, "\n-------------------\n");
            }
            print_encoded(partial + prev_len, o->encoding, stdout);
        }
        printf("\n");

        char final_buf[65536];
        llmtc_get_result_to_buffer(final_buf, sizeof(final_buf)); // 内容は使わない。状態をIDLEに戻すためだけに呼ぶ

        if (final_status == LLMTC_STATUS_ERROR) {
            char errbuf[1024];
            llmtc_get_last_error_to_buffer(errbuf, sizeof(errbuf));
            fprintf(stderr, "生成中にエラーが発生しました: %s\n", errbuf);
            exit_code = 1;
        } else if (final_status == LLMTC_STATUS_CANCELLED) {
            fprintf(stderr, "[Canceled]\n");
        }
    }

    llmtc_free();
    return exit_code;
}

int main(int argc, char** argv) {
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);

    int wargc = 0;
    LPWSTR* wargv = CommandLineToArgvW(GetCommandLineW(), &wargc);
    if (!wargv) { return 1; } // 失敗時はwargcの値も未規定なので、ここで即座に打ち切る
    char** uargv = (char**)malloc(sizeof(char*) * (size_t)wargc);
    if (!uargv) { LocalFree(wargv); return 1; }
    for (int i = 0; i < wargc; i++) {
        uargv[i] = wide_to_utf8(wargv[i]);
        if (!uargv[i]) {
            for (int j = 0; j < i; j++) free(uargv[j]);
            free(uargv);
            LocalFree(wargv);
            return 1;
        }
    }
    LocalFree(wargv);
    argc = wargc;
    argv = uargv;

    int exit_code = 0;

    if (argc < 2) {
        print_usage(argv[0]);
        exit_code = 1;
        goto cleanup;
    }

    {
        Options o;
        options_init_defaults(&o);

        int prc = parse_args(argc, argv, &o);
        if (prc == 1) { goto cleanup; }       // -h / -v で即終了
        if (prc < 0)  { exit_code = 1; goto cleanup; } // パースエラー

        // すべてのモードで --model が必須
        if (!o.model_path) {
            fprintf(stderr, "エラー: --model <path> の指定が必要です\n");
            fprintf(stderr, "(%s --help でオプション一覧を確認できます)\n", argv[0]);
            exit_code = 1;
            goto cleanup;
        }

        switch (o.mode) {
            case MODE_LOAD_ONLY: exit_code = run_load_check(&o); break;
            case MODE_INFO:      exit_code = run_model_info(&o); break;
            default:              exit_code = run_generate(&o);   break;
        }
    }

cleanup:
    for (int i = 0; i < argc; i++) free(uargv[i]);
    free(uargv);
    return exit_code;
}
