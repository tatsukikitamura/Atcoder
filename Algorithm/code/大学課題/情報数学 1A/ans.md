情報数学 2A レポート

学籍番号:1E23M023
名前:北村健紀
応用レベルまで


<コード>

 1 #include <stdio.h>
 2 #include <string.h>
 3 #include <stdbool.h>
 4
 5 typedef struct {
 6     char moto[100];
 7     char gyaku[100];
 8     bool dekita;
 9 } Kaibun;
10
11 void make_kaibun(Kaibun *ptr);
12 
13 
14 void make_kaibun(Kaibun *ptr) {
15     int len = strlen(ptr->moto);
16     
17    for (int i = len - 3, j = 0; i >= 0; i -= 3, j += 3) {
18         ptr->gyaku[j]     = ptr->moto[i];
19         ptr->gyaku[j + 1] = ptr->moto[i + 1];
20         ptr->gyaku[j + 2] = ptr->moto[i + 2];
21    }
22    ptr->gyaku[len] = '\0';
23     ptr->dekita = (len > 0 && strcmp(ptr->moto, ptr->gyaku) == 0);
24 }
25
26 int main(int argc, char *argv[]) {
27     Kaibun my_kaibun;
28     FILE *fp;
29     int i;
30
31     if (argc < 2) {
32         printf("コマンドライン引数に日本語の文字列か「-f ファイル名」を指定してください\n");
33         return 1;
34     }
35
36     if (argc == 3 && strcmp(argv[1], "-f") == 0) {
37         fp = fopen(argv[2], "r");
38         if (fp == NULL) {
39             printf("%sを開けませんでした\n", argv[2]);
40             return 2;
41         }
42
43         while (fgets(my_kaibun.moto, 100, fp) != NULL) {
44             int len = strlen(my_kaibun.moto);
45             if (len > 0 && my_kaibun.moto[len - 1] == '\n') {
46                 my_kaibun.moto[len - 1] = '\0';
47             }
48             puts(my_kaibun.moto);                    
49             make_kaibun(&my_kaibun);            
50             puts(my_kaibun.gyaku);            
51             if (my_kaibun.dekita) {
52                 printf("回文です\n\n");
53             } else {
54                 printf("回文ではありません\n\n");
55             }
56         }
57         fclose(fp);
58     } else {
59         for (i = 1; i < argc; i++) {
60             strcpy(my_kaibun.moto, argv[i]);
61             puts(my_kaibun.moto);   
62             make_kaibun(&my_kaibun);    
63             puts(my_kaibun.gyaku);
64             if (my_kaibun.dekita) {
65                 printf("回文です\n\n");
66             } else {
67                 printf("回文ではありません\n\n");
68             }
69         }
70     }
71
72     return 0;
73 }











<実行結果>

./prog40 -f kaibun.txt                                                                                 

たけやぶやけた
たけやぶやけた
回文です

あいうえお
おえういあ
回文ではありません

しんぶんし
しんぶんし
回文です

トマト
トマト
回文です

わたしまけましたわ
わたしまけましたわ
回文です

こんにちは
はちにんこ
回文ではありません






















解説


1～3行ではライブラリの機能を使用するために、stdio.h（入出力関数であるprintf, puts, fopenなどを使用するため）、string.h（文字列操作関数であるstrlen, strcpy, strcmpを使用するため）、stdbool.h（bool型のtrue/falseを使用するため）をincludeしている

5～9行では、回文判定に必要なデータをまとめて管理するために、Kaibunという構造体を定義する処理をしている。この構造体には、元の文字列を格納するmoto配列（100バイト）、逆順にした文字列を格納するgyaku配列（100バイト）、回文かどうかを表すbool型変数dekitaの3つのメンバを持たせている

11行では、コンパイラに関数の存在を事前に知らせるために、make_kaibun関数のプロトタイプ宣言をしている

14行では、逆順処理のループ回数を決定するために、strlenで元の文字列の長さを取得する

17～21行では、日本語文字列を逆順に並べるために、3バイトずつ末尾から取り出してgyakuに格納する処理をしている。UTF-8では日本語1文字は3バイトで構成されているため、1バイトずつではなく3バイト単位で処理する必要がある。具体的には、iを文字列の末尾（len-3）から開始し、3バイトずつ前に移動しながら、jを先頭から3バイトずつ後ろに移動させてコピーしている

22行では、逆順文字列を正しく終端するために、gyakuの末尾にnull文字を追加する

23行では、回文かどうかを判定するために、strcmpでmotoとgyakuを比較し、一致していればtrue、一致していなければfalseをdekitaに格納する

27～29行では、プログラム全体で使用する変数を準備するために、構造体変数my_kaibun、ファイルポインタfp、ループ変数iを宣言する

31～34行では、プログラムの誤った使用を防ぐために、コマンドライン引数が不足している場合（argcが2未満の場合）にエラーメッセージを表示してreturn 1で終了する

36行では、ファイルから読み込むモードかどうかを判定するために、strcmpで第1引数（argv[1]）が「-f」かどうかを確認する処理をしている。この条件分岐により、応用Aの機能と応用Bの機能を両立させている。-fオプションが指定された場合はファイルモードで動作し、そうでない場合は通常モード（コマンドライン引数処理）で動作する

37～41行では、指定されたファイルを読み込むために、fopenで第2引数（argv[2]）のファイルを読み込みモード（"r"）で開き、失敗した場合（fpがNULLの場合）はエラーメッセージを表示してreturn 2で終了する
43～47行では、ファイルから文字列を取得するために、fgetsで1行ずつmy_kaibun.motoに読み込み、末尾に改行文字がある場合はnull文字に置き換えて削除する

48行では、入力内容を確認するために、putsで元の文字列を画面に表示する

49行では、回文判定を行うために、make_kaibun関数にmy_kaibunのアドレスを渡して呼び出す

50行では、逆順にした結果を確認するために、putsで逆順文字列を画面に表示する

51～55行では、判定結果をユーザーに伝えるために、dekitaの値に応じて「回文です」または「回文ではありません」と表示する

57行では、使用したファイルリソースを解放するために、fcloseでファイルを閉じる

59～69行では、コマンドライン引数として渡された文字列を処理するために、forループでi=1からargc-1まで繰り返し、各引数を順番に回文判定する

60行では、引数の文字列を構造体に格納するために、strcpyでargv[i]をmy_kaibun.motoにコピーする

61行では、入力内容を確認するために、putsで元の文字列を画面に表示する

62行では、回文判定を行うために、make_kaibun関数にmy_kaibunのアドレスを渡して呼び出す
63行では、逆順にした結果を確認するために、putsで逆順文字列を画面に表示する

64～68行では、判定結果をユーザーに伝えるために、dekitaの値に応じて「回文です」または「回文ではありません」と表示する

72行では、プログラムが正常に終了したことをOSに伝えるために、return 0で終了コード0を返す