#include <stdio.h>
#include <string.h>
#include <stdbool.h>

typedef struct {
    char moto[100];
    char gyaku[100];
    bool dekita;
} Kaibun;

void make_kaibun(Kaibun *ptr);

void make_kaibun(Kaibun *ptr) {
    int len = strlen(ptr->moto);
    
    for (int i = len - 3, j = 0; i >= 0; i -= 3, j += 3) {
        ptr->gyaku[j]     = ptr->moto[i];
        ptr->gyaku[j + 1] = ptr->moto[i + 1];
        ptr->gyaku[j + 2] = ptr->moto[i + 2];
    }
    ptr->gyaku[len] = '\0';
    
    ptr->dekita = (len > 0 && strcmp(ptr->moto, ptr->gyaku) == 0);
}

int main(int argc, char *argv[]) {
    Kaibun my_kaibun;
    FILE *fp;
    int i;
    
    if (argc < 2) {
        printf("コマンドライン引数に日本語の文字列か「-f ファイル名」を指定してください\n");
        return 1;
    }
    
    if (argc == 3 && strcmp(argv[1], "-f") == 0) {
        fp = fopen(argv[2], "r");
        if (fp == NULL) {
            printf("%sを開けませんでした\n", argv[2]);
            return 2;
        }
        
        while (fgets(my_kaibun.moto, 100, fp) != NULL) {
            int len = strlen(my_kaibun.moto);
            if (len > 0 && my_kaibun.moto[len - 1] == '\n') {
                my_kaibun.moto[len - 1] = '\0';
            }
            
            puts(my_kaibun.moto);
                    
            make_kaibun(&my_kaibun);
            
            puts(my_kaibun.gyaku);
            
            if (my_kaibun.dekita) {
                printf("回文です\n\n");
            } else {
                printf("回文ではありません\n\n");
            }
        }
        
        fclose(fp);
    } else {
        for (i = 1; i < argc; i++) {
            strcpy(my_kaibun.moto, argv[i]);
            puts(my_kaibun.moto);   
            make_kaibun(&my_kaibun);    
            puts(my_kaibun.gyaku);
            
            if (my_kaibun.dekita) {
                printf("回文です\n\n");
            } else {
                printf("回文ではありません\n\n");
            }
        }
    }
    
    return 0;
}
