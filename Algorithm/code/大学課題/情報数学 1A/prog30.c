#include <stdio.h>
#include <string.h>

void safegets(char *head, int size, const char *prompt) {
    *head = '\0';
    printf("%s", prompt);
    if (fgets(head, size, stdin) == NULL) return;

    if (head[strlen(head) - 1] == '\n') {
        head[strlen(head) - 1] = '\0';
    }
    if (strlen(head) == size - 1) {
        printf("Input is too long\n");
    }
}


int main() {
    char moto[100], gyaku[100];
    safegets(moto, sizeof(moto),"入力内容：\n");
    for (int i = strlen(moto)-3,j = 0; i >= 0; i-=3, j +=3) {
        gyaku[j] = moto[i];
        gyaku[j+1] = moto[i+1];
        gyaku[j+2] = moto[i+2];
    }
    gyaku[strlen(moto)] = '\0';
    puts(gyaku);
    if (strcmp(moto, gyaku) == 0) {
        puts("回文です\n");
    } else {
        puts("回文ではありません\n");
    }
    return 0;
}
