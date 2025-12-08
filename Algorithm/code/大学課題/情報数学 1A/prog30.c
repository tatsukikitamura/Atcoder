#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <ctype.h>
#include <limits.h>
#include <float.h>
#include <stdbool.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <stdnoreturn.h>

int print_file(const char *filename) 
{
    FILE *fp;
    char buf[1000];

    fp = fopen(filename, "r");
    if (fp == NULL) {
        printf("ファイルを開けませんでした\n");
        return 1;
    }

    while (fgets(buf, sizeof(buf), fp) != NULL) {
    printf("%s", buf);
    }
    fclose(fp);
    return 0;
}



int main() {
    print_file("prog30.c");
    return 0;
}