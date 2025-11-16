#include <iostream>
#include <vector>

int main() {
    int N;
    std::cin >> N;
    std::vector<int> NLI(N);
    for (int i = 0; i < N; ++i) {
        std::cin >> NLI[i];
    }

    int count = 0;
    for (int x = 0; x < N; ++x) {
        for (int y = x + 1; y < N; ++y) {
            if ((y - x) == NLI[x] + NLI[y]) {
                count++;
            }
        }
    }

    std::cout << count << std::endl;

    return 0;
}