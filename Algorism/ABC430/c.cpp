#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <iterator> 

int main() {
    int N, A, B;
    std::cin >> N >> A >> B;
    std::string S;
    std::cin >> S;

    std::vector<int> sum_a(N + 1, 0);
    std::vector<int> sum_b(N + 1, 0);
    for (int i = 0; i < N; ++i) {
        sum_a[i + 1] = sum_a[i] + (S[i] == 'a' ? 1 : 0);
        sum_b[i + 1] = sum_b[i] + (S[i] == 'b' ? 1 : 0);
    }

    long long total_pairs = 0;


    for (int l = 1; l <= N - 1; ++l) {
        
        long long target_a = (long long)A + sum_a[l - 1];
        
        
        auto it_a = std::lower_bound(
            sum_a.begin() + l + 1, 
            sum_a.end(), 
            target_a
        );

        
        if (it_a == sum_a.end()) {
            continue; 
        }

        int r_start = std::distance(sum_a.begin(), it_a);

        long long target_b = (long long)B + sum_b[l - 1];

        auto it_b = std::lower_bound(
            sum_b.begin() + l + 1, 
            sum_b.end(), 
            target_b
        );
        
        int r_end;
       
        if (it_b == sum_b.end()) {
      
            r_end = N;
        } else {
           
            int r_break = std::distance(sum_b.begin(), it_b);
          
            r_end = r_break - 1;
        }

       
        if (r_start <= r_end) {
            total_pairs += (long long)(r_end - r_start + 1);
        }
    }

    std::cout << total_pairs << std::endl;

    return 0;
}