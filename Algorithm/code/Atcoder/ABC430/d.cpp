#include <iostream>
#include <vector>
#include <set>
#include <map>
#include <algorithm>

using namespace std;

int main() {
    int N;
    cin >> N;
    vector<long long> X(N);
    for (int i = 0; i < N; ++i) {
        cin >> X[i];
    }

    set<long long> coords;
    map<long long, long long> distances;

    coords.insert(0);
    distances[0] = 0; 

    long long total_sum = 0;

    for (int r = 0; r < N; ++r) {
        long long Xr = X[r];

        auto it_R = coords.lower_bound(Xr);
        auto it_L = prev(it_R);

        long long L = *it_L;
        long long R = -1; 
        if (it_R != coords.end()) {
            R = *it_R;
        }

        long long old_d_L = distances[L];
        long long new_d_L;
        if (it_L == coords.begin()) { 
            new_d_L = Xr - L;
        } else {
            long long L_left = *prev(it_L);
            new_d_L = min(L - L_left, Xr - L);
        }
        total_sum = total_sum - old_d_L + new_d_L;
        distances[L] = new_d_L;

        if (R != -1) {
            long long old_d_R = distances[R];
            long long new_d_R;
            auto it_R_right = next(it_R);
            if (it_R_right == coords.end()) { 
                new_d_R = R - Xr;
            } else {
                long long R_right = *it_R_right;
                new_d_R = min(R - Xr, R_right - R);
            }
            total_sum = total_sum - old_d_R + new_d_R;
            distances[R] = new_d_R;
        }

        long long new_d_r;
        if (R != -1) {
            new_d_r = min(Xr - L, R - Xr);
        } else { 
            new_d_r = Xr - L;
        }
        total_sum += new_d_r;
        distances[Xr] = new_d_r;

        coords.insert(Xr);

        cout << total_sum << endl;
    }

    return 0;
}