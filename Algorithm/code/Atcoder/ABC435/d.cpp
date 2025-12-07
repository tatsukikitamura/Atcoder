#include <iostream>
#include <vector>
#include <stack>
using namespace std;
struct Node {
    int index;
    vector<int> nears;
    bool arrival;      
    Node(int idx) : index(idx), arrival(false) {}
};
vector<Node> graph;

bool check(int start_node_idx) {
    stack<int> stk;
    stk.push(start_node_idx);
    vector<bool> visited(graph.size(), false);
    visited[start_node_idx] = true;

    while (!stk.empty()) {
        int curr_idx = stk.top();
        stk.pop();

        if (graph[curr_idx].arrival) {
            return true;
        }

    
        for (int near_idx : graph[curr_idx].nears) {
            if (graph[near_idx].arrival) {
                return true;
            }
            
              if (!visited[near_idx]) {
                visited[near_idx] = true;
                stk.push(near_idx);
            }
        }
    }
    return false;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int N, M;
    if (!(cin >> N >> M)) return 0;

    graph.reserve(N);
    for (int i = 0; i < N; ++i) {
        graph.emplace_back(i);
    }

    for (int i = 0; i < M; ++i) {
        int A, B;
        cin >> A >> B;
        graph[A - 1].nears.push_back(B - 1);
    }

    int Q;
    cin >> Q;
    for (int i = 0; i < Q; ++i) {
        int type, val;
        cin >> type >> val;

        if (type == 1) {
            graph[val - 1].arrival = true;
        } else if (type == 2) {
            if (check(val - 1)) {
                cout << "Yes" << "\n";
            } else {
                cout << "No" << "\n";
            }
        }
    }

    return 0;
}