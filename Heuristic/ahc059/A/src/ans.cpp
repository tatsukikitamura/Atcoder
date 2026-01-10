#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <queue>
#include <cmath>
#include <chrono>
#include <random>
#include <set>
using namespace std;

// XorShift乱数
struct XRand {
    uint32_t x, y, z, w;
    XRand() : x(314159265), y(358979323), z(846264338), w(327950288) {}
    void setSeed(uint32_t s) { x ^= s; for(int i=0;i<100;i++) next(); }
    uint32_t next() {
        uint32_t t = x ^ (x << 11);
        x = y; y = z; z = w;
        return w = w ^ (w >> 19) ^ t ^ (t >> 8);
    }
    int nextInt(int m) { return next() % m; }
    double nextDouble() { return (double)next() / UINT32_MAX; }
};

int N, M;
int A[20][20];

struct Pair {
    int id, r1, c1, r2, c2;
    Pair() : id(-1), r1(-1), c1(-1), r2(-1), c2(-1) {}
    pair<int,int> first(int flag) const { return flag == 0 ? make_pair(r1, c1) : make_pair(r2, c2); }
    pair<int,int> second(int flag) const { return flag == 0 ? make_pair(r2, c2) : make_pair(r1, c1); }
};
Pair Pairs[200];

struct Node {
    Node* parent;
    vector<Node*> childs;
    int id, flag;

    Node(Node* p, int i, int f) : parent(p), id(i), flag(f) {}

    Node* insert(int newId, int newFlag, int start, int end) {
        Node* newNode = new Node(this, newId, newFlag);
        for (int i = start; i < end; i++) {
            childs[i]->parent = newNode;
            newNode->childs.push_back(childs[i]);
        }
        childs.erase(childs.begin() + start, childs.begin() + end);
        childs.insert(childs.begin() + start, newNode);
        return newNode;
    }
};

chrono::high_resolution_clock::time_point startTime;
int LIMIT = 1900;

inline long long getElapsed() {
    return chrono::duration_cast<chrono::milliseconds>(
        chrono::high_resolution_clock::now() - startTime).count();
}

inline int dist(int r1, int c1, int r2, int c2) {
    return abs(r1 - r2) + abs(c1 - c2);
}

Node* Top;
Node* Nodes[200];
vector<int> NearPairs[200];
vector<pair<int,int>> SwapCandidates; // swap候補ペア
XRand rnd;
int BestMoves;
string BestOutput;

// ペア間距離の事前計算
int PairDist[200][2][200][2]; // PairDist[i][ei][j][ej] = Pairs[i]の端点ei と Pairs[j]の端点ej の距離
int TopDist[200][2]; // TopDist[i][ei] = (0,0) と Pairs[i]の端点ei の距離

void initPairDist() {
    for (int i = 0; i < 200; i++) {
        int r1i = Pairs[i].r1, c1i = Pairs[i].c1;
        int r2i = Pairs[i].r2, c2i = Pairs[i].c2;
        TopDist[i][0] = abs(r1i) + abs(c1i);
        TopDist[i][1] = abs(r2i) + abs(c2i);
        for (int j = 0; j < 200; j++) {
            int r1j = Pairs[j].r1, c1j = Pairs[j].c1;
            int r2j = Pairs[j].r2, c2j = Pairs[j].c2;
            PairDist[i][0][j][0] = abs(r1i - r1j) + abs(c1i - c1j);
            PairDist[i][0][j][1] = abs(r1i - r2j) + abs(c1i - c2j);
            PairDist[i][1][j][0] = abs(r2i - r1j) + abs(c2i - c1j);
            PairDist[i][1][j][1] = abs(r2i - r2j) + abs(c2i - c2j);
        }
    }
}

// ペア間距離を取得（id==-1はTop=(0,0)）
// ep: 0=flag側の端点(first), 1=1-flag側の端点(second)
inline int distPair(int idA, int epA, int idB, int epB) {
    if (idA == -1 && idB == -1) return 0;
    if (idA == -1) return TopDist[idB][epB];
    if (idB == -1) return TopDist[idA][epA];
    return PairDist[idA][epA][idB][epB];
}

// nodeのfirst端点のインデックス (0 or 1)
inline int firstEp(Node* node) { return node->flag; }
// nodeのsecond端点のインデックス (0 or 1)
inline int secondEp(Node* node) { return 1 - node->flag; }

// exp(-x)のテーブル (x = 0 ~ 10 を 1000分割)
// idx = (int)(delta / temp * 1000), delta > 0
// idx >= 10000 なら受理確率ほぼ0
constexpr int EXP_TABLE_SIZE = 10000;
constexpr double EXP_TABLE_SCALE = 1000.0;
double expTable[EXP_TABLE_SIZE];

void initExpTable() {
    for (int i = 0; i < EXP_TABLE_SIZE; i++) {
        expTable[i] = exp(-(double)i / EXP_TABLE_SCALE);
    }
}

pair<int,int> getFirst(Node* node) {
    if (node->id == -1) return {0, 0};
    return Pairs[node->id].first(node->flag);
}

pair<int,int> getSecond(Node* node) {
    if (node->id == -1) return {0, 0};
    return Pairs[node->id].second(node->flag);
}

int pairDist(int i, int j) {
    auto& p1 = Pairs[i];
    auto& p2 = Pairs[j];
    int d = INT_MAX;
    d = min(d, dist(p1.r1, p1.c1, p2.r1, p2.c1));
    d = min(d, dist(p1.r1, p1.c1, p2.r2, p2.c2));
    d = min(d, dist(p1.r2, p1.c2, p2.r1, p2.c1));
    d = min(d, dist(p1.r2, p1.c2, p2.r2, p2.c2));
    return d;
}

tuple<int,int,int> findBestInsertForNode(Node* parent, int pairId, int flag) {
    int fEp = flag, sEp = 1 - flag;
    int pId = parent->id, pFEp = firstEp(parent), pSEp = secondEp(parent);

    int childCount = parent->childs.size();
    int bestCost = INT_MAX, bestStart = 0, bestEnd = 0;

    int distFirstSecond = distPair(pairId, fEp, pairId, sEp);
    for (int pos = 0; pos <= childCount; pos++) {
        int prevId, prevEp, nextId, nextEp;
        if (pos == 0) { prevId = pId; prevEp = pFEp; }
        else { auto* c = parent->childs[pos-1]; prevId = c->id; prevEp = secondEp(c); }
        if (pos == childCount) { nextId = pId; nextEp = pSEp; }
        else { auto* c = parent->childs[pos]; nextId = c->id; nextEp = firstEp(c); }

        int cost = distPair(prevId, prevEp, pairId, fEp) + distFirstSecond
                 + distPair(pairId, sEp, nextId, nextEp) - distPair(prevId, prevEp, nextId, nextEp);
        if (cost < bestCost) {
            bestCost = cost; bestStart = pos; bestEnd = pos;
        }
    }

    if (childCount >= 1) {
        int minStartCost = INT_MAX, minStartIdx = -1;
        for (int e = 1; e <= childCount; e++) {
            int s = e - 1;
            int prevId, prevEp;
            if (s == 0) { prevId = pId; prevEp = pFEp; }
            else { auto* c = parent->childs[s-1]; prevId = c->id; prevEp = secondEp(c); }
            auto* childS = parent->childs[s];
            int startCost = distPair(prevId, prevEp, pairId, fEp)
                          + distPair(pairId, fEp, childS->id, firstEp(childS))
                          - distPair(prevId, prevEp, childS->id, firstEp(childS));
            if (startCost < minStartCost) { minStartCost = startCost; minStartIdx = s; }

            int nextId, nextEp;
            if (e == childCount) { nextId = pId; nextEp = pSEp; }
            else { auto* c = parent->childs[e]; nextId = c->id; nextEp = firstEp(c); }
            auto* childE = parent->childs[e-1];
            int endCost = distPair(childE->id, secondEp(childE), pairId, sEp)
                        + distPair(pairId, sEp, nextId, nextEp)
                        - distPair(childE->id, secondEp(childE), nextId, nextEp);

            int totalCost = minStartCost + endCost;
            if (totalCost < bestCost) {
                bestCost = totalCost; bestStart = minStartIdx; bestEnd = e;
            }
        }
    }
    return {bestCost, bestStart, bestEnd};
}

void findBestInsert(Node* node, int pairId, int& bestCost, Node*& bestParent, int& bestFlag, int& bestStart, int& bestEnd) {
    for (int flag = 0; flag <= 1; flag++) {
        auto [cost, start, end] = findBestInsertForNode(node, pairId, flag);
        if (cost < bestCost) {
            bestCost = cost; bestParent = node; bestFlag = flag; bestStart = start; bestEnd = end;
        }
    }
    for (auto* child : node->childs) {
        findBestInsert(child, pairId, bestCost, bestParent, bestFlag, bestStart, bestEnd);
    }
}

int calcRemoveCost(Node* node) {
    if (!node->parent || node->id == -1) return INT_MIN;
    auto* parent = node->parent;
    int idx = find(parent->childs.begin(), parent->childs.end(), node) - parent->childs.begin();

    int nId = node->id, nFEp = firstEp(node), nSEp = secondEp(node);
    int prevId, prevEp, nextId, nextEp;
    if (idx == 0) { prevId = parent->id; prevEp = firstEp(parent); }
    else { auto* c = parent->childs[idx-1]; prevId = c->id; prevEp = secondEp(c); }
    if (idx == (int)parent->childs.size() - 1) { nextId = parent->id; nextEp = secondEp(parent); }
    else { auto* c = parent->childs[idx+1]; nextId = c->id; nextEp = firstEp(c); }

    int childCount = node->childs.size();
    if (childCount == 0) {
        int currentCost = distPair(prevId, prevEp, nId, nFEp)
                        + distPair(nId, nFEp, nId, nSEp)
                        + distPair(nId, nSEp, nextId, nextEp);
        int afterCost = distPair(prevId, prevEp, nextId, nextEp);
        return currentCost - afterCost;
    } else {
        auto* cFirst = node->childs[0];
        auto* cLast = node->childs[childCount-1];
        int currentCost = distPair(prevId, prevEp, nId, nFEp)
                        + distPair(nId, nFEp, cFirst->id, firstEp(cFirst))
                        + distPair(cLast->id, secondEp(cLast), nId, nSEp)
                        + distPair(nId, nSEp, nextId, nextEp);
        int afterCost = distPair(prevId, prevEp, cFirst->id, firstEp(cFirst))
                      + distPair(cLast->id, secondEp(cLast), nextId, nextEp);
        return currentCost - afterCost;
    }
}

void removeNode(Node* node) {
    if (!node->parent || node->id == -1) return;
    auto* parent = node->parent;
    int idx = find(parent->childs.begin(), parent->childs.end(), node) - parent->childs.begin();
    parent->childs.erase(parent->childs.begin() + idx);
    for (int i = 0; i < (int)node->childs.size(); i++) {
        node->childs[i]->parent = parent;
        parent->childs.insert(parent->childs.begin() + idx + i, node->childs[i]);
    }
    Nodes[node->id] = nullptr;
}

int calcTotalMoves() {
    int moves = 0, curR = 0, curC = 0;
    function<void(Node*)> visit = [&](Node* node) {
        if (node->id == -1) {
            for (auto* child : node->childs) visit(child);
        } else {
            auto [fr, fc] = getFirst(node);
            auto [sr, sc] = getSecond(node);
            moves += dist(curR, curC, fr, fc);
            curR = fr; curC = fc;
            for (auto* child : node->childs) visit(child);
            moves += dist(curR, curC, sr, sc);
            curR = sr; curC = sc;
        }
    };
    visit(Top);
    return moves;
}

string makeOutput() {
    string result;
    int curR = 0, curC = 0;
    auto moveTo = [&](int destR, int destC) {
        while (curR < destR) { result += "D\n"; curR++; }
        while (curR > destR) { result += "U\n"; curR--; }
        while (curC < destC) { result += "R\n"; curC++; }
        while (curC > destC) { result += "L\n"; curC--; }
    };
    function<void(Node*)> visit = [&](Node* node) {
        if (node->id == -1) {
            for (auto* child : node->childs) visit(child);
        } else {
            auto [fr, fc] = getFirst(node);
            auto [sr, sc] = getSecond(node);
            moveTo(fr, fc); result += "Z\n";
            for (auto* child : node->childs) visit(child);
            moveTo(sr, sc); result += "Z\n";
        }
    };
    visit(Top);
    if (!result.empty() && result.back() == '\n') result.pop_back();
    return result;
}

void resetState() {
    delete Top;
    Top = new Node(nullptr, -1, 0);
    for (int i = 0; i < M; i++) Nodes[i] = nullptr;
}

void greedy(int startR, int startC) {
    for (int t = 0; t < M; t++) {
        int bestPairId = -1, bestCost = INT_MAX, bestFlag = 0, bestStart = 0, bestEnd = 0;
        Node* bestParent = nullptr;

        if (t == 0) {
            bestPairId = A[startR][startC];
            findBestInsert(Top, bestPairId, bestCost, bestParent, bestFlag, bestStart, bestEnd);
        } else {
            for (int pairId = 0; pairId < M; pairId++) {
                if (Nodes[pairId]) continue;
                int cost = INT_MAX, flag = 0, start = 0, end = 0;
                Node* parent = nullptr;
                findBestInsert(Top, pairId, cost, parent, flag, start, end);
                if (cost < bestCost) {
                    bestCost = cost; bestPairId = pairId; bestParent = parent;
                    bestFlag = flag; bestStart = start; bestEnd = end;
                }
            }
        }
        if (bestPairId != -1) {
            Nodes[bestPairId] = bestParent->insert(bestPairId, bestFlag, bestStart, bestEnd);
        }
    }
    BestMoves = calcTotalMoves();
    BestOutput = makeOutput();
    cerr << "Greedy(" << startR << "," << startC << "): " << BestMoves << " moves, " << getElapsed() << "ms" << endl;
}

// ノードの現在のコスト（前後との接続コスト）
int calcNodeCost(Node* node) {
    if (!node->parent || node->id == -1) return 0;
    auto* parent = node->parent;
    int idx = find(parent->childs.begin(), parent->childs.end(), node) - parent->childs.begin();

    int nId = node->id, nFEp = firstEp(node), nSEp = secondEp(node);
    int prevId, prevEp, nextId, nextEp;
    if (idx == 0) { prevId = parent->id; prevEp = firstEp(parent); }
    else { auto* c = parent->childs[idx-1]; prevId = c->id; prevEp = secondEp(c); }
    if (idx == (int)parent->childs.size() - 1) { nextId = parent->id; nextEp = secondEp(parent); }
    else { auto* c = parent->childs[idx+1]; nextId = c->id; nextEp = firstEp(c); }

    int cost = distPair(prevId, prevEp, nId, nFEp);
    if (node->childs.empty()) {
        cost += distPair(nId, nFEp, nId, nSEp);
    } else {
        auto* cFirst = node->childs[0];
        auto* cLast = node->childs.back();
        cost += distPair(nId, nFEp, cFirst->id, firstEp(cFirst));
        cost += distPair(cLast->id, secondEp(cLast), nId, nSEp);
    }
    cost += distPair(nId, nSEp, nextId, nextEp);
    return cost;
}

// ノードのid/flagを変えた場合のコスト
int calcNodeCostWith(Node* node, int newId, int newFlag) {
    if (!node->parent) return 0;
    auto* parent = node->parent;
    int idx = find(parent->childs.begin(), parent->childs.end(), node) - parent->childs.begin();

    int nFEp = newFlag, nSEp = 1 - newFlag;
    int prevId, prevEp, nextId, nextEp;
    if (idx == 0) { prevId = parent->id; prevEp = firstEp(parent); }
    else { auto* c = parent->childs[idx-1]; prevId = c->id; prevEp = secondEp(c); }
    if (idx == (int)parent->childs.size() - 1) { nextId = parent->id; nextEp = secondEp(parent); }
    else { auto* c = parent->childs[idx+1]; nextId = c->id; nextEp = firstEp(c); }

    int cost = distPair(prevId, prevEp, newId, nFEp);
    if (node->childs.empty()) {
        cost += distPair(newId, nFEp, newId, nSEp);
    } else {
        auto* cFirst = node->childs[0];
        auto* cLast = node->childs.back();
        cost += distPair(newId, nFEp, cFirst->id, firstEp(cFirst));
        cost += distPair(cLast->id, secondEp(cLast), newId, nSEp);
    }
    cost += distPair(newId, nSEp, nextId, nextEp);
    return cost;
}

// nodeAがnodeBの祖先か判定
bool isAncestor(Node* ancestor, Node* descendant) {
    Node* cur = descendant;
    while (cur) {
        if (cur == ancestor) return true;
        cur = cur->parent;
    }
    return false;
}

// 部分木を取り除いたときのコスト削減量（子は部分木と一緒に移動）
int calcRemoveSubtreeCost(Node* node) {
    if (!node->parent || node->id == -1) return INT_MIN;
    auto* parent = node->parent;
    int idx = find(parent->childs.begin(), parent->childs.end(), node) - parent->childs.begin();

    int nId = node->id, nFEp = firstEp(node), nSEp = secondEp(node);
    int prevId, prevEp, nextId, nextEp;
    if (idx == 0) { prevId = parent->id; prevEp = firstEp(parent); }
    else { auto* c = parent->childs[idx-1]; prevId = c->id; prevEp = secondEp(c); }
    if (idx == (int)parent->childs.size() - 1) { nextId = parent->id; nextEp = secondEp(parent); }
    else { auto* c = parent->childs[idx+1]; nextId = c->id; nextEp = firstEp(c); }

    // 現在: prev -> entry, exit -> next
    int currentCost = distPair(prevId, prevEp, nId, nFEp)
                    + distPair(nId, nSEp, nextId, nextEp);
    // 削除後: prev -> next
    int afterCost = distPair(prevId, prevEp, nextId, nextEp);

    return currentCost - afterCost;
}

// 部分木をparentから取り除く（子は維持）
void removeSubtree(Node* node) {
    if (!node->parent) return;
    auto* parent = node->parent;
    int idx = find(parent->childs.begin(), parent->childs.end(), node) - parent->childs.begin();
    parent->childs.erase(parent->childs.begin() + idx);
    node->parent = nullptr;
}

// 部分木をparentのpos位置に挿入
void insertSubtree(Node* parent, Node* subtree, int pos) {
    parent->childs.insert(parent->childs.begin() + pos, subtree);
    subtree->parent = parent;
}

// 部分木のflagを再帰的に反転（子配列もreverse）
void reverseSubtreeFlag(Node* node) {
    node->flag ^= 1;
    reverse(node->childs.begin(), node->childs.end());
    for (auto* child : node->childs) {
        reverseSubtreeFlag(child);
    }
}

// 部分木を挿入したときのコスト増加量（reversed=trueならentry/exit逆）
int calcInsertSubtreeCostEx(Node* parent, Node* subtree, int pos, bool reversed) {
    int sId = subtree->id;
    int entryEp = reversed ? secondEp(subtree) : firstEp(subtree);
    int exitEp = reversed ? firstEp(subtree) : secondEp(subtree);

    int prevId, prevEp, nextId, nextEp;
    if (pos == 0) { prevId = parent->id; prevEp = firstEp(parent); }
    else { auto* c = parent->childs[pos-1]; prevId = c->id; prevEp = secondEp(c); }
    if (pos == (int)parent->childs.size()) { nextId = parent->id; nextEp = secondEp(parent); }
    else { auto* c = parent->childs[pos]; nextId = c->id; nextEp = firstEp(c); }

    int beforeCost = distPair(prevId, prevEp, nextId, nextEp);
    int afterCost = distPair(prevId, prevEp, sId, entryEp) + distPair(sId, exitEp, nextId, nextEp);

    return afterCost - beforeCost;
}

// 部分木の最適挿入位置を探す（reversed=trueならentry/exit逆）
pair<int, int> findBestInsertPositionForSubtreeEx(Node* parent, Node* subtree, bool reversed) {
    int bestCost = INT_MAX, bestPos = 0;
    int childCount = parent->childs.size();
    for (int pos = 0; pos <= childCount; pos++) {
        int cost = calcInsertSubtreeCostEx(parent, subtree, pos, reversed);
        if (cost < bestCost) {
            bestCost = cost;
            bestPos = pos;
        }
    }
    return {bestCost, bestPos};
}

void simulatedAnnealing(long long timeLimit) {
    int loopCount = 0;
    int currentMoves = BestMoves;
    double startTemp = 1.0, endTemp = 0.001;
    long long startTime_ = getElapsed();
    long long duration = timeLimit - startTime_;

    while (getElapsed() < timeLimit) {
        double progress = (double)(getElapsed() - startTime_) / duration;
        double temp = startTemp + (endTemp - startTemp) * progress;

        // 7:1:1:1 = move:subtree:swap:subtreeSwap
        int choice = rnd.nextInt(10);

        // 10%でswap
        if (!SwapCandidates.empty() && choice == 0) {
            // swap
            auto [idA, idB] = SwapCandidates[rnd.nextInt(SwapCandidates.size())];
            Node* nodeA = Nodes[idA];
            Node* nodeB = Nodes[idB];
            if (!nodeA || !nodeB) { loopCount++; continue; }

            // 親子関係 or 兄弟ならスキップ
            if (isAncestor(nodeA, nodeB) || isAncestor(nodeB, nodeA)) { loopCount++; continue; }
            if (nodeA->parent == nodeB->parent) { loopCount++; continue; }

            // 現在のコスト
            int oldCostA = calcNodeCost(nodeA);
            int oldCostB = calcNodeCost(nodeB);

            // swap後の最適なflagを探す
            int bestDelta = INT_MAX;
            int bestFlagA = 0, bestFlagB = 0;
            for (int fa = 0; fa <= 1; fa++) {
                for (int fb = 0; fb <= 1; fb++) {
                    int newCostA = calcNodeCostWith(nodeA, idB, fb);
                    int newCostB = calcNodeCostWith(nodeB, idA, fa);
                    int delta = (newCostA + newCostB) - (oldCostA + oldCostB);
                    if (delta < bestDelta) {
                        bestDelta = delta;
                        bestFlagA = fa;
                        bestFlagB = fb;
                    }
                }
            }

            int expIdx = (int)(bestDelta / temp * EXP_TABLE_SCALE);
            bool accept = (bestDelta <= 0) || (expIdx < EXP_TABLE_SIZE && rnd.nextDouble() < expTable[expIdx]);
            if (accept) {
                // swap実行
                nodeA->id = idB; nodeA->flag = bestFlagB;
                nodeB->id = idA; nodeB->flag = bestFlagA;
                Nodes[idA] = nodeB;
                Nodes[idB] = nodeA;
                currentMoves += bestDelta;
                if (currentMoves < BestMoves) {
                    BestMoves = currentMoves;
                    BestOutput = makeOutput();
                    cerr << "SA(swap): " << BestMoves << " moves, " << getElapsed() << "ms" << endl;
                }
            }
            loopCount++;
            continue;
        }

        // 10%で部分木移動
        if (choice == 1) {
            int pairId = rnd.nextInt(M);
            Node* node = Nodes[pairId];
            if (!node || !node->parent) { loopCount++; continue; }
            if (node->childs.empty()) { loopCount++; continue; } // 子がいなければ通常moveと同じ

            // 挿入先を探す
            auto& nearList = NearPairs[pairId];
            if (nearList.empty()) { loopCount++; continue; }
            int targetPairId = nearList[rnd.nextInt(nearList.size())];
            Node* targetNode = Nodes[targetPairId];
            if (!targetNode) { loopCount++; continue; }

            // targetNodeがnodeの子孫ならスキップ
            if (isAncestor(node, targetNode)) { loopCount++; continue; }

            // 親オプション
            if (rnd.nextInt(2) == 0 && targetNode->parent) {
                targetNode = targetNode->parent;
            }
            if (isAncestor(node, targetNode)) { loopCount++; continue; }

            // コスト計算
            Node* oldParent = node->parent;
            int oldIdx = find(oldParent->childs.begin(), oldParent->childs.end(), node) - oldParent->childs.begin();

            int removeCost = calcRemoveSubtreeCost(node);
            removeSubtree(node);

            // 通常パターン
            auto [insertCost, bestPos] = findBestInsertPositionForSubtreeEx(targetNode, node, false);
            bool useReversed = false;

            // flag反転パターン（topでなければ試す）
            if (targetNode->id != -1) {
                auto [insertCostRev, bestPosRev] = findBestInsertPositionForSubtreeEx(targetNode, node, true);
                if (insertCostRev < insertCost) {
                    insertCost = insertCostRev;
                    bestPos = bestPosRev;
                    useReversed = true;
                }
            }

            int delta = insertCost - removeCost;

            int expIdx = (int)(delta / temp * EXP_TABLE_SCALE);
            bool accept = (delta <= 0) || (expIdx < EXP_TABLE_SIZE && rnd.nextDouble() < expTable[expIdx]);

            if (accept) {
                if (useReversed) reverseSubtreeFlag(node);
                insertSubtree(targetNode, node, bestPos);
                currentMoves += delta;
                if (currentMoves < BestMoves) {
                    BestMoves = currentMoves;
                    BestOutput = makeOutput();
                    cerr << "SA(subtree): " << BestMoves << " moves, " << getElapsed() << "ms" << endl;
                }
            } else {
                insertSubtree(oldParent, node, oldIdx);
            }
            loopCount++;
            continue;
        }

        // 10%で部分木swap
        if (!SwapCandidates.empty() && choice == 2) {
            auto [idA, idB] = SwapCandidates[rnd.nextInt(SwapCandidates.size())];
            Node* nodeA = Nodes[idA];
            Node* nodeB = Nodes[idB];
            if (!nodeA || !nodeB) { loopCount++; continue; }
            if (!nodeA->parent || !nodeB->parent) { loopCount++; continue; }

            // 親子関係 or 兄弟ならスキップ
            if (isAncestor(nodeA, nodeB) || isAncestor(nodeB, nodeA)) { loopCount++; continue; }
            if (nodeA->parent == nodeB->parent) { loopCount++; continue; }

            // 現在の位置を保存
            Node* parentA = nodeA->parent;
            Node* parentB = nodeB->parent;
            int idxA = find(parentA->childs.begin(), parentA->childs.end(), nodeA) - parentA->childs.begin();
            int idxB = find(parentB->childs.begin(), parentB->childs.end(), nodeB) - parentB->childs.begin();

            // removeコスト計算
            int removeCostA = calcRemoveSubtreeCost(nodeA);
            int removeCostB = calcRemoveSubtreeCost(nodeB);

            // 両方外す
            removeSubtree(nodeA);
            removeSubtree(nodeB);

            // swapした位置での挿入コスト計算
            auto [insertCostA, posA] = findBestInsertPositionForSubtreeEx(parentB, nodeA, false);
            auto [insertCostB, posB] = findBestInsertPositionForSubtreeEx(parentA, nodeB, false);

            int delta = (insertCostA + insertCostB) - (removeCostA + removeCostB);

            int expIdx = (int)(delta / temp * EXP_TABLE_SCALE);
            bool accept = (delta <= 0) || (expIdx < EXP_TABLE_SIZE && rnd.nextDouble() < expTable[expIdx]);

            if (accept) {
                // swap実行: AをBの元位置に、BをAの元位置に
                insertSubtree(parentB, nodeA, posA);
                insertSubtree(parentA, nodeB, posB);
                currentMoves += delta;
                if (currentMoves < BestMoves) {
                    BestMoves = currentMoves;
                    BestOutput = makeOutput();
                    cerr << "SA(subtreeSwap): " << BestMoves << " moves, " << getElapsed() << "ms" << endl;
                }
            } else {
                // 元に戻す
                insertSubtree(parentA, nodeA, idxA);
                insertSubtree(parentB, nodeB, idxB);
            }
            loopCount++;
            continue;
        }

        // move
        int pairId = rnd.nextInt(M);
        if (!Nodes[pairId]) continue;
        Node* node = Nodes[pairId];

        auto& nearList = NearPairs[pairId];
        if (nearList.empty()) continue;

        int targetPairId = nearList[rnd.nextInt(nearList.size())];
        Node* targetNode = Nodes[targetPairId];
        if (!targetNode) continue;

        if (rnd.nextInt(2) == 0 && targetNode->parent) {
            targetNode = targetNode->parent;
        }
        if (targetNode == node) continue;

        Node* oldParent = node->parent;
        int oldIdx = find(oldParent->childs.begin(), oldParent->childs.end(), node) - oldParent->childs.begin();
        int oldFlag = node->flag;
        int oldChildCount = node->childs.size();

        int removeCost = calcRemoveCost(node);
        removeNode(node);

        int insertCost = INT_MAX, bestFlag = 0, bestStart = 0, bestEnd = 0;
        for (int flag = 0; flag <= 1; flag++) {
            auto [cost, start, end] = findBestInsertForNode(targetNode, pairId, flag);
            if (cost < insertCost) {
                insertCost = cost; bestFlag = flag; bestStart = start; bestEnd = end;
            }
        }

        int newMoves = currentMoves - removeCost + insertCost;
        int delta = newMoves - currentMoves;

        int expIdx = (int)(delta / temp * EXP_TABLE_SCALE);
        bool accept = (delta <= 0) || (expIdx < EXP_TABLE_SIZE && rnd.nextDouble() < expTable[expIdx]);
        if (accept) {
            Nodes[pairId] = targetNode->insert(pairId, bestFlag, bestStart, bestEnd);
            currentMoves = newMoves;
            if (newMoves < BestMoves) {
                BestMoves = newMoves;
                BestOutput = makeOutput();
                cerr << "SA: " << BestMoves << " moves, " << getElapsed() << "ms" << endl;
            }
        } else {
            Nodes[pairId] = oldParent->insert(pairId, oldFlag, oldIdx, oldIdx + oldChildCount);
        }
        loopCount++;
    }
    cerr << "SA loops: " << loopCount << endl;
}

int main(int argc, char* argv[]) {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    if (argc > 1) rnd.setSeed(atoi(argv[1]));

    cin >> N;
    M = N * N / 2;

    for (int i = 0; i < M; i++) Pairs[i].id = i;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            cin >> A[i][j];
            int id = A[i][j];
            if (Pairs[id].r1 == -1) { Pairs[id].r1 = i; Pairs[id].c1 = j; }
            else { Pairs[id].r2 = i; Pairs[id].c2 = j; }
        }
    }

    startTime = chrono::high_resolution_clock::now();
    initExpTable();
    initPairDist();

    Top = new Node(nullptr, -1, 0);
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < M; j++) {
            if (i != j && pairDist(i, j) <= 3) NearPairs[i].push_back(j);
        }
    }

    // swap候補を列挙（入口出口の距離和が8以下）
    for (int i = 0; i < M; i++) {
        for (int j = i + 1; j < M; j++) {
            auto& pi = Pairs[i];
            auto& pj = Pairs[j];
            int d1 = dist(pi.r1, pi.c1, pj.r1, pj.c1) + dist(pi.r2, pi.c2, pj.r2, pj.c2);
            int d2 = dist(pi.r1, pi.c1, pj.r2, pj.c2) + dist(pi.r2, pi.c2, pj.r1, pj.c1);
            if (min(d1, d2) <= 8) {
                SwapCandidates.push_back({i, j});
            }
        }
    }

    greedy(0, 0);
    simulatedAnnealing(LIMIT);

    cout << BestOutput << endl;
    return 0;
}
