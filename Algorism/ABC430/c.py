import sys
from bisect import bisect_left

sys.setrecursionlimit(2000000)

class BIT:
    def __init__(self, n):
        self.n = n
        self.data = [0] * (n + 1)

    def add(self, i, x):
        while i <= self.n:
            self.data[i] += x
            i += i & (-i)

    def sum(self, i):
        s = 0
        while i > 0:
            s += self.data[i]
            i -= i & (-i)
        return s

    def query_range(self, l, r):
        if l > r:
            return 0
        return self.sum(r) - self.sum(l - 1)

class SegTreeCompressedBITs:
    def __init__(self, N, Pa, Pb):
        self.n_a = N
        
        pb_values_raw = [[] for _ in range(4 * (self.n_a + 1))]
        for i in range(N + 1):
            pa_i = Pa[i]
            pb_i = Pb[i]
            self._collect_pb_values(1, 0, self.n_a, pa_i, pb_i, pb_values_raw)

        self.tree = [None] * (4 * (self.n_a + 1))
        self._init_bits(1, 0, self.n_a, pb_values_raw)

    def _collect_pb_values(self, node_idx, L, R, pa_i, pb_i, pb_values_raw):
        if pa_i < L or pa_i > R:
            return
            
        pb_values_raw[node_idx].append(pb_i)
        
        if L == R:
            return

        M = (L + R) // 2
        if pa_i <= M:
            self._collect_pb_values(node_idx * 2, L, M, pa_i, pb_i, pb_values_raw)
        else:
            self._collect_pb_values(node_idx * 2 + 1, M + 1, R, pa_i, pb_i, pb_values_raw)

    def _init_bits(self, node_idx, L, R, pb_values_raw):
        raw_list = pb_values_raw[node_idx]
        
        if not raw_list:
            sorted_unique_pb = []
        else:
            raw_list.sort()
            sorted_unique_pb = [raw_list[0]]
            for k in range(1, len(raw_list)):
                if raw_list[k] != raw_list[k-1]:
                    sorted_unique_pb.append(raw_list[k])

        bit_size = len(sorted_unique_pb)
        self.tree[node_idx] = (BIT(bit_size), sorted_unique_pb)

        if L == R:
            return
        
        M = (L + R) // 2
        self._init_bits(node_idx * 2, L, M, pb_values_raw)
        self._init_bits(node_idx * 2 + 1, M + 1, R, pb_values_raw)

    def _get_pb_idx(self, node_idx, y):
        bit, sorted_pb_list = self.tree[node_idx]
        idx_0based = bisect_left(sorted_pb_list, y)
        if idx_0based < len(sorted_pb_list) and sorted_pb_list[idx_0based] == y:
            return idx_0based + 1
        return -1

    def _get_pb_idx_min(self, node_idx, y_min):
        bit, sorted_pb_list = self.tree[node_idx]
        idx_0based = bisect_left(sorted_pb_list, y_min)
        return idx_0based + 1

    def add(self, x, y):
        self._add(1, 0, self.n_a, x, y)

    def _add(self, node_idx, L, R, x, y):
        if x < L or x > R:
            return
            
        bit, sorted_pb_list = self.tree[node_idx]
        if sorted_pb_list:
            idx_1based = self._get_pb_idx(node_idx, y)
            if idx_1based != -1:
                 bit.add(idx_1based, 1)
        
        if L == R:
            return
            
        M = (L + R) // 2
        if x <= M:
            self._add(node_idx * 2, L, M, x, y)
        else:
            self._add(node_idx * 2 + 1, M + 1, R, x, y)

    def query(self, qx, qy):
        y_min = qy + 1
        return self._query(1, 0, self.n_a, 0, qx, y_min)

    def _query(self, node_idx, L, R, qL, qR, y_min):
        if R < qL or L > qR:
            return 0
        
        if qL <= L and R <= qR:
            bit, sorted_pb_list = self.tree[node_idx]
            if not sorted_pb_list:
                return 0
            
            idx_min_1based = self._get_pb_idx_min(node_idx, y_min)
            return bit.query_range(idx_min_1based, bit.n)
            
        M = (L + R) // 2
        res_L = self._query(node_idx * 2, L, M, qL, qR, y_min)
        res_R = self._query(node_idx * 2 + 1, M + 1, R, qL, qR, y_min)
        return res_L + res_R

def main():
    N, A, B = map(int, sys.stdin.readline().split())
    S = sys.stdin.readline().strip()

    Pa = [0] * (N + 1)
    Pb = [0] * (N + 1)
    for i in range(N):
        if S[i] == 'a':
            Pa[i+1] = Pa[i] + 1
            Pb[i+1] = Pb[i]
        else:
            Pa[i+1] = Pa[i]
            Pb[i+1] = Pb[i] + 1

    ds = SegTreeCompressedBITs(N, Pa, Pb)
    
    total_count = 0
    
    for r in range(1, N + 1):
        ds.add(Pa[r-1], Pb[r-1])
        
        Ca = Pa[r] - A
        Cb = Pb[r] - B
        
        count_i = ds.query(Ca, Cb)
        total_count += count_i
        
    print(total_count)

if __name__ == "__main__":
    main()