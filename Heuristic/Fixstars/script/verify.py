#!/usr/bin/env python3
import sys, os

def verify(tc_name, data_dir="data"):
    with open(f'{data_dir}/in-{tc_name}.txt') as f:
        tokens = f.read().split()
    with open(f'{data_dir}/out-{tc_name}.txt') as f:
        out_text = f.read()

    idx = 0
    N = int(tokens[idx]); idx += 1

    A = []
    for i in range(N):
        row = []
        for j in range(N):
            row.append(int(tokens[idx])); idx += 1
        A.append(row)

    v = []
    for i in range(N):
        row = []
        for k in range(128):
            row.append(int(tokens[idx])); idx += 1
        v.append(row)

    r = []
    for k in range(128):
        r.append(int(tokens[idx])); idx += 1

    # Parse output
    lines = out_text.strip().split('\n')
    party_size = int(lines[1].split('=')[1].strip())
    param_sum = int(lines[2].split('=')[1].strip())
    members_line = lines[4].strip()
    members = list(map(int, members_line.split()))

    print(f'=== {tc_name} === N={N}, PartySize={party_size}, T={param_sum}')
    print(f'  Members: {members}')

    # Clique check (1-indexed -> 0-indexed)
    m0 = [x - 1 for x in members]
    clique_ok = True
    for i in range(len(m0)):
        for j in range(i + 1, len(m0)):
            if A[m0[i]][m0[j]] != 1:
                print(f'  ERROR: {members[i]} and {members[j]} NOT friends')
                clique_ok = False
    print(f'  Clique: {"OK" if clique_ok else "FAIL"}')

    # BOSS check
    sums = [0] * 128
    for i in m0:
        for k in range(128):
            sums[k] += v[i][k]

    violations = sum(1 for k in range(128) if sums[k] < r[k])
    print(f'  BOSS: {"OK" if violations == 0 else f"FAIL ({violations} violations)"}')

    # T check
    total = sum(sums)
    match = total == param_sum
    print(f'  T: declared={param_sum}, computed={total} {"OK" if match else "MISMATCH"}')
    print()

for tc in ['small-1', 'small-2', 'large-1', 'small-8']:
    verify(tc)
