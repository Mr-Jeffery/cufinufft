#!/usr/bin/env python3
"""
Convert Matrix Market COO (.mtx) to dense real row-major text format.

Output format (used by test/cufft_dense2d_test.cu):
- Header line: "N1 N2"
- Then exactly N1*N2 lines
- One real value per line, row-major order
"""

import sys


def main() -> int:
    if len(sys.argv) != 3:
        print("Usage: mtx_coo_to_dense_real.py input.mtx output_dense_real.txt", file=sys.stderr)
        return 1

    in_path = sys.argv[1]
    out_path = sys.argv[2]

    with open(in_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    idx = 0
    while idx < len(lines) and lines[idx].startswith("%"):
        idx += 1

    nrows, ncols, _ = map(int, lines[idx].split())
    idx += 1

    dense = [0.0] * (nrows * ncols)

    while idx < len(lines):
        s = lines[idx].strip()
        idx += 1
        if not s or s.startswith("%"):
            continue
        parts = s.split()
        r = int(parts[0]) - 1
        c = int(parts[1]) - 1
        dense[r * ncols + c] = 1.0

    with open(out_path, "w", encoding="utf-8") as g:
        g.write(f"{nrows} {ncols}\n")
        for v in dense:
            g.write(f"{v}\n")

    print(f"Wrote dense real matrix with header: {out_path} ({nrows}x{ncols})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
