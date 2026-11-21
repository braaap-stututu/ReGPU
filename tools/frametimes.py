"""Frame-time statistics from a REGPU_FRAME_TIMES file ("log seconds, gap ms" per present).

    py -3 tools/frametimes.py times.csv [from_s to_s]

Prints the average fps, the frame-time percentiles (p50/p99/p99.9), the 1% and 0.1% lows (fps of the average of the
slowest 1% / 0.1% of frames, as frame-time tools report them), the worst frame, and the frames over 2x / 4x the median.
"""
import statistics
import sys


def main():
    rows = []
    with open(sys.argv[1]) as f:
        for line in f:
            try:
                t, g = line.split(',')
                rows.append((float(t), float(g)))
            except ValueError:
                pass
    if len(sys.argv) >= 4:
        a, b = float(sys.argv[2]), float(sys.argv[3])
        rows = [r for r in rows if a <= r[0] <= b]
    gaps = sorted((g for _, g in rows), reverse=True)
    n = len(gaps)
    if n < 100:
        print('too few frames:', n)
        return
    total = sum(gaps)
    med = statistics.median(gaps)

    def low(frac):
        k = max(1, int(n * frac))
        return 1000.0 / (sum(gaps[:k]) / k)

    def pct(p):
        return gaps[max(0, int(n * (1 - p)) - 1)]

    print(f'frames {n} over {total / 1000:.1f} s: avg {1000 * n / total:.1f} fps, median {med:.3f} ms')
    print(f'p99 {pct(0.99):.3f} ms, p99.9 {pct(0.999):.3f} ms, worst {gaps[0]:.2f} ms')
    print(f'1% low {low(0.01):.1f} fps, 0.1% low {low(0.001):.1f} fps')
    print(f'frames > 2x median {sum(g > 2 * med for g in gaps)}, > 4x {sum(g > 4 * med for g in gaps)}, > 20 ms {sum(g > 20 for g in gaps)}')
    worst = sorted(rows, key=lambda r: -r[1])[:12]
    print('slowest:', ', '.join(f'{g:.1f}@{t:.1f}' for t, g in sorted(worst)))


main()
