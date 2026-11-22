# RGP's Barriers table (rgpscroll.ps1 -Which tree on Overview -> Barriers) summarised, each barrier matched to the event
# after it (Event timing order, rgpscroll.ps1 -Which tree on Events -> Event timing).
#   py -3 rgpbarriers.py <ui-barriers.tsv> <ui-events.tsv>
import collections, re, sys

text = open(sys.argv[1], encoding='utf-8-sig').read()
lines = text.split('\n')
# (The header cells contain line breaks: the data rows start at the first line beginning with a number.)
rows = [l.split('\t') for l in lines if re.match(r'^\d+\t', l)]
cols = ['event', 'duration', 'drain', 'stalls', 'ds_decompress', 'hiz_resummarize', 'dcc_decompress', 'fmask_decompress',
        'fast_clear_eliminate', 'init_mask_ram', 'invalidated', 'flushed', 'type', 'reason']

def us(s):
    m = re.match(r'([\d.]+)', s or '')
    return float(m.group(1)) if m else 0.0

print(f'{len(rows)} barriers: duration {sum(us(r[1]) for r in rows):.0f} us, drain {sum(us(r[2]) for r in rows):.0f} us')
for ci in range(4, 10):
    hit = [r for r in rows if len(r) > ci and r[ci].strip()]
    if hit:
        print(f'  {cols[ci]}: {len(hit)} barriers, ' + ', '.join(f'#{r[0]} {r[ci]}' for r in hit[:8]))
for ci, name in ((3, 'stalls'), (11, 'flushed'), (12, 'type'), (13, 'reason')):
    print(f'{name}:', collections.Counter(r[ci].strip() if len(r) > ci else '' for r in rows).most_common(6))

order = []
for line in open(sys.argv[2], encoding='utf-8-sig'):
    m = re.match(r'^(\d+) (\S+)', line.split('\t')[0].strip())
    if m:
        order.append((int(m.group(1)), re.sub(r'\(.*', '', m.group(2))))
nxt = {}
for i, (eid, call) in enumerate(order):
    if call.startswith('vkCmdPipelineBarrier') and i + 1 < len(order):
        nxt[eid] = order[i + 1][1]
after = collections.defaultdict(lambda: [0, 0.0, 0.0])
for r in rows:
    a = after[nxt.get(int(r[0]), '?')]
    a[0] += 1
    a[1] += us(r[1])
    a[2] += us(r[2])
print('followed by (count, duration us, drain us):')
for k, (n, d, dr) in sorted(after.items(), key=lambda kv: -kv[1][1]):
    print(f'  {k:28} {n:4} {d:8.1f} {dr:8.1f}')
print('largest:')
for r in sorted(rows, key=lambda r: -us(r[1]))[:10]:
    print(f'  #{r[0]:5} duration {r[1]:>12} drain {r[2]:>12} stalls {r[3]:6} flushed {r[11] if len(r) > 11 else "":12} next {nxt.get(int(r[0]))}')
