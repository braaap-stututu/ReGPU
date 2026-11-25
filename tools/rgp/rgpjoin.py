# Joins RGP's Pipelines table (read from the window: rgpscroll.ps1 -Which tree) with the pipeline binaries' facts
# (rgppipes.py's pipelines.csv): where the captured GPU time goes by scratch, register use, instruction count, exports.
#   py -3 rgpjoin.py <ui-pipelines-all.tsv> <pipelines.csv>
import csv, re, statistics, sys

ui = [l.rstrip('\n').split('\t') for l in open(sys.argv[1], encoding='utf-8-sig')][1:]
facts = {r['api_hash']: r for r in csv.DictReader(open(sys.argv[2]))}
rows = []
for c in ui:
    if not c or not c[0].strip().isdigit():
        continue  # (a VS / FS sub-row)
    h = c[1].replace('0x', '').upper()
    dur = float(re.match(r'([\d.]+)', c[2]).group(1))
    ev = int(re.match(r'(\d+)', c[3]).group(1)) if c[3] else 0
    f = facts.get(h)
    rows.append((int(c[0]), h, dur, ev, f))
total = sum(r[2] for r in rows)
matched = [r for r in rows if r[4]]
print(f'{len(rows)} pipelines in the table, {len(matched)} matched to binaries; summed event time {total:.0f} us')

def share(name, pred):
    sel = [r for r in matched if pred(r[4])]
    t = sum(r[2] for r in sel)
    ev = sum(r[3] for r in sel)
    print(f'{name:44} {len(sel):4} pipelines  {t:8.0f} us ({100 * t / total:5.1f}%)  {ev:5} events  {t / max(ev, 1):6.2f} us/event')

i = lambda f, k: int(f.get(k) or 0)
share('vertex stage spills to scratch', lambda f: i(f, 'gs_scratch_memory_size') > 0)
share('no vertex scratch', lambda f: i(f, 'gs_scratch_memory_size') == 0)
share('vertex sgpr >= 100', lambda f: i(f, 'gs_sgpr_count') >= 100)
share('vertex sgpr < 100', lambda f: i(f, 'gs_sgpr_count') < 100)
share('pixel kill on', lambda f: f.get('kill_enable') == 'true')
share('pixel kill off', lambda f: f.get('kill_enable') == 'false')
for lo, hi in ((0, 1000), (1000, 2000), (2000, 3000), (3000, 99999)):
    share(f'vertex instructions {lo}-{hi}', lambda f, lo=lo, hi=hi: lo <= i(f, 'gs_ins') < hi)
for lo, hi in ((0, 1000), (1000, 3000), (3000, 6000), (6000, 99999)):
    share(f'pixel instructions {lo}-{hi}', lambda f, lo=lo, hi=hi: lo <= i(f, 'ps_ins') < hi)
print('\ntop 12 by time:')
for b, h, dur, ev, f in sorted(rows, key=lambda r: -r[2])[:12]:
    if f:
        print(f'  bucket {b:3} {h} {dur:7.1f} us {ev:4} ev {dur / max(ev, 1):6.2f} us/ev | vs sgpr {f["gs_sgpr_count"]:>3} vgpr {f["gs_vgpr_count"]:>3} '
              f'scratch {f["gs_scratch_memory_size"] or 0:>2} ins {f["gs_ins"]:>5} | ps vgpr {f["ps_vgpr_count"]:>3} ins {f["ps_ins"]:>5} kill {f["kill_enable"]} | exports {f["vs_export_count"]}')
