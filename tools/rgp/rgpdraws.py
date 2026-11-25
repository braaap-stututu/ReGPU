# Per-draw analysis of an RGP capture: events in order (rgpevents.ps1's events-by-pipeline table, by event ID), each
# with its pipeline's facts (rgppipes.py's pipelines.csv). Compares draws right after a pipeline switch with draws that
# keep their pipeline, by what changed (export count, vertex or pixel shader size), and lists the passes (from the
# Event timing list, rgpscroll.ps1 -Which tree on Events -> Event timing).
#   py -3 rgpdraws.py <events-by-pipeline.tsv> <pipelines.csv> [<ui-events.tsv>]
import csv, collections, re, statistics, sys

ev = list(csv.DictReader(open(sys.argv[1], encoding='utf-8-sig'), delimiter='\t'))
facts = {r['api_hash']: r for r in csv.DictReader(open(sys.argv[2]))}
for e in ev:
    e['id'] = int(e['event'])
    e['us'] = float(e['duration_us'] or 0)
    e['f'] = facts.get(e['api_hash'].replace('0x', '').upper())
ev.sort(key=lambda e: e['id'])
draws = [e for e in ev if e['call'].startswith('vkCmdDraw')]
print(f'{len(ev)} events with a pipeline, {len(draws)} draws; summed draw time {sum(d["us"] for d in draws):.0f} us')

def med(v):
    return statistics.median(v) if v else 0
groups = collections.defaultdict(list)
prev = None
for d in draws:
    f, pf = d['f'], prev['f'] if prev else None
    if prev is None or d['api_hash'] != prev['api_hash']:
        kind = 'switch'
        if f and pf:
            kind += ', exports ' + ('same' if f['vs_export_count'] == pf['vs_export_count'] else 'changed')
    else:
        kind = 'same pipeline'
    groups[kind].append(d['us'])
    prev = d
print('\ndraw duration by what came before (RGP duration: start to end, overlapping draws included):')
for k, v in sorted(groups.items()):
    print(f'  {k:32} {len(v):5} draws  median {med(v):6.2f} us  mean {statistics.mean(v):6.2f} us  sum {sum(v):8.0f} us')

# Small draws only (under 64 indices or vertices): their time is mostly fixed cost, not work.
def count(call):
    m = re.match(r'vkCmdDraw(?:Indexed)?\((\d+)', call)
    return int(m.group(1)) if m else 0
small = collections.defaultdict(list)
prev = None
for d in draws:
    if count(d['call']) <= 64:
        small['switch' if prev is None or d['api_hash'] != prev['api_hash'] else 'same pipeline'].append(d['us'])
    prev = d
print('\nsmall draws (<= 64 indices or vertices):')
for k, v in small.items():
    print(f'  {k:32} {len(v):5} draws  median {med(v):6.2f} us  mean {statistics.mean(v):6.2f} us')

# By the vertex shader's scratch and register use, small draws only.
by = collections.defaultdict(list)
for d in draws:
    if count(d['call']) <= 64 and d['f']:
        f = d['f']
        by[('vs scratch' if int(f['gs_scratch_memory_size'] or 0) else 'vs no scratch') + (', vs sgpr>=100' if int(f['gs_sgpr_count'] or 0) >= 100 else ', vs sgpr<100')].append(d['us'])
print('\nsmall draws by vertex shader:')
for k, v in sorted(by.items()):
    print(f'  {k:32} {len(v):5} draws  median {med(v):6.2f} us  mean {statistics.mean(v):6.2f} us')

if len(sys.argv) > 3:
    # Passes: the Event timing list's group rows ("Color pass N", "Depth-only pass N", ...) and the events under them.
    order = []
    cur = None
    for line in open(sys.argv[3], encoding='utf-8-sig'):
        name = line.split('\t')[0].strip()
        m = re.match(r'^(\d+) (vk\w+)', name)
        if m:
            order.append((int(m.group(1)), cur))
        elif name:
            cur = name
    pass_of = dict(order)
    per = collections.OrderedDict()
    prev = None
    for d in draws:
        p = pass_of.get(d['id'], '?')
        s = per.setdefault(p, [0, 0, 0.0])
        s[0] += 1
        s[1] += prev is not None and d['api_hash'] != prev['api_hash']
        s[2] += d['us']
        prev = d
    print('\npasses with the most draws (draws, pipeline switches, summed draw time):')
    for p, (n, sw, t) in sorted(per.items(), key=lambda kv: -kv[1][0])[:12]:
        print(f'  {p:28} {n:5} draws {sw:5} switches {t:8.0f} us')
