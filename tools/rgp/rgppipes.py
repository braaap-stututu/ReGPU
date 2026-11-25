# Per-pipeline facts from an .rgp capture's pipeline binaries (extract them first: rgpchunks.py <rgp> --extract <dir>):
# for each, the API PSO hash (RGP's Pipelines "Hash" column, from the PsoCorrelation chunk), the vertex (NGG "gs") and
# pixel stages' scalar/vector registers and scratch, the export count, the pixel inputs, kill, and instruction counts.
# Writes <dir>/pipelines.csv and prints a summary of what drives context rolls and register pressure.
#   py -3 rgppipes.py <dir> [--hash 0x56FD04EAA0834D8B ...]
import collections, concurrent.futures as cf, csv, os, re, subprocess, sys

# The Radeon Developer Tool Suite's LLVM tools: RDTS_LC_BIN, else <RDTS_DIR>/utils/lc/opencl/bin.
T = os.environ.get('RDTS_LC_BIN') or os.path.join(os.environ.get('RDTS_DIR', '.'), 'utils', 'lc', 'opencl', 'bin')
d = sys.argv[1]
want = [h.upper().replace('0X', '') for h in sys.argv[sys.argv.index('--hash') + 1:]] if '--hash' in sys.argv else []

# PsoCorrelation: each internal hash (16 bytes, as stored in a CodeObject header at offset 8) is preceded by its
# 8-byte API PSO hash.
corr = open(os.path.join(d, 'PsoCorrelation.bin'), 'rb').read()
elfs = sorted(f for f in os.listdir(d) if f.endswith('.elf'))
api_of = {}
for f in elfs:
    hdr = open(os.path.join(d, f[:-4] + '.hdr'), 'rb').read()
    internal = hdr[8:24]
    pos = corr.find(internal)
    if pos >= 8:
        api_of[f] = corr[pos - 8:pos][::-1].hex().upper()

FIELDS = ['sgpr_count', 'vgpr_count', 'scratch_memory_size']
def facts(f):
    p = os.path.join(d, f)
    notes = subprocess.run([os.path.join(T, 'llvm-readobj.exe'), '--notes', p], capture_output=True, text=True).stdout
    r = {'file': f, 'api_hash': api_of.get(f, '')}
    # The hardware stages block: ".hardware_stages:" then ".gs:" / ".ps:" sections with their fields.
    stage = None
    in_hw = False
    for line in notes.splitlines():
        s = line.strip()
        if s == '.hardware_stages:':
            in_hw = True
            continue
        if in_hw:
            m = re.match(r'\.(gs|ps|vs|hs|es|ls|cs):$', s)
            if m:
                stage = m.group(1)
                continue
            m = re.match(r'\.(\w+):\s+(\S+)', s)
            if m and stage and m.group(1) in FIELDS:
                r[f'{stage}_{m.group(1)}'] = int(m.group(2), 0) if re.match(r'^-?(0x)?[0-9a-fA-F]+$', m.group(2)) else m.group(2)
            if re.match(r'\.(shaders|pipelines|api|registers|graphics_registers|internal_pipeline_hash):', s) and not s.startswith('.hardware'):
                in_hw = False
        for key in ('vs_export_count', 'num_interps', 'kill_enable', 'z_export_enable'):
            m = re.match(r'\.' + key + r':\s+(\S+)', s)
            if m and key not in r:
                r[key] = m.group(1)
    dis = subprocess.run([os.path.join(T, 'llvm-objdump.exe'), '-d', '--mcpu=gfx1100', p], capture_output=True, text=True).stdout
    cur = None
    for line in dis.splitlines():
        m = re.match(r'^[0-9a-f]+ <(_amdgpu_\w+)>:', line)
        if m:
            cur = m.group(1)
            continue
        if cur and re.match(r'^\s+[a-z_0-9]+ .*//', line):
            k = 'gs_ins' if 'gs' in cur else 'ps_ins' if 'ps' in cur else 'other_ins'
            r[k] = r.get(k, 0) + 1
    return r

with cf.ThreadPoolExecutor(16) as ex:
    rows = list(ex.map(facts, elfs))
keys = ['api_hash', 'gs_sgpr_count', 'gs_vgpr_count', 'gs_scratch_memory_size', 'gs_ins', 'ps_sgpr_count', 'ps_vgpr_count',
        'ps_scratch_memory_size', 'ps_ins', 'vs_export_count', 'num_interps', 'kill_enable', 'z_export_enable', 'file']
with open(os.path.join(d, 'pipelines.csv'), 'w', newline='') as fh:
    w = csv.DictWriter(fh, keys, extrasaction='ignore')
    w.writeheader()
    w.writerows(rows)

g = [r for r in rows if 'gs_sgpr_count' in r]
print(f'{len(rows)} binaries, {len(g)} graphics pipelines, {len(api_of)} with an API hash')
def dist(key, rs=g):
    c = collections.Counter(r.get(key) for r in rs)
    return ', '.join(f'{k}: {v}' for k, v in sorted(c.items(), key=lambda kv: (kv[0] is None, kv[0] if not isinstance(kv[0], str) else 0)))
sp = [r for r in g if r.get('gs_scratch_memory_size', 0)]
pp = [r for r in g if r.get('ps_scratch_memory_size', 0)]
print(f'vertex stage with scratch: {len(sp)} (sizes {dist("gs_scratch_memory_size", sp)}); pixel stage with scratch: {len(pp)}')
print('vertex sgpr >= 100:', sum(1 for r in g if r.get('gs_sgpr_count', 0) >= 100), ' pixel sgpr >= 100:', sum(1 for r in g if r.get('ps_sgpr_count', 0) >= 100))
import statistics
for k in ('gs_sgpr_count', 'gs_vgpr_count', 'gs_ins', 'ps_sgpr_count', 'ps_vgpr_count', 'ps_ins'):
    v = sorted(r.get(k, 0) for r in g)
    print(f'{k:16} median {v[len(v) // 2]:5}  p90 {v[int(len(v) * 0.9)]:5}  max {v[-1]:5}')
print('vs_export_count:', dist('vs_export_count'))
print('num_interps:    ', dist('num_interps'))
print('kill_enable:    ', dist('kill_enable'))
print('scratch vs not, median vertex instructions:', statistics.median([r.get('gs_ins', 0) for r in sp]) if sp else '-',
      statistics.median([r.get('gs_ins', 0) for r in g if not r.get('gs_scratch_memory_size', 0)]))
for h in want:
    for r in rows:
        if r['api_hash'].endswith(h[-16:]):
            print(h, {k: r.get(k) for k in keys})
