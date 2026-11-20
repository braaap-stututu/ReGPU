# The Vulkan conversion's progress (docs/vulkan-conversion-plan.md, step 6): D3D12 types and calls left in a source
# file, by top-level function, most first. Usage: py -3 tools/d3d12_left.py src/gpu/renderer.cpp [how many to list]
import re, sys
src = open(sys.argv[1], encoding='utf-8', errors='replace').read().split('\n')
tok = re.compile(r'ID3D12|D3D12_|DXGI_|IDXGI|->(?:Set[A-Z]\w*|Draw\w*|Copy\w*|ResourceBarrier|Create\w*|Map|Unmap|Execute\w*|Signal|GetGPUVirtualAddress|Clear\w*View|OMSet\w*|IASet\w*|RSSet\w*|Dispatch|ResolveSubresource\w*)\b')
name_re = re.compile(r'([A-Za-z_][\w:~]*)\s*\(')
blocks = []
start = None; name = None
for i, line in enumerate(src, 1):
    if line and not line[0].isspace() and line[0] not in '}/#{' and start is None:
        if line.startswith(('struct ', 'class ')):
            start, name = i, line.split()[1].rstrip('{;').strip()
        else:
            m = name_re.search(line)
            if m and not line.rstrip().endswith(';'):
                start, name = i, m.group(1)
    elif line.startswith('}') and start is not None:
        blocks.append((start, i, name))
        start = None
res = []
for s, e, n in blocks:
    c = sum(len(tok.findall(l)) for l in src[s-1:e])
    if c: res.append((c, s, e, n))
res.sort(reverse=True)
print('functions with D3D12:', len(res), 'tokens', sum(r[0] for r in res))
for c, s, e, n in res[:int(sys.argv[2]) if len(sys.argv) > 2 else 40]:
    print(f'{c:5d} {s:6d}-{e:<6d} {e-s+1:5d}L {n}')
