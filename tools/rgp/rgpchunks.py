# Reads an .rgp capture (AMD's RDF container: header "AMD_RDF ", u32 version, u32 reserved, i64 directory offset,
# i64 directory size; 64-byte directory entries {char id[16], u32, u32, i64 header offset, i64 header size,
# i64 data offset, i64 data size, i64}). Lists the chunks; --extract <dir> writes every CodeObject chunk's ELF (the
# pipeline binaries the capture saw) as <internal hash>.elf, and the PsoCorrelation and COLoadEvent chunks raw.
#   py -3 rgpchunks.py <file.rgp> [--extract <dir>]
import os, struct, sys

path = sys.argv[1]
data = open(path, 'rb').read()
ident, version, _res, dir_off, dir_size = struct.unpack_from('<8sIIqq', data, 0)
print(f'{ident!r} version {version}, {dir_size // 64} chunks, file {len(data) // 1024} KB')
ENTRY = struct.Struct('<16sIIqqqqq')
entries = []
for i in range(dir_size // ENTRY.size):
    cid, a, b, hoff, hsize, doff, dsize, usize = ENTRY.unpack_from(data, dir_off + i * ENTRY.size)
    entries.append((cid.rstrip(b'\0').decode('ascii', 'replace'), hoff, hsize, doff, dsize))
summary = {}
for e in entries:
    s = summary.setdefault(e[0], [0, 0])
    s[0] += 1
    s[1] += e[4]
for name, (n, size) in summary.items():
    print(f'{name:20} x{n:5}  {size // 1024:8} KB')

if '--extract' in sys.argv:
    out = sys.argv[sys.argv.index('--extract') + 1]
    os.makedirs(out, exist_ok=True)
    n = 0
    for name, hoff, hsize, doff, dsize in entries:
        body = data[doff:doff + dsize]
        if name == 'CodeObject':
            hdr = data[hoff:hoff + hsize]
            # Header: u32 (pci id / flags), u32, then the 128-bit internal pipeline hash.
            h = hdr[8:24][::-1].hex().upper() if len(hdr) >= 24 else f'{n:04}'
            if body[:4] == b'\x7fELF':
                open(os.path.join(out, f'{h}.elf'), 'wb').write(body)
                open(os.path.join(out, f'{h}.hdr'), 'wb').write(hdr)
                n += 1
        elif name in ('PsoCorrelation', 'COLoadEvent', 'ApiInfo', 'AsicInfo'):
            open(os.path.join(out, f'{name}.bin'), 'wb').write(body)
            open(os.path.join(out, f'{name}.hdr'), 'wb').write(data[hoff:hoff + hsize])
    print('extracted', n, 'pipeline ELFs to', out)
