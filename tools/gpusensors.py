# Reads the AMD GPU's sensors (clocks, temperatures, power, activity) through the driver's ADL library, read-only.
#   py -3 gpusensors.py dump              every supported sensor of every adapter, once
#   py -3 gpusensors.py log SECONDS [OUT]  one line a second for the dGPU: gfx clock, mem clock, temps, power, activity
import ctypes, sys, time

adl = ctypes.WinDLL('atiadlxx.dll')
MALLOC = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_int)
_keep = []
@MALLOC
def _malloc(n):
    b = ctypes.create_string_buffer(n)
    _keep.append(b)
    return ctypes.addressof(b)

ctx = ctypes.c_void_p()
assert adl.ADL2_Main_Control_Create(_malloc, 1, ctypes.byref(ctx)) == 0, 'ADL init failed'

class Sensor(ctypes.Structure):
    _fields_ = [('supported', ctypes.c_int), ('value', ctypes.c_int)]
class Output(ctypes.Structure):
    _fields_ = [('size', ctypes.c_int), ('sensors', Sensor * 256)]

# ADLSensorType (adl_defines.h)
NAMES = {1: 'gfxclk', 2: 'memclk', 3: 'socclk', 8: 'temp_edge', 9: 'temp_mem', 10: 'temp_vrvddc', 11: 'temp_vrmvdd',
         14: 'fan_rpm', 15: 'fan_pct', 19: 'gfx_activity', 20: 'mem_activity', 21: 'gfx_mV', 23: 'asic_W',
         27: 'temp_hotspot', 29: 'board_W', 73: 'power_W?'}

n = ctypes.c_int()
adl.ADL2_Adapter_NumberOfAdapters_Get(ctx, ctypes.byref(n))

def read(i):
    o = Output()
    o.size = ctypes.sizeof(Output)
    if adl.ADL2_New_QueryPMLogData_Get(ctx, i, ctypes.byref(o)) != 0:
        return None
    return {k: o.sensors[k].value for k in range(256) if o.sensors[k].supported}

def dgpu():
    best = None
    for i in range(n.value):
        s = read(i)
        if s and 27 in s and best is None:  # (the iGPU reports no hotspot temperature)
            best = (i, s)
    return best[0]

if sys.argv[1] == 'dump':
    seen = set()
    for i in range(n.value):
        s = read(i)
        if not s or tuple(sorted(s)) in seen:
            continue
        seen.add(tuple(sorted(s)))
        print('adapter', i, ' '.join(f'{NAMES.get(k, k)}={v}' for k, v in s.items()))
else:
    seconds = int(sys.argv[2])
    out = open(sys.argv[3], 'w') if len(sys.argv) > 3 else sys.stdout
    i = dgpu()
    keys = [1, 2, 8, 27, 9, 73, 19, 20, 15]
    print('t ' + ' '.join(NAMES.get(k, str(k)) for k in keys), file=out, flush=True)
    t0 = time.time()
    while time.time() - t0 < seconds:
        s = read(i) or {}
        print(f'{time.time() - t0:.0f} ' + ' '.join(str(s.get(k, '-')) for k in keys), file=out, flush=True)
        time.sleep(1)
