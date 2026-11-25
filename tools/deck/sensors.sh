#!/bin/sh
# The Deck's APU once a second: time, GPU clock (MHz), GPU busy %, APU power (W), GPU temperature (C), the
# fastest CPU core's clock (MHz). The Deck throttles: log this beside every measurement and compare runs only at
# the same clocks and power.
#
#   tools/deck/sensors.sh [seconds] [out file]      (default 600 s, to stdout)
SECONDS_MAX=${1:-600}
OUT=${2:-/dev/stdout}
DEV=
for d in /sys/class/drm/card*/device; do
    [ "$(cat "$d/vendor" 2>/dev/null)" = "0x1002" ] && { DEV=$d; break; }
done
[ -n "$DEV" ] || { echo "sensors.sh: no AMD GPU under /sys/class/drm" >&2; exit 1; }
HW=$(ls -d "$DEV"/hwmon/hwmon* 2>/dev/null | head -1)
echo "time sclk_mhz busy_pct power_w temp_c cpu_max_mhz" > "$OUT"
i=0
while [ $i -lt "$SECONDS_MAX" ]; do
    sclk=$(grep '\*' "$DEV/pp_dpm_sclk" 2>/dev/null | head -1 | sed 's/.*: *\([0-9]*\)Mhz.*/\1/')
    busy=$(cat "$DEV/gpu_busy_percent" 2>/dev/null)
    pw=$(cat "$HW/power1_average" 2>/dev/null || cat "$HW/power1_input" 2>/dev/null)
    [ -n "$pw" ] && pw=$((pw / 1000000))
    t=$(cat "$HW/temp1_input" 2>/dev/null)
    [ -n "$t" ] && t=$((t / 1000))
    cpu=$(cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq 2>/dev/null | sort -n | tail -1)
    [ -n "$cpu" ] && cpu=$((cpu / 1000))
    echo "$(date +%T) ${sclk:-?} ${busy:-?} ${pw:-?} ${t:-?} ${cpu:-?}" >> "$OUT"
    i=$((i + 1))
    sleep 1
done
