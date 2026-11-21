#pragma once
// Short waits that stay out of Wine's wineserver: REGPU_WINE_WAITS=1 (off by default; a port turns it on for Wine
// platforms, e.g. FM3's Steam Deck profile). Read at every wait (live).
//
// Under Wine a waitable timer is a wineserver object: every SetWaitableTimer plus its wait is a round trip to the
// server process. ReGPU's vblank ticks (twice the frame cap: 720 Hz at a 360 cap), the command processor's idle
// sleeps, the frame limiter and a port's short sleeps all use high-resolution timers, so the server process becomes a
// busy thread of its own (FM3 on a Steam Deck under Proton: wineserver at 24% of a core in a race). With this switch:
//   - a sleep is NtDelayExecution, not alertable: in Wine that is a nanosleep in the calling thread;
//   - "an event or a timeout" is NtWaitForSingleObject with a relative timeout: with ntsync (Proton 10+, Linux 6.14+)
//     one ioctl with a nanosecond timeout, no server.
// Both measured on the Deck (Proton 11): a 100 us sleep took 169 us against the timer's 203 us. On Windows these
// calls round to the system timer resolution (0.5-1 ms), which is why it is a Wine-only switch: without Wine
// (no wine_get_version in ntdll) it stays off even when set.
#include <windows.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace regpu::winewait
{
using NtDelayExecutionFn = LONG(NTAPI*)(BOOLEAN alertable, PLARGE_INTEGER interval);
using NtWaitForSingleObjectFn = LONG(NTAPI*)(HANDLE handle, BOOLEAN alertable, PLARGE_INTEGER timeout);

struct Api
{
    bool wine = false;
    NtDelayExecutionFn delay = nullptr;
    NtWaitForSingleObjectFn wait = nullptr;
};

inline const Api& Get()
{
    static const Api api = [] {
        Api a;
        HMODULE ntdll = GetModuleHandleW(L"ntdll.dll");
        if (!ntdll || !GetProcAddress(ntdll, "wine_get_version"))
            return a;
        a.delay = reinterpret_cast<NtDelayExecutionFn>(GetProcAddress(ntdll, "NtDelayExecution"));
        a.wait = reinterpret_cast<NtWaitForSingleObjectFn>(GetProcAddress(ntdll, "NtWaitForSingleObject"));
        a.wine = a.delay && a.wait;
        return a;
    }();
    return api;
}

// REGPU_WINE_WAITS is set, this is Wine, and the ntdll calls exist. The variable is read at every call (a getenv
// is nothing next to the wait), so a port may set it after the first waits: threads that waited before a port's
// start-up code ran kept the timer path for good when it was read once.
inline bool On()
{
    if (!Get().wine)
        return false;
    const char* v = getenv("REGPU_WINE_WAITS");
    return v && *v && strcmp(v, "0") != 0;
}

// Sleeps about `us` microseconds (only when On()).
inline void SleepUs(int64_t us)
{
    LARGE_INTEGER interval;
    interval.QuadPart = -(us > 0 ? us : 0) * 10; // relative, 100 ns units
    Get().delay(FALSE, &interval);
}

// Waits for `handle` at most `us` microseconds (only when On()): WAIT_OBJECT_0 when it was signalled, else
// WAIT_TIMEOUT.
inline DWORD WaitUs(HANDLE handle, int64_t us)
{
    LARGE_INTEGER timeout;
    timeout.QuadPart = -(us > 0 ? us : 0) * 10;
    return Get().wait(handle, FALSE, &timeout) == 0 ? WAIT_OBJECT_0 : WAIT_TIMEOUT;
}
} // namespace regpu::winewait
