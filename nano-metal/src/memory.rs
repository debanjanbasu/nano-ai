//! Process memory queries used to size the engine for the device it runs on.

use objc2_metal::{MTLCreateSystemDefaultDevice, MTLDevice};

#[cfg(target_os = "ios")]
#[expect(unsafe_code)]
unsafe extern "C" {
    fn os_proc_available_memory() -> usize;
}

#[expect(unsafe_code)]
unsafe extern "C" {
    fn proc_pid_rusage(pid: i32, flavor: i32, buffer: *mut u64) -> i32;
}

/// Bytes this process may still use: the jetsam headroom on iOS, Metal's recommended GPU
/// working set elsewhere (about three quarters of RAM on Apple silicon Macs).
#[must_use]
pub fn budget() -> u64 {
    #[cfg(target_os = "ios")]
    {
        #[expect(unsafe_code)]
        // SAFETY: argument-free libSystem query.
        let headroom = unsafe { os_proc_available_memory() } as u64;
        if headroom > 0 {
            return headroom;
        }
    }
    MTLCreateSystemDefaultDevice().map_or(0, |d| d.recommendedMaxWorkingSetSize())
}

/// Kernel memory counters for this process.
#[derive(Clone, Copy, Debug)]
pub struct Usage {
    /// Physical footprint in bytes: the figure jetsam enforces and Xcode's memory gauge shows.
    pub footprint: u64,
    /// Highest footprint so far.
    pub peak_footprint: u64,
    /// Pages read back in from storage (file-backed weights that were evicted count here).
    pub page_ins: u64,
}

#[must_use]
pub fn usage() -> Option<Usage> {
    const RUSAGE_INFO_V4: i32 = 4;
    // `rusage_info_v4` is 37 u64 words: `ri_pageins` is word 6, `ri_phys_footprint` word 9 and
    // `ri_lifetime_max_phys_footprint` word 30.
    let mut info = [0_u64; 40];
    #[expect(unsafe_code)]
    // SAFETY: `info` is larger than `rusage_info_v4`, which the kernel fills for our own pid.
    let rc = unsafe {
        proc_pid_rusage(
            std::process::id().cast_signed(),
            RUSAGE_INFO_V4,
            info.as_mut_ptr(),
        )
    };
    (rc == 0).then_some(Usage {
        footprint: info[9],
        peak_footprint: info[30],
        page_ins: info[6],
    })
}
