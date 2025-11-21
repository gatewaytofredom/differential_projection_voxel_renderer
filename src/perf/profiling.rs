/// Instrumentation and profiling infrastructure for microoptimization
/// Provides function call counting and hardware performance counter integration
use std::sync::atomic::{AtomicU64, Ordering};

/// Thread-safe performance counters for function call tracking
pub struct FunctionCounters {
    // Meshing counters
    pub mesh_chunk_calls: AtomicU64,
    pub greedy_mesh_slice_calls: AtomicU64,
    pub generate_binary_masks_calls: AtomicU64,
    pub find_chunk_calls: AtomicU64,
    pub has_solid_neighbor_calls: AtomicU64,

    // Rasterization counters
    pub render_triangle_calls: AtomicU64,
    pub render_triangle_clipped: AtomicU64,
    pub render_triangle_culled: AtomicU64,
    pub set_pixel_attempts: AtomicU64,
    pub set_pixel_depth_passed: AtomicU64,
    pub set_pixel_depth_failed: AtomicU64,

    // Framebuffer counters
    pub framebuffer_clear_calls: AtomicU64,

    // Cache and performance metrics
    pub total_pixels_tested: AtomicU64,
    pub total_triangles_processed: AtomicU64,
}

impl FunctionCounters {
    pub const fn new() -> Self {
        Self {
            mesh_chunk_calls: AtomicU64::new(0),
            greedy_mesh_slice_calls: AtomicU64::new(0),
            generate_binary_masks_calls: AtomicU64::new(0),
            find_chunk_calls: AtomicU64::new(0),
            has_solid_neighbor_calls: AtomicU64::new(0),
            render_triangle_calls: AtomicU64::new(0),
            render_triangle_clipped: AtomicU64::new(0),
            render_triangle_culled: AtomicU64::new(0),
            set_pixel_attempts: AtomicU64::new(0),
            set_pixel_depth_passed: AtomicU64::new(0),
            set_pixel_depth_failed: AtomicU64::new(0),
            framebuffer_clear_calls: AtomicU64::new(0),
            total_pixels_tested: AtomicU64::new(0),
            total_triangles_processed: AtomicU64::new(0),
        }
    }

    /// Reset all counters to zero
    pub fn reset(&self) {
        self.mesh_chunk_calls.store(0, Ordering::Relaxed);
        self.greedy_mesh_slice_calls.store(0, Ordering::Relaxed);
        self.generate_binary_masks_calls.store(0, Ordering::Relaxed);
        self.find_chunk_calls.store(0, Ordering::Relaxed);
        self.has_solid_neighbor_calls.store(0, Ordering::Relaxed);
        self.render_triangle_calls.store(0, Ordering::Relaxed);
        self.render_triangle_clipped.store(0, Ordering::Relaxed);
        self.render_triangle_culled.store(0, Ordering::Relaxed);
        self.set_pixel_attempts.store(0, Ordering::Relaxed);
        self.set_pixel_depth_passed.store(0, Ordering::Relaxed);
        self.set_pixel_depth_failed.store(0, Ordering::Relaxed);
        self.framebuffer_clear_calls.store(0, Ordering::Relaxed);
        self.total_pixels_tested.store(0, Ordering::Relaxed);
        self.total_triangles_processed.store(0, Ordering::Relaxed);
    }

    /// Get snapshot of all counters
    pub fn snapshot(&self) -> CounterSnapshot {
        CounterSnapshot {
            mesh_chunk_calls: self.mesh_chunk_calls.load(Ordering::Relaxed),
            greedy_mesh_slice_calls: self.greedy_mesh_slice_calls.load(Ordering::Relaxed),
            generate_binary_masks_calls: self.generate_binary_masks_calls.load(Ordering::Relaxed),
            find_chunk_calls: self.find_chunk_calls.load(Ordering::Relaxed),
            has_solid_neighbor_calls: self.has_solid_neighbor_calls.load(Ordering::Relaxed),
            render_triangle_calls: self.render_triangle_calls.load(Ordering::Relaxed),
            render_triangle_clipped: self.render_triangle_clipped.load(Ordering::Relaxed),
            render_triangle_culled: self.render_triangle_culled.load(Ordering::Relaxed),
            set_pixel_attempts: self.set_pixel_attempts.load(Ordering::Relaxed),
            set_pixel_depth_passed: self.set_pixel_depth_passed.load(Ordering::Relaxed),
            set_pixel_depth_failed: self.set_pixel_depth_failed.load(Ordering::Relaxed),
            framebuffer_clear_calls: self.framebuffer_clear_calls.load(Ordering::Relaxed),
            total_pixels_tested: self.total_pixels_tested.load(Ordering::Relaxed),
            total_triangles_processed: self.total_triangles_processed.load(Ordering::Relaxed),
        }
    }
}

/// Snapshot of counter values at a point in time
#[derive(Debug, Clone, Copy)]
pub struct CounterSnapshot {
    pub mesh_chunk_calls: u64,
    pub greedy_mesh_slice_calls: u64,
    pub generate_binary_masks_calls: u64,
    pub find_chunk_calls: u64,
    pub has_solid_neighbor_calls: u64,
    pub render_triangle_calls: u64,
    pub render_triangle_clipped: u64,
    pub render_triangle_culled: u64,
    pub set_pixel_attempts: u64,
    pub set_pixel_depth_passed: u64,
    pub set_pixel_depth_failed: u64,
    pub framebuffer_clear_calls: u64,
    pub total_pixels_tested: u64,
    pub total_triangles_processed: u64,
}

impl CounterSnapshot {
    /// Print formatted report
    pub fn print_report(&self) {
        println!("\n=== Performance Counters Report ===");
        println!("\nMeshing Operations:");
        println!("  mesh_chunk calls:           {:12}", self.mesh_chunk_calls);
        println!("  greedy_mesh_slice calls:    {:12}", self.greedy_mesh_slice_calls);
        println!("  generate_binary_masks calls:{:12}", self.generate_binary_masks_calls);
        println!("  find_chunk calls:           {:12}", self.find_chunk_calls);
        println!("  has_solid_neighbor calls:   {:12}", self.has_solid_neighbor_calls);

        println!("\nRasterization Operations:");
        println!("  render_triangle calls:      {:12}", self.render_triangle_calls);
        println!("  triangles clipped:          {:12}", self.render_triangle_clipped);
        println!("  triangles culled:           {:12}", self.render_triangle_culled);
        println!("  total triangles processed:  {:12}", self.total_triangles_processed);

        println!("\nPixel Operations:");
        println!("  set_pixel attempts:         {:12}", self.set_pixel_attempts);
        println!("  depth test passed:          {:12}", self.set_pixel_depth_passed);
        println!("  depth test failed:          {:12}", self.set_pixel_depth_failed);
        if self.set_pixel_attempts > 0 {
            let pass_rate = (self.set_pixel_depth_passed as f64 / self.set_pixel_attempts as f64) * 100.0;
            println!("  depth test pass rate:       {:11.2}%", pass_rate);
        }
        println!("  total pixels tested:        {:12}", self.total_pixels_tested);

        println!("\nFramebuffer Operations:");
        println!("  framebuffer clear calls:    {:12}", self.framebuffer_clear_calls);

        println!();
    }
}

/// Global function counters instance
pub static FUNCTION_COUNTERS: FunctionCounters = FunctionCounters::new();

/// Macro for incrementing a counter (only when profiling feature is enabled)
#[macro_export]
macro_rules! count_call {
    ($counter:expr) => {
        #[cfg(feature = "profiling")]
        {
            $counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
    };
}

/// Macro for adding to a counter (only when profiling feature is enabled)
#[macro_export]
macro_rules! count_add {
    ($counter:expr, $value:expr) => {
        #[cfg(feature = "profiling")]
        {
            $counter.fetch_add($value, std::sync::atomic::Ordering::Relaxed);
        }
    };
}

/// Hardware performance counter wrapper for benchmarking
#[cfg(feature = "profiling")]
pub mod hardware {
    use perf_event::{Builder, Counter};

    pub struct PerfCounters {
        pub cpu_cycles: Option<Counter>,
        pub instructions: Option<Counter>,
        pub cache_references: Option<Counter>,
        pub cache_misses: Option<Counter>,
        pub branch_instructions: Option<Counter>,
        pub branch_misses: Option<Counter>,
    }

    impl PerfCounters {
        pub fn new() -> Self {
            Self {
                cpu_cycles: Builder::new().kind(perf_event::events::Hardware::CPU_CYCLES).build().ok(),
                instructions: Builder::new().kind(perf_event::events::Hardware::INSTRUCTIONS).build().ok(),
                cache_references: Builder::new().kind(perf_event::events::Hardware::CACHE_REFERENCES).build().ok(),
                cache_misses: Builder::new().kind(perf_event::events::Hardware::CACHE_MISSES).build().ok(),
                branch_instructions: Builder::new().kind(perf_event::events::Hardware::BRANCH_INSTRUCTIONS).build().ok(),
                branch_misses: Builder::new().kind(perf_event::events::Hardware::BRANCH_MISSES).build().ok(),
            }
        }

        pub fn enable_all(&mut self) {
            if let Some(ref mut c) = self.cpu_cycles { let _ = c.enable(); }
            if let Some(ref mut c) = self.instructions { let _ = c.enable(); }
            if let Some(ref mut c) = self.cache_references { let _ = c.enable(); }
            if let Some(ref mut c) = self.cache_misses { let _ = c.enable(); }
            if let Some(ref mut c) = self.branch_instructions { let _ = c.enable(); }
            if let Some(ref mut c) = self.branch_misses { let _ = c.enable(); }
        }

        pub fn disable_all(&mut self) {
            if let Some(ref mut c) = self.cpu_cycles { let _ = c.disable(); }
            if let Some(ref mut c) = self.instructions { let _ = c.disable(); }
            if let Some(ref mut c) = self.cache_references { let _ = c.disable(); }
            if let Some(ref mut c) = self.cache_misses { let _ = c.disable(); }
            if let Some(ref mut c) = self.branch_instructions { let _ = c.disable(); }
            if let Some(ref mut c) = self.branch_misses { let _ = c.disable(); }
        }

        pub fn reset_all(&mut self) {
            if let Some(ref mut c) = self.cpu_cycles { let _ = c.reset(); }
            if let Some(ref mut c) = self.instructions { let _ = c.reset(); }
            if let Some(ref mut c) = self.cache_references { let _ = c.reset(); }
            if let Some(ref mut c) = self.cache_misses { let _ = c.reset(); }
            if let Some(ref mut c) = self.branch_instructions { let _ = c.reset(); }
            if let Some(ref mut c) = self.branch_misses { let _ = c.reset(); }
        }

        pub fn read_all(&mut self) -> PerfSnapshot {
            PerfSnapshot {
                cpu_cycles: self.cpu_cycles.as_mut().and_then(|c| c.read().ok()).unwrap_or(0),
                instructions: self.instructions.as_mut().and_then(|c| c.read().ok()).unwrap_or(0),
                cache_references: self.cache_references.as_mut().and_then(|c| c.read().ok()).unwrap_or(0),
                cache_misses: self.cache_misses.as_mut().and_then(|c| c.read().ok()).unwrap_or(0),
                branch_instructions: self.branch_instructions.as_mut().and_then(|c| c.read().ok()).unwrap_or(0),
                branch_misses: self.branch_misses.as_mut().and_then(|c| c.read().ok()).unwrap_or(0),
            }
        }
    }

    #[derive(Debug, Clone, Copy)]
    pub struct PerfSnapshot {
        pub cpu_cycles: u64,
        pub instructions: u64,
        pub cache_references: u64,
        pub cache_misses: u64,
        pub branch_instructions: u64,
        pub branch_misses: u64,
    }

    impl PerfSnapshot {
        pub fn print_report(&self) {
            println!("\n=== Hardware Performance Counters ===");
            println!("CPU Cycles:            {:16}", self.cpu_cycles);
            println!("Instructions:          {:16}", self.instructions);

            if self.cpu_cycles > 0 {
                let ipc = self.instructions as f64 / self.cpu_cycles as f64;
                println!("IPC (Instructions/Cycle): {:13.3}", ipc);
            }

            println!("\nCache Performance:");
            println!("Cache References:      {:16}", self.cache_references);
            println!("Cache Misses:          {:16}", self.cache_misses);

            if self.cache_references > 0 {
                let hit_rate = ((self.cache_references - self.cache_misses) as f64
                               / self.cache_references as f64) * 100.0;
                let miss_rate = (self.cache_misses as f64 / self.cache_references as f64) * 100.0;
                println!("Cache Hit Rate:        {:13.2}%", hit_rate);
                println!("Cache Miss Rate:       {:13.2}%", miss_rate);
            }

            println!("\nBranch Performance:");
            println!("Branch Instructions:   {:16}", self.branch_instructions);
            println!("Branch Misses:         {:16}", self.branch_misses);

            if self.branch_instructions > 0 {
                let prediction_rate = ((self.branch_instructions - self.branch_misses) as f64
                                      / self.branch_instructions as f64) * 100.0;
                println!("Branch Prediction:     {:13.2}%", prediction_rate);
            }

            println!();
        }
    }
}
