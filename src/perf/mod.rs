/// Performance measurement utilities
/// Each rendering stage is timed and logged for optimization analysis
pub mod profiling;

pub use profiling::{CounterSnapshot, FunctionCounters, FUNCTION_COUNTERS};

use std::time::{Duration, Instant};

pub struct PerfTimer {
    name: &'static str,
    start: Instant,
}

impl PerfTimer {
    #[inline]
    pub fn new(name: &'static str) -> Self {
        Self {
            name,
            start: Instant::now(),
        }
    }

    #[inline]
    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }
}

impl Drop for PerfTimer {
    fn drop(&mut self) {
        let elapsed = self.elapsed();
        println!("[PERF] {}: {:.2}μs", self.name, elapsed.as_micros());
    }
}

/// Performance statistics accumulator
pub struct PerfStats {
    pub face_culling_us: f64,
    pub greedy_meshing_us: f64,
    pub mesh_building_us: f64,
    pub rasterization_us: f64,
    pub total_us: f64,
}

impl PerfStats {
    pub fn new() -> Self {
        Self {
            face_culling_us: 0.0,
            greedy_meshing_us: 0.0,
            mesh_building_us: 0.0,
            rasterization_us: 0.0,
            total_us: 0.0,
        }
    }

    pub fn print_summary(&self) {
        println!("\n========== PERFORMANCE SUMMARY ==========");
        println!(
            "Face Culling:    {:8.2}μs ({:5.1}%)",
            self.face_culling_us,
            (self.face_culling_us / self.total_us) * 100.0
        );
        println!(
            "Greedy Meshing:  {:8.2}μs ({:5.1}%)",
            self.greedy_meshing_us,
            (self.greedy_meshing_us / self.total_us) * 100.0
        );
        println!(
            "Mesh Building:   {:8.2}μs ({:5.1}%)",
            self.mesh_building_us,
            (self.mesh_building_us / self.total_us) * 100.0
        );
        println!(
            "Rasterization:   {:8.2}μs ({:5.1}%)",
            self.rasterization_us,
            (self.rasterization_us / self.total_us) * 100.0
        );
        println!("─────────────────────────────────────────");
        println!("Total:           {:8.2}μs", self.total_us);
        println!("=========================================\n");
    }
}

/// Macro for easy performance measurement
#[macro_export]
macro_rules! perf_scope {
    ($name:expr) => {
        let _timer = $crate::perf::PerfTimer::new($name);
    };
}
