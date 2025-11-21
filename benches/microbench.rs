/// Comprehensive micro-benchmarks with hardware performance counter integration
/// Provides detailed profiling data for hot-path optimization
use criterion::{black_box, criterion_group, Criterion};
use glam::{IVec3, Vec3};
use std::sync::Mutex;
use voxel_engine::{
    BinaryGreedyMesher, BlockData, BlockType, Camera, Chunk, Framebuffer, Rasterizer,
    CHUNK_SIZE, FUNCTION_COUNTERS, CounterSnapshot,
};

// Thread-safe storage for collected statistics
lazy_static::lazy_static! {
    static ref COLLECTED_STATS: Mutex<Vec<(String, CounterSnapshot)>> = Mutex::new(Vec::new());
}

/// Benchmark with function call counting and detailed reporting
fn bench_greedy_mesh_slice_with_counters(c: &mut Criterion) {
    let mut group = c.benchmark_group("greedy_mesh_slice_detailed");

    // Test various data patterns to understand cache behavior
    let patterns = [
        ("empty", [0u32; CHUNK_SIZE]),
        ("full", [!0u32; CHUNK_SIZE]),
        ("checkerboard", {
            let mut data = [0u32; CHUNK_SIZE];
            for i in 0..CHUNK_SIZE {
                data[i] = if i % 2 == 0 { !0u32 } else { 0 };
            }
            data
        }),
        ("sparse", {
            let mut data = [0u32; CHUNK_SIZE];
            for i in 0..CHUNK_SIZE {
                data[i] = 0x80000001; // Only first and last bit
            }
            data
        }),
    ];

    for (name, data) in patterns.iter() {
        group.bench_function(*name, |b| {
            b.iter(|| {
                FUNCTION_COUNTERS.reset();
                black_box(BinaryGreedyMesher::greedy_mesh_slice(black_box(data)));
            });
        });

        // Collect stats after benchmark completes
        let snapshot = FUNCTION_COUNTERS.snapshot();
        let label = format!("greedy_mesh_slice/{}", name);
        COLLECTED_STATS.lock().unwrap().push((label, snapshot));
        FUNCTION_COUNTERS.reset();
    }

    group.finish();
}

/// Benchmark rasterization inner loop with depth test profiling
fn bench_rasterize_triangle_depth_test(c: &mut Criterion) {
    let mut group = c.benchmark_group("rasterize_depth_test");

    // Generate different mesh densities
    let densities = [
        ("sparse_terrain", Chunk::generate_terrain(IVec3::ZERO)),
        ("half_solid", {
            let mut chunk = Chunk::uniform(IVec3::ZERO, BlockType::Air);
            for z in 0..CHUNK_SIZE {
                for y in 0..CHUNK_SIZE / 2 {
                    for x in 0..CHUNK_SIZE {
                        chunk.set_block(x, y, z, BlockData::new(BlockType::Stone));
                    }
                }
            }
            chunk
        }),
    ];

    for (name, chunk) in densities.iter() {
        if let Some(mesh) = BinaryGreedyMesher::mesh_chunk(chunk) {
            group.bench_function(*name, |b| {
                let mut framebuffer = Framebuffer::new(1280, 720);
                let camera = Camera::new(Vec3::new(0.0, 10.0, 20.0), 1280.0 / 720.0);
                let mut rasterizer = Rasterizer::new();
                let view_proj = camera.view_projection_matrix();

                b.iter(|| {
                    FUNCTION_COUNTERS.reset();
                    framebuffer.clear(0xFF87CEEB);
                    rasterizer.render_mesh(black_box(&mesh), black_box(&view_proj), &mut framebuffer);
                });
            });

            // Collect stats after benchmark completes
            let snapshot = FUNCTION_COUNTERS.snapshot();
            let label = format!("rasterize_depth_test/{}", name);
            COLLECTED_STATS.lock().unwrap().push((label, snapshot));
            FUNCTION_COUNTERS.reset();
        }
    }

    group.finish();
}

/// Benchmark meshing with neighbor lookups
fn bench_neighbor_lookup_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("neighbor_lookups");

    let sizes = [1, 9, 27]; // 1x1x1, 3x3x1, 3x3x3

    for &size in &sizes {
        let label = format!("chunks_{}", size);

        group.bench_function(&label, |b| {
            let mut chunks = Vec::new();
            let dim = (size as f64).cbrt().ceil() as i32;

            for x in 0..dim {
                for y in 0..dim {
                    for z in 0..dim {
                        if chunks.len() >= size {
                            break;
                        }
                        chunks.push(Chunk::generate_terrain(IVec3::new(x, y, z)));
                    }
                }
            }

            let chunk_refs: Vec<&Chunk> = chunks.iter().collect();
            let center_chunk = &chunks[0];

            b.iter(|| {
                FUNCTION_COUNTERS.reset();
                black_box(BinaryGreedyMesher::mesh_chunk_in_world(
                    black_box(center_chunk),
                    black_box(&chunk_refs),
                ));
            });
        });

        // Collect stats after benchmark completes
        let snapshot = FUNCTION_COUNTERS.snapshot();
        let full_label = format!("neighbor_lookups/{}", label);
        COLLECTED_STATS.lock().unwrap().push((full_label, snapshot));
        FUNCTION_COUNTERS.reset();
    }

    group.finish();
}

/// Benchmark framebuffer operations
fn bench_framebuffer_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("framebuffer_operations");

    group.bench_function("clear_1280x720", |b| {
        let mut framebuffer = Framebuffer::new(1280, 720);

        b.iter(|| {
            FUNCTION_COUNTERS.reset();
            framebuffer.clear(black_box(0xFF87CEEB));
        });
    });

    // Collect stats after benchmark completes
    let snapshot = FUNCTION_COUNTERS.snapshot();
    let label = "framebuffer_operations/clear_1280x720".to_string();
    COLLECTED_STATS.lock().unwrap().push((label, snapshot));
    FUNCTION_COUNTERS.reset();

    group.bench_function("set_pixel_tight_loop", |b| {
        let mut framebuffer = Framebuffer::new(1280, 720);
        b.iter(|| {
            for y in 0..100 {
                for x in 0..100 {
                    unsafe {
                        framebuffer.set_pixel_unchecked(x, y, 0xFF00FF00, 0.5);
                    }
                }
            }
        });
    });

    group.finish();
}

// Custom function to print summary after all benchmarks complete
fn print_profiling_summary() {
    let stats = COLLECTED_STATS.lock().unwrap();

    if stats.is_empty() {
        return;
    }

    println!("\n");
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("                    PROFILING SUMMARY (--features profiling)");
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!();

    // Group by benchmark type
    for (label, snapshot) in stats.iter() {
        println!("──────────────────────────────────────────────────────────────────────────");
        println!("  Benchmark: {}", label);
        println!("──────────────────────────────────────────────────────────────────────────");

        // Meshing stats
        if snapshot.mesh_chunk_calls > 0 || snapshot.greedy_mesh_slice_calls > 0 {
            println!("  Meshing Operations:");
            if snapshot.mesh_chunk_calls > 0 {
                println!("    mesh_chunk calls:           {:>12}", snapshot.mesh_chunk_calls);
            }
            if snapshot.greedy_mesh_slice_calls > 0 {
                println!("    greedy_mesh_slice calls:    {:>12}", snapshot.greedy_mesh_slice_calls);
            }
            if snapshot.generate_binary_masks_calls > 0 {
                println!("    generate_binary_masks:      {:>12}", snapshot.generate_binary_masks_calls);
            }
            if snapshot.find_chunk_calls > 0 {
                println!("    find_chunk calls:           {:>12}", snapshot.find_chunk_calls);
            }
            if snapshot.has_solid_neighbor_calls > 0 {
                println!("    has_solid_neighbor calls:   {:>12}", snapshot.has_solid_neighbor_calls);
            }
            println!();
        }

        // Rasterization stats
        if snapshot.render_triangle_calls > 0 || snapshot.total_triangles_processed > 0 {
            println!("  Rasterization Operations:");
            if snapshot.render_triangle_calls > 0 {
                println!("    render_triangle calls:      {:>12}", snapshot.render_triangle_calls);
            }
            if snapshot.total_triangles_processed > 0 {
                println!("    total triangles processed:  {:>12}", snapshot.total_triangles_processed);
            }
            if snapshot.render_triangle_clipped > 0 {
                println!("    triangles clipped:          {:>12}", snapshot.render_triangle_clipped);
            }
            if snapshot.render_triangle_culled > 0 {
                println!("    triangles culled:           {:>12}", snapshot.render_triangle_culled);
            }
            println!();
        }

        // Pixel operations
        if snapshot.set_pixel_attempts > 0 || snapshot.total_pixels_tested > 0 {
            println!("  Pixel Operations:");
            if snapshot.total_pixels_tested > 0 {
                println!("    total pixels tested:        {:>12}", snapshot.total_pixels_tested);
            }
            if snapshot.set_pixel_attempts > 0 {
                println!("    set_pixel attempts:         {:>12}", snapshot.set_pixel_attempts);
                println!("    depth tests passed:         {:>12}", snapshot.set_pixel_depth_passed);
                println!("    depth tests failed:         {:>12}", snapshot.set_pixel_depth_failed);

                let pass_rate = (snapshot.set_pixel_depth_passed as f64
                    / snapshot.set_pixel_attempts as f64) * 100.0;
                println!("    depth test pass rate:       {:>11.2}%", pass_rate);
            }
            println!();
        }

        // Framebuffer operations
        if snapshot.framebuffer_clear_calls > 0 {
            println!("  Framebuffer Operations:");
            println!("    clear calls:                {:>12}", snapshot.framebuffer_clear_calls);
            println!();
        }
    }

    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("  Tip: Run without --features profiling for pure performance benchmarks");
    println!("  Tip: Use 'perf stat cargo bench' for hardware counter analysis");
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!();
}

criterion_group!{
    name = benches;
    config = Criterion::default();
    targets =
    bench_greedy_mesh_slice_with_counters,
    bench_rasterize_triangle_depth_test,
    bench_neighbor_lookup_patterns,
    bench_framebuffer_ops
}

// Custom main to print summary after all benchmarks
fn main() {
    benches();
    print_profiling_summary();
}
