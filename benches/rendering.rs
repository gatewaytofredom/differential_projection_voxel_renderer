/// Benchmark suite for rendering pipeline
/// Tests performance of software rasterization and hot-path primitives.
use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use glam::{IVec3, Mat4, Vec3, Vec4};
use voxel_engine::{
    BinaryGreedyMesher, Camera, Chunk, ChunkMesh, Framebuffer, Rasterizer, ShadingConfig,
    CHUNK_SIZE, BlockType,
};
use voxel_engine::meshing::Vertex;
use voxel_engine::rendering::simd_vertex;

fn bench_rasterize_single_chunk(c: &mut Criterion) {
    c.bench_function("rasterize_single_chunk", |b| {
        let chunk = Chunk::generate_terrain(IVec3::ZERO);
        let mesh = BinaryGreedyMesher::mesh_chunk(&chunk).unwrap();
        let mut framebuffer = Framebuffer::new(1280, 720);
        let camera = Camera::new(Vec3::new(0.0, 10.0, 20.0), 1280.0 / 720.0);
        let mut rasterizer = Rasterizer::new();
        let view_proj = camera.view_projection_matrix();

        b.iter(|| {
            framebuffer.clear(0xFF87CEEB);
            rasterizer.render_mesh(black_box(&mesh), black_box(&view_proj), &mut framebuffer);
        });
    });
}

fn bench_framebuffer_clear(c: &mut Criterion) {
    c.bench_function("framebuffer_clear", |b| {
        let mut framebuffer = Framebuffer::new(1280, 720);

        b.iter(|| {
            framebuffer.clear(black_box(0xFF87CEEB));
        });
    });
}

fn bench_framebuffer_set_pixel(c: &mut Criterion) {
    c.bench_function("framebuffer_set_pixel", |b| {
        let mut framebuffer = Framebuffer::new(1280, 720);
        let color = 0xFF00FF00;
        let depth = 0.5;

        b.iter(|| {
            black_box(framebuffer.set_pixel(100, 100, color, depth));
        });
    });
}

fn bench_framebuffer_set_pixel_unchecked(c: &mut Criterion) {
    c.bench_function("framebuffer_set_pixel_unchecked", |b| {
        let mut framebuffer = Framebuffer::new(1280, 720);
        let color = 0xFF00FF00;
        let depth = 0.5;

        b.iter(|| unsafe {
            black_box(framebuffer.set_pixel_unchecked(100, 100, color, depth));
        });
    });
}

fn bench_rasterize_world_3x3x3(c: &mut Criterion) {
    c.bench_function("rasterize_world_3x3x3", |b| {
        // Build a small world of chunks and meshes (similar to main.rs)
        let mut chunks = Vec::new();
        for cx in -1..=1 {
            for cy in -1..=1 {
                for cz in -1..=1 {
                    chunks.push(Chunk::generate_terrain(IVec3::new(cx, cy, cz)));
                }
            }
        }

        let chunk_refs: Vec<&Chunk> = chunks.iter().collect();
        let meshes: Vec<ChunkMesh> = BinaryGreedyMesher::mesh_world(&chunk_refs);

        let mut framebuffer = Framebuffer::new(1280, 720);
        let camera = Camera::new(
            Vec3::new(
                (CHUNK_SIZE as f32) * 1.5,
                (CHUNK_SIZE as f32),
                (CHUNK_SIZE as f32) * 2.5,
            ),
            1280.0 / 720.0,
        );
        let mut rasterizer = Rasterizer::new();
        let view_proj = camera.view_projection_matrix();

        b.iter(|| {
            framebuffer.clear(0xFF87CEEB);
            for mesh in &meshes {
                rasterizer.render_mesh(black_box(mesh), black_box(&view_proj), &mut framebuffer);
            }
        });
    });
}

fn bench_vertex_decompression(c: &mut Criterion) {
    let mut group = c.benchmark_group("vertex_decompression");

    // Test various vertex counts (common in real meshes)
    for vertex_count in [64, 256, 1024, 4096] {
        // Generate test vertices
        let vertices: Vec<Vertex> = (0..vertex_count)
            .map(|i| Vertex::from_local_coords(
                ((i * 7) % 32) as u8,
                ((i * 11) % 32) as u8,
                ((i * 13) % 32) as u8,
                BlockType::Stone,
                0,
                0,
                1.0,
            ))
            .collect();

        let chunk_offset = Vec3::new(64.0, 32.0, 128.0);
        let camera = Camera::new(Vec3::new(96.0, 64.0, 160.0), 16.0 / 9.0);
        let view_proj = camera.view_projection_matrix();
        let mut output = vec![Vec4::ZERO; vertex_count];

        group.bench_with_input(
            BenchmarkId::new("simd", vertex_count),
            &vertex_count,
            |b, _| {
                b.iter(|| {
                    simd_vertex::decompress_and_transform_vertices(
                        black_box(&vertices),
                        black_box(chunk_offset),
                        black_box(&view_proj),
                        black_box(&mut output),
                    );
                });
            },
        );
    }

    group.finish();
}

fn bench_parallel_rendering_single_threaded(c: &mut Criterion) {
    c.bench_function("parallel_rendering/single_thread", |b| {
        // Generate a realistic workload: 9 terrain chunks
        let mut chunks = Vec::new();
        for cx in -1..=1 {
            for cz in -1..=1 {
                chunks.push(Chunk::generate_terrain(IVec3::new(cx, 0, cz)));
            }
        }

        let chunk_refs: Vec<&Chunk> = chunks.iter().collect();
        let mut meshes = Vec::new();
        for chunk in &chunks {
            if let Some(mesh) = BinaryGreedyMesher::mesh_chunk_in_world(chunk, &chunk_refs) {
                meshes.push(mesh);
            }
        }

        let mut framebuffer = Framebuffer::new(1280, 720);
        let mut rasterizer = Rasterizer::new();
        let camera = Camera::new(Vec3::new(0.0, 10.0, 20.0), 1280.0 / 720.0);
        let view_proj = camera.view_projection_matrix();

        b.iter(|| {
            framebuffer.clear(0xFF87CEEB);

            // Single-threaded: use single stripe
            let mut slices = framebuffer.split_into_stripes(1);
            if let Some(slice) = slices.first_mut() {
                for mesh in &meshes {
                    rasterizer.render_mesh_into_slice(black_box(mesh), black_box(&view_proj), slice);
                }
            }
        });
    });
}

fn bench_parallel_rendering_multi_threaded(c: &mut Criterion) {
    c.bench_function("parallel_rendering/multi_thread", |b| {
        use rayon::prelude::*;

        // Generate a realistic workload: 9 terrain chunks
        let mut chunks = Vec::new();
        for cx in -1..=1 {
            for cz in -1..=1 {
                chunks.push(Chunk::generate_terrain(IVec3::new(cx, 0, cz)));
            }
        }

        let chunk_refs: Vec<&Chunk> = chunks.iter().collect();
        let mut meshes = Vec::new();
        for chunk in &chunks {
            if let Some(mesh) = BinaryGreedyMesher::mesh_chunk_in_world(chunk, &chunk_refs) {
                meshes.push(mesh);
            }
        }

        let mut framebuffer = Framebuffer::new(1280, 720);
        let rasterizer = Rasterizer::new();
        let camera = Camera::new(Vec3::new(0.0, 10.0, 20.0), 1280.0 / 720.0);
        let view_proj = camera.view_projection_matrix();

        // Capture configuration
        let enable_shading = rasterizer.enable_shading;
        let backface_culling = rasterizer.backface_culling;
        let shading_config = rasterizer.shading;

        b.iter(|| {
            framebuffer.clear(0xFF87CEEB);

            // Multi-threaded: one stripe per thread
            let stripe_count = rayon::current_num_threads().max(1);
            let mut slices = framebuffer.split_into_stripes(stripe_count);

            slices.par_iter_mut().for_each(|slice| {
                let mut local_rasterizer = Rasterizer::new();
                local_rasterizer.enable_shading = enable_shading;
                local_rasterizer.backface_culling = backface_culling;
                local_rasterizer.shading = shading_config;

                for mesh in &meshes {
                    local_rasterizer.render_mesh_into_slice(black_box(mesh), black_box(&view_proj), slice);
                }
            });
        });
    });
}

criterion_group!(
    benches,
    bench_rasterize_single_chunk,
    bench_framebuffer_clear,
    bench_framebuffer_set_pixel,
    bench_framebuffer_set_pixel_unchecked,
    bench_rasterize_world_3x3x3,
    bench_vertex_decompression,
    bench_parallel_rendering_single_threaded,
    bench_parallel_rendering_multi_threaded
);
criterion_main!(benches);
