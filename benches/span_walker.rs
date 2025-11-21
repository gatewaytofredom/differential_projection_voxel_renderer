use criterion::{black_box, criterion_group, criterion_main, Criterion};
use voxel_engine::rendering::differential_projection::ProjectedPacket;
use voxel_engine::rendering::framebuffer::Framebuffer;
use voxel_engine::rendering::span_walker::SpanWalkerRasterizer;

/// Benchmark span walker rasterization of a single quad
fn bench_span_walker_single_quad(c: &mut Criterion) {
    let width = 1920;
    let height = 1080;
    let mut fb = Framebuffer::new(width, height);

    // Create a projected packet with a single quad
    let mut projected = ProjectedPacket::new();
    projected.count = 1;
    projected.screen_x_min[0] = -0.5;
    projected.screen_y_min[0] = -0.5;
    projected.screen_x_max[0] = 0.5;
    projected.screen_y_max[0] = 0.5;
    projected.depth_near[0] = 0.5;
    projected.block_type[0] = 1;
    projected.visibility_mask = 1;

    let span_walker = SpanWalkerRasterizer::new(width as u32, height as u32);

    c.bench_function("span_walker_single_quad", |b| {
        b.iter(|| {
            let mut slice = fb.as_full_slice_mut();
            span_walker.rasterize_projected_packet(
                black_box(&projected),
                black_box(&mut slice),
            );
        })
    });
}

/// Benchmark span walker with a full packet (32 quads)
fn bench_span_walker_full_packet(c: &mut Criterion) {
    let width = 1920;
    let height = 1080;
    let mut fb = Framebuffer::new(width, height);

    // Create a projected packet with 32 quads in a grid
    let mut projected = ProjectedPacket::new();
    projected.count = 32;

    // Create a 4x8 grid of quads
    for i in 0..32 {
        let row = i / 8;
        let col = i % 8;

        // Map to NDC space [-0.9, 0.9]
        let x_min = -0.9 + (col as f32) * 0.225;
        let y_min = -0.9 + (row as f32) * 0.45;
        let x_max = x_min + 0.2;
        let y_max = y_min + 0.4;

        projected.screen_x_min[i as usize] = x_min;
        projected.screen_y_min[i as usize] = y_min;
        projected.screen_x_max[i as usize] = x_max;
        projected.screen_y_max[i as usize] = y_max;
        projected.depth_near[i as usize] = 0.5;
        projected.block_type[i as usize] = ((i % 4) + 1) as u8;
    }

    projected.visibility_mask = 0xFFFFFFFF; // All visible

    let span_walker = SpanWalkerRasterizer::new(width as u32, height as u32);

    c.bench_function("span_walker_full_packet_32_quads", |b| {
        b.iter(|| {
            let mut slice = fb.as_full_slice_mut();
            span_walker.rasterize_projected_packet(
                black_box(&projected),
                black_box(&mut slice),
            );
        })
    });
}

/// Benchmark span walker with partially visible packet (culled)
fn bench_span_walker_culled_packet(c: &mut Criterion) {
    let width = 1920;
    let height = 1080;
    let mut fb = Framebuffer::new(width, height);

    // Create a projected packet with 32 quads, but only 8 visible
    let mut projected = ProjectedPacket::new();
    projected.count = 32;

    for i in 0..32 {
        projected.screen_x_min[i as usize] = -0.5;
        projected.screen_y_min[i as usize] = -0.5;
        projected.screen_x_max[i as usize] = 0.5;
        projected.screen_y_max[i as usize] = 0.5;
        projected.depth_near[i as usize] = 0.5;
        projected.block_type[i as usize] = 1;
    }

    // Only first 8 quads visible
    projected.visibility_mask = 0x000000FF;

    let span_walker = SpanWalkerRasterizer::new(width as u32, height as u32);

    c.bench_function("span_walker_culled_packet_8_of_32_visible", |b| {
        b.iter(|| {
            let mut slice = fb.as_full_slice_mut();
            span_walker.rasterize_projected_packet(
                black_box(&projected),
                black_box(&mut slice),
            );
        })
    });
}

/// Benchmark span walker with small quads (high overhead)
fn bench_span_walker_tiny_quads(c: &mut Criterion) {
    let width = 1920;
    let height = 1080;
    let mut fb = Framebuffer::new(width, height);

    // Create a projected packet with 32 tiny quads (5x5 pixels each)
    let mut projected = ProjectedPacket::new();
    projected.count = 32;

    for i in 0..32 {
        let x = (i % 16) as f32 / 16.0 * 1.8 - 0.9;
        let y = (i / 16) as f32 / 16.0 * 1.8 - 0.9;

        projected.screen_x_min[i as usize] = x;
        projected.screen_y_min[i as usize] = y;
        projected.screen_x_max[i as usize] = x + 0.01; // ~10 pixels
        projected.screen_y_max[i as usize] = y + 0.01;
        projected.depth_near[i as usize] = 0.5;
        projected.block_type[i as usize] = ((i % 4) + 1) as u8;
    }

    projected.visibility_mask = 0xFFFFFFFF;

    let span_walker = SpanWalkerRasterizer::new(width as u32, height as u32);

    c.bench_function("span_walker_tiny_quads_32x10px", |b| {
        b.iter(|| {
            let mut slice = fb.as_full_slice_mut();
            span_walker.rasterize_projected_packet(
                black_box(&projected),
                black_box(&mut slice),
            );
        })
    });
}

/// Benchmark span walker with large quads (low overhead)
fn bench_span_walker_large_quads(c: &mut Criterion) {
    let width = 1920;
    let height = 1080;
    let mut fb = Framebuffer::new(width, height);

    // Create a projected packet with 4 large quads (each 1/4 of screen)
    let mut projected = ProjectedPacket::new();
    projected.count = 4;

    // Top-left
    projected.screen_x_min[0] = -0.95;
    projected.screen_y_min[0] = -0.95;
    projected.screen_x_max[0] = 0.0;
    projected.screen_y_max[0] = 0.0;

    // Top-right
    projected.screen_x_min[1] = 0.0;
    projected.screen_y_min[1] = -0.95;
    projected.screen_x_max[1] = 0.95;
    projected.screen_y_max[1] = 0.0;

    // Bottom-left
    projected.screen_x_min[2] = -0.95;
    projected.screen_y_min[2] = 0.0;
    projected.screen_x_max[2] = 0.0;
    projected.screen_y_max[2] = 0.95;

    // Bottom-right
    projected.screen_x_min[3] = 0.0;
    projected.screen_y_min[3] = 0.0;
    projected.screen_x_max[3] = 0.95;
    projected.screen_y_max[3] = 0.95;

    for i in 0..4 {
        projected.depth_near[i] = 0.5;
        projected.block_type[i] = (i + 1) as u8;
    }

    projected.visibility_mask = 0x0F; // First 4 visible

    let span_walker = SpanWalkerRasterizer::new(width as u32, height as u32);

    c.bench_function("span_walker_large_quads_4x500kpx", |b| {
        b.iter(|| {
            let mut slice = fb.as_full_slice_mut();
            span_walker.rasterize_projected_packet(
                black_box(&projected),
                black_box(&mut slice),
            );
        })
    });
}

criterion_group!(
    benches,
    bench_span_walker_single_quad,
    bench_span_walker_full_packet,
    bench_span_walker_culled_packet,
    bench_span_walker_tiny_quads,
    bench_span_walker_large_quads,
);
criterion_main!(benches);
