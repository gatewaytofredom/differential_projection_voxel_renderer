/// Benchmark suite for world management and large-scale rendering
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use glam::{IVec3, Vec3};
use voxel_engine::{
    BinaryGreedyMesher, Camera, Framebuffer, Rasterizer, World, WorldConfig, CHUNK_SIZE,
};

fn bench_world_generation(c: &mut Criterion) {
    let mut group = c.benchmark_group("world_generation");

    for &size in &[5, 10, 15] {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| {
                let mut world = World::new(WorldConfig::default());
                let half = size / 2;
                world.generate_region(
                    IVec3::new(-half, -1, -half),
                    IVec3::new(half, 1, half),
                );
                black_box(world.chunk_count())
            });
        });
    }
    group.finish();
}

fn bench_view_distance_update(c: &mut Criterion) {
    c.bench_function("world_update_view_distance", |b| {
        let mut world = World::new(WorldConfig {
            view_distance: 8,
            ..Default::default()
        });

        // Pre-generate a large region
        world.generate_region(IVec3::new(-15, -3, -15), IVec3::new(15, 3, 15));

        b.iter(|| {
            // Simulate camera movement
            let camera_pos = Vec3::new(100.0, 50.0, 100.0);
            black_box(world.update(camera_pos))
        });
    });
}

fn bench_large_world_meshing(c: &mut Criterion) {
    let mut group = c.benchmark_group("large_world_meshing");

    for &view_dist in &[4, 8, 12] {
        group.bench_with_input(
            BenchmarkId::from_parameter(view_dist),
            &view_dist,
            |b, &view_dist| {
                let mut world = World::new(WorldConfig {
                    view_distance: view_dist,
                    ..Default::default()
                });

                // Generate region
                world.generate_region(
                    IVec3::new(-view_dist, -2, -view_dist),
                    IVec3::new(view_dist, 2, view_dist),
                );

                b.iter(|| {
                    let visible_chunks = world.get_visible_chunks(Vec3::ZERO);
                    black_box(BinaryGreedyMesher::mesh_world(&visible_chunks))
                });
            },
        );
    }
    group.finish();
}

fn bench_large_world_rendering(c: &mut Criterion) {
    c.bench_function("render_large_world_16x5x16", |b| {
        let mut world = World::new(WorldConfig::default());
        world.generate_region(IVec3::new(-8, -2, -8), IVec3::new(8, 2, 8));

        let visible_chunks = world.get_visible_chunks(Vec3::ZERO);
        let meshes = BinaryGreedyMesher::mesh_world(&visible_chunks);

        let mut framebuffer = Framebuffer::new(1280, 720);
        let camera = Camera::new(
            Vec3::new(
                (CHUNK_SIZE as f32) * 2.0,
                (CHUNK_SIZE as f32) * 1.0,
                (CHUNK_SIZE as f32) * 3.0,
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

fn bench_world_chunk_visibility(c: &mut Criterion) {
    c.bench_function("world_get_visible_chunks", |b| {
        let mut world = World::new(WorldConfig {
            view_distance: 12,
            ..Default::default()
        });

        // Generate large world
        world.generate_region(IVec3::new(-20, -5, -20), IVec3::new(20, 5, 20));

        b.iter(|| {
            let camera_pos = Vec3::new(50.0, 25.0, 50.0);
            black_box(world.get_visible_chunks(camera_pos))
        });
    });
}

criterion_group!(
    benches,
    bench_world_generation,
    bench_view_distance_update,
    bench_large_world_meshing,
    bench_large_world_rendering,
    bench_world_chunk_visibility,
);
criterion_main!(benches);
