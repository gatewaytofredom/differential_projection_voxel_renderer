/// Benchmark suite for meshing algorithms
/// Tests performance of binary greedy meshing across different scenarios
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use glam::IVec3;
use voxel_engine::{BinaryGreedyMesher, BlockData, BlockType, Chunk, CHUNK_SIZE};

fn bench_mesh_uniform_air(c: &mut Criterion) {
    c.bench_function("mesh_uniform_air", |b| {
        let chunk = Chunk::uniform(IVec3::ZERO, BlockType::Air);
        b.iter(|| BinaryGreedyMesher::mesh_chunk(black_box(&chunk)));
    });
}

fn bench_mesh_uniform_solid(c: &mut Criterion) {
    c.bench_function("mesh_uniform_solid", |b| {
        let chunk = Chunk::uniform(IVec3::ZERO, BlockType::Stone);
        b.iter(|| BinaryGreedyMesher::mesh_chunk(black_box(&chunk)));
    });
}

fn bench_mesh_terrain(c: &mut Criterion) {
    c.bench_function("mesh_terrain", |b| {
        let chunk = Chunk::generate_terrain(IVec3::ZERO);
        b.iter(|| BinaryGreedyMesher::mesh_chunk(black_box(&chunk)));
    });
}

fn bench_mesh_dense_solid(c: &mut Criterion) {
    c.bench_function("mesh_dense_solid", |b| {
        // Chunk that is almost entirely solid, exercising worst-case meshing.
        let mut chunk = Chunk::uniform(IVec3::ZERO, BlockType::Air);
        for z in 0..CHUNK_SIZE {
            for y in 0..CHUNK_SIZE {
                for x in 0..CHUNK_SIZE {
                    chunk.set_block(x, y, z, BlockData::new(BlockType::Stone));
                }
            }
        }

        b.iter(|| BinaryGreedyMesher::mesh_chunk(black_box(&chunk)));
    });
}

fn bench_greedy_slice_dense(c: &mut Criterion) {
    c.bench_function("greedy_slice_dense", |b| {
        let mut data = [0u32; CHUNK_SIZE];
        for row in 0..CHUNK_SIZE {
            data[row] = !0u32;
        }

        b.iter(|| {
            black_box(BinaryGreedyMesher::greedy_mesh_slice(black_box(&data)));
        });
    });
}

fn bench_mesh_multiple_chunks(c: &mut Criterion) {
    let mut group = c.benchmark_group("mesh_multiple_chunks");

    for size in [1, 3, 5, 10].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let mut chunks = Vec::new();
            for x in 0..size {
                for y in 0..size {
                    for z in 0..size {
                        chunks.push(Chunk::generate_terrain(IVec3::new(x, y, z)));
                    }
                }
            }

            let chunk_refs: Vec<&Chunk> = chunks.iter().collect();

            b.iter(|| {
                BinaryGreedyMesher::mesh_world(black_box(&chunk_refs));
            });
        });
    }
    group.finish();
}

fn bench_mesh_chunk_in_world_center(c: &mut Criterion) {
    c.bench_function("mesh_chunk_in_world_center_3x3x3", |b| {
        // World with full neighbour information; measure a single center chunk.
        let mut chunks = Vec::new();
        for cx in -1..=1 {
            for cy in -1..=1 {
                for cz in -1..=1 {
                    chunks.push(Chunk::generate_terrain(IVec3::new(cx, cy, cz)));
                }
            }
        }

        let mut center_index = 0;
        for (i, chunk) in chunks.iter().enumerate() {
            if chunk.position == IVec3::ZERO {
                center_index = i;
                break;
            }
        }

        let chunk_refs: Vec<&Chunk> = chunks.iter().collect();
        let center_chunk = &chunks[center_index];

        b.iter(|| {
            black_box(BinaryGreedyMesher::mesh_chunk_in_world(
                black_box(center_chunk),
                black_box(&chunk_refs),
            ));
        });
    });
}

criterion_group!(
    benches,
    bench_mesh_uniform_air,
    bench_mesh_uniform_solid,
    bench_mesh_terrain,
    bench_mesh_dense_solid,
    bench_greedy_slice_dense,
    bench_mesh_multiple_chunks,
    bench_mesh_chunk_in_world_center
);
criterion_main!(benches);
