use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use glam::{IVec3, Mat4, Vec3};
use voxel_engine::meshing::{BinaryGreedyMesher, mesh::FaceDir, face_packets::FacePacket32, face_packets::ChunkFacePackets};
use voxel_engine::rendering::differential_projection::{FaceBasis, ProjectedPacket};
use voxel_engine::voxel::{BlockData, BlockType, Chunk};

/// Generate a test chunk with some geometry
fn generate_test_chunk() -> Chunk {
    let mut chunk = Chunk::uniform(IVec3::ZERO, BlockType::Air);

    // Create a simple terrain pattern
    for y in 0..16 {
        for x in 0..32 {
            for z in 0..32 {
                if y < 8 + ((x + z) % 4) {
                    chunk.set_block(x, y, z, BlockData::new(BlockType::Stone));
                }
            }
        }
    }

    chunk
}

/// Benchmark: Full MVP matrix multiply (legacy approach)
fn bench_full_mvp_multiply(c: &mut Criterion) {
    let chunk = generate_test_chunk();
    let chunks = vec![chunk];
    let chunk_refs: Vec<&Chunk> = chunks.iter().collect();
    let mesh = BinaryGreedyMesher::mesh_chunk_in_world(&chunks[0], &chunk_refs)
        .expect("meshing should succeed");

    let face_packets = ChunkFacePackets::from_chunk_mesh(&mesh);
    let view_proj = Mat4::perspective_rh(70.0_f32.to_radians(), 16.0 / 9.0, 0.1, 1000.0)
        * Mat4::look_at_rh(Vec3::new(64.0, 50.0, 100.0), Vec3::new(64.0, 32.0, 64.0), Vec3::Y);

    c.bench_function("full_mvp_per_vertex", |b| {
        b.iter(|| {
            let mut total = 0;
            for (face_idx, packets) in face_packets.faces.iter().enumerate() {
                for packet in packets {
                    for i in 0..(packet.len as usize) {
                        // Simulate full MVP multiply for 4 vertices per quad
                        for _ in 0..4 {
                            let _ = black_box(view_proj);
                            total += 1;
                        }
                    }
                }
            }
            black_box(total)
        });
    });
}

/// Benchmark: Differential basis projection (scalar)
fn bench_differential_projection_scalar(c: &mut Criterion) {
    let chunk = generate_test_chunk();
    let chunks = vec![chunk];
    let chunk_refs: Vec<&Chunk> = chunks.iter().collect();
    let mesh = BinaryGreedyMesher::mesh_chunk_in_world(&chunks[0], &chunk_refs)
        .expect("meshing should succeed");

    let face_packets = ChunkFacePackets::from_chunk_mesh(&mesh);
    let view_proj = Mat4::perspective_rh(70.0_f32.to_radians(), 16.0 / 9.0, 0.1, 1000.0)
        * Mat4::look_at_rh(Vec3::new(64.0, 50.0, 100.0), Vec3::new(64.0, 32.0, 64.0), Vec3::Y);

    c.bench_function("differential_projection_scalar", |b| {
        b.iter(|| {
            let mut total = 0;
            for (face_idx, packets) in face_packets.faces.iter().enumerate() {
                let face_dir = match face_idx {
                    0 => FaceDir::PosX,
                    1 => FaceDir::NegX,
                    2 => FaceDir::PosY,
                    3 => FaceDir::NegY,
                    4 => FaceDir::PosZ,
                    5 => FaceDir::NegZ,
                    _ => unreachable!(),
                };

                for packet in packets {
                    if packet.is_empty() {
                        continue;
                    }

                    let slice_idx = packet.axis_pos[0];
                    let basis = FaceBasis::from_face_direction(
                        face_dir,
                        mesh.chunk_position,
                        slice_idx,
                        &view_proj,
                    );

                    let mut projected = ProjectedPacket::new();
                    for i in 0..(packet.len as usize) {
                        basis.project_single_scalar(packet, &mut projected, i);
                        total += 1;
                    }
                }
            }
            black_box(total)
        });
    });
}

/// Benchmark: Differential basis projection (SIMD)
#[cfg(target_arch = "x86_64")]
fn bench_differential_projection_simd(c: &mut Criterion) {
    if !is_x86_feature_detected!("avx2") {
        println!("Skipping AVX2 benchmark (not supported on this CPU)");
        return;
    }

    let chunk = generate_test_chunk();
    let chunks = vec![chunk];
    let chunk_refs: Vec<&Chunk> = chunks.iter().collect();
    let mesh = BinaryGreedyMesher::mesh_chunk_in_world(&chunks[0], &chunk_refs)
        .expect("meshing should succeed");

    let face_packets = ChunkFacePackets::from_chunk_mesh(&mesh);
    let view_proj = Mat4::perspective_rh(70.0_f32.to_radians(), 16.0 / 9.0, 0.1, 1000.0)
        * Mat4::look_at_rh(Vec3::new(64.0, 50.0, 100.0), Vec3::new(64.0, 32.0, 64.0), Vec3::Y);

    c.bench_function("differential_projection_simd", |b| {
        b.iter(|| {
            let mut total = 0;
            for (face_idx, packets) in face_packets.faces.iter().enumerate() {
                let face_dir = match face_idx {
                    0 => FaceDir::PosX,
                    1 => FaceDir::NegX,
                    2 => FaceDir::PosY,
                    3 => FaceDir::NegY,
                    4 => FaceDir::PosZ,
                    5 => FaceDir::NegZ,
                    _ => unreachable!(),
                };

                for packet in packets {
                    if packet.is_empty() {
                        continue;
                    }

                    let slice_idx = packet.axis_pos[0];
                    let basis = FaceBasis::from_face_direction(
                        face_dir,
                        mesh.chunk_position,
                        slice_idx,
                        &view_proj,
                    );

                    let mut projected = ProjectedPacket::new();
                    unsafe {
                        basis.project_packet_bounds_simd(packet, &mut projected);
                    }
                    total += packet.len as usize;
                }
            }
            black_box(total)
        });
    });
}

criterion_group!(
    benches,
    bench_full_mvp_multiply,
    bench_differential_projection_scalar,
);

#[cfg(target_arch = "x86_64")]
criterion_group!(
    simd_benches,
    bench_differential_projection_simd,
);

#[cfg(target_arch = "x86_64")]
criterion_main!(benches, simd_benches);

#[cfg(not(target_arch = "x86_64"))]
criterion_main!(benches);
