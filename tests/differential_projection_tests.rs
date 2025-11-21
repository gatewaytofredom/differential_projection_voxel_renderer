use voxel_engine::*;
use glam::{IVec3, Mat4, Vec3, Vec4};
use voxel_engine::meshing::{BinaryGreedyMesher, mesh::FaceDir, face_packets::FacePacket32, face_packets::ChunkFacePackets};
use voxel_engine::rendering::differential_projection::{FaceBasis, ProjectedPacket};

/// Helper: Generate a random view-projection matrix
fn random_view_proj_matrix(seed: u64) -> Mat4 {
    use std::hash::{Hash, Hasher};
    use std::collections::hash_map::DefaultHasher;

    let mut hasher = DefaultHasher::new();
    seed.hash(&mut hasher);
    let hash = hasher.finish();

    let yaw = ((hash & 0xFF) as f32 / 255.0) * 2.0 * std::f32::consts::PI;
    let pitch = (((hash >> 8) & 0xFF) as f32 / 255.0 - 0.5) * std::f32::consts::PI * 0.9;
    let distance = 50.0 + ((hash >> 16) & 0xFF) as f32;

    let position = Vec3::new(
        distance * yaw.cos() * pitch.cos(),
        distance * pitch.sin(),
        distance * yaw.sin() * pitch.cos(),
    );

    let view = Mat4::look_at_rh(position, Vec3::ZERO, Vec3::Y);
    let proj = Mat4::perspective_rh(70.0_f32.to_radians(), 16.0 / 9.0, 0.1, 1000.0);

    proj * view
}

/// Helper: Full matrix multiply for a quad vertex (reference implementation)
/// This must match the coordinate systems defined in differential_projection.rs face_coordinate_system
fn full_mvp_multiply(
    face_dir: FaceDir,
    chunk_pos: IVec3,
    slice_idx: u8,
    u: u8,
    v: u8,
    view_proj: &Mat4,
) -> Vec4 {
    let chunk_size = 32.0;
    let chunk_world = chunk_pos.as_vec3() * chunk_size;
    let slice = slice_idx as f32;

    // Calculate world position based on face direction
    // Must match face_coordinate_system: origin + u*tangent + v*bitangent
    let world_pos = match face_dir {
        FaceDir::PosX => {
            // origin = (slice, 0, 0), tangent = Y, bitangent = Z
            chunk_world + Vec3::new(slice, u as f32, v as f32)
        }
        FaceDir::NegX => {
            // origin = (slice, 0, 0), tangent = Y, bitangent = NEG_Z
            chunk_world + Vec3::new(slice, u as f32, -(v as f32))
        }
        FaceDir::PosY => {
            // origin = (0, slice, 0), tangent = X, bitangent = Z
            chunk_world + Vec3::new(u as f32, slice, v as f32)
        }
        FaceDir::NegY => {
            // origin = (0, slice, 0), tangent = X, bitangent = NEG_Z
            chunk_world + Vec3::new(u as f32, slice, -(v as f32))
        }
        FaceDir::PosZ => {
            // origin = (0, 0, slice), tangent = X, bitangent = Y
            chunk_world + Vec3::new(u as f32, v as f32, slice)
        }
        FaceDir::NegZ => {
            // origin = (0, 0, slice), tangent = NEG_X, bitangent = Y
            chunk_world + Vec3::new(-(u as f32), v as f32, slice)
        }
    };

    view_proj.mul_vec4(Vec4::from((world_pos, 1.0)))
}

#[test]
fn test_basis_vectors_match_full_transform() {
    // Test: Differential basis projection should match full MVP multiply
    let chunk_pos = IVec3::new(5, 10, -3);
    let view_proj = random_view_proj_matrix(12345);
    let face_dir = FaceDir::PosY;
    let slice_idx = 15;

    // Calculate basis vectors
    let basis = FaceBasis::from_face_direction(face_dir, chunk_pos, slice_idx, &view_proj);

    // Test 100 random (u, v) coordinates
    for i in 0..100 {
        let u = ((i * 7) % 32) as u8;
        let v = ((i * 13) % 32) as u8;

        // Method 1: Differential basis (fast path)
        let pos_diff = basis.project_point(u as f32, v as f32);

        // Method 2: Full matrix multiply (reference)
        let pos_ref = full_mvp_multiply(face_dir, chunk_pos, slice_idx, u, v, &view_proj);

        // Assert: Results match within floating-point tolerance
        let tolerance = 0.001;
        assert!(
            (pos_diff.x - pos_ref.x).abs() < tolerance,
            "X mismatch at ({}, {}): {} vs {}",
            u,
            v,
            pos_diff.x,
            pos_ref.x
        );
        assert!(
            (pos_diff.y - pos_ref.y).abs() < tolerance,
            "Y mismatch at ({}, {}): {} vs {}",
            u,
            v,
            pos_diff.y,
            pos_ref.y
        );
        assert!(
            (pos_diff.z - pos_ref.z).abs() < tolerance,
            "Z mismatch at ({}, {}): {} vs {}",
            u,
            v,
            pos_diff.z,
            pos_ref.z
        );
        assert!(
            (pos_diff.w - pos_ref.w).abs() < tolerance,
            "W mismatch at ({}, {}): {} vs {}",
            u,
            v,
            pos_diff.w,
            pos_ref.w
        );
    }
}

#[test]
fn test_all_face_directions() {
    let view_proj = Mat4::perspective_rh(90.0_f32.to_radians(), 16.0 / 9.0, 0.1, 1000.0);
    let chunk_pos = IVec3::ZERO;

    for &face_dir in &[
        FaceDir::PosX,
        FaceDir::NegX,
        FaceDir::PosY,
        FaceDir::NegY,
        FaceDir::PosZ,
        FaceDir::NegZ,
    ] {
        let basis = FaceBasis::from_face_direction(face_dir, chunk_pos, 0, &view_proj);

        // Test several points
        for u in [0, 10, 20, 31] {
            for v in [0, 10, 20, 31] {
                let pos_diff = basis.project_point(u as f32, v as f32);
                let pos_ref = full_mvp_multiply(face_dir, chunk_pos, 0, u as u8, v as u8, &view_proj);

                let tolerance = 0.01;
                if !((pos_diff.x - pos_ref.x).abs() < tolerance
                    && (pos_diff.y - pos_ref.y).abs() < tolerance
                    && (pos_diff.z - pos_ref.z).abs() < tolerance
                    && (pos_diff.w - pos_ref.w).abs() < tolerance)
                {
                    println!("Face {:?} at ({}, {})", face_dir, u, v);
                    println!("  Differential: {:?}", pos_diff);
                    println!("  Reference:    {:?}", pos_ref);
                    println!("  Diff: ({}, {}, {}, {})",
                        pos_diff.x - pos_ref.x,
                        pos_diff.y - pos_ref.y,
                        pos_diff.z - pos_ref.z,
                        pos_diff.w - pos_ref.w);
                    panic!("Face {:?} at ({}, {}) mismatch", face_dir, u, v);
                }
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[test]
fn test_simd_projection_matches_scalar() {
    if !is_x86_feature_detected!("avx2") {
        println!("Skipping AVX2 test (not supported on this CPU)");
        return;
    }

    let basis = FaceBasis::from_face_direction(
        FaceDir::PosY,
        IVec3::ZERO,
        0,
        &Mat4::IDENTITY,
    );

    let mut packet = FacePacket32::new();
    // Fill with test data (8 quads)
    for i in 0..8 {
        packet.u_min[i] = (i * 3) as u8;
        packet.v_min[i] = (i * 2) as u8;
        packet.u_len[i] = 4;
        packet.v_len[i] = 5;
        packet.block_type[i] = 1;
    }
    packet.len = 8;

    // Project with SIMD
    let mut projected_simd = ProjectedPacket::new();
    unsafe {
        basis.project_packet_bounds_simd(&packet, &mut projected_simd);
    }

    // Project with scalar (reference)
    let mut projected_scalar = ProjectedPacket::new();
    projected_scalar.count = packet.len;
    for i in 0..8 {
        basis.project_single_scalar(&packet, &mut projected_scalar, i);
    }
    projected_scalar.block_type[..8].copy_from_slice(&packet.block_type[..8]);

    // Compare results
    let tolerance = 0.001;
    for i in 0..8 {
        assert!(
            (projected_simd.screen_x_min[i] - projected_scalar.screen_x_min[i]).abs() < tolerance,
            "x_min[{}] mismatch: {} vs {}",
            i,
            projected_simd.screen_x_min[i],
            projected_scalar.screen_x_min[i]
        );
        assert!(
            (projected_simd.screen_y_min[i] - projected_scalar.screen_y_min[i]).abs() < tolerance,
            "y_min[{}] mismatch: {} vs {}",
            i,
            projected_simd.screen_y_min[i],
            projected_scalar.screen_y_min[i]
        );
        assert!(
            (projected_simd.screen_x_max[i] - projected_scalar.screen_x_max[i]).abs() < tolerance,
            "x_max[{}] mismatch",
            i
        );
        assert!(
            (projected_simd.screen_y_max[i] - projected_scalar.screen_y_max[i]).abs() < tolerance,
            "y_max[{}] mismatch",
            i
        );
        assert!(
            (projected_simd.depth_near[i] - projected_scalar.depth_near[i]).abs() < tolerance,
            "depth[{}] mismatch",
            i
        );
    }
}

#[cfg(target_arch = "x86_64")]
#[test]
fn test_simd_projection_with_realistic_view() {
    if !is_x86_feature_detected!("avx2") {
        println!("Skipping AVX2 test (not supported on this CPU)");
        return;
    }

    // Realistic camera setup
    let view = Mat4::look_at_rh(
        Vec3::new(64.0, 50.0, 100.0),
        Vec3::new(64.0, 32.0, 64.0),
        Vec3::Y,
    );
    let proj = Mat4::perspective_rh(70.0_f32.to_radians(), 16.0 / 9.0, 0.1, 1000.0);
    let view_proj = proj * view;

    let chunk_pos = IVec3::new(2, 1, 2);
    let basis = FaceBasis::from_face_direction(FaceDir::PosY, chunk_pos, 16, &view_proj);

    let mut packet = FacePacket32::new();
    for i in 0..32 {
        packet.u_min[i] = (i % 32) as u8;
        packet.v_min[i] = (i / 4) as u8;
        packet.u_len[i] = 1 + (i % 3) as u8;
        packet.v_len[i] = 1 + (i % 4) as u8;
        packet.block_type[i] = 1;
    }
    packet.len = 32;

    // Project with SIMD
    let mut projected_simd = ProjectedPacket::new();
    unsafe {
        basis.project_packet_bounds_simd(&packet, &mut projected_simd);
    }

    // Project with scalar
    let mut projected_scalar = ProjectedPacket::new();
    projected_scalar.count = packet.len;
    for i in 0..32 {
        basis.project_single_scalar(&packet, &mut projected_scalar, i);
    }
    projected_scalar.block_type[..32].copy_from_slice(&packet.block_type[..32]);

    // Compare
    let tolerance = 0.001;
    for i in 0..32 {
        assert!(
            (projected_simd.screen_x_min[i] - projected_scalar.screen_x_min[i]).abs() < tolerance,
            "Packet [{}] x_min mismatch",
            i
        );
        assert!(
            (projected_simd.screen_y_min[i] - projected_scalar.screen_y_min[i]).abs() < tolerance,
            "Packet [{}] y_min mismatch",
            i
        );
    }
}

#[test]
fn test_projection_with_chunk_mesh() {
    // Test with actual chunk mesh data
    let mut chunk = Chunk::uniform(IVec3::new(2, 1, 3), BlockType::Air);

    // Create a simple pattern
    for y in 0..5 {
        for x in 0..5 {
            chunk.set_block(x, y, 5, BlockData::new(BlockType::Stone));
        }
    }

    let chunks = vec![chunk];
    let chunk_refs: Vec<&Chunk> = chunks.iter().collect();
    let mesh = BinaryGreedyMesher::mesh_chunk_in_world(&chunks[0], &chunk_refs)
        .expect("meshing should succeed");

    // Convert to face packets
    let face_packets = ChunkFacePackets::from_chunk_mesh(&mesh);

    // Test projection for one face direction
    let view_proj = Mat4::perspective_rh(70.0_f32.to_radians(), 16.0 / 9.0, 0.1, 1000.0);

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

            let slice_idx = packet.axis_pos[0]; // First quad's slice index
            let basis = FaceBasis::from_face_direction(
                face_dir,
                mesh.chunk_position,
                slice_idx,
                &view_proj,
            );

            let mut projected = ProjectedPacket::new();

            #[cfg(target_arch = "x86_64")]
            {
                if is_x86_feature_detected!("avx2") {
                    unsafe {
                        basis.project_packet_bounds_simd(packet, &mut projected);
                    }
                } else {
                    // Scalar fallback
                    projected.count = packet.len;
                    for i in 0..(packet.len as usize) {
                        basis.project_single_scalar(packet, &mut projected, i);
                    }
                }
            }

            #[cfg(not(target_arch = "x86_64"))]
            {
                projected.count = packet.len;
                for i in 0..(packet.len as usize) {
                    basis.project_single_scalar(packet, &mut projected, i);
                }
            }

            // Verify projected bounds are reasonable
            for i in 0..(packet.len as usize) {
                assert!(
                    projected.screen_x_min[i] <= projected.screen_x_max[i],
                    "Invalid X bounds"
                );
                assert!(
                    projected.screen_y_min[i] <= projected.screen_y_max[i],
                    "Invalid Y bounds"
                );
                // NDC should be in range [-1, 1] (or slightly outside due to perspective)
                assert!(
                    projected.screen_x_min[i].abs() < 10.0,
                    "X coordinate out of reasonable range: {}",
                    projected.screen_x_min[i]
                );
            }
        }
    }
}

#[test]
fn test_backface_culling() {
    // Camera looking down -Z
    let camera_pos = Vec3::new(0.0, 0.0, 10.0);
    let view = Mat4::look_at_rh(camera_pos, Vec3::ZERO, Vec3::Y);
    let proj = Mat4::perspective_rh(70.0_f32.to_radians(), 16.0 / 9.0, 0.1, 1000.0);
    let view_proj = proj * view;

    let chunk_pos = IVec3::ZERO;

    // Front-facing (+Z face, towards camera)
    let basis_front = FaceBasis::from_face_direction(FaceDir::PosZ, chunk_pos, 0, &view_proj);

    // Back-facing (-Z face, away from camera)
    let basis_back = FaceBasis::from_face_direction(FaceDir::NegZ, chunk_pos, 0, &view_proj);

    // Note: The backface culling test may need adjustment based on the actual
    // implementation. For now, we just verify the normal direction makes sense.
    println!("Front normal.z: {}", basis_front.normal.z);
    println!("Back normal.z: {}", basis_back.normal.z);

    // The sign of normal.z should differ between front and back faces
    assert_ne!(
        basis_front.normal.z.signum(),
        basis_back.normal.z.signum(),
        "Front and back face normals should have opposite Z signs"
    );
}

#[test]
fn test_projection_multiple_slices() {
    // Test that different slices produce different origins
    let view_proj = Mat4::IDENTITY;
    let chunk_pos = IVec3::ZERO;

    let basis_slice_0 = FaceBasis::from_face_direction(FaceDir::PosY, chunk_pos, 0, &view_proj);
    let basis_slice_15 = FaceBasis::from_face_direction(FaceDir::PosY, chunk_pos, 15, &view_proj);
    let basis_slice_31 = FaceBasis::from_face_direction(FaceDir::PosY, chunk_pos, 31, &view_proj);

    // Origins should differ by slice index along the Y axis (for PosY face)
    assert!((basis_slice_0.origin.y - 0.0).abs() < 0.001);
    assert!((basis_slice_15.origin.y - 15.0).abs() < 0.001);
    assert!((basis_slice_31.origin.y - 31.0).abs() < 0.001);

    // Tangent and bitangent should be the same
    assert_eq!(basis_slice_0.tangent, basis_slice_15.tangent);
    assert_eq!(basis_slice_0.bitangent, basis_slice_31.bitangent);
}
