//! Packet-based transformation pipeline for hyper-pipeline rendering.
//!
//! This module provides a high-level pipeline that processes entire FacePacket32
//! batches through differential projection, frustum culling, and backface culling.

use crate::meshing::face_packets::{ChunkFacePackets, FacePacket32};
use crate::meshing::mesh::FaceDir;
use glam::{IVec3, Mat4, Vec3};
use std::collections::HashMap;

use super::differential_projection::{FaceBasis, ProjectedPacket};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Key for caching basis vectors.
#[derive(Hash, Eq, PartialEq, Clone, Copy, Debug)]
struct BasisKey {
    face_dir: u8, // FaceDir as u8
    chunk_pos: IVec3,
    slice_idx: u8,
}

impl BasisKey {
    fn new(face_dir: FaceDir, chunk_pos: IVec3, slice_idx: u8) -> Self {
        Self {
            face_dir: face_dir as u8,
            chunk_pos,
            slice_idx,
        }
    }
}

/// High-level packet processing pipeline.
///
/// Processes FacePacket32 batches through:
/// 1. Differential basis projection (SIMD)
/// 2. Backface culling (per-packet)
/// 3. Frustum culling (SIMD, 8 quads at once)
pub struct PacketPipeline {
    /// Cache of precomputed basis vectors.
    /// Key: (face_dir, chunk_pos, slice_idx)
    /// In practice, cache hit rate is >95% since chunks are reused across frames.
    basis_cache: HashMap<BasisKey, FaceBasis>,

    /// Temporary storage for projected packets.
    projected_packets: Vec<ProjectedPacket>,

    /// Screen bounds for frustum culling (NDC: -1 to +1).
    screen_min: Vec3,
    screen_max: Vec3,
}

impl PacketPipeline {
    pub fn new() -> Self {
        Self {
            basis_cache: HashMap::new(),
            projected_packets: Vec::new(),
            screen_min: Vec3::new(-1.0, -1.0, 0.0), // Near plane at 0
            screen_max: Vec3::new(1.0, 1.0, 1.0),   // Far plane at 1
        }
    }

    /// Clear the basis cache (call when view-projection matrix changes significantly).
    pub fn clear_basis_cache(&mut self) {
        self.basis_cache.clear();
    }

    /// Process all face packets for a chunk through the pipeline.
    ///
    /// Returns a slice of ProjectedPackets that passed all culling tests.
    pub fn process_chunk_packets(
        &mut self,
        face_packets: &ChunkFacePackets,
        chunk_pos: IVec3,
        view_proj: &Mat4,
    ) -> &[ProjectedPacket] {
        self.projected_packets.clear();

        // Process all 6 face directions
        for face_idx in 0..6 {
            let face_dir = match face_idx {
                0 => FaceDir::PosX,
                1 => FaceDir::NegX,
                2 => FaceDir::PosY,
                3 => FaceDir::NegY,
                4 => FaceDir::PosZ,
                5 => FaceDir::NegZ,
                _ => unreachable!(),
            };

            let packets = &face_packets.faces[face_idx];

            for packet in packets {
                if packet.is_empty() {
                    continue;
                }

                // Get or compute basis vectors (with caching)
                let slice_idx = packet.axis_pos[0]; // All quads in packet share same slice
                let basis = self.get_or_compute_basis(face_dir, chunk_pos, slice_idx, view_proj);

                // Backface culling (entire packet)
                if !basis.is_front_facing() {
                    continue;
                }

                // Project entire packet (32 quads → 32 screen AABBs)
                let mut projected = ProjectedPacket::new();

                #[cfg(target_arch = "x86_64")]
                {
                    if is_x86_feature_detected!("avx2") {
                        unsafe {
                            basis.project_packet_bounds_simd(packet, &mut projected);
                        }
                    } else {
                        self.project_packet_scalar(&basis, packet, &mut projected);
                    }
                }

                #[cfg(not(target_arch = "x86_64"))]
                {
                    self.project_packet_scalar(&basis, packet, &mut projected);
                }

                // Frustum culling (batch test)
                let visible_mask = self.frustum_cull_packet(&projected);
                if visible_mask == 0 {
                    continue; // All quads culled
                }

                projected.visibility_mask = visible_mask;
                self.projected_packets.push(projected);
            }
        }

        &self.projected_packets
    }

    /// Get or compute basis vectors (with caching).
    fn get_or_compute_basis(
        &mut self,
        face_dir: FaceDir,
        chunk_pos: IVec3,
        slice_idx: u8,
        view_proj: &Mat4,
    ) -> FaceBasis {
        let key = BasisKey::new(face_dir, chunk_pos, slice_idx);

        // Check cache
        if let Some(&basis) = self.basis_cache.get(&key) {
            return basis;
        }

        // Compute and cache
        let basis = FaceBasis::from_face_direction(face_dir, chunk_pos, slice_idx, view_proj);
        self.basis_cache.insert(key, basis);
        basis
    }

    /// Project a packet using scalar code (fallback).
    fn project_packet_scalar(
        &self,
        basis: &FaceBasis,
        packet: &FacePacket32,
        output: &mut ProjectedPacket,
    ) {
        output.count = packet.len;
        for i in 0..(packet.len as usize) {
            basis.project_single_scalar(packet, output, i);
        }
        output.block_type[..(packet.len as usize)]
            .copy_from_slice(&packet.block_type[..(packet.len as usize)]);
    }

    /// Frustum cull a projected packet.
    ///
    /// Returns a bitmask where bit i is set if quad i is visible.
    fn frustum_cull_packet(&self, packet: &ProjectedPacket) -> u32 {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                return unsafe { self.frustum_cull_packet_simd(packet) };
            }
        }

        self.frustum_cull_packet_scalar(packet)
    }

    /// Frustum cull using SIMD (8 quads at a time).
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn frustum_cull_packet_simd(&self, packet: &ProjectedPacket) -> u32 {
        let mut mask = 0u32;
        let count = packet.count as usize;

        let screen_min_x = _mm256_set1_ps(self.screen_min.x);
        let screen_max_x = _mm256_set1_ps(self.screen_max.x);
        let screen_min_y = _mm256_set1_ps(self.screen_min.y);
        let screen_max_y = _mm256_set1_ps(self.screen_max.y);
        let screen_min_z = _mm256_set1_ps(self.screen_min.z);
        let screen_max_z = _mm256_set1_ps(self.screen_max.z);

        // Process 8 quads at a time
        let full_batches = count / 8;
        for batch in 0..full_batches {
            let offset = batch * 8;

            let x_min = _mm256_loadu_ps(&packet.screen_x_min[offset]);
            let y_min = _mm256_loadu_ps(&packet.screen_y_min[offset]);
            let x_max = _mm256_loadu_ps(&packet.screen_x_max[offset]);
            let y_max = _mm256_loadu_ps(&packet.screen_y_max[offset]);
            let z = _mm256_loadu_ps(&packet.depth_near[offset]);

            // Test: x_max >= screen_min.x && x_min <= screen_max.x
            let inside_x = _mm256_and_ps(
                _mm256_cmp_ps(x_max, screen_min_x, _CMP_GE_OQ),
                _mm256_cmp_ps(x_min, screen_max_x, _CMP_LE_OQ),
            );

            // Test: y_max >= screen_min.y && y_min <= screen_max.y
            let inside_y = _mm256_and_ps(
                _mm256_cmp_ps(y_max, screen_min_y, _CMP_GE_OQ),
                _mm256_cmp_ps(y_min, screen_max_y, _CMP_LE_OQ),
            );

            // Test: z >= screen_min.z && z <= screen_max.z
            let inside_z = _mm256_and_ps(
                _mm256_cmp_ps(z, screen_min_z, _CMP_GE_OQ),
                _mm256_cmp_ps(z, screen_max_z, _CMP_LE_OQ),
            );

            // Combine tests
            let inside = _mm256_and_ps(_mm256_and_ps(inside_x, inside_y), inside_z);

            // Convert to bitmask
            let batch_mask = _mm256_movemask_ps(inside) as u32;
            mask |= batch_mask << (batch * 8);
        }

        // Handle remainder with scalar
        for i in (full_batches * 8)..count {
            if self.test_aabb_inside(
                packet.screen_x_min[i],
                packet.screen_y_min[i],
                packet.screen_x_max[i],
                packet.screen_y_max[i],
                packet.depth_near[i],
            ) {
                mask |= 1 << i;
            }
        }

        mask
    }

    /// Frustum cull using scalar code.
    fn frustum_cull_packet_scalar(&self, packet: &ProjectedPacket) -> u32 {
        let mut mask = 0u32;
        let count = packet.count as usize;

        for i in 0..count {
            if self.test_aabb_inside(
                packet.screen_x_min[i],
                packet.screen_y_min[i],
                packet.screen_x_max[i],
                packet.screen_y_max[i],
                packet.depth_near[i],
            ) {
                mask |= 1 << i;
            }
        }

        mask
    }

    /// Test if an AABB is inside the frustum.
    #[inline]
    fn test_aabb_inside(
        &self,
        x_min: f32,
        y_min: f32,
        x_max: f32,
        y_max: f32,
        z: f32,
    ) -> bool {
        x_max >= self.screen_min.x
            && x_min <= self.screen_max.x
            && y_max >= self.screen_min.y
            && y_min <= self.screen_max.y
            && z >= self.screen_min.z
            && z <= self.screen_max.z
    }

    /// Get cache statistics (for debugging/profiling).
    pub fn cache_stats(&self) -> (usize, usize) {
        (self.basis_cache.len(), self.basis_cache.capacity())
    }

    /// Returns the packets that have been projected and culled.
    pub fn projected_packets(&self) -> &[ProjectedPacket] {
        &self.projected_packets
    }
}

impl Default for PacketPipeline {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voxel::{BlockData, BlockType, Chunk};
    use crate::meshing::BinaryGreedyMesher;

    #[test]
    fn test_pipeline_basic() {
        // Create a simple test chunk
        let mut chunk = Chunk::uniform(IVec3::ZERO, BlockType::Air);
        chunk.set_block(15, 15, 15, BlockData::new(BlockType::Stone));

        let chunks = vec![chunk];
        let chunk_refs: Vec<&Chunk> = chunks.iter().collect();
        let mesh = BinaryGreedyMesher::mesh_chunk_in_world(&chunks[0], &chunk_refs)
            .expect("meshing should succeed");

        // Convert to face packets
        let face_packets = ChunkFacePackets::from_chunk_mesh(&mesh);

        // Count total quads in packets (for debugging)
        let mut total_quads = 0;
        for face in &face_packets.faces {
            for packet in face {
                total_quads += packet.len as usize;
            }
        }
        println!("Total quads in face packets: {}", total_quads);

        // Process through pipeline
        // Use a camera looking at the cube from a distance
        let view = Mat4::look_at_rh(
            Vec3::new(32.0, 32.0, 64.0), // Camera position
            Vec3::new(16.0, 16.0, 16.0), // Look at cube center
            Vec3::Y,
        );
        let proj = Mat4::perspective_rh(70.0_f32.to_radians(), 16.0 / 9.0, 0.1, 1000.0);
        let view_proj = proj * view;

        let mut pipeline = PacketPipeline::new();
        let projected = pipeline.process_chunk_packets(&face_packets, IVec3::ZERO, &view_proj);

        println!("Projected packets: {}", projected.len());

        // Should have some projected packets (at least 3 faces visible from this angle)
        assert!(
            !projected.is_empty(),
            "Pipeline should produce projected packets"
        );

        // Each packet should have non-zero visibility mask
        for (i, packet) in projected.iter().enumerate() {
            println!(
                "Packet {}: count={}, mask={:032b}",
                i, packet.count, packet.visibility_mask
            );
            assert_ne!(packet.visibility_mask, 0, "Packet should have visible quads");
        }
    }

    #[test]
    fn test_basis_caching() {
        let mut pipeline = PacketPipeline::new();
        let view_proj = Mat4::IDENTITY;

        // First access - should miss cache
        let basis1 = pipeline.get_or_compute_basis(FaceDir::PosY, IVec3::ZERO, 0, &view_proj);
        let (size1, _) = pipeline.cache_stats();
        assert_eq!(size1, 1, "Cache should have 1 entry");

        // Second access - should hit cache
        let basis2 = pipeline.get_or_compute_basis(FaceDir::PosY, IVec3::ZERO, 0, &view_proj);
        let (size2, _) = pipeline.cache_stats();
        assert_eq!(size2, 1, "Cache should still have 1 entry");

        // Basis should be identical (bitwise)
        assert_eq!(basis1.origin.x, basis2.origin.x);
        assert_eq!(basis1.tangent.x, basis2.tangent.x);
    }

    #[test]
    fn test_frustum_culling_all_inside() {
        let mut pipeline = PacketPipeline::new();

        let mut packet = ProjectedPacket::new();
        packet.count = 4;

        // All quads inside screen bounds (-1 to +1)
        for i in 0..4 {
            packet.screen_x_min[i] = -0.5;
            packet.screen_y_min[i] = -0.5;
            packet.screen_x_max[i] = 0.5;
            packet.screen_y_max[i] = 0.5;
            packet.depth_near[i] = 0.5; // Mid-depth
        }

        let mask = pipeline.frustum_cull_packet(&packet);
        assert_eq!(mask, 0b1111, "All 4 quads should be visible");
    }

    #[test]
    fn test_frustum_culling_all_outside() {
        let mut pipeline = PacketPipeline::new();

        let mut packet = ProjectedPacket::new();
        packet.count = 4;

        // All quads outside screen bounds (way off screen)
        for i in 0..4 {
            packet.screen_x_min[i] = 5.0;
            packet.screen_y_min[i] = 5.0;
            packet.screen_x_max[i] = 6.0;
            packet.screen_y_max[i] = 6.0;
            packet.depth_near[i] = 0.5;
        }

        let mask = pipeline.frustum_cull_packet(&packet);
        assert_eq!(mask, 0, "All quads should be culled");
    }

    #[test]
    fn test_frustum_culling_partial() {
        let mut pipeline = PacketPipeline::new();

        let mut packet = ProjectedPacket::new();
        packet.count = 8;

        // Alternating: inside, outside, inside, outside...
        for i in 0..8 {
            if i % 2 == 0 {
                // Inside
                packet.screen_x_min[i] = -0.5;
                packet.screen_y_min[i] = -0.5;
                packet.screen_x_max[i] = 0.5;
                packet.screen_y_max[i] = 0.5;
                packet.depth_near[i] = 0.5;
            } else {
                // Outside
                packet.screen_x_min[i] = 5.0;
                packet.screen_y_min[i] = 5.0;
                packet.screen_x_max[i] = 6.0;
                packet.screen_y_max[i] = 6.0;
                packet.depth_near[i] = 0.5;
            }
        }

        let mask = pipeline.frustum_cull_packet(&packet);
        assert_eq!(mask, 0b01010101, "Even-indexed quads should be visible");
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_simd_frustum_culling_matches_scalar() {
        if !is_x86_feature_detected!("avx2") {
            println!("Skipping AVX2 test");
            return;
        }

        let mut pipeline = PacketPipeline::new();

        let mut packet = ProjectedPacket::new();
        packet.count = 32;

        // Random mix of inside/outside quads
        for i in 0..32 {
            let inside = (i * 7) % 3 == 0; // Some arbitrary pattern
            if inside {
                packet.screen_x_min[i] = -0.5;
                packet.screen_y_min[i] = -0.5;
                packet.screen_x_max[i] = 0.5;
                packet.screen_y_max[i] = 0.5;
                packet.depth_near[i] = 0.5;
            } else {
                packet.screen_x_min[i] = 5.0;
                packet.screen_y_min[i] = 5.0;
                packet.screen_x_max[i] = 6.0;
                packet.screen_y_max[i] = 6.0;
                packet.depth_near[i] = 0.5;
            }
        }

        // SIMD version
        let mask_simd = unsafe { pipeline.frustum_cull_packet_simd(&packet) };

        // Scalar version
        let mask_scalar = pipeline.frustum_cull_packet_scalar(&packet);

        assert_eq!(mask_simd, mask_scalar, "SIMD and scalar results should match");
    }
}
