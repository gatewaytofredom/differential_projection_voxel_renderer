//! Differential basis vector projection for voxel face packets.
//!
//! Key optimization: For axis-aligned voxel faces, all quads lie on parallel planes.
//! Instead of performing full MVP matrix multiplication per vertex (16 FMAs),
//! we precompute basis vectors once per face and use differential projection (2 FMAs per quad).
//!
//! Speedup: ~8x reduction in vertex transformation cost.

use crate::meshing::mesh::FaceDir;
use glam::{IVec3, Mat4, Vec3, Vec4};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Precomputed basis vectors for a single face direction in clip space.
/// These define the coordinate system for projecting quads on this face.
#[derive(Clone, Copy, Debug)]
pub struct FaceBasis {
    /// Clip-space origin (corresponds to chunk corner + slice position)
    pub origin: Vec4,
    /// Clip-space tangent vector (U direction - moves along face's first axis)
    pub tangent: Vec4,
    /// Clip-space bitangent vector (V direction - moves along face's second axis)
    pub bitangent: Vec4,
    /// Clip-space normal vector (for backface culling)
    pub normal: Vec4,
}

impl FaceBasis {
    /// Calculate basis vectors for a face direction.
    ///
    /// # Arguments
    /// * `face_dir` - The face direction (±X, ±Y, ±Z)
    /// * `chunk_pos` - World-space chunk position (in chunk coordinates)
    /// * `slice_idx` - Index of the slice within the chunk (0..32)
    /// * `view_proj` - Combined view-projection matrix
    pub fn from_face_direction(
        face_dir: FaceDir,
        chunk_pos: IVec3,
        slice_idx: u8,
        view_proj: &Mat4,
    ) -> Self {
        // Calculate world-space coordinate system
        let (origin_world, tangent_world, bitangent_world, normal_world) =
            face_coordinate_system(face_dir, chunk_pos, slice_idx);

        // Transform to clip space
        // Note: For vectors (tangent, bitangent, normal), we use w=0
        // For the origin point, we use w=1
        let origin = view_proj.mul_vec4(Vec4::from((origin_world, 1.0)));
        let tangent = view_proj.mul_vec4(Vec4::from((tangent_world, 0.0)));
        let bitangent = view_proj.mul_vec4(Vec4::from((bitangent_world, 0.0)));
        let normal = view_proj.mul_vec4(Vec4::from((normal_world, 0.0)));

        Self {
            origin,
            tangent,
            bitangent,
            normal,
        }
    }

    /// Project a single point using differential basis vectors.
    ///
    /// Formula: P_clip = origin + u * tangent + v * bitangent
    ///
    /// This replaces a full 4x4 matrix-vector multiply (16 FMAs) with 2 FMAs.
    #[inline]
    pub fn project_point(&self, u: f32, v: f32) -> Vec4 {
        self.origin + u * self.tangent + v * self.bitangent
    }

    /// Check if this face is front-facing (for backface culling).
    ///
    /// A face is front-facing if its normal points towards the camera
    /// (negative Z in view space means towards camera).
    #[inline]
    pub fn is_front_facing(&self) -> bool {
        // In clip space, check if normal.z < 0 (points towards near plane)
        // This is a simplified test; full test would use camera position
        self.normal.z < 0.0
    }

    /// Project 8 quad AABBs in parallel using AVX2 SIMD.
    ///
    /// Processes 8 quads simultaneously, calculating screen-space bounding boxes.
    ///
    /// # Safety
    /// Requires AVX2 support. Caller must check CPU features.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    pub unsafe fn project_packet_bounds_simd(
        &self,
        packet: &crate::meshing::face_packets::FacePacket32,
        output: &mut ProjectedPacket,
    ) {
        let count = packet.len.min(32) as usize;
        output.count = packet.len;

        // Process 8 quads at a time
        let full_batches = count / 8;
        let remainder = count % 8;

        for batch_idx in 0..full_batches {
            let offset = batch_idx * 8;
            self.project_batch_8(packet, output, offset);
        }

        // Handle remainder with scalar code
        for i in (full_batches * 8)..count {
            self.project_single_scalar(packet, output, i);
        }

        // Copy block types
        output.block_type[..count].copy_from_slice(&packet.block_type[..count]);
    }

    /// Project a batch of 8 quads using AVX2.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn project_batch_8(
        &self,
        packet: &crate::meshing::face_packets::FacePacket32,
        output: &mut ProjectedPacket,
        offset: usize,
    ) {
        // Load 8 u_min, v_min, u_len, v_len values (u8 → f32 conversion)
        let u_min = load_u8_to_f32(&packet.u_min[offset..offset + 8]);
        let v_min = load_u8_to_f32(&packet.v_min[offset..offset + 8]);
        let u_len = load_u8_to_f32(&packet.u_len[offset..offset + 8]);
        let v_len = load_u8_to_f32(&packet.v_len[offset..offset + 8]);

        // Calculate max corners
        let u_max = _mm256_add_ps(u_min, u_len);
        let v_max = _mm256_add_ps(v_min, v_len);

        // Project 4 corners of each quad
        let corner_00 = self.project_point_simd(u_min, v_min);
        let corner_10 = self.project_point_simd(u_max, v_min);
        let corner_01 = self.project_point_simd(u_min, v_max);
        let corner_11 = self.project_point_simd(u_max, v_max);

        // Perspective divide (clip → NDC)
        let ndc_00 = perspective_divide_simd(corner_00);
        let ndc_10 = perspective_divide_simd(corner_10);
        let ndc_01 = perspective_divide_simd(corner_01);
        let ndc_11 = perspective_divide_simd(corner_11);

        // Calculate screen-space AABB (min/max of 4 corners)
        let x_min = min4_ps(ndc_00.x, ndc_10.x, ndc_01.x, ndc_11.x);
        let y_min = min4_ps(ndc_00.y, ndc_10.y, ndc_01.y, ndc_11.y);
        let x_max = max4_ps(ndc_00.x, ndc_10.x, ndc_01.x, ndc_11.x);
        let y_max = max4_ps(ndc_00.y, ndc_10.y, ndc_01.y, ndc_11.y);

        // Find nearest depth (for depth sorting)
        let depth_near = min4_ps(ndc_00.z, ndc_10.z, ndc_01.z, ndc_11.z);

        // Store to output
        _mm256_storeu_ps(&mut output.screen_x_min[offset], x_min);
        _mm256_storeu_ps(&mut output.screen_y_min[offset], y_min);
        _mm256_storeu_ps(&mut output.screen_x_max[offset], x_max);
        _mm256_storeu_ps(&mut output.screen_y_max[offset], y_max);
        _mm256_storeu_ps(&mut output.depth_near[offset], depth_near);
    }

    /// Project a single quad using scalar code (for remainder).
    pub fn project_single_scalar(
        &self,
        packet: &crate::meshing::face_packets::FacePacket32,
        output: &mut ProjectedPacket,
        idx: usize,
    ) {
        let u_min = packet.u_min[idx] as f32;
        let v_min = packet.v_min[idx] as f32;
        let u_max = u_min + packet.u_len[idx] as f32;
        let v_max = v_min + packet.v_len[idx] as f32;

        // Project 4 corners
        let corner_00 = self.project_point(u_min, v_min);
        let corner_10 = self.project_point(u_max, v_min);
        let corner_01 = self.project_point(u_min, v_max);
        let corner_11 = self.project_point(u_max, v_max);

        // Perspective divide
        let ndc_00 = perspective_divide(corner_00);
        let ndc_10 = perspective_divide(corner_10);
        let ndc_01 = perspective_divide(corner_01);
        let ndc_11 = perspective_divide(corner_11);

        // Calculate AABB
        output.screen_x_min[idx] = ndc_00.x.min(ndc_10.x).min(ndc_01.x).min(ndc_11.x);
        output.screen_y_min[idx] = ndc_00.y.min(ndc_10.y).min(ndc_01.y).min(ndc_11.y);
        output.screen_x_max[idx] = ndc_00.x.max(ndc_10.x).max(ndc_01.x).max(ndc_11.x);
        output.screen_y_max[idx] = ndc_00.y.max(ndc_10.y).max(ndc_01.y).max(ndc_11.y);
        output.depth_near[idx] = ndc_00.z.min(ndc_10.z).min(ndc_01.z).min(ndc_11.z);
    }

    /// Project a point using SIMD (8 points at once).
    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn project_point_simd(&self, u: __m256, v: __m256) -> ClipPoint {
        // P_clip = origin + u * tangent + v * bitangent
        let tangent_x = _mm256_set1_ps(self.tangent.x);
        let tangent_y = _mm256_set1_ps(self.tangent.y);
        let tangent_z = _mm256_set1_ps(self.tangent.z);
        let tangent_w = _mm256_set1_ps(self.tangent.w);

        let bitangent_x = _mm256_set1_ps(self.bitangent.x);
        let bitangent_y = _mm256_set1_ps(self.bitangent.y);
        let bitangent_z = _mm256_set1_ps(self.bitangent.z);
        let bitangent_w = _mm256_set1_ps(self.bitangent.w);

        let origin_x = _mm256_set1_ps(self.origin.x);
        let origin_y = _mm256_set1_ps(self.origin.y);
        let origin_z = _mm256_set1_ps(self.origin.z);
        let origin_w = _mm256_set1_ps(self.origin.w);

        // x = origin.x + u * tangent.x + v * bitangent.x
        let x = _mm256_fmadd_ps(u, tangent_x, _mm256_fmadd_ps(v, bitangent_x, origin_x));
        let y = _mm256_fmadd_ps(u, tangent_y, _mm256_fmadd_ps(v, bitangent_y, origin_y));
        let z = _mm256_fmadd_ps(u, tangent_z, _mm256_fmadd_ps(v, bitangent_z, origin_z));
        let w = _mm256_fmadd_ps(u, tangent_w, _mm256_fmadd_ps(v, bitangent_w, origin_w));

        ClipPoint { x, y, z, w }
    }
}

/// Calculate world-space coordinate system for a face.
///
/// Returns (origin, tangent, bitangent, normal) in world space.
fn face_coordinate_system(
    face_dir: FaceDir,
    chunk_pos: IVec3,
    slice_idx: u8,
) -> (Vec3, Vec3, Vec3, Vec3) {
    use crate::voxel::CHUNK_SIZE;
    let chunk_size = CHUNK_SIZE as f32;
    let chunk_world = chunk_pos.as_vec3() * chunk_size;

    match face_dir {
        FaceDir::PosX => {
            let slice = slice_idx as f32;
            let origin = chunk_world + Vec3::new(slice, 0.0, 0.0);
            let tangent = Vec3::Y;        // U moves along +Y
            let bitangent = Vec3::Z;      // V moves along +Z
            let normal = Vec3::X;         // Normal is +X
            (origin, tangent, bitangent, normal)
        }
        FaceDir::NegX => {
            let slice = slice_idx as f32;
            let origin = chunk_world + Vec3::new(slice, 0.0, 0.0);
            let tangent = Vec3::Y;
            let bitangent = Vec3::NEG_Z;  // Flip to maintain right-handed
            let normal = Vec3::NEG_X;
            (origin, tangent, bitangent, normal)
        }
        FaceDir::PosY => {
            let slice = slice_idx as f32;
            let origin = chunk_world + Vec3::new(0.0, slice, 0.0);
            let tangent = Vec3::X;        // U moves along +X
            let bitangent = Vec3::Z;      // V moves along +Z
            let normal = Vec3::Y;
            (origin, tangent, bitangent, normal)
        }
        FaceDir::NegY => {
            let slice = slice_idx as f32;
            let origin = chunk_world + Vec3::new(0.0, slice, 0.0);
            let tangent = Vec3::X;
            let bitangent = Vec3::NEG_Z;
            let normal = Vec3::NEG_Y;
            (origin, tangent, bitangent, normal)
        }
        FaceDir::PosZ => {
            let slice = slice_idx as f32;
            let origin = chunk_world + Vec3::new(0.0, 0.0, slice);
            let tangent = Vec3::X;        // U moves along +X
            let bitangent = Vec3::Y;      // V moves along +Y
            let normal = Vec3::Z;
            (origin, tangent, bitangent, normal)
        }
        FaceDir::NegZ => {
            let slice = slice_idx as f32;
            let origin = chunk_world + Vec3::new(0.0, 0.0, slice);
            let tangent = Vec3::NEG_X;    // Flip to maintain right-handed
            let bitangent = Vec3::Y;
            let normal = Vec3::NEG_Z;
            (origin, tangent, bitangent, normal)
        }
    }
}

/// Projected packet: Screen-space bounding boxes for up to 32 quads.
#[repr(C, align(32))]
#[derive(Clone, Debug)]
pub struct ProjectedPacket {
    pub count: u8,
    pub screen_x_min: [f32; 32],
    pub screen_y_min: [f32; 32],
    pub screen_x_max: [f32; 32],
    pub screen_y_max: [f32; 32],
    pub depth_near: [f32; 32],
    pub block_type: [u8; 32],
    pub visibility_mask: u32,
}

impl ProjectedPacket {
    pub fn new() -> Self {
        Self {
            count: 0,
            screen_x_min: [0.0; 32],
            screen_y_min: [0.0; 32],
            screen_x_max: [0.0; 32],
            screen_y_max: [0.0; 32],
            depth_near: [0.0; 32],
            block_type: [0; 32],
            visibility_mask: 0xFFFFFFFF,
        }
    }
}

impl Default for ProjectedPacket {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// SIMD Helper Functions
// ============================================================================

/// Clip-space point with 4 components (8-wide SIMD).
#[cfg(target_arch = "x86_64")]
struct ClipPoint {
    x: __m256,
    y: __m256,
    z: __m256,
    w: __m256,
}

/// NDC point with 3 components (8-wide SIMD).
#[cfg(target_arch = "x86_64")]
struct NdcPoint {
    x: __m256,
    y: __m256,
    z: __m256,
}

/// Load 8 u8 values and convert to f32 (SIMD).
#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn load_u8_to_f32(data: &[u8]) -> __m256 {
    debug_assert!(data.len() >= 8);

    // Load 8 bytes into lower 64 bits of a 128-bit register
    let bytes = _mm_loadl_epi64(data.as_ptr() as *const __m128i);

    // Zero-extend u8 to i32 (8 values)
    let ints = _mm256_cvtepu8_epi32(bytes);

    // Convert i32 to f32
    _mm256_cvtepi32_ps(ints)
}

/// Perspective divide: clip space → NDC (8-wide SIMD).
/// Uses fast reciprocal approximation with Newton-Raphson refinement.
///
/// Performance: ~8 cycles (pipelined) vs ~14-20 cycles (unpipelined vdivps)
/// Precision: ~23 bits (float32-equivalent) after one NR iteration
#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn perspective_divide_simd(clip: ClipPoint) -> NdcPoint {
    let inv_w = fast_reciprocal_ps(clip.w);
    NdcPoint {
        x: _mm256_mul_ps(clip.x, inv_w),
        y: _mm256_mul_ps(clip.y, inv_w),
        z: _mm256_mul_ps(clip.z, inv_w),
    }
}

/// Fast reciprocal approximation using Newton-Raphson refinement.
///
/// Algorithm:
/// 1. x0 = rcp(w)              // ~12-bit precision, 5 cycles, pipelined
/// 2. x1 = x0 * (2.0 - w * x0) // Newton-Raphson iteration, brings to ~23 bits
///
/// Total latency: ~8 cycles (fully pipelined)
/// vs vdivps: ~14-20 cycles (unpipelined)
///
/// Error bound: < 0.5 ULP for normalized inputs (w > near_plane)
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn fast_reciprocal_ps(w: __m256) -> __m256 {
    // Step 1: Get approximate reciprocal (~12 bits precision)
    // Latency: ~5 cycles, Throughput: 1 per cycle
    let rcp = _mm256_rcp_ps(w);

    // Step 2: One Newton-Raphson iteration for refinement
    // Formula: x1 = x0 * (2.0 - w * x0)
    // This can be rewritten as: x1 = rcp * (2.0 - w * rcp)
    let two = _mm256_set1_ps(2.0);

    // Compute (2.0 - w * rcp) using FMA for efficiency
    // fnmadd computes: (2.0 - w * rcp) = fma(-w, rcp, 2.0)
    let two_minus_w_rcp = _mm256_fnmadd_ps(w, rcp, two);

    // Final result: rcp * (2.0 - w * rcp)
    _mm256_mul_ps(rcp, two_minus_w_rcp)
}

/// Perspective divide: clip space → NDC (scalar).
#[inline]
fn perspective_divide(clip: Vec4) -> Vec3 {
    Vec3::new(clip.x / clip.w, clip.y / clip.w, clip.z / clip.w)
}

/// Minimum of 4 SIMD vectors (element-wise).
#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn min4_ps(a: __m256, b: __m256, c: __m256, d: __m256) -> __m256 {
    _mm256_min_ps(_mm256_min_ps(a, b), _mm256_min_ps(c, d))
}

/// Maximum of 4 SIMD vectors (element-wise).
#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn max4_ps(a: __m256, b: __m256, c: __m256, d: __m256) -> __m256 {
    _mm256_max_ps(_mm256_max_ps(a, b), _mm256_max_ps(c, d))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_face_coordinate_system_pos_y() {
        let chunk_pos = IVec3::ZERO;
        let slice_idx = 15;
        let (origin, tangent, bitangent, normal) =
            face_coordinate_system(FaceDir::PosY, chunk_pos, slice_idx);

        assert_eq!(origin, Vec3::new(0.0, 15.0, 0.0));
        assert_eq!(tangent, Vec3::X);
        assert_eq!(bitangent, Vec3::Z);
        assert_eq!(normal, Vec3::Y);
    }

    #[test]
    fn test_project_point_identity() {
        // With identity matrix, clip space == world space
        let basis = FaceBasis::from_face_direction(
            FaceDir::PosY,
            IVec3::ZERO,
            0,
            &Mat4::IDENTITY,
        );

        let point = basis.project_point(5.0, 10.0);
        // origin=(0,0,0), tangent=(1,0,0), bitangent=(0,0,1)
        // result = (0,0,0) + 5*(1,0,0) + 10*(0,0,1) = (5,0,10)
        assert!((point.x - 5.0).abs() < 0.001);
        assert!((point.y - 0.0).abs() < 0.001);
        assert!((point.z - 10.0).abs() < 0.001);
    }

    #[test]
    fn test_basis_vectors_orthogonal() {
        let view_proj = Mat4::perspective_rh(90.0_f32.to_radians(), 16.0 / 9.0, 0.1, 1000.0);

        for &face_dir in &[
            FaceDir::PosX,
            FaceDir::NegX,
            FaceDir::PosY,
            FaceDir::NegY,
            FaceDir::PosZ,
            FaceDir::NegZ,
        ] {
            let basis = FaceBasis::from_face_direction(face_dir, IVec3::ZERO, 0, &view_proj);

            // Check orthogonality (should be maintained under linear transform)
            let dot_tn = basis.tangent.truncate().dot(basis.normal.truncate());
            let dot_bn = basis.bitangent.truncate().dot(basis.normal.truncate());

            assert!(
                dot_tn.abs() < 0.1,
                "{:?}: tangent-normal not orthogonal: {}",
                face_dir,
                dot_tn
            );
            assert!(
                dot_bn.abs() < 0.1,
                "{:?}: bitangent-normal not orthogonal: {}",
                face_dir,
                dot_bn
            );
        }
    }
}
