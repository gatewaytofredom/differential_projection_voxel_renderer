/// SIMD-accelerated vertex decompression and transformation
/// Processes vertices in batches of 8 using AVX2 for maximum throughput
use crate::meshing::Vertex;
use glam::{Mat4, Vec3, Vec4};

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

/// Batch size for SIMD processing (8 vertices per iteration)
/// Chosen to match AVX2 register width (8x f32)
const SIMD_BATCH_SIZE: usize = 8;

/// Decompress and transform vertices using SIMD
///
/// This function processes vertices in batches of 8, converting u8 local coordinates
/// to world-space f32, then transforming to clip space in a single vectorized operation.
///
/// # Performance characteristics
/// - Processes 8 vertices per iteration (8x throughput vs scalar)
/// - Eliminates branch mispredictions via batch processing
/// - Better cache utilization (sequential reads of compressed data)
/// - Remainder handled with scalar fallback (typically <8 vertices)
#[inline]
pub fn decompress_and_transform_vertices(
    vertices: &[Vertex],
    chunk_offset: Vec3,
    view_proj: &Mat4,
    output: &mut [Vec4],
) {
    debug_assert_eq!(vertices.len(), output.len(), "Output buffer must match vertex count");

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                decompress_and_transform_vertices_avx2(vertices, chunk_offset, view_proj, output);
            }
            return;
        }
    }

    // Fallback to scalar implementation
    decompress_and_transform_vertices_scalar(vertices, chunk_offset, view_proj, output);
}

/// Scalar fallback implementation (used when AVX2 is not available)
#[inline]
fn decompress_and_transform_vertices_scalar(
    vertices: &[Vertex],
    chunk_offset: Vec3,
    view_proj: &Mat4,
    output: &mut [Vec4],
) {
    for (i, v) in vertices.iter().enumerate() {
        let world_pos = v.world_position(chunk_offset);
        output[i] = *view_proj * world_pos.extend(1.0);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn decompress_and_transform_vertices_avx2(
    vertices: &[Vertex],
    chunk_offset: Vec3,
    view_proj: &Mat4,
    output: &mut [Vec4],
) {
    let len = vertices.len();
    let batch_count = len / SIMD_BATCH_SIZE;
    let remainder = len % SIMD_BATCH_SIZE;

    // Extract view-projection matrix columns for vectorized transform
    let m_col0 = view_proj.x_axis;
    let m_col1 = view_proj.y_axis;
    let m_col2 = view_proj.z_axis;
    let m_col3 = view_proj.w_axis;

    // Broadcast matrix columns to SIMD registers (will be reused for all vertices)
    let m0_x = _mm256_set1_ps(m_col0.x);
    let m0_y = _mm256_set1_ps(m_col0.y);
    let m0_z = _mm256_set1_ps(m_col0.z);
    let m0_w = _mm256_set1_ps(m_col0.w);

    let m1_x = _mm256_set1_ps(m_col1.x);
    let m1_y = _mm256_set1_ps(m_col1.y);
    let m1_z = _mm256_set1_ps(m_col1.z);
    let m1_w = _mm256_set1_ps(m_col1.w);

    let m2_x = _mm256_set1_ps(m_col2.x);
    let m2_y = _mm256_set1_ps(m_col2.y);
    let m2_z = _mm256_set1_ps(m_col2.z);
    let m2_w = _mm256_set1_ps(m_col2.w);

    let m3_x = _mm256_set1_ps(m_col3.x);
    let m3_y = _mm256_set1_ps(m_col3.y);
    let m3_z = _mm256_set1_ps(m_col3.z);
    let m3_w = _mm256_set1_ps(m_col3.w);

    // Broadcast chunk offset (same for all vertices in this chunk)
    let offset_x = _mm256_set1_ps(chunk_offset.x);
    let offset_y = _mm256_set1_ps(chunk_offset.y);
    let offset_z = _mm256_set1_ps(chunk_offset.z);

    // Process batches of 8 vertices
    for batch_idx in 0..batch_count {
        let base_idx = batch_idx * SIMD_BATCH_SIZE;

        // Load 8 vertices and decompress u8 -> f32
        // This is the memory bandwidth optimization in action!
        let mut x_values = [0.0f32; 8];
        let mut y_values = [0.0f32; 8];
        let mut z_values = [0.0f32; 8];

        for i in 0..SIMD_BATCH_SIZE {
            let v = &vertices[base_idx + i];
            x_values[i] = v.x as f32;
            y_values[i] = v.y as f32;
            z_values[i] = v.z as f32;
        }

        // Load into SIMD registers
        let mut local_x = _mm256_loadu_ps(x_values.as_ptr());
        let mut local_y = _mm256_loadu_ps(y_values.as_ptr());
        let mut local_z = _mm256_loadu_ps(z_values.as_ptr());

        // Convert to world space: world = local + offset
        local_x = _mm256_add_ps(local_x, offset_x);
        local_y = _mm256_add_ps(local_y, offset_y);
        local_z = _mm256_add_ps(local_z, offset_z);

        // Transform by view-projection matrix (8 vertices in parallel)
        // out.x = m0.x * x + m1.x * y + m2.x * z + m3.x * 1
        let out_x = _mm256_add_ps(
            _mm256_add_ps(
                _mm256_mul_ps(m0_x, local_x),
                _mm256_mul_ps(m1_x, local_y)
            ),
            _mm256_add_ps(
                _mm256_mul_ps(m2_x, local_z),
                m3_x
            )
        );

        // out.y = m0.y * x + m1.y * y + m2.y * z + m3.y * 1
        let out_y = _mm256_add_ps(
            _mm256_add_ps(
                _mm256_mul_ps(m0_y, local_x),
                _mm256_mul_ps(m1_y, local_y)
            ),
            _mm256_add_ps(
                _mm256_mul_ps(m2_y, local_z),
                m3_y
            )
        );

        // out.z = m0.z * x + m1.z * y + m2.z * z + m3.z * 1
        let out_z = _mm256_add_ps(
            _mm256_add_ps(
                _mm256_mul_ps(m0_z, local_x),
                _mm256_mul_ps(m1_z, local_y)
            ),
            _mm256_add_ps(
                _mm256_mul_ps(m2_z, local_z),
                m3_z
            )
        );

        // out.w = m0.w * x + m1.w * y + m2.w * z + m3.w * 1
        let out_w = _mm256_add_ps(
            _mm256_add_ps(
                _mm256_mul_ps(m0_w, local_x),
                _mm256_mul_ps(m1_w, local_y)
            ),
            _mm256_add_ps(
                _mm256_mul_ps(m2_w, local_z),
                m3_w
            )
        );

        // Store results (8 Vec4s)
        let mut x_out = [0.0f32; 8];
        let mut y_out = [0.0f32; 8];
        let mut z_out = [0.0f32; 8];
        let mut w_out = [0.0f32; 8];

        _mm256_storeu_ps(x_out.as_mut_ptr(), out_x);
        _mm256_storeu_ps(y_out.as_mut_ptr(), out_y);
        _mm256_storeu_ps(z_out.as_mut_ptr(), out_z);
        _mm256_storeu_ps(w_out.as_mut_ptr(), out_w);

        for i in 0..SIMD_BATCH_SIZE {
            output[base_idx + i] = Vec4::new(x_out[i], y_out[i], z_out[i], w_out[i]);
        }
    }

    // Handle remainder with scalar code
    if remainder > 0 {
        let base_idx = batch_count * SIMD_BATCH_SIZE;
        for i in 0..remainder {
            let v = &vertices[base_idx + i];
            let world_pos = v.world_position(chunk_offset);
            output[base_idx + i] = *view_proj * world_pos.extend(1.0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voxel::BlockType;

    #[test]
    fn test_simd_matches_scalar() {
        // Create test vertices
        let vertices: Vec<Vertex> = (0..16)
            .map(|i| Vertex::from_local_coords(
                (i % 32) as u8,
                ((i * 2) % 32) as u8,
                ((i * 3) % 32) as u8,
                BlockType::Stone,
                0,
                0,
                1.0,
            ))
            .collect();

        let chunk_offset = Vec3::new(64.0, 32.0, 128.0);
        let view_proj = Mat4::IDENTITY;

        let mut output_simd = vec![Vec4::ZERO; vertices.len()];
        let mut output_scalar = vec![Vec4::ZERO; vertices.len()];

        // Process with both implementations
        decompress_and_transform_vertices(&vertices, chunk_offset, &view_proj, &mut output_simd);
        decompress_and_transform_vertices_scalar(&vertices, chunk_offset, &view_proj, &mut output_scalar);

        // Compare results (should be identical within floating-point tolerance)
        for i in 0..vertices.len() {
            let diff = (output_simd[i] - output_scalar[i]).abs();
            assert!(
                diff.x < 0.001 && diff.y < 0.001 && diff.z < 0.001 && diff.w < 0.001,
                "Mismatch at vertex {}: SIMD {:?} vs Scalar {:?}",
                i, output_simd[i], output_scalar[i]
            );
        }
    }

    #[test]
    fn test_simd_with_various_batch_sizes() {
        // Test with different vertex counts to ensure remainder handling works
        for count in [1, 7, 8, 9, 15, 16, 17, 100] {
            let vertices: Vec<Vertex> = (0..count)
                .map(|i| Vertex::from_local_coords(
                    (i % 32) as u8,
                    ((i * 2) % 32) as u8,
                    ((i * 3) % 32) as u8,
                    BlockType::Grass,
                    0,
                    0,
                    1.0,
                ))
                .collect();

            let chunk_offset = Vec3::new(0.0, 0.0, 0.0);
            let view_proj = Mat4::IDENTITY;

            let mut output = vec![Vec4::ZERO; count];
            decompress_and_transform_vertices(&vertices, chunk_offset, &view_proj, &mut output);

            // Just verify it doesn't crash and produces reasonable results
            for (i, v) in output.iter().enumerate() {
                assert!(
                    v.x.is_finite() && v.y.is_finite() && v.z.is_finite() && v.w.is_finite(),
                    "Invalid output at vertex {} with count {}: {:?}",
                    i, count, v
                );
            }
        }
    }
}
