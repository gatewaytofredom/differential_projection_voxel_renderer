//! Span-based rasterization for axis-aligned voxel quads.
//!
//! This module provides an optimized rasterization path for voxel quads that exploits
//! the fact that vertical edges in world space remain vertical in screen space (when
//! the camera is upright). This allows us to use a much simpler span-walking algorithm
//! instead of full triangle rasterization with barycentric interpolation.
//!
//! Key optimization: Process 8 trapezoids simultaneously using SIMD.

use super::framebuffer::FrameSlice;
use super::differential_projection::ProjectedPacket;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Batch of 8 trapezoids for SIMD processing.
///
/// Each trapezoid represents a projected axis-aligned quad.
/// Layout: Structure of Arrays for efficient SIMD access.
#[repr(C, align(32))]
#[derive(Clone, Debug)]
pub struct TrapezoidBatch {
    /// Number of active trapezoids (1-8)
    pub count: u8,

    /// Current X coordinate of left edge (8 trapezoids)
    pub left_x: [f32; 8],

    /// Current X coordinate of right edge
    pub right_x: [f32; 8],

    /// Change in X per scanline (left edge)
    pub left_slope: [f32; 8],

    /// Change in X per scanline (right edge)
    pub right_slope: [f32; 8],

    /// Starting Y coordinate for each trapezoid
    pub start_y: [f32; 8],

    /// Ending Y coordinate for each trapezoid
    pub end_y: [f32; 8],

    /// Constant depth per quad (for depth testing)
    pub depth: [f32; 8],

    /// Material ID / color for each quad
    pub color: [u32; 8],

    /// Active bitmask (bit i = trapezoid i is active)
    pub active_mask: u8,
}

impl TrapezoidBatch {
    pub fn new() -> Self {
        Self {
            count: 0,
            left_x: [0.0; 8],
            right_x: [0.0; 8],
            left_slope: [0.0; 8],
            right_slope: [0.0; 8],
            start_y: [0.0; 8],
            end_y: [0.0; 8],
            depth: [0.0; 8],
            color: [0; 8],
            active_mask: 0,
        }
    }

    /// Check if any trapezoids are still active.
    #[inline]
    pub fn is_active(&self) -> bool {
        self.active_mask != 0
    }

    /// Update active mask based on current Y position.
    #[inline]
    pub fn update_active_mask(&mut self, current_y: f32) {
        let mut mask = 0u8;
        for i in 0..(self.count as usize) {
            if current_y >= self.start_y[i] && current_y < self.end_y[i] {
                mask |= 1 << i;
            }
        }
        self.active_mask = mask;
    }
}

impl Default for TrapezoidBatch {
    fn default() -> Self {
        Self::new()
    }
}

/// Span walker rasterizer.
///
/// Processes projected quads using a scanline-based approach that's
/// optimized for axis-aligned geometry.
pub struct SpanWalkerRasterizer {
    // Configuration
    viewport_width: u32,
    viewport_height: u32,
}

impl SpanWalkerRasterizer {
    pub fn new(viewport_width: u32, viewport_height: u32) -> Self {
        Self {
            viewport_width,
            viewport_height,
        }
    }

    /// Rasterize a projected packet into a framebuffer slice.
    ///
    /// This is the "fast path" for axis-aligned quads.
    pub fn rasterize_projected_packet(
        &self,
        projected: &ProjectedPacket,
        framebuffer: &mut FrameSlice,
    ) {
        // Convert ProjectedPacket to trapezoid batches
        let batches = self.setup_trapezoid_batches(projected);

        // Rasterize each batch
        for mut batch in batches {
            self.rasterize_batch(framebuffer, &mut batch);
        }
    }

    /// Convert a ProjectedPacket into TrapezoidBatch structs.
    fn setup_trapezoid_batches(&self, projected: &ProjectedPacket) -> Vec<TrapezoidBatch> {
        let mut batches = Vec::new();
        let mut current_batch = TrapezoidBatch::new();

        let vp_width = self.viewport_width as f32;
        let vp_height = self.viewport_height as f32;

        // FIX: Add a small epsilon to close sub-pixel gaps between adjacent quads.
        // Without this, floating point errors can cause the end of one quad to be
        // slightly less than the start of the next (e.g., 100.4999 vs 100.5001),
        // causing pixel centers at 100.5 to be skipped.
        const EPSILON: f32 = 0.001;

        for i in 0..(projected.count as usize) {
            // Check visibility mask
            if (projected.visibility_mask & (1 << i)) == 0 {
                continue;
            }

            // Convert NDC to screen space
            let screen_x_min = ((projected.screen_x_min[i] + 1.0) * 0.5 * vp_width).max(0.0);
            let screen_y_min = ((1.0 - projected.screen_y_max[i]) * 0.5 * vp_height).max(0.0);

            // Apply epsilon to max bounds
            let screen_x_max = ((projected.screen_x_max[i] + 1.0) * 0.5 * vp_width + EPSILON).min(vp_width);
            let screen_y_max = ((1.0 - projected.screen_y_min[i]) * 0.5 * vp_height + EPSILON).min(vp_height);

            // Skip if completely outside viewport
            if screen_x_min >= vp_width
                || screen_y_min >= vp_height
                || screen_x_max <= 0.0
                || screen_y_max <= 0.0
            {
                continue;
            }

            // For axis-aligned quads, edges are vertical (slope = 0)
            let batch_idx = current_batch.count as usize;
            current_batch.left_x[batch_idx] = screen_x_min;
            current_batch.right_x[batch_idx] = screen_x_max;
            current_batch.left_slope[batch_idx] = 0.0; // Vertical edge
            current_batch.right_slope[batch_idx] = 0.0; // Vertical edge
            current_batch.start_y[batch_idx] = screen_y_min;
            current_batch.end_y[batch_idx] = screen_y_max;
            current_batch.depth[batch_idx] = projected.depth_near[i];
            current_batch.color[batch_idx] = self.get_block_color(projected.block_type[i]);
            current_batch.active_mask |= 1 << batch_idx;
            current_batch.count += 1;

            // Flush batch when full (8 trapezoids)
            if current_batch.count == 8 {
                batches.push(current_batch);
                current_batch = TrapezoidBatch::new();
            }
        }

        // Push remaining batch
        if current_batch.count > 0 {
            batches.push(current_batch);
        }

        batches
    }

    /// Rasterize a single batch of trapezoids.
    fn rasterize_batch(&self, framebuffer: &mut FrameSlice, batch: &mut TrapezoidBatch) {
        if batch.count == 0 {
            return;
        }

        // Find the vertical bounds of the entire batch
        let mut min_y = f32::INFINITY;
        let mut max_y = f32::NEG_INFINITY;

        for i in 0..(batch.count as usize) {
            min_y = min_y.min(batch.start_y[i]);
            max_y = max_y.max(batch.end_y[i]);
        }

        let mut current_y = min_y.floor() as i32;
        let end_y = max_y.ceil() as i32;

        // Scanline loop: Run until we pass the bottom of the lowest quad
        // Do NOT rely on batch.is_active() for the loop condition, as gaps
        // or sub-pixel starts can cause false negatives at the start/middle.
        while current_y < end_y {
            // Check bounds
            if current_y >= framebuffer.height() as i32 {
                break;
            }

            if current_y >= 0 {
                // CRITICAL FIX: Sample at PIXEL CENTER (y + 0.5)
                // Previously, we sampled at 'current_y' (top of pixel).
                // If a quad starts at 10.1, checking 10.0 >= 10.1 fails, skipping the row.
                // Checking 10.5 >= 10.1 succeeds, correctly drawing the row.
                // This fixes the "horizontal line" artifacts caused by sub-pixel precision.
                batch.update_active_mask(current_y as f32 + 0.5);

                // Only draw if something is actually active on this scanline
                if batch.is_active() {
                    for i in 0..(batch.count as usize) {
                        if (batch.active_mask & (1 << i)) == 0 {
                            continue;
                        }

                        let x_start = batch.left_x[i].round() as i32;
                        let x_end = batch.right_x[i].round() as i32;
                        let depth = batch.depth[i];
                        let color = batch.color[i];

                        // Emit horizontal span
                        framebuffer.fill_span(current_y as usize, x_start, x_end, depth, color);
                    }
                }
            }

            // Advance to next scanline
            // Note: We advance slopes even when not active to maintain correct X positions
            // for quads that start later (handles vertical gaps)
            for i in 0..(batch.count as usize) {
                batch.left_x[i] += batch.left_slope[i];
                batch.right_x[i] += batch.right_slope[i];
            }

            current_y += 1;
        }
    }

    /// Get color for a block type.
    fn get_block_color(&self, block_type: u8) -> u32 {
        use crate::voxel::BlockType;

        match BlockType::from_u8(block_type) {
            BlockType::Air => 0x00000000,
            BlockType::Grass => 0x00FF00FF, // Green
            BlockType::Dirt => 0x8B4513FF,  // Brown
            BlockType::Stone => 0x808080FF, // Gray
        }
    }
}

/// Extension to FrameSlice for span filling.
impl<'a> FrameSlice<'a> {
    /// Fill a horizontal span with depth testing.
    ///
    /// This is the core operation of the span walker.
    pub fn fill_span(
        &mut self,
        y: usize,
        x_start: i32,
        x_end: i32,
        depth: f32,
        color: u32,
    ) {
        let width = self.width() as i32;

        // Clamp to framebuffer bounds
        let x_start = x_start.max(0).min(width - 1);
        let x_end = x_end.max(0).min(width);

        if x_start >= x_end {
            return;
        }

        let row_offset = y * self.width();

        // Scalar loop (SIMD version in Phase 5)
        for x in x_start..x_end {
            let idx = row_offset + x as usize;

            // Depth test
            if depth < self.depth_at(idx) {
                self.set_depth_at(idx, depth);
                self.set_color_at(idx, color);
            }
        }
    }

    /// SIMD-accelerated span fill (AVX2).
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    pub unsafe fn fill_span_simd(
        &mut self,
        y: usize,
        x_start: i32,
        x_end: i32,
        depth: f32,
        color: u32,
    ) {
        let width = self.width() as i32;

        // Clamp to framebuffer bounds
        let x_start = x_start.max(0).min(width - 1);
        let x_end = x_end.max(0).min(width);

        if x_start >= x_end {
            return;
        }

        let row_offset = y * self.width();
        let depth_vec = _mm256_set1_ps(depth);
        // let color_vec = _mm256_set1_epi32(color as i32);

        let mut x = x_start as usize;

        // SIMD loop (8 pixels at a time)
        while x + 8 <= x_end as usize {
            let idx = row_offset + x;

            // Load current depth
            let current_depth = _mm256_loadu_ps(&self.depth[idx] as *const f32);

            // Depth test (8 pixels)
            let pass_mask = _mm256_cmp_ps(depth_vec, current_depth, _CMP_LT_OQ);
            let mask_bits = _mm256_movemask_ps(pass_mask);

            if mask_bits != 0 {
                // Store depth (masked store not available, use scalar for now)
                for lane in 0..8 {
                    if (mask_bits & (1 << lane)) != 0 {
                        self.set_depth_at(idx + lane, depth);
                        self.set_color_at(idx + lane, color);
                    }
                }
            }

            x += 8;
        }

        // Scalar tail
        for x in x..(x_end as usize) {
            let idx = row_offset + x;
            if depth < self.depth_at(idx) {
                self.set_depth_at(idx, depth);
                self.set_color_at(idx, color);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rendering::Framebuffer;

    #[test]
    fn test_trapezoid_batch_creation() {
        let batch = TrapezoidBatch::new();
        assert_eq!(batch.count, 0);
        assert_eq!(batch.active_mask, 0);
        assert!(!batch.is_active());
    }

    #[test]
    fn test_trapezoid_batch_active_mask() {
        let mut batch = TrapezoidBatch::new();
        batch.count = 4;
        batch.start_y = [0.0, 5.0, 10.0, 15.0, 0.0, 0.0, 0.0, 0.0];
        batch.end_y = [10.0, 15.0, 20.0, 25.0, 0.0, 0.0, 0.0, 0.0];
        batch.active_mask = 0b1111;

        // At y=12, trapezoids 1 and 2 should be active.
        batch.update_active_mask(12.0);
        assert_eq!(batch.active_mask, 0b0110);

        // At y=22, only trapezoid 3 should be active
        batch.update_active_mask(22.0);
        assert_eq!(batch.active_mask, 0b1000);
    }

    #[test]
    fn test_span_fill_basic() {
        let mut fb = Framebuffer::new(64, 64);
        let mut slice = fb.as_full_slice_mut();

        // Fill a horizontal span
        slice.fill_span(32, 10, 50, 0.5, 0xFF0000FF);

        // Verify pixels inside span
        for x in 10..50 {
            assert_eq!(slice.color_at(32 * 64 + x), 0xFF0000FF);
            assert_eq!(slice.depth_at(32 * 64 + x), 0.5);
        }

        // Verify pixels outside span are unchanged
        assert_eq!(slice.color_at(32 * 64 + 9), 0);
        assert_eq!(slice.color_at(32 * 64 + 50), 0);
    }

    #[test]
    fn test_span_fill_depth_test() {
        let mut fb = Framebuffer::new(64, 64);
        let mut slice = fb.as_full_slice_mut();

        // Fill first span at depth 0.5
        slice.fill_span(32, 10, 50, 0.5, 0xFF0000FF);

        // Try to fill same span at depth 0.7 (should fail depth test)
        slice.fill_span(32, 10, 50, 0.7, 0x00FF00FF);

        // Should still be red (first color)
        assert_eq!(slice.color_at(32 * 64 + 25), 0xFF0000FF);
        assert_eq!(slice.depth_at(32 * 64 + 25), 0.5);

        // Fill same span at depth 0.3 (should pass depth test)
        slice.fill_span(32, 10, 50, 0.3, 0x0000FFFF);

        // Should now be blue (new color)
        assert_eq!(slice.color_at(32 * 64 + 25), 0x0000FFFF);
        assert_eq!(slice.depth_at(32 * 64 + 25), 0.3);
    }

    #[test]
    fn test_span_walker_simple_quad() {
        let mut fb = Framebuffer::new(128, 128);
        let mut slice = fb.as_full_slice_mut();

        let walker = SpanWalkerRasterizer::new(128, 128);

        // Create a simple projected packet with one quad
        let mut projected = ProjectedPacket::new();
        projected.count = 1;
        projected.screen_x_min[0] = -0.5; // NDC
        projected.screen_y_min[0] = -0.5;
        projected.screen_x_max[0] = 0.5;
        projected.screen_y_max[0] = 0.5;
        projected.depth_near[0] = 0.5;
        projected.block_type[0] = 1; // Grass
        projected.visibility_mask = 1;

        walker.rasterize_projected_packet(&projected, &mut slice);

        // Verify center pixel was filled
        let center_idx = 64 * 128 + 64;
        assert_ne!(slice.color_at(center_idx), 0, "Center pixel should be filled");
        assert_eq!(slice.depth_at(center_idx), 0.5);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_simd_span_fill_matches_scalar() {
        if !is_x86_feature_detected!("avx2") {
            println!("Skipping AVX2 test");
            return;
        }

        let mut fb_scalar = Framebuffer::new(128, 128);
        let mut fb_simd = Framebuffer::new(128, 128);

        let mut slice_scalar = fb_scalar.as_full_slice_mut();
        let mut slice_simd = fb_simd.as_full_slice_mut();

        // Fill with scalar
        slice_scalar.fill_span(64, 10, 100, 0.5, 0xFF00FFFF);

        // Fill with SIMD
        unsafe {
            slice_simd.fill_span_simd(64, 10, 100, 0.5, 0xFF00FFFF);
        }

        // Compare results
        for x in 0..128 {
            let idx = 64 * 128 + x;
            assert_eq!(
                slice_scalar.color_at(idx),
                slice_simd.color_at(idx),
                "Color mismatch at x={}",
                x
            );
            assert_eq!(
                slice_scalar.depth_at(idx),
                slice_simd.depth_at(idx),
                "Depth mismatch at x={}",
                x
            );
        }
    }
}
