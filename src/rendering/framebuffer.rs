/// Framebuffer for software rendering
/// Stores color and depth information
///
/// Memory layout optimized for cache efficiency:
/// - Hot metadata (width, height) stored first for bounds checking
/// - Buffers are stored as separate Vecs to allow independent access patterns
use crate::count_call;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{
    _mm256_set1_epi32, _mm256_set1_ps, _mm256_storeu_ps, _mm256_storeu_si256, _mm_set1_epi32,
    _mm_set1_ps, _mm_storeu_ps, _mm_storeu_si128,
};

/// View into a contiguous set of rows in the framebuffer.
/// Used for multi-core rasterization where each worker owns a disjoint slice.
pub struct FrameSlice<'a> {
    pub width: usize,
    pub full_height: usize,
    pub y0: usize,
    pub height: usize,
    pub color: &'a mut [u32],
    pub depth: &'a mut [f32],
}

impl<'a> FrameSlice<'a> {
    /// Perform a depth test at (x, y_global) and, if it passes, update depth and
    /// return the linear index into the local color buffer. Returns None if the
    /// pixel lies outside this slice or fails the depth test.
    #[inline]
    pub unsafe fn test_depth_and_get_index(
        &mut self,
        x: usize,
        y_global: usize,
        depth: f32,
    ) -> Option<usize> {
        if y_global < self.y0 {
            return None;
        }
        let y_local = y_global - self.y0;
        if y_local >= self.height {
            return None;
        }

        let index = y_local * self.width + x;
        if depth < self.depth[index] {
            self.depth[index] = depth;
            Some(index)
        } else {
            None
        }
    }

    #[inline]
    pub fn write_color(&mut self, index: usize, color: u32) {
        self.color[index] = color;
    }

    /// Get raw pointers to the start of a specific row within the slice.
    ///
    /// # Safety
    /// Caller must ensure:
    /// - y_local < self.height
    /// - The returned pointers are only used for the current row
    /// - No other mutable access to the same memory occurs concurrently
    #[inline(always)]
    pub unsafe fn get_row_pointers(&mut self, y_local: usize) -> (*mut u32, *mut f32) {
        // A slice is contiguous in memory representing a specific Y range
        let start_index = y_local * self.width;
        (
            self.color.as_mut_ptr().add(start_index),
            self.depth.as_mut_ptr().add(start_index),
        )
    }

    /// Get slice bounds: (x0, y0, x1, y1) in global framebuffer coordinates
    #[inline(always)]
    pub fn bounds(&self) -> (usize, usize, usize, usize) {
        (0, self.y0, self.width, self.y0 + self.height)
    }

    // Helper methods for span walker
    #[inline]
    pub fn width(&self) -> usize {
        self.width
    }

    #[inline]
    pub fn height(&self) -> usize {
        self.height
    }

    #[inline]
    pub fn color_at(&self, idx: usize) -> u32 {
        self.color[idx]
    }

    #[inline]
    pub fn depth_at(&self, idx: usize) -> f32 {
        self.depth[idx]
    }

    #[inline]
    pub fn set_color_at(&mut self, idx: usize, color: u32) {
        self.color[idx] = color;
    }

    #[inline]
    pub fn set_depth_at(&mut self, idx: usize, depth: f32) {
        self.depth[idx] = depth;
    }

    #[inline]
    pub fn depth_slice_at(&self, idx: usize) -> &[f32] {
        &self.depth[idx..]
    }
}

/// View into a rectangular tile of the framebuffer.
/// Unlike `FrameSlice`, tiles partition both X and Y. Tiles are designed
/// for cache-friendly, tile-based rasterization. Internally they use raw
/// pointers into the backing buffers; callers must ensure tiles do not
/// overlap when used in parallel.
pub struct FrameTile {
    pub width: usize,
    pub full_height: usize,
    pub x0: usize,
    pub y0: usize,
    pub tile_width: usize,
    pub tile_height: usize,
    color_ptr: *mut u32,
    depth_ptr: *mut f32,
}

// Safety: FrameTile is Send + Sync because it only carries raw pointers
// to the underlying framebuffer and (x0, y0, tile_width, tile_height)
// ensure that tiles are used on disjoint pixel regions when processed
// in parallel.
unsafe impl Send for FrameTile {}
unsafe impl Sync for FrameTile {}

impl FrameTile {
    /// Perform a depth test at global pixel (x, y) and, if it lies within
    /// this tile and passes, update depth and return the linear index into
    /// the framebuffer color buffer. Returns None if the pixel lies outside
    /// this tile or fails the depth test.
    #[inline]
    pub unsafe fn test_depth_and_get_index(
        &mut self,
        x: usize,
        y: usize,
        depth: f32,
    ) -> Option<usize> {
        if x < self.x0 || x >= self.x0 + self.tile_width {
            return None;
        }
        if y < self.y0 || y >= self.y0 + self.tile_height {
            return None;
        }

        let index = y * self.width + x;

        let depth_ref = &mut *self.depth_ptr.add(index);
        if depth < *depth_ref {
            *depth_ref = depth;
            Some(index)
        } else {
            None
        }
    }

    #[inline]
    pub fn write_color(&mut self, index: usize, color: u32) {
        unsafe {
            *self.color_ptr.add(index) = color;
        }
    }

    /// Get raw pointers to the start of a specific row within the tile.
    ///
    /// # Safety
    /// Caller must ensure:
    /// - y_global is within the framebuffer bounds
    /// - The returned pointers are only used for pixels within the tile's x range [x0, x0+tile_width)
    /// - No other mutable access to the same memory occurs concurrently
    #[inline(always)]
    pub unsafe fn get_row_pointers(&mut self, y_global: usize) -> (*mut u32, *mut f32) {
        // Calculate the global index for the start of this row
        let start_index = y_global * self.width;

        (
            self.color_ptr.add(start_index),
            self.depth_ptr.add(start_index),
        )
    }
}

pub struct Framebuffer {
    // Hot data: used for every bounds check and index calculation
    pub width: usize,
    pub height: usize,
    // Color and depth buffers - accessed with different patterns
    // Separate allocation allows better cache utilization when only one is needed
    pub color_buffer: Vec<u32>, // ARGB format
    pub depth_buffer: Vec<f32>,
}

impl Framebuffer {
    pub fn new(width: usize, height: usize) -> Self {
        let pixel_count = width * height;
        Self {
            width,
            height,
            color_buffer: vec![0; pixel_count],
            depth_buffer: vec![f32::INFINITY; pixel_count],
        }
    }

    /// Clear color and depth buffers
    pub fn clear(&mut self, clear_color: u32) {
        // count_call!(FUNCTION_COUNTERS.framebuffer_clear_calls);
        self.clear_impl(clear_color);
    }

    #[inline]
    fn clear_impl(&mut self, clear_color: u32) {
        #[cfg(target_arch = "x86_64")]
        {
            // Prefer AVX (8 pixels per iteration) when available,
            // otherwise fall back to SSE2 (4 pixels per iteration).
            if std::arch::is_x86_feature_detected!("avx") {
                unsafe {
                    return self.clear_simd_avx(clear_color);
                }
            }
            if std::arch::is_x86_feature_detected!("sse2") {
                unsafe {
                    return self.clear_simd_sse2(clear_color);
                }
            }
        }

        // Generic scalar fallback for non-x86_64 or CPUs without SIMD.
        self.color_buffer.fill(clear_color);
        self.depth_buffer.fill(f32::INFINITY);
    }

    /// SIMD-accelerated clear for x86_64 with SSE2.
    /// Clears 4 pixels per iteration using vector stores.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "sse2")]
    unsafe fn clear_simd_sse2(&mut self, clear_color: u32) {
        let len = self.color_buffer.len();

        // Process color buffer as u32 lanes
        let mut i = 0usize;
        let color_vec = _mm_set1_epi32(clear_color as i32);
        while i + 4 <= len {
            let ptr = self.color_buffer.as_mut_ptr().add(i) as *mut _;
            _mm_storeu_si128(ptr, color_vec);
            i += 4;
        }
        // Tail
        for j in i..len {
            self.color_buffer[j] = clear_color;
        }

        // Process depth buffer as f32 lanes
        let len_d = self.depth_buffer.len();
        let mut k = 0usize;
        let depth_vec = _mm_set1_ps(f32::INFINITY);
        while k + 4 <= len_d {
            let ptr = self.depth_buffer.as_mut_ptr().add(k);
            _mm_storeu_ps(ptr, depth_vec);
            k += 4;
        }
        for j in k..len_d {
            self.depth_buffer[j] = f32::INFINITY;
        }
    }

    /// SIMD-accelerated clear for x86_64 with AVX.
    /// Clears 8 pixels per iteration using 256-bit vector stores.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx")]
    unsafe fn clear_simd_avx(&mut self, clear_color: u32) {
        let len = self.color_buffer.len();

        // Process color buffer as u32 lanes (8 pixels per iteration)
        let mut i = 0usize;
        let color_vec = _mm256_set1_epi32(clear_color as i32);
        while i + 8 <= len {
            let ptr = self.color_buffer.as_mut_ptr().add(i) as *mut _;
            _mm256_storeu_si256(ptr, color_vec);
            i += 8;
        }
        // Tail
        for j in i..len {
            self.color_buffer[j] = clear_color;
        }

        // Process depth buffer as f32 lanes
        let len_d = self.depth_buffer.len();
        let mut k = 0usize;
        let depth_vec = _mm256_set1_ps(f32::INFINITY);
        while k + 8 <= len_d {
            let ptr = self.depth_buffer.as_mut_ptr().add(k);
            _mm256_storeu_ps(ptr, depth_vec);
            k += 8;
        }
        for j in k..len_d {
            self.depth_buffer[j] = f32::INFINITY;
        }
    }

    /// Set pixel with depth test
    #[inline]
    pub fn set_pixel(&mut self, x: usize, y: usize, color: u32, depth: f32) -> bool {
        if x >= self.width || y >= self.height {
            return false;
        }

        let index = y * self.width + x;

        // Depth test
        if depth < self.depth_buffer[index] {
            self.color_buffer[index] = color;
            self.depth_buffer[index] = depth;
            true
        } else {
            false
        }
    }

    /// Set pixel with depth test, without bounds checking.
    /// Callers must guarantee x,y are within the framebuffer.
    #[inline]
    pub unsafe fn set_pixel_unchecked(
        &mut self,
        x: usize,
        y: usize,
        color: u32,
        depth: f32,
    ) -> bool {
        let index = y * self.width + x;

        if depth < self.depth_buffer[index] {
            self.color_buffer[index] = color;
            self.depth_buffer[index] = depth;
            true
        } else {
            false
        }
    }

    /// Set pixel without depth test (for UI, etc.)
    #[inline]
    pub fn set_pixel_no_depth(&mut self, x: usize, y: usize, color: u32) {
        if x < self.width && y < self.height {
            let index = y * self.width + x;
            self.color_buffer[index] = color;
        }
    }

    /// Get color buffer as slice
    pub fn color_buffer_slice(&self) -> &[u32] {
        &self.color_buffer
    }

    /// Create a FrameSlice covering the entire framebuffer (for testing/simple cases)
    pub fn as_full_slice_mut(&mut self) -> FrameSlice<'_> {
        FrameSlice {
            width: self.width,
            full_height: self.height,
            y0: 0,
            height: self.height,
            color: &mut self.color_buffer,
            depth: &mut self.depth_buffer,
        }
    }

    /// Resize framebuffer
    pub fn resize(&mut self, width: usize, height: usize) {
        self.width = width;
        self.height = height;
        let pixel_count = width * height;
        self.color_buffer.resize(pixel_count, 0);
        self.depth_buffer.resize(pixel_count, f32::INFINITY);
    }

    /// Split the framebuffer into horizontal stripes for multi-core rendering.
    /// Each stripe owns a disjoint subset of rows, so they can be rendered in parallel.
    pub fn split_into_stripes(&mut self, stripes: usize) -> Vec<FrameSlice<'_>> {
        let stripes = stripes.max(1);
        let width = self.width;
        let height = self.height;

        let mut slices = Vec::with_capacity(stripes);

        let mut remaining_color: &mut [u32] = self.color_buffer.as_mut_slice();
        let mut remaining_depth: &mut [f32] = self.depth_buffer.as_mut_slice();

        let mut y0 = 0usize;
        let min_rows_per_stripe = (height + stripes - 1) / stripes;

        for _ in 0..stripes {
            if y0 >= height {
                break;
            }
            let remaining_rows = height - y0;
            let rows = remaining_rows.min(min_rows_per_stripe);
            let pixels = rows * width;

            let (color_head, color_tail) = remaining_color.split_at_mut(pixels);
            let (depth_head, depth_tail) = remaining_depth.split_at_mut(pixels);

            slices.push(FrameSlice {
                width,
                full_height: height,
                y0,
                height: rows,
                color: color_head,
                depth: depth_head,
            });

            remaining_color = color_tail;
            remaining_depth = depth_tail;
            y0 += rows;
        }

        slices
    }

    /// Split the framebuffer into 2D tiles for cache-friendly, tile-based rendering.
    /// Tiles partition both X and Y dimensions; each tile owns a disjoint rectangle
    /// of pixels, making them suitable for parallel processing without overlap.
    pub fn split_into_tiles(&mut self, tile_width: usize, tile_height: usize) -> Vec<FrameTile> {
        let tile_width = tile_width.max(1);
        let tile_height = tile_height.max(1);

        let width = self.width;
        let height = self.height;

        let color_ptr = self.color_buffer.as_mut_ptr();
        let depth_ptr = self.depth_buffer.as_mut_ptr();

        let mut tiles = Vec::new();

        let mut y0 = 0usize;
        while y0 < height {
            let h = (height - y0).min(tile_height);
            let mut x0 = 0usize;
            while x0 < width {
                let w = (width - x0).min(tile_width);
                tiles.push(FrameTile {
                    width,
                    full_height: height,
                    x0,
                    y0,
                    tile_width: w,
                    tile_height: h,
                    color_ptr,
                    depth_ptr,
                });
                x0 += tile_width;
            }
            y0 += tile_height;
        }

        tiles
    }
}

/// Convert RGB to ARGB u32
#[inline]
pub const fn rgb_to_u32(r: u8, g: u8, b: u8) -> u32 {
    0xFF000000 | ((r as u32) << 16) | ((g as u32) << 8) | (b as u32)
}

/// Apply simple lighting based on AO
#[inline]
pub fn apply_ao(color: [u8; 3], ao: u8) -> u32 {
    // AO ranges from 0 (darkest) to 3 (no occlusion)
    let factor = match ao {
        0 => 0.4,
        1 => 0.6,
        2 => 0.8,
        _ => 1.0,
    };

    let r = (color[0] as f32 * factor) as u8;
    let g = (color[1] as f32 * factor) as u8;
    let b = (color[2] as f32 * factor) as u8;

    rgb_to_u32(r, g, b)
}
