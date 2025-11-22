/// Macrotile-based rendering system for L2 cache optimization
///
/// Key Design Principles:
/// 1. Tile size: 128×128 pixels (64KB depth + 64KB color = 128KB fits in L2 cache)
/// 2. Binning: Pre-compute which meshes touch which macrotiles
/// 3. Thread-local bins: Avoid mutex contention during geometry processing
/// 4. Large geometry bypass: Handle screen-filling primitives separately
///
/// Performance Benefits:
/// - L2 cache residency: Entire tile depth/color buffers stay hot in L2
/// - Reduced memory bandwidth: Write to main framebuffer only once per tile
/// - Better parallelism: Independent tiles can be processed on different cores
/// - Scalability: Works well from 720p to 4K

use glam::{Mat4, Vec3};
use std::sync::Arc;
use super::rasterizer::PixelTarget;

/// Macrotile size in pixels (128×128 = 16,384 pixels)
/// Memory footprint: 64KB depth + 64KB color = 128KB total (fits in 256KB L2 cache)
pub const MACROTILE_SIZE: usize = 128;

/// Threshold for "large primitive" bypass
/// If a mesh covers more than this fraction of the screen, render it directly
/// without binning (to avoid duplicating work across many tiles)
pub const LARGE_PRIMITIVE_SCREEN_FRACTION: f32 = 0.25;

/// L2-resident macrotile with depth and color buffers
#[repr(C, align(64))]
pub struct MacroTile {
    /// Top-left X coordinate in framebuffer space
    pub x0: usize,
    /// Top-left Y coordinate in framebuffer space
    pub y0: usize,
    /// Actual width of this tile (may be < MACROTILE_SIZE at edges)
    pub width: usize,
    /// Actual height of this tile (may be < MACROTILE_SIZE at edges)
    pub height: usize,
    /// Full framebuffer width (for coordinate mapping)
    pub fb_width: usize,
    /// Full framebuffer height (for coordinate mapping)
    pub fb_height: usize,
    /// Color buffer (128×128 × 4 bytes = 64KB)
    pub color: Box<[u32]>,
    /// Depth buffer (128×128 × 4 bytes = 64KB)
    pub depth: Box<[f32]>,
}

impl MacroTile {
    /// Create a new macrotile at the specified position
    pub fn new(
        x0: usize,
        y0: usize,
        width: usize,
        height: usize,
        fb_width: usize,
        fb_height: usize,
    ) -> Self {
        let capacity = width * height;
        let color = vec![0u32; capacity].into_boxed_slice();
        let depth = vec![f32::INFINITY; capacity].into_boxed_slice();

        Self {
            x0,
            y0,
            width,
            height,
            fb_width,
            fb_height,
            color,
            depth,
        }
    }

    /// Clear the tile to the specified color
    #[inline]
    pub fn clear(&mut self, clear_color: u32) {
        self.color.fill(clear_color);
        self.depth.fill(f32::INFINITY);
    }

    /// Test depth at a pixel and return the buffer index if passed
    #[inline]
    pub fn test_depth(&mut self, x: usize, y: usize, depth: f32) -> Option<usize> {
        // Convert to tile-local coordinates
        let local_x = x.wrapping_sub(self.x0);
        let local_y = y.wrapping_sub(self.y0);

        if local_x >= self.width || local_y >= self.height {
            return None;
        }

        let idx = local_y * self.width + local_x;

        if depth < self.depth[idx] {
            self.depth[idx] = depth;
            Some(idx)
        } else {
            None
        }
    }

    /// Write color to a pixel (assumes depth test already passed)
    #[inline]
    pub fn write_color(&mut self, index: usize, color: u32) {
        self.color[index] = color;
    }

    /// Get the index for a pixel at (x, y) in tile-local coordinates
    #[inline]
    fn local_index(&self, local_x: usize, local_y: usize) -> usize {
        local_y * self.width + local_x
    }

    /// Flush tile contents to the main framebuffer
    ///
    /// This is the only point where we write to main memory, minimizing bandwidth
    pub fn flush_to_framebuffer(&self, framebuffer: &mut [u32], fb_width: usize) {
        for y in 0..self.height {
            let src_offset = y * self.width;
            let dst_offset = (self.y0 + y) * fb_width + self.x0;

            let src_slice = &self.color[src_offset..src_offset + self.width];
            let dst_slice = &mut framebuffer[dst_offset..dst_offset + self.width];

            dst_slice.copy_from_slice(src_slice);
        }
    }
}

/// Mesh identifier for binning
#[derive(Copy, Clone, Debug)]
pub struct MeshId(pub usize);

/// Macrotile binning system
///
/// Tracks which meshes need to be rendered into which tiles
pub struct MacroTileBins {
    /// Number of tiles horizontally
    pub tiles_x: usize,
    /// Number of tiles vertically
    pub tiles_y: usize,
    /// Total number of tiles
    pub tile_count: usize,
    /// Per-tile mesh lists (tile_index -> list of mesh indices)
    pub bins: Vec<Vec<MeshId>>,
    /// Large primitives that bypass binning (rendered to all tiles)
    pub large_primitives: Vec<MeshId>,
}

impl MacroTileBins {
    /// Create a new binning system for the given framebuffer dimensions
    pub fn new(fb_width: usize, fb_height: usize) -> Self {
        let tiles_x = (fb_width + MACROTILE_SIZE - 1) / MACROTILE_SIZE;
        let tiles_y = (fb_height + MACROTILE_SIZE - 1) / MACROTILE_SIZE;
        let tile_count = tiles_x * tiles_y;

        let bins = vec![Vec::new(); tile_count];

        Self {
            tiles_x,
            tiles_y,
            tile_count,
            bins,
            large_primitives: Vec::new(),
        }
    }

    /// Clear all bins (reuse allocations)
    pub fn clear(&mut self) {
        for bin in &mut self.bins {
            bin.clear();
        }
        self.large_primitives.clear();
    }

    /// Add a mesh to the appropriate tiles based on its screen-space AABB
    ///
    /// Returns true if the mesh was added to bins, false if it was marked as large primitive
    pub fn add_mesh(
        &mut self,
        mesh_id: MeshId,
        screen_min_x: i32,
        screen_min_y: i32,
        screen_max_x: i32,
        screen_max_y: i32,
        fb_width: usize,
        fb_height: usize,
    ) -> bool {
        // Clamp to framebuffer bounds
        let min_x = screen_min_x.max(0) as usize;
        let min_y = screen_min_y.max(0) as usize;
        let max_x = (screen_max_x.min(fb_width as i32 - 1)) as usize;
        let max_y = (screen_max_y.min(fb_height as i32 - 1)) as usize;

        if min_x > max_x || min_y > max_y {
            return false; // Off-screen
        }

        // Calculate coverage
        let coverage_pixels = (max_x - min_x + 1) * (max_y - min_y + 1);
        let total_pixels = fb_width * fb_height;
        let coverage_fraction = coverage_pixels as f32 / total_pixels as f32;

        // Large primitive bypass: if it covers >25% of the screen, don't bin it
        if coverage_fraction > LARGE_PRIMITIVE_SCREEN_FRACTION {
            self.large_primitives.push(mesh_id);
            return false;
        }

        // Determine which tiles this mesh overlaps
        let start_tile_x = min_x / MACROTILE_SIZE;
        let start_tile_y = min_y / MACROTILE_SIZE;
        let end_tile_x = max_x / MACROTILE_SIZE;
        let end_tile_y = max_y / MACROTILE_SIZE;

        // Add to all overlapping tiles
        for ty in start_tile_y..=end_tile_y.min(self.tiles_y - 1) {
            for tx in start_tile_x..=end_tile_x.min(self.tiles_x - 1) {
                let tile_idx = ty * self.tiles_x + tx;
                self.bins[tile_idx].push(mesh_id);
            }
        }

        true
    }

    /// Get the mesh list for a specific tile
    #[inline]
    pub fn get_bin(&self, tile_x: usize, tile_y: usize) -> &[MeshId] {
        let idx = tile_y * self.tiles_x + tile_x;
        &self.bins[idx]
    }

    /// Get macrotile coordinates for a given tile index
    #[inline]
    pub fn tile_rect(&self, tile_x: usize, tile_y: usize, fb_width: usize, fb_height: usize) -> (usize, usize, usize, usize) {
        let x0 = tile_x * MACROTILE_SIZE;
        let y0 = tile_y * MACROTILE_SIZE;
        let x1 = (x0 + MACROTILE_SIZE).min(fb_width);
        let y1 = (y0 + MACROTILE_SIZE).min(fb_height);
        let width = x1 - x0;
        let height = y1 - y0;

        (x0, y0, width, height)
    }
}

/// Thread-local binning workspace
///
/// Each thread maintains its own bins during geometry processing,
/// then merges into the global bins to avoid mutex contention
pub struct ThreadLocalBins {
    /// Per-thread bins (one set per worker thread)
    thread_bins: Vec<MacroTileBins>,
}

impl ThreadLocalBins {
    /// Create thread-local bins for the given framebuffer size and thread count
    pub fn new(fb_width: usize, fb_height: usize, thread_count: usize) -> Self {
        let thread_bins = (0..thread_count)
            .map(|_| MacroTileBins::new(fb_width, fb_height))
            .collect();

        Self { thread_bins }
    }

    /// Get a mutable reference to a thread's bins
    pub fn get_thread_bins(&mut self, thread_id: usize) -> &mut MacroTileBins {
        &mut self.thread_bins[thread_id]
    }

    /// Merge all thread-local bins into a single global bin
    pub fn merge(&self, global_bins: &mut MacroTileBins) {
        global_bins.clear();

        // Merge large primitives
        for thread_bin in &self.thread_bins {
            global_bins.large_primitives.extend_from_slice(&thread_bin.large_primitives);
        }

        // Merge per-tile bins
        for tile_idx in 0..global_bins.tile_count {
            for thread_bin in &self.thread_bins {
                global_bins.bins[tile_idx].extend_from_slice(&thread_bin.bins[tile_idx]);
            }
        }
    }

    /// Clear all thread-local bins
    pub fn clear_all(&mut self) {
        for bin in &mut self.thread_bins {
            bin.clear();
        }
    }
}

/// Implement PixelTarget trait for MacroTile to enable rasterization directly into tiles
impl PixelTarget for MacroTile {
    #[inline]
    fn width(&self) -> usize {
        self.fb_width
    }

    #[inline]
    fn full_height(&self) -> usize {
        self.fb_height
    }

    #[inline]
    fn rect(&self) -> (usize, usize, usize, usize) {
        (self.x0, self.y0, self.width, self.height)
    }

    #[inline]
    unsafe fn test_depth_and_get_index(
        &mut self,
        x: usize,
        y: usize,
        depth: f32,
    ) -> Option<usize> {
        // Convert to tile-local coordinates
        let local_x = x.wrapping_sub(self.x0);
        let local_y = y.wrapping_sub(self.y0);

        // Bounds check
        if local_x >= self.width || local_y >= self.height {
            return None;
        }

        let idx = self.local_index(local_x, local_y);

        // Depth test
        if depth < self.depth[idx] {
            self.depth[idx] = depth;
            Some(idx)
        } else {
            None
        }
    }

    #[inline]
    fn write_color(&mut self, index: usize, color: u32) {
        self.color[index] = color;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_macrotile_creation() {
        let tile = MacroTile::new(0, 0, 128, 128, 1280, 720);
        assert_eq!(tile.x0, 0);
        assert_eq!(tile.y0, 0);
        assert_eq!(tile.width, 128);
        assert_eq!(tile.height, 128);
        assert_eq!(tile.color.len(), 128 * 128);
        assert_eq!(tile.depth.len(), 128 * 128);
    }

    #[test]
    fn test_macrotile_clear() {
        let mut tile = MacroTile::new(0, 0, 128, 128, 1280, 720);
        tile.clear(0xFF0000FF);

        assert_eq!(tile.color[0], 0xFF0000FF);
        assert_eq!(tile.color[100], 0xFF0000FF);
        assert_eq!(tile.depth[0], f32::INFINITY);
    }

    #[test]
    fn test_macrotile_depth_test() {
        let mut tile = MacroTile::new(0, 0, 128, 128, 1280, 720);

        // First write should pass
        let idx = tile.test_depth(10, 10, 0.5);
        assert!(idx.is_some());

        // Closer depth should pass
        let idx = tile.test_depth(10, 10, 0.3);
        assert!(idx.is_some());

        // Further depth should fail
        let idx = tile.test_depth(10, 10, 0.7);
        assert!(idx.is_none());
    }

    #[test]
    fn test_binning_system() {
        let mut bins = MacroTileBins::new(1280, 720);

        // 1280 / 128 = 10 tiles wide
        // 720 / 128 = 6 tiles tall (5.625, rounded up)
        assert_eq!(bins.tiles_x, 10);
        assert_eq!(bins.tiles_y, 6);
        assert_eq!(bins.tile_count, 60);
    }

    #[test]
    fn test_mesh_binning_small_mesh() {
        let mut bins = MacroTileBins::new(1280, 720);

        // Small mesh in top-left tile (0, 0) to (100, 100)
        bins.add_mesh(MeshId(0), 0, 0, 100, 100, 1280, 720);

        // Should be in tile (0, 0) only
        let bin = bins.get_bin(0, 0);
        assert_eq!(bin.len(), 1);
        assert_eq!(bin[0].0, 0);

        // Should NOT be in tile (1, 0)
        let bin = bins.get_bin(1, 0);
        assert_eq!(bin.len(), 0);
    }

    #[test]
    fn test_mesh_binning_crosses_tiles() {
        let mut bins = MacroTileBins::new(1280, 720);

        // Mesh that crosses 4 tiles: (64, 64) to (192, 192)
        // Covers tiles (0,0), (1,0), (0,1), (1,1)
        bins.add_mesh(MeshId(0), 64, 64, 192, 192, 1280, 720);

        assert_eq!(bins.get_bin(0, 0).len(), 1);
        assert_eq!(bins.get_bin(1, 0).len(), 1);
        assert_eq!(bins.get_bin(0, 1).len(), 1);
        assert_eq!(bins.get_bin(1, 1).len(), 1);
        assert_eq!(bins.get_bin(2, 0).len(), 0);
    }

    #[test]
    fn test_large_primitive_bypass() {
        let mut bins = MacroTileBins::new(1280, 720);

        // Large mesh covering >25% of screen (entire screen in this case)
        let added = bins.add_mesh(MeshId(0), 0, 0, 1279, 719, 1280, 720);

        assert!(!added); // Should NOT be added to bins
        assert_eq!(bins.large_primitives.len(), 1);
        assert_eq!(bins.large_primitives[0].0, 0);
    }

    #[test]
    fn test_thread_local_bins_merge() {
        let mut thread_bins = ThreadLocalBins::new(1280, 720, 4);

        // Thread 0 adds mesh 0
        thread_bins.get_thread_bins(0).add_mesh(MeshId(0), 0, 0, 100, 100, 1280, 720);

        // Thread 1 adds mesh 1
        thread_bins.get_thread_bins(1).add_mesh(MeshId(1), 150, 0, 250, 100, 1280, 720);

        // Merge into global bins
        let mut global_bins = MacroTileBins::new(1280, 720);
        thread_bins.merge(&mut global_bins);

        // Tile (0, 0) should have mesh 0
        assert!(global_bins.get_bin(0, 0).iter().any(|m| m.0 == 0));

        // Tile (1, 0) should have mesh 1
        assert!(global_bins.get_bin(1, 0).iter().any(|m| m.0 == 1));
    }
}
