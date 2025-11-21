use std::f32;

/// Low-resolution depth buffer used for conservative chunk-level occlusion.
/// Stores the minimum depth seen in each cell so that distant chunks that are
/// fully behind nearer geometry can be culled.
pub struct OcclusionBuffer {
    screen_width: usize,
    screen_height: usize,
    grid_width: usize,
    grid_height: usize,
    cells: Vec<f32>,
}

impl OcclusionBuffer {
    pub fn new(screen_width: usize, screen_height: usize, grid_width: usize, grid_height: usize) -> Self {
        let size = grid_width * grid_height;
        Self {
            screen_width,
            screen_height,
            grid_width,
            grid_height,
            cells: vec![f32::INFINITY; size],
        }
    }

    /// Resize to match the current framebuffer size while keeping grid resolution.
    pub fn resize(&mut self, screen_width: usize, screen_height: usize) {
        self.screen_width = screen_width;
        self.screen_height = screen_height;
        self.clear();
    }

    /// Clear occlusion data for a new frame.
    #[inline]
    pub fn clear(&mut self) {
        self.cells.fill(f32::INFINITY);
    }

    /// Update the occlusion buffer at the given pixel position with a depth value.
    /// This records the minimum depth per cell for conservative occlusion.
    #[inline]
    pub fn update(&mut self, x: usize, y: usize, depth: f32) {
        if x >= self.screen_width || y >= self.screen_height {
            return;
        }

        let cx = x * self.grid_width / self.screen_width;
        let cy = y * self.grid_height / self.screen_height;
        let index = cy * self.grid_width + cx;

        let cell = &mut self.cells[index];
        if depth < *cell {
            *cell = depth;
        }
    }

    /// Mark a screen-space rectangle as containing geometry at approximately
    /// `depth`. Used in a pre-pass to build a conservative occluder map.
    #[inline]
    pub fn mark_rect(
        &mut self,
        mut min_x: i32,
        mut min_y: i32,
        mut max_x: i32,
        mut max_y: i32,
        depth: f32,
    ) {
        if self.screen_width == 0 || self.screen_height == 0 {
            return;
        }

        if max_x < 0 || max_y < 0 || min_x >= self.screen_width as i32 || min_y >= self.screen_height as i32 {
            return;
        }

        min_x = min_x.max(0);
        min_y = min_y.max(0);
        max_x = max_x.min(self.screen_width as i32 - 1);
        max_y = max_y.min(self.screen_height as i32 - 1);

        if min_x > max_x || min_y > max_y {
            return;
        }

        let cx0 = (min_x as usize * self.grid_width) / self.screen_width;
        let cx1 = (max_x as usize * self.grid_width) / self.screen_width;
        let cy0 = (min_y as usize * self.grid_height) / self.screen_height;
        let cy1 = (max_y as usize * self.grid_height) / self.screen_height;

        for cy in cy0..=cy1 {
            for cx in cx0..=cx1 {
                let index = cy * self.grid_width + cx;
                let cell = &mut self.cells[index];
                if depth < *cell {
                    *cell = depth;
                }
            }
        }
    }

    /// Returns true if a chunk whose screen-space bounding rectangle is
    /// [min_x, max_x] x [min_y, max_y] and whose nearest depth is `near_depth`
    /// is conservatively considered fully occluded by existing geometry.
    #[inline]
    pub fn is_occluded(
        &self,
        mut min_x: i32,
        mut min_y: i32,
        mut max_x: i32,
        mut max_y: i32,
        near_depth: f32,
    ) -> bool {
        if self.screen_width == 0 || self.screen_height == 0 {
            return false;
        }

        // Clamp rect to screen bounds
        if max_x < 0 || max_y < 0 || min_x >= self.screen_width as i32 || min_y >= self.screen_height as i32 {
            return false;
        }

        min_x = min_x.max(0);
        min_y = min_y.max(0);
        max_x = max_x.min(self.screen_width as i32 - 1);
        max_y = max_y.min(self.screen_height as i32 - 1);

        if min_x > max_x || min_y > max_y {
            return false;
        }

        let cx0 = (min_x as usize * self.grid_width) / self.screen_width;
        let cx1 = (max_x as usize * self.grid_width) / self.screen_width;
        let cy0 = (min_y as usize * self.grid_height) / self.screen_height;
        let cy1 = (max_y as usize * self.grid_height) / self.screen_height;

        // Require occluders to be meaningfully closer in depth to avoid
        // over-aggressive culling from low-resolution cells.
        let epsilon = 0.005;

        for cy in cy0..=cy1 {
            for cx in cx0..=cx1 {
                let index = cy * self.grid_width + cx;
                let cell_depth = self.cells[index];

                // If any overlapping cell does NOT have strictly nearer depth,
                // we cannot guarantee full occlusion.
                if !(cell_depth < near_depth - epsilon) {
                    return false;
                }
            }
        }

        true
    }
}
