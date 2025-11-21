/// Software rasterizer using scanline algorithm
/// Optimized for cache locality and minimal branching
use super::framebuffer::{Framebuffer, FrameSlice, FrameTile};
use super::shading::ShadingConfig;
use super::texture::TextureAtlas;
use super::simd_vertex;
use crate::meshing::{ChunkMesh, FaceDir, TinyQuad, Vertex};
use crate::count_call;
use crate::perf::FUNCTION_COUNTERS;
use std::sync::Arc;
use glam::{Mat4, Vec2, Vec3, Vec4};
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{_mm_add_ps, _mm_mul_ps, _mm_set_ps, _mm_set1_ps, _mm_storeu_ps};

// A clipped quad (4 verts) against one plane can result in at most 5 vertices.
// We use 8 to be safe and aligned.
const MAX_POLY_VERTS: usize = 8;
const NEAR_W_EPS: f32 = 0.001;

/// SIMD-friendly projected quad (SoA layout) used by the span renderer.
/// Keeps the four corners in packed Vec4 registers and stores pre-multiplied
/// attributes for perspective-correct interpolation.
#[repr(C, align(16))]
#[allow(dead_code)]
struct ProjectedQuad {
    xs: Vec4,
    ys: Vec4,
    zs: Vec4,
    inv_w: Vec4,
    us: Vec4,
    vs: Vec4,
    block_type: crate::voxel::BlockType,
    light: f32,
}

/// Edge state for scanline span rasterization.
#[derive(Copy, Clone)]
#[allow(dead_code)]
struct SpanEdge {
    x: f32,
    z: f32,
    u_over_w: f32,
    v_over_w: f32,
    inv_w: f32,
    dx: f32,
    dz: f32,
    du_over_w: f32,
    dv_over_w: f32,
    dinv_w: f32,
}

/// Abstraction over a render target that supports depth-tested pixel writes.
pub trait PixelTarget {
    /// Full framebuffer width (stride for indexing).
    fn width(&self) -> usize;
    /// Full framebuffer height (used for NDC -> screen mapping).
    fn full_height(&self) -> usize;
    /// Rectangle covered by this target in framebuffer coordinates:
    /// (x0, y0, width, height).
    fn rect(&self) -> (usize, usize, usize, usize);
    unsafe fn test_depth_and_get_index(
        &mut self,
        x: usize,
        y: usize,
        depth: f32,
    ) -> Option<usize>;
    fn write_color(&mut self, index: usize, color: u32);
}

impl<'a> PixelTarget for FrameSlice<'a> {
    #[inline]
    fn width(&self) -> usize {
        self.width
    }

    #[inline]
    fn full_height(&self) -> usize {
        self.full_height
    }

    #[inline]
    fn rect(&self) -> (usize, usize, usize, usize) {
        (0, self.y0, self.width, self.height)
    }

    #[inline]
    unsafe fn test_depth_and_get_index(
        &mut self,
        x: usize,
        y: usize,
        depth: f32,
    ) -> Option<usize> {
        FrameSlice::test_depth_and_get_index(self, x, y, depth)
    }

    #[inline]
    fn write_color(&mut self, index: usize, color: u32) {
        FrameSlice::write_color(self, index, color);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voxel::BlockType;

    struct TestTarget {
        width: usize,
        height: usize,
        color: Vec<u32>,
        depth: Vec<f32>,
        pub pixels_written: usize,
    }

    impl TestTarget {
        fn new(width: usize, height: usize) -> Self {
            let len = width * height;
            Self {
                width,
                height,
                color: vec![0; len],
                depth: vec![f32::INFINITY; len],
                pixels_written: 0,
            }
        }
    }

    impl PixelTarget for TestTarget {
        fn width(&self) -> usize {
            self.width
        }

        fn full_height(&self) -> usize {
            self.height
        }

        fn rect(&self) -> (usize, usize, usize, usize) {
            (0, 0, self.width, self.height)
        }

        unsafe fn test_depth_and_get_index(
            &mut self,
            x: usize,
            y: usize,
            depth: f32,
        ) -> Option<usize> {
            if x >= self.width || y >= self.height {
                return None;
            }
            let idx = y * self.width + x;
            if depth < self.depth[idx] {
                self.depth[idx] = depth;
                Some(idx)
            } else {
                None
            }
        }

        fn write_color(&mut self, index: usize, color: u32) {
            self.color[index] = color;
            self.pixels_written += 1;
        }
    }

    #[test]
    fn triangle_crossing_near_plane_is_clipped_and_drawn() {
        let mut rasterizer = Rasterizer::new();
        rasterizer.backface_culling = false; // Focus on clipping behavior
        let mut target = TestTarget::new(8, 8);

        // One vertex behind the near plane (w <= 0), others in front.
        let p0 = Vec4::new(-0.5, -0.5, 0.0, 1.0);
        let p1 = Vec4::new(0.5, -0.5, 0.0, 1.0);
        let p2 = Vec4::new(-0.5, 0.5, 0.0, 0.0); // Behind near plane

        let drawn = rasterizer.render_triangle_from_clip(
            p0,
            p1,
            p2,
            BlockType::Grass,
            1.0,
            &mut target,
        );

        assert!(drawn, "Triangle crossing near plane should be clipped, not dropped");
        assert!(
            target.pixels_written > 0,
            "Clipped triangle should rasterize some pixels"
        );
    }

    #[test]
    fn triangle_fully_behind_near_plane_is_rejected() {
        let rasterizer = Rasterizer::new();
        let mut target = TestTarget::new(8, 8);

        // All vertices behind the near plane (w <= 0)
        let p0 = Vec4::new(-0.5, -0.5, 0.0, 0.0);
        let p1 = Vec4::new(0.5, -0.5, 0.0, 0.0);
        let p2 = Vec4::new(-0.5, 0.5, 0.0, -0.1);

        let drawn = rasterizer.render_triangle_from_clip(
            p0,
            p1,
            p2,
            BlockType::Grass,
            1.0,
            &mut target,
        );

        assert!(
            !drawn,
            "Triangle fully behind near plane should be rejected"
        );
        assert_eq!(
            target.pixels_written, 0,
            "No pixels should be written when triangle is fully clipped"
        );
    }

    #[test]
    fn quad_crossing_near_plane_is_clipped_and_drawn() {
        let mut rasterizer = Rasterizer::new();
        rasterizer.backface_culling = false; // Focus on clipping behavior
        let mut target = TestTarget::new(8, 8);

        let quad = [
            Vec4::new(-0.5, -0.5, 0.0, 1.0),
            Vec4::new(0.5, -0.5, 0.0, 1.0),
            Vec4::new(0.5, 0.5, 0.0, 1.0),
            Vec4::new(-0.5, 0.5, 0.0, -0.2), // behind near plane
        ];

        let drawn = rasterizer.render_convex_polygon(
            &quad,
            BlockType::Grass,
            1.0,
            &mut target,
        );

        assert!(drawn, "Quad crossing near plane should render after clipping");
        assert!(
            target.pixels_written > 0,
            "Clipped quad should rasterize pixels"
        );
    }

    #[test]
    fn degenerate_polygon_is_skipped() {
        let rasterizer = Rasterizer::new();
        let mut target = TestTarget::new(8, 8);

        // All vertices lie on the same line in screen space.
        let quad = [
            Vec4::new(-1.0, 0.0, 0.0, 1.0),
            Vec4::new(1.0, 0.0, 0.0, 1.0),
            Vec4::new(1.0, 0.0, 0.0, 1.0),
            Vec4::new(-1.0, 0.0, 0.0, 1.0),
        ];

        let drawn = rasterizer.render_convex_polygon(
            &quad,
            BlockType::Grass,
            1.0,
            &mut target,
        );

        assert!(!drawn, "Degenerate polygon should be skipped");
        assert_eq!(target.pixels_written, 0);
    }
}

impl PixelTarget for FrameTile {
    #[inline]
    fn width(&self) -> usize {
        self.width
    }

    #[inline]
    fn full_height(&self) -> usize {
        self.full_height
    }

    #[inline]
    fn rect(&self) -> (usize, usize, usize, usize) {
        (self.x0, self.y0, self.tile_width, self.tile_height)
    }

    #[inline]
    unsafe fn test_depth_and_get_index(
        &mut self,
        x: usize,
        y: usize,
        depth: f32,
    ) -> Option<usize> {
        FrameTile::test_depth_and_get_index(self, x, y, depth)
    }

    #[inline]
    fn write_color(&mut self, index: usize, color: u32) {
        FrameTile::write_color(self, index, color);
    }
}

/// Lightweight projected point for rasterization
/// This is used instead of full Vertex structs to minimize memory traffic
/// Fits in 16 bytes (1 cache line per 4 points with AVX)
#[derive(Copy, Clone)]
struct ProjectedPoint {
    clip: Vec4, // 16 bytes
}

/// Internal representation of a vertex in clip space
/// used during near-plane clipping.
#[derive(Copy, Clone)]
struct ClipVertex {
    vertex: Vertex,
    clip_pos: Vec4,
}

/// Clip-space vertex with attached texture coordinates for near-plane clipping.
#[derive(Copy, Clone)]
struct ClipTexturedVertex {
    clip_pos: Vec4,
    uv: glam::Vec2,
}

#[cfg(target_arch = "x86_64")]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum SimdMode {
    Scalar,
    Sse2,
}

pub struct Rasterizer {
    pub backface_culling: bool,
    pub shading: ShadingConfig,
    /// If false, skip per-pixel lighting and render
    /// geometry with a flat, unlit color.
    pub enable_shading: bool,
    /// Shared texture atlas for all block types.
    pub atlas: Arc<TextureAtlas>,
    // Scratch buffer for clip-space positions so we only transform
    // each vertex once per mesh rather than per triangle.
    clip_space_positions: Vec<Vec4>,
    #[cfg(target_arch = "x86_64")]
    simd_mode: SimdMode,
}

impl Rasterizer {
    pub fn new() -> Self {
        Self::new_with_atlas(Arc::new(TextureAtlas::default()))
    }

    /// Create a rasterizer with a specific texture atlas.
    /// This is useful when sharing an atlas across worker rasterizers.
    pub fn new_with_atlas(atlas: Arc<TextureAtlas>) -> Self {
        #[cfg(target_arch = "x86_64")]
        let simd_mode = if std::arch::is_x86_feature_detected!("sse2") {
            SimdMode::Sse2
        } else {
            SimdMode::Scalar
        };

        Self {
            backface_culling: true,
            shading: ShadingConfig::default(),
            enable_shading: true,
            atlas,
            clip_space_positions: Vec::new(),
            #[cfg(target_arch = "x86_64")]
            simd_mode,
        }
    }

    #[inline]
    fn is_camera_level(&self, camera_up: Option<Vec3>) -> bool {
        // Treat camera as "level" when its up vector aligns with world up.
        // When no hint is provided we assume the common FPS case (level).
        const LEVEL_THRESHOLD: f32 = 0.995;
        camera_up.map_or(true, |up| up.y.abs() >= LEVEL_THRESHOLD)
    }

    /// Render a mesh to the full framebuffer (single-threaded path).
    pub fn render_mesh(
        &mut self,
        mesh: &ChunkMesh,
        view_proj: &Mat4,
        framebuffer: &mut Framebuffer,
    ) {
        let mut slices = framebuffer.split_into_stripes(1);
        if let Some(slice) = slices.first_mut() {
            self.render_mesh_into_target(mesh, view_proj, slice, None);
        }
    }

    /// Version of `render_mesh` that accepts a camera up vector to decide
    /// whether the span renderer (fast path) can be used safely.
    pub fn render_mesh_with_up(
        &mut self,
        mesh: &ChunkMesh,
        view_proj: &Mat4,
        framebuffer: &mut Framebuffer,
        camera_up: Vec3,
    ) {
        let mut slices = framebuffer.split_into_stripes(1);
        if let Some(slice) = slices.first_mut() {
            self.render_mesh_into_target(mesh, view_proj, slice, Some(camera_up));
        }
    }

    /// Render a mesh into a specific framebuffer slice (used for multi-core stripes).
    pub fn render_mesh_into_slice(
        &mut self,
        mesh: &ChunkMesh,
        view_proj: &Mat4,
        slice: &mut FrameSlice<'_>,
    ) {
        self.render_mesh_into_target(mesh, view_proj, slice, None);
    }

    /// Render a mesh into a framebuffer tile (used for tile-based parallel rendering).
    pub fn render_mesh_into_tile(
        &mut self,
        mesh: &ChunkMesh,
        view_proj: &Mat4,
        tile: &mut FrameTile,
    ) {
        // Use the generic path so we can reuse fast span rendering logic.
        self.render_mesh_into_target(mesh, view_proj, tile, None);
    }

    /// Optimized tile rendering with direct pointer access
    fn render_mesh_into_tile_optimized(
        &mut self,
        mesh: &ChunkMesh,
        view_proj: &Mat4,
        tile: &mut FrameTile,
    ) {
        if mesh.is_empty() {
            return;
        }

        let chunk_offset = mesh.world_offset();

        // Process each face direction
        let face_dirs = [
            FaceDir::PosX,
            FaceDir::NegX,
            FaceDir::PosY,
            FaceDir::NegY,
            FaceDir::PosZ,
            FaceDir::NegZ,
        ];

        for &face_dir in &face_dirs {
            let face_list = mesh.face_list(face_dir);

            if face_list.is_empty() {
                continue;
            }

            let normal = face_dir.normal();
            let light = self.compute_face_lighting(face_dir);

            for (slice_idx, quads) in face_list.slice_quads.iter().enumerate() {
                if quads.is_empty() {
                    continue;
                }

                let slice_pos = if face_dir.is_positive() {
                    (slice_idx + 1) as u8
                } else {
                    slice_idx as u8
                };

                for quad in quads {
                    self.render_tiny_quad_optimized(
                        quad,
                        face_dir,
                        slice_pos,
                        chunk_offset,
                        normal,
                        light,
                        view_proj,
                        tile,
                    );
                }
            }
        }
    }

    /// Optimized quad rendering for tiles with direct pointer access
    fn render_tiny_quad_optimized(
        &mut self,
        quad: &TinyQuad,
        face_dir: FaceDir,
        slice_pos: u8,
        chunk_offset: glam::Vec3,
        _normal: glam::Vec3,
        light: f32,
        view_proj: &Mat4,
        tile: &mut FrameTile,
    ) {
        use crate::voxel::BlockType;

        let u = quad.u();
        let v = quad.v();
        let w = quad.width();
        let h = quad.height();
        let block_type = BlockType::from_u8(quad.block_type());

        // Generate 4 local positions
        let local_positions: [(f32, f32, f32); 4] = match face_dir {
            FaceDir::PosX => [
                (slice_pos as f32, u as f32, v as f32),
                (slice_pos as f32, (u + w) as f32, v as f32),
                (slice_pos as f32, (u + w) as f32, (v + h) as f32),
                (slice_pos as f32, u as f32, (v + h) as f32),
            ],
            FaceDir::NegX => [
                (slice_pos as f32, u as f32, v as f32),
                (slice_pos as f32, u as f32, (v + h) as f32),
                (slice_pos as f32, (u + w) as f32, (v + h) as f32),
                (slice_pos as f32, (u + w) as f32, v as f32),
            ],
            FaceDir::PosY => [
                (u as f32, slice_pos as f32, v as f32),
                (u as f32, slice_pos as f32, (v + h) as f32),
                ((u + w) as f32, slice_pos as f32, (v + h) as f32),
                ((u + w) as f32, slice_pos as f32, v as f32),
            ],
            FaceDir::NegY => [
                (u as f32, slice_pos as f32, v as f32),
                ((u + w) as f32, slice_pos as f32, v as f32),
                ((u + w) as f32, slice_pos as f32, (v + h) as f32),
                (u as f32, slice_pos as f32, (v + h) as f32),
            ],
            FaceDir::PosZ => [
                (u as f32, v as f32, slice_pos as f32),
                ((u + w) as f32, v as f32, slice_pos as f32),
                ((u + w) as f32, (v + h) as f32, slice_pos as f32),
                (u as f32, (v + h) as f32, slice_pos as f32),
            ],
            FaceDir::NegZ => [
                (u as f32, v as f32, slice_pos as f32),
                (u as f32, (v + h) as f32, slice_pos as f32),
                ((u + w) as f32, (v + h) as f32, slice_pos as f32),
                ((u + w) as f32, v as f32, slice_pos as f32),
            ],
        };

        let u_start = u as f32;
        let v_start = v as f32;
        let u_end = (u + w) as f32;
        let v_end = (v + h) as f32;

        let uvs: [Vec2; 4] = match face_dir {
            FaceDir::PosX => [
                Vec2::new(u_start, v_start),
                Vec2::new(u_end, v_start),
                Vec2::new(u_end, v_end),
                Vec2::new(u_start, v_end),
            ],
            FaceDir::NegX => [
                Vec2::new(u_start, v_start),
                Vec2::new(u_start, v_end),
                Vec2::new(u_end, v_end),
                Vec2::new(u_end, v_start),
            ],
            FaceDir::PosY => [
                Vec2::new(u_start, v_start),
                Vec2::new(u_start, v_end),
                Vec2::new(u_end, v_end),
                Vec2::new(u_end, v_start),
            ],
            FaceDir::NegY => [
                Vec2::new(u_start, v_start),
                Vec2::new(u_end, v_start),
                Vec2::new(u_end, v_end),
                Vec2::new(u_start, v_end),
            ],
            FaceDir::PosZ => [
                Vec2::new(u_start, v_start),
                Vec2::new(u_end, v_start),
                Vec2::new(u_end, v_end),
                Vec2::new(u_start, v_end),
            ],
            FaceDir::NegZ => [
                Vec2::new(u_start, v_start),
                Vec2::new(u_start, v_end),
                Vec2::new(u_end, v_end),
                Vec2::new(u_end, v_start),
            ],
        };

        // Transform to clip space
        let mut clip_pos = [Vec4::ZERO; 4];
        for i in 0..4 {
            let local = local_positions[i];
            let world = glam::Vec3::new(
                chunk_offset.x + local.0,
                chunk_offset.y + local.1,
                chunk_offset.z + local.2,
            );
            clip_pos[i] = *view_proj * world.extend(1.0);
        }

        // Split quad into two triangles
        let tri_indices = [(0usize, 1usize, 2usize), (0usize, 2usize, 3usize)];
        for &(i0, i1, i2) in &tri_indices {
            self.render_triangle_textured_tile_optimized(
                clip_pos[i0],
                clip_pos[i1],
                clip_pos[i2],
                uvs[i0],
                uvs[i1],
                uvs[i2],
                block_type,
                light,
                tile,
            );
        }
    }

    /// Generic entry for rendering into any pixel target (stripes or tiles).
    fn render_mesh_into_target<T: PixelTarget>(
        &mut self,
        mesh: &ChunkMesh,
        view_proj: &Mat4,
        target: &mut T,
        camera_up: Option<Vec3>,
    ) {
        // Check if using new format (TinyQuad-based faces) or legacy format
        #[allow(deprecated)]
        let use_legacy = !mesh.vertices.is_empty() || !mesh.indices.is_empty();

        if use_legacy {
            #[allow(deprecated)]
            self.render_mesh_legacy(mesh, view_proj, target);
        } else {
            let is_level = self.is_camera_level(camera_up);
            self.render_mesh_tiny_quads(mesh, view_proj, target, is_level);
        }
    }

    /// Legacy rendering path for vertex/index based meshes
    #[allow(deprecated)]
    fn render_mesh_legacy<T: PixelTarget>(
        &mut self,
        mesh: &ChunkMesh,
        view_proj: &Mat4,
        target: &mut T,
    ) {
        if mesh.vertices.is_empty() || mesh.indices.is_empty() {
            return;
        }

        // Ensure scratch buffer has enough space and transform each vertex once.
        let vertex_count = mesh.vertices.len();
        if self.clip_space_positions.len() < vertex_count {
            self.clip_space_positions
                .resize(vertex_count, Vec4::new(0.0, 0.0, 0.0, 0.0));
        }

        // Get chunk world offset for vertex decompression
        let chunk_offset = mesh.world_offset();

        // Transform vertices: decompress u8 coords -> world position -> clip space
        // Using SIMD-accelerated batch processing (8 vertices at a time with AVX2)
        simd_vertex::decompress_and_transform_vertices(
            &mesh.vertices,
            chunk_offset,
            view_proj,
            &mut self.clip_space_positions[0..vertex_count],
        );

        // Process triangles
        for i in (0..mesh.indices.len()).step_by(3) {
            let i0 = mesh.indices[i] as usize;
            let i1 = mesh.indices[i + 1] as usize;
            let i2 = mesh.indices[i + 2] as usize;

            let v0 = &mesh.vertices[i0];
            let v1 = &mesh.vertices[i1];
            let v2 = &mesh.vertices[i2];

            let p0_clip = self.clip_space_positions[i0];
            let p1_clip = self.clip_space_positions[i1];
            let p2_clip = self.clip_space_positions[i2];

            self.render_triangle_pretransformed(
                v0,
                v1,
                v2,
                p0_clip,
                p1_clip,
                p2_clip,
                target,
            );
        }
    }

    /// Clip a convex polygon against the near plane (w >= NEAR_W_EPS).
    /// Returns the number of output vertices written to `output`.
    fn clip_polygon_near(input: &[Vec4], output: &mut [Vec4; MAX_POLY_VERTS]) -> usize {
        if input.is_empty() {
            return 0;
        }

        let mut out_len = 0usize;
        let mut prev = *input.last().unwrap();
        let mut prev_inside = prev.w >= NEAR_W_EPS;

        for &curr in input {
            let curr_inside = curr.w >= NEAR_W_EPS;
            match (prev_inside, curr_inside) {
                (true, true) => {
                    output[out_len] = curr;
                    out_len += 1;
                }
                (true, false) => {
                    let inter = Self::intersect_near_vec4(prev, curr, NEAR_W_EPS);
                    output[out_len] = inter;
                    out_len += 1;
                }
                (false, true) => {
                    let inter = Self::intersect_near_vec4(prev, curr, NEAR_W_EPS);
                    output[out_len] = inter;
                    out_len += 1;
                    output[out_len] = curr;
                    out_len += 1;
                }
                (false, false) => {}
            }

            prev = curr;
            prev_inside = curr_inside;
        }

        out_len
    }

    /// Render a convex polygon (tri/quad/pentagon after clipping).
    /// Triangulates into a fan for correctness (perspective-correct depth).
    /// Returns true if any pixels were drawn.
    fn render_convex_polygon<T: PixelTarget>(
        &self,
        clip_vertices: &[Vec4],
        block_type: crate::voxel::BlockType,
        light: f32,
        target: &mut T,
    ) -> bool {
        if clip_vertices.len() < 3 {
            return false;
        }

        // Clip against near plane
        let mut clipped = [Vec4::ZERO; MAX_POLY_VERTS];
        let clipped_len = Self::clip_polygon_near(clip_vertices, &mut clipped);
        if clipped_len < 3 {
            return false;
        }

        let mut any_drawn = false;
        for i in 1..(clipped_len - 1) {
            if self.render_triangle_from_clip(
                clipped[0],
                clipped[i],
                clipped[i + 1],
                block_type,
                light,
                target,
            ) {
                any_drawn = true;
            }
        }
        any_drawn
    }

    /// New rendering path for TinyQuad-based face-direction-separated meshes
    pub fn render_mesh_tiny_quads<T: PixelTarget>(
        &mut self,
        mesh: &ChunkMesh,
        view_proj: &Mat4,
        target: &mut T,
        use_span_renderer: bool,
    ) {
        if mesh.is_empty() {
            return;
        }

        let chunk_offset = mesh.world_offset();

        // Process each face direction
        let face_dirs = [
            FaceDir::PosX,
            FaceDir::NegX,
            FaceDir::PosY,
            FaceDir::NegY,
            FaceDir::PosZ,
            FaceDir::NegZ,
        ];

        for &face_dir in &face_dirs {
            let face_list = mesh.face_list(face_dir);

            if face_list.is_empty() {
                continue;
            }

            // Face-list level culling: project its local AABB and skip if it doesn't touch this target.
            let mut min_local = face_list.min;
            let mut max_local = face_list.max;
            if min_local.x > max_local.x || min_local.y > max_local.y || min_local.z > max_local.z {
                continue;
            }

            // Convert local AABB to world space
            let world_min = Vec3::new(
                chunk_offset.x + min_local.x as f32,
                chunk_offset.y + min_local.y as f32,
                chunk_offset.z + min_local.z as f32,
            );
            let world_max = Vec3::new(
                chunk_offset.x + max_local.x as f32,
                chunk_offset.y + max_local.y as f32,
                chunk_offset.z + max_local.z as f32,
            );

            // Project bbox corners to screen and compute rect
            let corners = [
                Vec3::new(world_min.x, world_min.y, world_min.z),
                Vec3::new(world_max.x, world_min.y, world_min.z),
                Vec3::new(world_min.x, world_max.y, world_min.z),
                Vec3::new(world_max.x, world_max.y, world_min.z),
                Vec3::new(world_min.x, world_min.y, world_max.z),
                Vec3::new(world_max.x, world_min.y, world_max.z),
                Vec3::new(world_min.x, world_max.y, world_max.z),
                Vec3::new(world_max.x, world_max.y, world_max.z),
            ];

            let mut rect_min_x = i32::MAX;
            let mut rect_min_y = i32::MAX;
            let mut rect_max_x = i32::MIN;
            let mut rect_max_y = i32::MIN;
            let mut any_behind = false;

            for corner in &corners {
                let clip = *view_proj * corner.extend(1.0);

                if clip.w < 0.001 {
                    any_behind = true;
                }

                if clip.w.abs() > 1e-4 {
                    let ndc = clip / clip.w;
                    let sx = (ndc.x + 1.0) * 0.5 * target.width() as f32;
                    let sy = (1.0 - ndc.y) * 0.5 * target.full_height() as f32;

                    rect_min_x = rect_min_x.min(sx.floor() as i32);
                    rect_max_x = rect_max_x.max(sx.ceil() as i32);
                    rect_min_y = rect_min_y.min(sy.floor() as i32);
                    rect_max_y = rect_max_y.max(sy.ceil() as i32);
                }
            }

            if !any_behind {
                // Intersect with target rect
                let (tx0, ty0, tw, th) = target.rect();
                let tx1 = (tx0 + tw - 1) as i32;
                let ty1 = (ty0 + th - 1) as i32;

                if rect_max_x < tx0 as i32
                    || rect_min_x > tx1
                    || rect_max_y < ty0 as i32
                    || rect_min_y > ty1
                {
                    continue;
                }
            }
            let normal = face_dir.normal();

            // Compute lighting for this face direction
            let light = self.compute_face_lighting(face_dir);

            // Process each slice
            for (slice_idx, quads) in face_list.slice_quads.iter().enumerate() {
                if quads.is_empty() {
                    continue;
                }

                // Convert slice index back to actual position
                // For positive faces, we stored (axis_pos - 1), so we need to add 1 back
                // For negative faces, we stored axis_pos directly
                let slice_pos = if face_dir.is_positive() {
                    (slice_idx + 1) as u8
                } else {
                    slice_idx as u8
                };

                // Render each quad in this slice
                for quad in quads {
                    if use_span_renderer {
                        self.render_tiny_quad_span(
                            quad,
                            face_dir,
                            slice_pos,
                            chunk_offset,
                            light,
                            view_proj,
                            target,
                        );
                    } else {
                        self.render_tiny_quad(
                            quad,
                            face_dir,
                            slice_pos,
                            chunk_offset,
                            normal,
                            light,
                            view_proj,
                            target,
                        );
                    }
                }
            }
        }
    }

    /// Render a single TinyQuad as two triangles
    fn render_tiny_quad<T: PixelTarget>(
        &mut self,
        quad: &TinyQuad,
        face_dir: FaceDir,
        slice_pos: u8,
        chunk_offset: glam::Vec3,
        _normal: glam::Vec3,
        light: f32,
        view_proj: &Mat4,
        target: &mut T,
    ) {
        use crate::voxel::BlockType;

        // Extract quad parameters
        let u = quad.u();
        let v = quad.v();
        let w = quad.width();
        let h = quad.height();
        let block_type = BlockType::from_u8(quad.block_type());

        // Generate 4 local positions directly without Vertex intermediate
        // This eliminates the memory expansion from TinyQuad (3 bytes) -> 4x Vertex (32 bytes)
        let local_positions: [(f32, f32, f32); 4] = match face_dir {
            FaceDir::PosX => [
                (slice_pos as f32, u as f32, v as f32),
                (slice_pos as f32, (u + w) as f32, v as f32),
                (slice_pos as f32, (u + w) as f32, (v + h) as f32),
                (slice_pos as f32, u as f32, (v + h) as f32),
            ],
            FaceDir::NegX => [
                (slice_pos as f32, u as f32, v as f32),
                (slice_pos as f32, u as f32, (v + h) as f32),
                (slice_pos as f32, (u + w) as f32, (v + h) as f32),
                (slice_pos as f32, (u + w) as f32, v as f32),
            ],
            FaceDir::PosY => [
                (u as f32, slice_pos as f32, v as f32),
                (u as f32, slice_pos as f32, (v + h) as f32),
                ((u + w) as f32, slice_pos as f32, (v + h) as f32),
                ((u + w) as f32, slice_pos as f32, v as f32),
            ],
            FaceDir::NegY => [
                (u as f32, slice_pos as f32, v as f32),
                ((u + w) as f32, slice_pos as f32, v as f32),
                ((u + w) as f32, slice_pos as f32, (v + h) as f32),
                (u as f32, slice_pos as f32, (v + h) as f32),
            ],
            FaceDir::PosZ => [
                (u as f32, v as f32, slice_pos as f32),
                ((u + w) as f32, v as f32, slice_pos as f32),
                ((u + w) as f32, (v + h) as f32, slice_pos as f32),
                (u as f32, (v + h) as f32, slice_pos as f32),
            ],
            FaceDir::NegZ => [
                (u as f32, v as f32, slice_pos as f32),
                (u as f32, (v + h) as f32, slice_pos as f32),
                ((u + w) as f32, (v + h) as f32, slice_pos as f32),
                ((u + w) as f32, v as f32, slice_pos as f32),
            ],
        };

        // Generate UVs based on face direction and greedy quad extents.
        // U/V are derived from the TinyQuad's (u, v, w, h) parameters,
        // allowing large quads (e.g., 32x4) to tile the 8x8 micro-texture.
        let u_start = u as f32;
        let v_start = v as f32;
        let u_end = (u + w) as f32;
        let v_end = (v + h) as f32;

        let uvs: [Vec2; 4] = match face_dir {
            // X faces: map TinyQuad (u,v) directly
            FaceDir::PosX => [
                Vec2::new(u_start, v_start),
                Vec2::new(u_end, v_start),
                Vec2::new(u_end, v_end),
                Vec2::new(u_start, v_end),
            ],
            FaceDir::NegX => [
                Vec2::new(u_start, v_start),
                Vec2::new(u_start, v_end),
                Vec2::new(u_end, v_end),
                Vec2::new(u_end, v_start),
            ],
            // Y faces: X is U, Z is V
            FaceDir::PosY => [
                Vec2::new(u_start, v_start),
                Vec2::new(u_start, v_end),
                Vec2::new(u_end, v_end),
                Vec2::new(u_end, v_start),
            ],
            FaceDir::NegY => [
                Vec2::new(u_start, v_start),
                Vec2::new(u_end, v_start),
                Vec2::new(u_end, v_end),
                Vec2::new(u_start, v_end),
            ],
            // Z faces: X is U, Y is V
            FaceDir::PosZ => [
                Vec2::new(u_start, v_start),
                Vec2::new(u_end, v_start),
                Vec2::new(u_end, v_end),
                Vec2::new(u_start, v_end),
            ],
            FaceDir::NegZ => [
                Vec2::new(u_start, v_start),
                Vec2::new(u_start, v_end),
                Vec2::new(u_end, v_end),
                Vec2::new(u_end, v_start),
            ],
        };

        // Transform directly to clip space
        let mut clip_pos = [Vec4::ZERO; 4];
        for i in 0..4 {
            let local = local_positions[i];
            let world = glam::Vec3::new(
                chunk_offset.x + local.0,
                chunk_offset.y + local.1,
                chunk_offset.z + local.2,
            );
            clip_pos[i] = *view_proj * world.extend(1.0);
        }

        // Split the quad into two triangles and render with texturing.
        let tri_indices = [(0usize, 1usize, 2usize), (0usize, 2usize, 3usize)];
        for &(i0, i1, i2) in &tri_indices {
            let p0 = clip_pos[i0];
            let p1 = clip_pos[i1];
            let p2 = clip_pos[i2];

            let uv0 = uvs[i0];
            let uv1 = uvs[i1];
            let uv2 = uvs[i2];

            self.render_triangle_from_clip_textured(
                p0, p1, p2, uv0, uv1, uv2, block_type, light, target,
            );
        }
    }

    /// Span-based rasterization path for level cameras (fast path).
    /// Falls back to barycentric depth test if roll is detected elsewhere.
    fn render_tiny_quad_span<T: PixelTarget>(
        &mut self,
        quad: &TinyQuad,
        face_dir: FaceDir,
        slice_pos: u8,
        chunk_offset: glam::Vec3,
        light: f32,
        view_proj: &Mat4,
        target: &mut T,
    ) {
        use crate::voxel::BlockType;

        let u = quad.u();
        let v = quad.v();
        let w = quad.width();
        let h = quad.height();
        let block_type = BlockType::from_u8(quad.block_type());

        let local_positions: [(f32, f32, f32); 4] = match face_dir {
            FaceDir::PosX => [
                (slice_pos as f32, u as f32, v as f32),
                (slice_pos as f32, (u + w) as f32, v as f32),
                (slice_pos as f32, (u + w) as f32, (v + h) as f32),
                (slice_pos as f32, u as f32, (v + h) as f32),
            ],
            FaceDir::NegX => [
                (slice_pos as f32, u as f32, v as f32),
                (slice_pos as f32, u as f32, (v + h) as f32),
                (slice_pos as f32, (u + w) as f32, (v + h) as f32),
                (slice_pos as f32, (u + w) as f32, v as f32),
            ],
            FaceDir::PosY => [
                (u as f32, slice_pos as f32, v as f32),
                (u as f32, slice_pos as f32, (v + h) as f32),
                ((u + w) as f32, slice_pos as f32, (v + h) as f32),
                ((u + w) as f32, slice_pos as f32, v as f32),
            ],
            FaceDir::NegY => [
                (u as f32, slice_pos as f32, v as f32),
                ((u + w) as f32, slice_pos as f32, v as f32),
                ((u + w) as f32, slice_pos as f32, (v + h) as f32),
                (u as f32, slice_pos as f32, (v + h) as f32),
            ],
            FaceDir::PosZ => [
                (u as f32, v as f32, slice_pos as f32),
                ((u + w) as f32, v as f32, slice_pos as f32),
                ((u + w) as f32, (v + h) as f32, slice_pos as f32),
                (u as f32, (v + h) as f32, slice_pos as f32),
            ],
            FaceDir::NegZ => [
                (u as f32, v as f32, slice_pos as f32),
                (u as f32, (v + h) as f32, slice_pos as f32),
                ((u + w) as f32, (v + h) as f32, slice_pos as f32),
                ((u + w) as f32, v as f32, slice_pos as f32),
            ],
        };

        let u_start = u as f32;
        let v_start = v as f32;
        let u_end = (u + w) as f32;
        let v_end = (v + h) as f32;

        let uvs: [Vec2; 4] = match face_dir {
            FaceDir::PosX => [
                Vec2::new(u_start, v_start),
                Vec2::new(u_end, v_start),
                Vec2::new(u_end, v_end),
                Vec2::new(u_start, v_end),
            ],
            FaceDir::NegX => [
                Vec2::new(u_start, v_start),
                Vec2::new(u_start, v_end),
                Vec2::new(u_end, v_end),
                Vec2::new(u_end, v_start),
            ],
            FaceDir::PosY => [
                Vec2::new(u_start, v_start),
                Vec2::new(u_start, v_end),
                Vec2::new(u_end, v_end),
                Vec2::new(u_end, v_start),
            ],
            FaceDir::NegY => [
                Vec2::new(u_start, v_start),
                Vec2::new(u_end, v_start),
                Vec2::new(u_end, v_end),
                Vec2::new(u_start, v_end),
            ],
            FaceDir::PosZ => [
                Vec2::new(u_start, v_start),
                Vec2::new(u_end, v_start),
                Vec2::new(u_end, v_end),
                Vec2::new(u_start, v_end),
            ],
            FaceDir::NegZ => [
                Vec2::new(u_start, v_start),
                Vec2::new(u_start, v_end),
                Vec2::new(u_end, v_end),
                Vec2::new(u_end, v_start),
            ],
        };

        // Transform to clip space
        let mut clip_pos = [Vec4::ZERO; 4];
        for i in 0..4 {
            let local = local_positions[i];
            let world = glam::Vec3::new(
                chunk_offset.x + local.0,
                chunk_offset.y + local.1,
                chunk_offset.z + local.2,
            );
            clip_pos[i] = *view_proj * world.extend(1.0);
        }

        let tri_indices = [(0usize, 1usize, 2usize), (0usize, 2usize, 3usize)];
        for &(i0, i1, i2) in &tri_indices {
            let p0 = clip_pos[i0];
            let p1 = clip_pos[i1];
            let p2 = clip_pos[i2];

            let uv0 = uvs[i0];
            let uv1 = uvs[i1];
            let uv2 = uvs[i2];

            self.render_triangle_span_from_clip(
                p0, p1, p2, uv0, uv1, uv2, block_type, light, target,
            );
        }
    }

    /// Compute lighting value for a face based on its direction
    fn compute_face_lighting(&self, face_dir: FaceDir) -> f32 {
        // Default light direction: Vec3::new(0.4, 1.0, 0.3).normalize()
        const LIGHT_DIR_X: f32 = 0.35634832;
        const LIGHT_DIR_Y: f32 = 0.8908708;
        const LIGHT_DIR_Z: f32 = 0.2672612;
        const AMBIENT: f32 = 0.35;
        const DIFFUSE: f32 = 0.65;

        let normal = face_dir.normal();
        let lambert = (normal.x * LIGHT_DIR_X + normal.y * LIGHT_DIR_Y + normal.z * LIGHT_DIR_Z).max(0.0);
        let light = AMBIENT + DIFFUSE * lambert;
        light.clamp(0.0, 1.0)
    }

    /// Fast span renderer for textured triangles. Assumes camera roll is near zero.
    fn render_triangle_span_from_clip<T: PixelTarget>(
        &self,
        p0_clip: Vec4,
        p1_clip: Vec4,
        p2_clip: Vec4,
        uv0: glam::Vec2,
        uv1: glam::Vec2,
        uv2: glam::Vec2,
        block_type: crate::voxel::BlockType,
        light: f32,
        target: &mut T,
    ) -> bool {
        count_call!(FUNCTION_COUNTERS.render_triangle_calls);
        count_call!(FUNCTION_COUNTERS.total_triangles_processed);

        // Clip against near plane first
        let tri = [
            ClipTexturedVertex {
                clip_pos: p0_clip,
                uv: uv0,
            },
            ClipTexturedVertex {
                clip_pos: p1_clip,
                uv: uv1,
            },
            ClipTexturedVertex {
                clip_pos: p2_clip,
                uv: uv2,
            },
        ];

        let (tri_count, clipped_tris) = Self::clip_triangle_near_textured(&tri, NEAR_W_EPS);
        if tri_count == 0 {
            count_call!(FUNCTION_COUNTERS.render_triangle_clipped);
            return false;
        }

        let fb_width = target.width() as f32;
        let fb_height = target.full_height() as f32;
        let (rect_x0, rect_y0, rect_w, rect_h) = target.rect();
        let rect_x1 = rect_x0 + rect_w - 1;
        let rect_y1 = rect_y0 + rect_h - 1;

        let tex_id = block_type.texture_id();
        let texture = unsafe { self.atlas.textures.get_unchecked(tex_id) };

        let mut any_drawn = false;

        for tri in clipped_tris.iter().take(tri_count) {
            // Perspective divide
            let verts_ndc = [
                tri[0].clip_pos / tri[0].clip_pos.w,
                tri[1].clip_pos / tri[1].clip_pos.w,
                tri[2].clip_pos / tri[2].clip_pos.w,
            ];

            // Backface culling
            if self.backface_culling {
                let v01 = verts_ndc[1] - verts_ndc[0];
                let v02 = verts_ndc[2] - verts_ndc[0];
                let cross_z = v01.x * v02.y - v01.y * v02.x;
                if cross_z <= 0.0 {
                    count_call!(FUNCTION_COUNTERS.render_triangle_culled);
                    continue;
                }
            }

            // Convert to screen space
            let mut verts_screen = [Vec2::ZERO; 3];
            for i in 0..3 {
                verts_screen[i] = Self::ndc_to_screen(verts_ndc[i].truncate().truncate(), fb_width, fb_height);
            }

            // Screen-space bounding box
            let mut min_y = verts_screen[0].y.min(verts_screen[1].y).min(verts_screen[2].y);
            let mut max_y = verts_screen[0].y.max(verts_screen[1].y).max(verts_screen[2].y);

            // Clamp to target rectangle
            min_y = min_y.max(rect_y0 as f32);
            max_y = max_y.min(rect_y1 as f32);

            if min_y > max_y {
                continue;
            }

            // Prepare per-vertex attributes for interpolation
            #[derive(Copy, Clone)]
            struct SpanVertex {
                x: f32,
                y: f32,
                z: f32,
                u_over_w: f32,
                v_over_w: f32,
                inv_w: f32,
            }

            let verts_span = [
                SpanVertex {
                    x: verts_screen[0].x,
                    y: verts_screen[0].y,
                    z: verts_ndc[0].z,
                    u_over_w: tri[0].uv.x / verts_ndc[0].w,
                    v_over_w: tri[0].uv.y / verts_ndc[0].w,
                    inv_w: 1.0 / verts_ndc[0].w,
                },
                SpanVertex {
                    x: verts_screen[1].x,
                    y: verts_screen[1].y,
                    z: verts_ndc[1].z,
                    u_over_w: tri[1].uv.x / verts_ndc[1].w,
                    v_over_w: tri[1].uv.y / verts_ndc[1].w,
                    inv_w: 1.0 / verts_ndc[1].w,
                },
                SpanVertex {
                    x: verts_screen[2].x,
                    y: verts_screen[2].y,
                    z: verts_ndc[2].z,
                    u_over_w: tri[2].uv.x / verts_ndc[2].w,
                    v_over_w: tri[2].uv.y / verts_ndc[2].w,
                    inv_w: 1.0 / verts_ndc[2].w,
                },
            ];

            // Integer scan range
            // Use conservative bounding box (floor/ceil) then sample at pixel centers
            // This matches the barycentric renderer's approach
            let y_start = min_y.floor() as i32;
            let y_end = max_y.ceil() as i32;

            for y in y_start..=y_end {
                // Respect target rectangle
                if y < rect_y0 as i32 || y > rect_y1 as i32 {
                    continue;
                }

                let y_center = y as f32 + 0.5;

                // Intersections with scanline
                let mut points: [SpanVertex; 2] = [verts_span[0]; 2];
                let mut count = 0usize;

                for i in 0..3 {
                    let v0 = verts_span[i];
                    let v1 = verts_span[(i + 1) % 3];
                    let y0 = v0.y;
                    let y1 = v1.y;

                    // Half-open interval test to avoid double counting shared vertices
                    if (y0 <= y_center && y_center < y1) || (y1 <= y_center && y_center < y0) {
                        let dy = y1 - y0;
                        if dy.abs() < 1e-6 {
                            continue;
                        }
                        let t = (y_center - y0) / dy;
                        let lerp = |a: f32, b: f32| a + (b - a) * t;
                        points[count] = SpanVertex {
                            x: lerp(v0.x, v1.x),
                            y: y_center,
                            z: lerp(v0.z, v1.z),
                            u_over_w: lerp(v0.u_over_w, v1.u_over_w),
                            v_over_w: lerp(v0.v_over_w, v1.v_over_w),
                            inv_w: lerp(v0.inv_w, v1.inv_w),
                        };
                        count += 1;
                        if count == 2 {
                            break;
                        }
                    }
                }

                if count < 2 {
                    continue;
                }

                // Sort left/right
                if points[0].x > points[1].x {
                    points.swap(0, 1);
                }

                let x_start_f = points[0].x.max(rect_x0 as f32);
                let x_end_f = points[1].x.min(rect_x1 as f32);

                let x_start = x_start_f.ceil() as i32;
                let x_end = x_end_f.floor() as i32;

                if x_start > x_end {
                    continue;
                }

                let span_width = points[1].x - points[0].x;
                if span_width.abs() < 1e-6 {
                    continue;
                }

                let inv_span = 1.0 / span_width;

                // Initial values at pixel center of x_start
                let offset = (x_start as f32 + 0.5) - points[0].x;
                let mut z_val = points[0].z + (points[1].z - points[0].z) * inv_span * offset;
                let mut u_over_w = points[0].u_over_w + (points[1].u_over_w - points[0].u_over_w) * inv_span * offset;
                let mut v_over_w = points[0].v_over_w + (points[1].v_over_w - points[0].v_over_w) * inv_span * offset;
                let mut inv_w = points[0].inv_w + (points[1].inv_w - points[0].inv_w) * inv_span * offset;

                let step_z = (points[1].z - points[0].z) * inv_span;
                let step_u_over_w = (points[1].u_over_w - points[0].u_over_w) * inv_span;
                let step_v_over_w = (points[1].v_over_w - points[0].v_over_w) * inv_span;
                let step_inv_w = (points[1].inv_w - points[0].inv_w) * inv_span;

                for x in x_start..=x_end {
                    count_call!(FUNCTION_COUNTERS.set_pixel_attempts);
                    count_call!(FUNCTION_COUNTERS.total_pixels_tested);

                    if let Some(idx) = unsafe { target.test_depth_and_get_index(x as usize, y as usize, z_val) } {
                        let inv_w_interp = inv_w;
                        let u = u_over_w / inv_w_interp;
                        let v = v_over_w / inv_w_interp;

                        let tex_u = ((u * 8.0) as i32 & 7) as u8;
                        let tex_v = ((v * 8.0) as i32 & 7) as u8;

                        let mut color = texture.sample(tex_u, tex_v);
                        if self.enable_shading {
                            color = self.shading.shade_color_u32(color, light);
                        }

                        target.write_color(idx, color);
                        count_call!(FUNCTION_COUNTERS.set_pixel_depth_passed);
                        any_drawn = true;
                    } else {
                        count_call!(FUNCTION_COUNTERS.set_pixel_depth_failed);
                    }

                    z_val += step_z;
                    u_over_w += step_u_over_w;
                    v_over_w += step_v_over_w;
                    inv_w += step_inv_w;
                }
            }
        }

        any_drawn
    }

    /// Render a triangle from clip space positions with raw block data
    /// This is the optimized path that avoids Vertex intermediate structs
    fn render_triangle_from_clip<T: PixelTarget>(
        &self,
        p0_clip: Vec4,
        p1_clip: Vec4,
        p2_clip: Vec4,
        block_type: crate::voxel::BlockType,
        light: f32,
        target: &mut T,
    ) -> bool {
        count_call!(FUNCTION_COUNTERS.render_triangle_calls);
        count_call!(FUNCTION_COUNTERS.total_triangles_processed);

        // Proper near-plane clipping (w = NEAR_W_EPS) so triangles crossing the near plane
        // are rasterized instead of being dropped entirely.
        let tri = [p0_clip, p1_clip, p2_clip];
        let (tri_count, clipped_tris) = Self::clip_triangle_near_points(tri, NEAR_W_EPS);
        if tri_count == 0 {
            count_call!(FUNCTION_COUNTERS.render_triangle_clipped);
            return false;
        }

        let fb_width = target.width() as f32;
        let fb_height = target.full_height() as f32;
        let base_color = block_type.color();
        let unlit_color = self.shading.shade_color(base_color, 1.0);
        let lit_color = if self.enable_shading {
            self.shading.shade_color(base_color, light)
        } else {
            unlit_color
        };

        let mut any_drawn = false;

        for tri in clipped_tris.iter().take(tri_count) {
            let p0_clip = tri[0];
            let p1_clip = tri[1];
            let p2_clip = tri[2];

            let w0 = p0_clip.w;
            let w1 = p1_clip.w;
            let w2 = p2_clip.w;

            // Perspective divide to NDC
            let p0_ndc = p0_clip / w0;
            let p1_ndc = p1_clip / w1;
            let p2_ndc = p2_clip / w2;

            // Backface culling (in NDC space)
            if self.backface_culling {
                let v01 = p1_ndc - p0_ndc;
                let v02 = p2_ndc - p0_ndc;
                let cross_z = v01.x * v02.y - v01.y * v02.x;
                if cross_z <= 0.0 {
                    count_call!(FUNCTION_COUNTERS.render_triangle_culled);
                    continue;
                }
            }

            // Convert to screen space
            let p0_xy = Self::ndc_to_screen(p0_ndc.truncate().truncate(), fb_width, fb_height);
            let p1_xy = Self::ndc_to_screen(p1_ndc.truncate().truncate(), fb_width, fb_height);
            let p2_xy = Self::ndc_to_screen(p2_ndc.truncate().truncate(), fb_width, fb_height);

            let z0 = p0_ndc.z;
            let z1 = p1_ndc.z;
            let z2 = p2_ndc.z;

            // Compute bounding box in screen space
            let mut min_x = p0_xy.x.min(p1_xy.x).min(p2_xy.x).floor() as i32;
            let mut max_x = p0_xy.x.max(p1_xy.x).max(p2_xy.x).ceil() as i32;
            let mut min_y = p0_xy.y.min(p1_xy.y).min(p2_xy.y).floor() as i32;
            let mut max_y = p0_xy.y.max(p1_xy.y).max(p2_xy.y).ceil() as i32;

            // Clip to full framebuffer
            let fb_w_i = target.width() as i32;
            let fb_h_i = target.full_height() as i32;
            min_x = min_x.max(0);
            max_x = max_x.min(fb_w_i - 1);
            min_y = min_y.max(0);
            max_y = max_y.min(fb_h_i - 1);

            // Intersect with this target's rectangle (stripe or tile)
            let (tx0, ty0, tw, th) = target.rect();
            let tx1 = (tx0 + tw - 1) as i32;
            let ty1 = (ty0 + th - 1) as i32;

            min_x = min_x.max(tx0 as i32);
            max_x = max_x.min(tx1);
            min_y = min_y.max(ty0 as i32);
            max_y = max_y.min(ty1);

            // Skip if completely outside this target
            if min_x > max_x || min_y > max_y {
                continue;
            }

            // Pre-compute edge functions
            let area = Self::edge_function(p0_xy, p1_xy, p2_xy);
            if area <= 0.0 {
                continue; // Degenerate triangle
            }

            // Sub-pixel triangle culling
            const MIN_TRIANGLE_AREA: f32 = 0.1;
            if area < MIN_TRIANGLE_AREA {
                count_call!(FUNCTION_COUNTERS.render_triangle_culled);
                continue;
            }

            let inv_area = 1.0 / area;

            // Precompute edge deltas
            let edge0_dx = p2_xy.y - p1_xy.y;
            let edge0_dy = p1_xy.x - p2_xy.x;
            let edge1_dx = p0_xy.y - p2_xy.y;
            let edge1_dy = p2_xy.x - p0_xy.x;
            let edge2_dx = p1_xy.y - p0_xy.y;
            let edge2_dy = p0_xy.x - p1_xy.x;

            // Evaluate barycentric coordinates at top-left pixel center
            let start_x = min_x as f32 + 0.5;
            let start_y = min_y as f32 + 0.5;

            let p_start = Vec2::new(start_x, start_y);
            let mut w0_row = Self::edge_function(p1_xy, p2_xy, p_start);
            let mut w1_row = Self::edge_function(p2_xy, p0_xy, p_start);
            let mut w2_row = Self::edge_function(p0_xy, p1_xy, p_start);

            // Scanline rasterization
            for y in min_y..=max_y {
                let mut w0 = w0_row;
                let mut w1 = w1_row;
                let mut w2 = w2_row;

                for x in min_x..=max_x {
                    if w0 >= 0.0 && w1 >= 0.0 && w2 >= 0.0 {
                        count_call!(FUNCTION_COUNTERS.set_pixel_attempts);
                        count_call!(FUNCTION_COUNTERS.total_pixels_tested);

                        let bw0 = w0 * inv_area;
                        let bw1 = w1 * inv_area;
                        let bw2 = w2 * inv_area;

                        let depth = bw0 * z0 + bw1 * z1 + bw2 * z2;

                        if let Some(idx) = unsafe {
                            target.test_depth_and_get_index(x as usize, y as usize, depth)
                        } {
                            target.write_color(idx, lit_color);
                            count_call!(FUNCTION_COUNTERS.set_pixel_depth_passed);
                            any_drawn = true;
                        } else {
                            count_call!(FUNCTION_COUNTERS.set_pixel_depth_failed);
                        }
                    }

                    w0 += edge0_dx;
                    w1 += edge1_dx;
                    w2 += edge2_dx;
                }

                w0_row += edge0_dy;
                w1_row += edge1_dy;
                w2_row += edge2_dy;
            }
        }

        any_drawn
    }

    /// Optimized triangle rendering for tiles using raw pointer arithmetic
    /// This eliminates bounds checks and reduces memory indirections
    fn render_triangle_textured_tile_optimized(
        &self,
        p0_clip: Vec4,
        p1_clip: Vec4,
        p2_clip: Vec4,
        uv0: Vec2,
        uv1: Vec2,
        uv2: Vec2,
        block_type: crate::voxel::BlockType,
        light: f32,
        tile: &mut FrameTile,
    ) -> bool {
        count_call!(FUNCTION_COUNTERS.render_triangle_calls);
        count_call!(FUNCTION_COUNTERS.total_triangles_processed);

        // Near-plane clipping
        let tri = [
            ClipTexturedVertex {
                clip_pos: p0_clip,
                uv: uv0,
            },
            ClipTexturedVertex {
                clip_pos: p1_clip,
                uv: uv1,
            },
            ClipTexturedVertex {
                clip_pos: p2_clip,
                uv: uv2,
            },
        ];

        let (tri_count, clipped_tris) = Self::clip_triangle_near_textured(&tri, NEAR_W_EPS);
        if tri_count == 0 {
            count_call!(FUNCTION_COUNTERS.render_triangle_clipped);
            return false;
        }

        let fb_width = tile.width as f32;
        let fb_height = tile.full_height as f32;

        let tex_id = block_type.texture_id();
        let texture = unsafe { self.atlas.textures.get_unchecked(tex_id) };

        let mut any_drawn = false;

        for tri in clipped_tris.iter().take(tri_count) {
            let v0 = tri[0];
            let v1 = tri[1];
            let v2 = tri[2];

            let p0_clip = v0.clip_pos;
            let p1_clip = v1.clip_pos;
            let p2_clip = v2.clip_pos;

            let w0_c = p0_clip.w;
            let w1_c = p1_clip.w;
            let w2_c = p2_clip.w;

            // Perspective divide
            let p0_ndc = p0_clip / w0_c;
            let p1_ndc = p1_clip / w1_c;
            let p2_ndc = p2_clip / w2_c;

            // Backface culling
            if self.backface_culling {
                let v01 = p1_ndc - p0_ndc;
                let v02 = p2_ndc - p0_ndc;
                let cross_z = v01.x * v02.y - v01.y * v02.x;
                if cross_z <= 0.0 {
                    count_call!(FUNCTION_COUNTERS.render_triangle_culled);
                    continue;
                }
            }

            // Screen space
            let p0_xy = Self::ndc_to_screen(p0_ndc.truncate().truncate(), fb_width, fb_height);
            let p1_xy = Self::ndc_to_screen(p1_ndc.truncate().truncate(), fb_width, fb_height);
            let p2_xy = Self::ndc_to_screen(p2_ndc.truncate().truncate(), fb_width, fb_height);

            let z0 = p0_ndc.z;
            let z1 = p1_ndc.z;
            let z2 = p2_ndc.z;

            // Bounding box
            let mut min_x = p0_xy.x.min(p1_xy.x).min(p2_xy.x).floor() as i32;
            let mut max_x = p0_xy.x.max(p1_xy.x).max(p2_xy.x).ceil() as i32;
            let mut min_y = p0_xy.y.min(p1_xy.y).min(p2_xy.y).floor() as i32;
            let mut max_y = p0_xy.y.max(p1_xy.y).max(p2_xy.y).ceil() as i32;

            // Clip to framebuffer
            min_x = min_x.max(0).min(fb_width as i32 - 1);
            max_x = max_x.max(0).min(fb_width as i32 - 1);
            min_y = min_y.max(0).min(fb_height as i32 - 1);
            max_y = max_y.max(0).min(fb_height as i32 - 1);

            // Clip to tile
            min_x = min_x.max(tile.x0 as i32);
            max_x = max_x.min((tile.x0 + tile.tile_width - 1) as i32);
            min_y = min_y.max(tile.y0 as i32);
            max_y = max_y.min((tile.y0 + tile.tile_height - 1) as i32);

            if min_x > max_x || min_y > max_y {
                continue;
            }

            // Edge function
            let area = Self::edge_function(p0_xy, p1_xy, p2_xy);
            if area <= 0.0 {
                continue;
            }

            const MIN_TRIANGLE_AREA: f32 = 0.1;
            if area < MIN_TRIANGLE_AREA {
                count_call!(FUNCTION_COUNTERS.render_triangle_culled);
                continue;
            }

            let inv_area = 1.0 / area;

            // Edge deltas
            let edge0_dx = p2_xy.y - p1_xy.y;
            let edge0_dy = p1_xy.x - p2_xy.x;
            let edge1_dx = p0_xy.y - p2_xy.y;
            let edge1_dy = p2_xy.x - p0_xy.x;
            let edge2_dx = p1_xy.y - p0_xy.y;
            let edge2_dy = p0_xy.x - p1_xy.x;

            let uv0 = v0.uv;
            let uv1 = v1.uv;
            let uv2 = v2.uv;

            // Perspective-correct UV setup
            let inv_w0 = 1.0 / w0_c;
            let inv_w1 = 1.0 / w1_c;
            let inv_w2 = 1.0 / w2_c;

            let u0_over_w = uv0.x * inv_w0;
            let u1_over_w = uv1.x * inv_w1;
            let u2_over_w = uv2.x * inv_w2;

            let v0_over_w = uv0.y * inv_w0;
            let v1_over_w = uv1.y * inv_w1;
            let v2_over_w = uv2.y * inv_w2;

            // Barycentric setup
            let start_x = min_x as f32 + 0.5;
            let start_y = min_y as f32 + 0.5;

            let p_start = Vec2::new(start_x, start_y);
            let mut w0_row = Self::edge_function(p1_xy, p2_xy, p_start);
            let mut w1_row = Self::edge_function(p2_xy, p0_xy, p_start);
            let mut w2_row = Self::edge_function(p0_xy, p1_xy, p_start);

            // OPTIMIZED SCANLINE LOOP WITH RAW POINTERS
            for y in min_y..=max_y {
                let mut w0 = w0_row;
                let mut w1 = w1_row;
                let mut w2 = w2_row;

                // Get row pointers once per scanline
                let (mut color_ptr, mut depth_ptr) =
                    unsafe { tile.get_row_pointers(y as usize) };

                // Offset to min_x
                unsafe {
                    color_ptr = color_ptr.add(min_x as usize);
                    depth_ptr = depth_ptr.add(min_x as usize);
                }

                for _x in min_x..=max_x {
                    if w0 >= 0.0 && w1 >= 0.0 && w2 >= 0.0 {
                        count_call!(FUNCTION_COUNTERS.set_pixel_attempts);
                        count_call!(FUNCTION_COUNTERS.total_pixels_tested);

                        let bw0 = w0 * inv_area;
                        let bw1 = w1 * inv_area;
                        let bw2 = w2 * inv_area;

                        let depth = bw0 * z0 + bw1 * z1 + bw2 * z2;

                        unsafe {
                            // Direct pointer access - no bounds checks, no multiplication
                            if depth < *depth_ptr {
                                *depth_ptr = depth;

                                // Perspective-correct UV interpolation
                                let inv_w_interp = bw0 * inv_w0 + bw1 * inv_w1 + bw2 * inv_w2;

                                let u = (bw0 * u0_over_w + bw1 * u1_over_w + bw2 * u2_over_w)
                                    / inv_w_interp;
                                let v = (bw0 * v0_over_w + bw1 * v1_over_w + bw2 * v2_over_w)
                                    / inv_w_interp;

                                // 8x8 texture space with wrap
                                let tex_u = ((u * 8.0) as i32 & 7) as u8;
                                let tex_v = ((v * 8.0) as i32 & 7) as u8;

                                let mut color = texture.sample(tex_u, tex_v);

                                if self.enable_shading {
                                    color = self.shading.shade_color_u32(color, light);
                                }

                                *color_ptr = color;

                                count_call!(FUNCTION_COUNTERS.set_pixel_depth_passed);
                                any_drawn = true;
                            } else {
                                count_call!(FUNCTION_COUNTERS.set_pixel_depth_failed);
                            }

                            // Advance pointers
                            color_ptr = color_ptr.add(1);
                            depth_ptr = depth_ptr.add(1);
                        }
                    } else {
                        unsafe {
                            color_ptr = color_ptr.add(1);
                            depth_ptr = depth_ptr.add(1);
                        }
                    }

                    w0 += edge0_dx;
                    w1 += edge1_dx;
                    w2 += edge2_dx;
                }

                w0_row += edge0_dy;
                w1_row += edge1_dy;
                w2_row += edge2_dy;
            }
        }

        any_drawn
    }

    /// Render a triangle from clip space with per-vertex texture coordinates.
    /// Uses the MicroTexture-based atlas for per-pixel sampling.
    fn render_triangle_from_clip_textured<T: PixelTarget>(
        &self,
        p0_clip: Vec4,
        p1_clip: Vec4,
        p2_clip: Vec4,
        uv0: glam::Vec2,
        uv1: glam::Vec2,
        uv2: glam::Vec2,
        block_type: crate::voxel::BlockType,
        light: f32,
        target: &mut T,
    ) -> bool {
        count_call!(FUNCTION_COUNTERS.render_triangle_calls);
        count_call!(FUNCTION_COUNTERS.total_triangles_processed);

        // Near-plane clipping with attribute interpolation.
        let tri = [
            ClipTexturedVertex {
                clip_pos: p0_clip,
                uv: uv0,
            },
            ClipTexturedVertex {
                clip_pos: p1_clip,
                uv: uv1,
            },
            ClipTexturedVertex {
                clip_pos: p2_clip,
                uv: uv2,
            },
        ];

        let (tri_count, clipped_tris) = Self::clip_triangle_near_textured(&tri, NEAR_W_EPS);
        if tri_count == 0 {
            count_call!(FUNCTION_COUNTERS.render_triangle_clipped);
            return false;
        }

        let fb_width = target.width() as f32;
        let fb_height = target.full_height() as f32;

        // Resolve MicroTexture once per triangle batch.
        let tex_id = block_type.texture_id();
        let texture = unsafe { self.atlas.textures.get_unchecked(tex_id) };

        let mut any_drawn = false;

        for tri in clipped_tris.iter().take(tri_count) {
            let v0 = tri[0];
            let v1 = tri[1];
            let v2 = tri[2];

            let p0_clip = v0.clip_pos;
            let p1_clip = v1.clip_pos;
            let p2_clip = v2.clip_pos;

            let w0 = p0_clip.w;
            let w1 = p1_clip.w;
            let w2 = p2_clip.w;

            // Perspective divide to NDC
            let p0_ndc = p0_clip / w0;
            let p1_ndc = p1_clip / w1;
            let p2_ndc = p2_clip / w2;

            // Backface culling (in NDC space)
            if self.backface_culling {
                let v01 = p1_ndc - p0_ndc;
                let v02 = p2_ndc - p0_ndc;
                let cross_z = v01.x * v02.y - v01.y * v02.x;
                if cross_z <= 0.0 {
                    count_call!(FUNCTION_COUNTERS.render_triangle_culled);
                    continue;
                }
            }

            // Convert to screen space
            let p0_xy = Self::ndc_to_screen(p0_ndc.truncate().truncate(), fb_width, fb_height);
            let p1_xy = Self::ndc_to_screen(p1_ndc.truncate().truncate(), fb_width, fb_height);
            let p2_xy = Self::ndc_to_screen(p2_ndc.truncate().truncate(), fb_width, fb_height);

            let z0 = p0_ndc.z;
            let z1 = p1_ndc.z;
            let z2 = p2_ndc.z;

            // Compute bounding box in screen space
            let mut min_x = p0_xy.x.min(p1_xy.x).min(p2_xy.x).floor() as i32;
            let mut max_x = p0_xy.x.max(p1_xy.x).max(p2_xy.x).ceil() as i32;
            let mut min_y = p0_xy.y.min(p1_xy.y).min(p2_xy.y).floor() as i32;
            let mut max_y = p0_xy.y.max(p1_xy.y).max(p2_xy.y).ceil() as i32;

            // Clip to full framebuffer
            let fb_w_i = target.width() as i32;
            let fb_h_i = target.full_height() as i32;
            min_x = min_x.max(0);
            max_x = max_x.min(fb_w_i - 1);
            min_y = min_y.max(0);
            max_y = max_y.min(fb_h_i - 1);

            // Intersect with this target's rectangle (stripe or tile)
            let (tx0, ty0, tw, th) = target.rect();
            let tx1 = (tx0 + tw - 1) as i32;
            let ty1 = (ty0 + th - 1) as i32;

            min_x = min_x.max(tx0 as i32);
            max_x = max_x.min(tx1);
            min_y = min_y.max(ty0 as i32);
            max_y = max_y.min(ty1);

            // Skip if completely outside this target
            if min_x > max_x || min_y > max_y {
                continue;
            }

            // Pre-compute edge functions
            let area = Self::edge_function(p0_xy, p1_xy, p2_xy);
            if area <= 0.0 {
                continue; // Degenerate triangle
            }

            // Sub-pixel triangle culling with conservative threshold
            const MIN_TRIANGLE_AREA: f32 = 0.1;
            if area < MIN_TRIANGLE_AREA {
                count_call!(FUNCTION_COUNTERS.render_triangle_culled);
                continue;
            }

            let inv_area = 1.0 / area;

            // Precompute edge deltas
            let edge0_dx = p2_xy.y - p1_xy.y;
            let edge0_dy = p1_xy.x - p2_xy.x;
            let edge1_dx = p0_xy.y - p2_xy.y;
            let edge1_dy = p2_xy.x - p0_xy.x;
            let edge2_dx = p1_xy.y - p0_xy.y;
            let edge2_dy = p0_xy.x - p1_xy.x;

            let uv0 = v0.uv;
            let uv1 = v1.uv;
            let uv2 = v2.uv;

            // Precompute perspective-correct UV factors.
            let inv_w0 = 1.0 / w0;
            let inv_w1 = 1.0 / w1;
            let inv_w2 = 1.0 / w2;

            let u0_over_w = uv0.x * inv_w0;
            let u1_over_w = uv1.x * inv_w1;
            let u2_over_w = uv2.x * inv_w2;

            let v0_over_w = uv0.y * inv_w0;
            let v1_over_w = uv1.y * inv_w1;
            let v2_over_w = uv2.y * inv_w2;

            // Evaluate barycentric coordinates at top-left pixel center
            let start_x = min_x as f32 + 0.5;
            let start_y = min_y as f32 + 0.5;

            let p_start = Vec2::new(start_x, start_y);
            let mut w0_row = Self::edge_function(p1_xy, p2_xy, p_start);
            let mut w1_row = Self::edge_function(p2_xy, p0_xy, p_start);
            let mut w2_row = Self::edge_function(p0_xy, p1_xy, p_start);

            // Optimized scanline rasterization using unchecked pointer arithmetic
            // to eliminate bounds checks and reduce memory latency
            for y in min_y..=max_y {
                let mut w0 = w0_row;
                let mut w1 = w1_row;
                let mut w2 = w2_row;

                // Calculate row pointer offset once per scanline
                let row_offset = y as usize * target.width();

                for x in min_x..=max_x {
                    if w0 >= 0.0 && w1 >= 0.0 && w2 >= 0.0 {
                        count_call!(FUNCTION_COUNTERS.set_pixel_attempts);
                        count_call!(FUNCTION_COUNTERS.total_pixels_tested);

                        let bw0 = w0 * inv_area;
                        let bw1 = w1 * inv_area;
                        let bw2 = w2 * inv_area;

                        let depth = bw0 * z0 + bw1 * z1 + bw2 * z2;

                        if let Some(idx) = unsafe {
                            target.test_depth_and_get_index(x as usize, y as usize, depth)
                        } {
                            // Perspective-correct interpolate UV in local voxel-space coordinates.
                            let inv_w_interp =
                                bw0 * inv_w0 + bw1 * inv_w1 + bw2 * inv_w2;

                            let u = (bw0 * u0_over_w + bw1 * u1_over_w + bw2 * u2_over_w)
                                / inv_w_interp;
                            let v = (bw0 * v0_over_w + bw1 * v1_over_w + bw2 * v2_over_w)
                                / inv_w_interp;

                            // Map to 8x8 MicroTexture space with wrap.
                            let tex_u = ((u * 8.0) as i32 & 7) as u8;
                            let tex_v = ((v * 8.0) as i32 & 7) as u8;

                            let mut color = texture.sample(tex_u, tex_v);

                            if self.enable_shading {
                                color = self.shading.shade_color_u32(color, light);
                            }

                            target.write_color(idx, color);

                            count_call!(FUNCTION_COUNTERS.set_pixel_depth_passed);
                            any_drawn = true;
                        } else {
                            count_call!(FUNCTION_COUNTERS.set_pixel_depth_failed);
                        }
                    }

                    w0 += edge0_dx;
                    w1 += edge1_dx;
                    w2 += edge2_dx;
                }

                w0_row += edge0_dy;
                w1_row += edge1_dy;
                w2_row += edge2_dy;
            }
        }

        any_drawn
    }

    /// Render a single triangle
    fn render_triangle_pretransformed<T: PixelTarget>(
        &self,
        v0: &Vertex,
        v1: &Vertex,
        v2: &Vertex,
        p0_clip: Vec4,
        p1_clip: Vec4,
        p2_clip: Vec4,
        target: &mut T,
    ) -> bool {
        count_call!(FUNCTION_COUNTERS.render_triangle_calls);
        count_call!(FUNCTION_COUNTERS.total_triangles_processed);

        // Perform near-plane clipping in clip space (against w = NEAR_W_EPS)
        let tri = [
            ClipVertex {
                vertex: *v0,
                clip_pos: p0_clip,
            },
            ClipVertex {
                vertex: *v1,
                clip_pos: p1_clip,
            },
            ClipVertex {
                vertex: *v2,
                clip_pos: p2_clip,
            },
        ];

        let (tri_count, clipped_tris) = Self::clip_triangle_near(&tri, NEAR_W_EPS);
        if tri_count == 0 {
            count_call!(FUNCTION_COUNTERS.render_triangle_clipped);
            return false;
        }

        let mut any_drawn = false;

        for tri in clipped_tris.iter().take(tri_count) {
            let v0 = tri[0].vertex;
            let v1 = tri[1].vertex;
            let v2 = tri[2].vertex;

            let p0_clip = tri[0].clip_pos;
            let p1_clip = tri[1].clip_pos;
            let p2_clip = tri[2].clip_pos;

            let w0 = p0_clip.w;
            let w1 = p1_clip.w;
            let w2 = p2_clip.w;

            // Early reject if triangle is completely behind the camera
            if w0 <= 0.0 && w1 <= 0.0 && w2 <= 0.0 {
                continue;
            }

            // Perspective divide to NDC
            let p0_ndc = p0_clip / w0;
            let p1_ndc = p1_clip / w1;
            let p2_ndc = p2_clip / w2;

            // Backface culling (in NDC space)
            if self.backface_culling {
                let v01 = p1_ndc - p0_ndc;
                let v02 = p2_ndc - p0_ndc;
                let cross_z = v01.x * v02.y - v01.y * v02.x;
                if cross_z <= 0.0 {
                    count_call!(FUNCTION_COUNTERS.render_triangle_culled);
                    continue;
                }
            }

            // Convert to screen space (keep XY as Vec2, Z separate)
            let fb_width = target.width() as f32;
            let fb_height = target.full_height() as f32;

            let p0_xy = Self::ndc_to_screen(p0_ndc.truncate().truncate(), fb_width, fb_height);
            let p1_xy = Self::ndc_to_screen(p1_ndc.truncate().truncate(), fb_width, fb_height);
            let p2_xy = Self::ndc_to_screen(p2_ndc.truncate().truncate(), fb_width, fb_height);

            let z0 = p0_ndc.z;
            let z1 = p1_ndc.z;
            let z2 = p2_ndc.z;

            // Compute bounding box in screen space
            let mut min_x = p0_xy.x.min(p1_xy.x).min(p2_xy.x).floor() as i32;
            let mut max_x = p0_xy.x.max(p1_xy.x).max(p2_xy.x).ceil() as i32;
            let mut min_y = p0_xy.y.min(p1_xy.y).min(p2_xy.y).floor() as i32;
            let mut max_y = p0_xy.y.max(p1_xy.y).max(p2_xy.y).ceil() as i32;

            // Clip to full framebuffer
            let fb_w_i = target.width() as i32;
            let fb_h_i = target.full_height() as i32;
            min_x = min_x.max(0);
            max_x = max_x.min(fb_w_i - 1);
            min_y = min_y.max(0);
            max_y = max_y.min(fb_h_i - 1);

            // Intersect with this target's rectangle (stripe or tile)
            let (tx0, ty0, tw, th) = target.rect();
            let tx1 = (tx0 + tw - 1) as i32;
            let ty1 = (ty0 + th - 1) as i32;

            min_x = min_x.max(tx0 as i32);
            max_x = max_x.min(tx1);
            min_y = min_y.max(ty0 as i32);
            max_y = max_y.min(ty1);

            // Skip if completely outside this target
            if min_x > max_x || min_y > max_y {
                continue;
            }

            // Pre-compute edge functions
            let area = Self::edge_function(p0_xy, p1_xy, p2_xy);
            if area <= 0.0 {
                continue; // Degenerate triangle
            }

            // Sub-pixel triangle culling with ultra-conservative threshold
            //
            // We use a very small threshold (0.05 pixels) to avoid creating gaps in
            // continuous surfaces. At 0.5 pixels, adjacent small triangles on distant
            // planes would both be culled, leaving visible holes.
            //
            // At 0.05 pixels (area = 0.1 in 2x calculation), we only cull triangles
            // that are truly invisible, not just small. This preserves visual quality
            // while still rejecting degenerate/micro triangles.
            const MIN_TRIANGLE_AREA: f32 = 0.1; // 0.05 pixels (area is 2x actual)
            if area < MIN_TRIANGLE_AREA {
                count_call!(FUNCTION_COUNTERS.render_triangle_culled);
                continue;
            }

            let inv_area = 1.0 / area;

            // Precompute edge deltas for incremental barycentric evaluation
            // w0(x, y) = edge(p1, p2, (x, y))
            //   => ∂w0/∂x = (p2.y - p1.y), ∂w0/∂y = -(p2.x - p1.x)
            let edge0_dx = p2_xy.y - p1_xy.y;
            let edge0_dy = p1_xy.x - p2_xy.x;
            // w1(x, y) = edge(p2, p0, (x, y))
            let edge1_dx = p0_xy.y - p2_xy.y;
            let edge1_dy = p2_xy.x - p0_xy.x;
            // w2(x, y) = edge(p0, p1, (x, y))
            let edge2_dx = p1_xy.y - p0_xy.y;
            let edge2_dy = p0_xy.x - p1_xy.x;

            // Derive base color from BlockType instead of per-vertex color
            use crate::voxel::BlockType;
            let block_type = BlockType::from_u8(v0.block_type);
            let base_color = block_type.color();

            // Convert quantized light (u8) to float for interpolation
            let l0 = v0.light as f32 / 255.0;
            let l1 = v1.light as f32 / 255.0;
            let l2 = v2.light as f32 / 255.0;

            // Pre-pack an unlit color for the common case where shading
            // is disabled to avoid per-pixel light computation.
            let unlit_color = self.shading.shade_color(base_color, 1.0);

            // Evaluate barycentric coordinates at top-left pixel center
            let start_x = min_x as f32 + 0.5;
            let start_y = min_y as f32 + 0.5;

            let p_start = Vec2::new(start_x, start_y);
            let mut w0_row = Self::edge_function(p1_xy, p2_xy, p_start);
            let mut w1_row = Self::edge_function(p2_xy, p0_xy, p_start);
            let mut w2_row = Self::edge_function(p0_xy, p1_xy, p_start);

            // Scanline rasterization with incremental barycentrics
            for y in min_y..=max_y {
                let mut w0 = w0_row;
                let mut w1 = w1_row;
                let mut w2 = w2_row;

                #[cfg(target_arch = "x86_64")]
                if self.simd_mode == SimdMode::Sse2 {
                    // SIMD-accelerated scanline: compute depth/light for 4 pixels at once,
                    // then apply depth test and color writes per-lane.
                    unsafe {
                        let mut x = min_x;

                        while x + 3 <= max_x {
                            // Precompute barycentrics for the 4 pixels in this group.
                            let w0_0 = w0;
                            let w0_1 = w0_0 + edge0_dx;
                            let w0_2 = w0_1 + edge0_dx;
                            let w0_3 = w0_2 + edge0_dx;

                            let w1_0 = w1;
                            let w1_1 = w1_0 + edge1_dx;
                            let w1_2 = w1_1 + edge1_dx;
                            let w1_3 = w1_2 + edge1_dx;

                            let w2_0 = w2;
                            let w2_1 = w2_0 + edge2_dx;
                            let w2_2 = w2_1 + edge2_dx;
                            let w2_3 = w2_2 + edge2_dx;

                            let w0_vec = _mm_set_ps(w0_3, w0_2, w0_1, w0_0);
                            let w1_vec = _mm_set_ps(w1_3, w1_2, w1_1, w1_0);
                            let w2_vec = _mm_set_ps(w2_3, w2_2, w2_1, w2_0);

                            let inv_area_vec = _mm_set1_ps(inv_area);

                            // Barycentric weights
                            let bw0_vec = _mm_mul_ps(w0_vec, inv_area_vec);
                            let bw1_vec = _mm_mul_ps(w1_vec, inv_area_vec);
                            let bw2_vec = _mm_mul_ps(w2_vec, inv_area_vec);

                            // Depth = bw0*z0 + bw1*z1 + bw2*z2
                            let z0_vec = _mm_set1_ps(z0);
                            let z1_vec = _mm_set1_ps(z1);
                            let z2_vec = _mm_set1_ps(z2);

                            let d0 = _mm_mul_ps(bw0_vec, z0_vec);
                            let d1 = _mm_mul_ps(bw1_vec, z1_vec);
                            let d2 = _mm_mul_ps(bw2_vec, z2_vec);
                            let depth_vec = _mm_add_ps(_mm_add_ps(d0, d1), d2);

                            let mut depth_arr = [0.0f32; 4];
                            _mm_storeu_ps(depth_arr.as_mut_ptr(), depth_vec);

                            let mut light_arr = [1.0f32; 4];
                            if self.enable_shading {
                                let l0_vec = _mm_set1_ps(l0);
                                let l1_vec = _mm_set1_ps(l1);
                                let l2_vec = _mm_set1_ps(l2);

                                let ll0 = _mm_mul_ps(bw0_vec, l0_vec);
                                let ll1 = _mm_mul_ps(bw1_vec, l1_vec);
                                let ll2 = _mm_mul_ps(bw2_vec, l2_vec);
                                let light_vec = _mm_add_ps(_mm_add_ps(ll0, ll1), ll2);

                                _mm_storeu_ps(light_arr.as_mut_ptr(), light_vec);
                            }

                            // Per-lane depth test and color write.
                            for lane in 0..4 {
                                let px = x + lane;
                                let w0_l = match lane {
                                    0 => w0_0,
                                    1 => w0_1,
                                    2 => w0_2,
                                    _ => w0_3,
                                };
                                let w1_l = match lane {
                                    0 => w1_0,
                                    1 => w1_1,
                                    2 => w1_2,
                                    _ => w1_3,
                                };
                                let w2_l = match lane {
                                    0 => w2_0,
                                    1 => w2_1,
                                    2 => w2_2,
                                    _ => w2_3,
                                };

                                if w0_l >= 0.0 && w1_l >= 0.0 && w2_l >= 0.0 {
                                    count_call!(FUNCTION_COUNTERS.set_pixel_attempts);
                                    count_call!(FUNCTION_COUNTERS.total_pixels_tested);

                                    let depth = depth_arr[lane as usize];

                                    if let Some(idx) = unsafe {
                                        target.test_depth_and_get_index(
                                            px as usize,
                                            y as usize,
                                            depth,
                                        )
                                    } {
                                        let color = if self.enable_shading {
                                            let light = light_arr[lane as usize];
                                            self.shading.shade_color(base_color, light)
                                        } else {
                                            unlit_color
                                        };

                                        target.write_color(idx, color);

                                        count_call!(FUNCTION_COUNTERS.set_pixel_depth_passed);
                                        any_drawn = true;
                                    } else {
                                        count_call!(FUNCTION_COUNTERS.set_pixel_depth_failed);
                                    }
                                }
                            }

                            // Advance barycentrics and x for next group.
                            w0 += edge0_dx * 4.0;
                            w1 += edge1_dx * 4.0;
                            w2 += edge2_dx * 4.0;
                            x += 4;
                        }

                        // Scalar tail for remaining pixels on this scanline.
                        while x <= max_x {
                            if w0 >= 0.0 && w1 >= 0.0 && w2 >= 0.0 {
                                count_call!(FUNCTION_COUNTERS.set_pixel_attempts);
                                count_call!(FUNCTION_COUNTERS.total_pixels_tested);

                                let bw0 = w0 * inv_area;
                                let bw1 = w1 * inv_area;
                                let bw2 = w2 * inv_area;

                                let depth = bw0 * z0 + bw1 * z1 + bw2 * z2;

                                if let Some(idx) = unsafe {
                                    target.test_depth_and_get_index(
                                        x as usize,
                                        y as usize,
                                        depth,
                                    )
                                } {
                                    let color = if self.enable_shading {
                                        let light = bw0 * l0 + bw1 * l1 + bw2 * l2;
                                        self.shading.shade_color(base_color, light)
                                    } else {
                                        unlit_color
                                    };

                                    target.write_color(idx, color);

                                    count_call!(FUNCTION_COUNTERS.set_pixel_depth_passed);
                                    any_drawn = true;
                                } else {
                                    count_call!(FUNCTION_COUNTERS.set_pixel_depth_failed);
                                }
                            }

                            w0 += edge0_dx;
                            w1 += edge1_dx;
                            w2 += edge2_dx;
                            x += 1;
                        }
                    }
                } else {
                    // Scalar fallback path (non-x86_64 or SIMD disabled).
                    for x in min_x..=max_x {
                        if w0 >= 0.0 && w1 >= 0.0 && w2 >= 0.0 {
                            count_call!(FUNCTION_COUNTERS.set_pixel_attempts);
                            count_call!(FUNCTION_COUNTERS.total_pixels_tested);

                            let bw0 = w0 * inv_area;
                            let bw1 = w1 * inv_area;
                            let bw2 = w2 * inv_area;

                            let depth = bw0 * z0 + bw1 * z1 + bw2 * z2;

                            // Safe because we clamp bounding box to framebuffer
                            if let Some(idx) = unsafe {
                                target.test_depth_and_get_index(
                                    x as usize,
                                    y as usize,
                                    depth,
                                )
                            } {
                                // Depth test passed: now compute final color.
                                let color = if self.enable_shading {
                                    let light = bw0 * l0 + bw1 * l1 + bw2 * l2;
                                    self.shading.shade_color(base_color, light)
                                } else {
                                    unlit_color
                                };

                                target.write_color(idx, color);

                                count_call!(FUNCTION_COUNTERS.set_pixel_depth_passed);
                                any_drawn = true;
                            } else {
                                count_call!(FUNCTION_COUNTERS.set_pixel_depth_failed);
                            }
                        }

                        w0 += edge0_dx;
                        w1 += edge1_dx;
                        w2 += edge2_dx;
                    }
                }

                #[cfg(not(target_arch = "x86_64"))]
                {
                    // Scalar path for non-x86_64 architectures.
                    for x in min_x..=max_x {
                        if w0 >= 0.0 && w1 >= 0.0 && w2 >= 0.0 {
                            count_call!(FUNCTION_COUNTERS.set_pixel_attempts);
                            count_call!(FUNCTION_COUNTERS.total_pixels_tested);

                            let bw0 = w0 * inv_area;
                            let bw1 = w1 * inv_area;
                            let bw2 = w2 * inv_area;

                            let depth = bw0 * z0 + bw1 * z1 + bw2 * z2;

                            if let Some(idx) = unsafe {
                                target.test_depth_and_get_index(
                                    x as usize,
                                    y as usize,
                                    depth,
                                )
                            } {
                                let color = if self.enable_shading {
                                    let light = bw0 * l0 + bw1 * l1 + bw2 * l2;
                                    self.shading.shade_color(base_color, light)
                                } else {
                                    unlit_color
                                };

                                target.write_color(idx, color);

                                count_call!(FUNCTION_COUNTERS.set_pixel_depth_passed);
                                any_drawn = true;
                            } else {
                                count_call!(FUNCTION_COUNTERS.set_pixel_depth_failed);
                            }
                        }

                        w0 += edge0_dx;
                        w1 += edge1_dx;
                        w2 += edge2_dx;
                    }
                }

                w0_row += edge0_dy;
                w1_row += edge1_dy;
                w2_row += edge2_dy;
            }
        }

        any_drawn
    }

    /// Convert NDC coordinates to screen space
    #[inline]
    fn ndc_to_screen(ndc: Vec2, width: f32, height: f32) -> Vec2 {
        Vec2::new(
            (ndc.x + 1.0) * 0.5 * width,
            (1.0 - ndc.y) * 0.5 * height, // Flip Y for screen coordinates
        )
    }

    /// Edge function for barycentric coordinates
    /// Returns 2x the signed area of the triangle
    #[inline]
    fn edge_function(a: Vec2, b: Vec2, c: Vec2) -> f32 {
        (c.x - a.x) * (b.y - a.y) - (c.y - a.y) * (b.x - a.x)
    }

    /// Intersect edge AB with the near-plane w = threshold in clip space (Vec4-only version).
    #[inline]
    fn intersect_near_vec4(a: Vec4, b: Vec4, threshold: f32) -> Vec4 {
        let wa = a.w;
        let wb = b.w;
        let t = (threshold - wa) / (wb - wa);
        a + (b - a) * t
    }

    /// Clip a triangle against the near plane in clip space (w = threshold).
    /// Returns (triangle_count, triangles). triangle_count is 0, 1, or 2.
    fn clip_triangle_near_points(
        tri: [Vec4; 3],
        threshold: f32,
    ) -> (usize, [[Vec4; 3]; 2]) {
        let mut output = [tri[0]; 4];
        let mut out_len = 0usize;

        let mut prev = tri[2];
        let mut prev_inside = prev.w >= threshold;

        for &curr in tri.iter() {
            let curr_inside = curr.w >= threshold;

            match (prev_inside, curr_inside) {
                (true, true) => {
                    output[out_len] = curr;
                    out_len += 1;
                }
                (true, false) => {
                    let inter = Self::intersect_near_vec4(prev, curr, threshold);
                    output[out_len] = inter;
                    out_len += 1;
                }
                (false, true) => {
                    let inter = Self::intersect_near_vec4(prev, curr, threshold);
                    output[out_len] = inter;
                    out_len += 1;
                    output[out_len] = curr;
                    out_len += 1;
                }
                (false, false) => {}
            }

            prev = curr;
            prev_inside = curr_inside;
        }

        let mut tris = [[tri[0]; 3]; 2];

        match out_len {
            0 => (0, tris),
            3 => {
                tris[0] = [output[0], output[1], output[2]];
                (1, tris)
            }
            4 => {
                tris[0] = [output[0], output[1], output[2]];
                tris[1] = [output[0], output[2], output[3]];
                (2, tris)
            }
            _ => (0, tris),
        }
    }

    /// Intersect edge AB with the near-plane w = threshold in clip space
    /// for textured vertices.
    #[inline]
    fn intersect_near_textured(
        a: &ClipTexturedVertex,
        b: &ClipTexturedVertex,
        threshold: f32,
    ) -> ClipTexturedVertex {
        let wa = a.clip_pos.w;
        let wb = b.clip_pos.w;
        let t = (threshold - wa) / (wb - wa);

        let clip_pos = a.clip_pos + (b.clip_pos - a.clip_pos) * t;
        let uv = a.uv + (b.uv - a.uv) * t;

        ClipTexturedVertex { clip_pos, uv }
    }

    /// Clip a triangle with texture coordinates against the near plane (w = threshold).
    /// Returns (triangle_count, triangles). triangle_count is 0, 1, or 2.
    fn clip_triangle_near_textured(
        tri: &[ClipTexturedVertex; 3],
        threshold: f32,
    ) -> (usize, [[ClipTexturedVertex; 3]; 2]) {
        let mut output = [tri[0]; 4];
        let mut out_len = 0usize;

        let mut prev = tri[2];
        let mut prev_inside = prev.clip_pos.w >= threshold;

        for &curr in tri.iter() {
            let curr_inside = curr.clip_pos.w >= threshold;

            match (prev_inside, curr_inside) {
                (true, true) => {
                    output[out_len] = curr;
                    out_len += 1;
                }
                (true, false) => {
                    let inter = Self::intersect_near_textured(&prev, &curr, threshold);
                    output[out_len] = inter;
                    out_len += 1;
                }
                (false, true) => {
                    let inter = Self::intersect_near_textured(&prev, &curr, threshold);
                    output[out_len] = inter;
                    out_len += 1;
                    output[out_len] = curr;
                    out_len += 1;
                }
                (false, false) => {}
            }

            prev = curr;
            prev_inside = curr_inside;
        }

        let mut tris = [[tri[0]; 3]; 2];

        match out_len {
            0 => (0, tris),
            3 => {
                tris[0] = [output[0], output[1], output[2]];
                (1, tris)
            }
            4 => {
                tris[0] = [output[0], output[1], output[2]];
                tris[1] = [output[0], output[2], output[3]];
                (2, tris)
            }
            _ => (0, tris),
        }
    }

    /// Intersect edge AB with the near-plane w = threshold in clip space.
    #[inline]
    fn intersect_near(a: &ClipVertex, b: &ClipVertex, threshold: f32) -> ClipVertex {
        let wa = a.clip_pos.w;
        let wb = b.clip_pos.w;
        let t = (threshold - wa) / (wb - wa);

        // Interpolate clip position
        let clip_pos = a.clip_pos + (b.clip_pos - a.clip_pos) * t;

        // Interpolate local position (u8 -> f32 -> interpolate -> f32, leave as f32 temporarily)
        let a_pos = a.vertex.local_position();
        let b_pos = b.vertex.local_position();
        let pos = a_pos + (b_pos - a_pos) * t;

        // Clamp interpolated position back to u8 range for storage
        // In practice, clipped vertices will be near the edge, so this should be safe
        let vertex = Vertex {
            x: pos.x.clamp(0.0, 32.0) as u8,
            y: pos.y.clamp(0.0, 32.0) as u8,
            z: pos.z.clamp(0.0, 32.0) as u8,
            block_type: a.vertex.block_type,
            light: a.vertex.light,
            packed: a.vertex.packed,
            padding: 0,
        };

        ClipVertex { vertex, clip_pos }
    }

    /// Clip a triangle against the near plane in clip space (w = threshold).
    /// Returns (triangle_count, triangles). triangle_count is 0, 1, or 2.
    fn clip_triangle_near(
        tri: &[ClipVertex; 3],
        threshold: f32,
    ) -> (usize, [[ClipVertex; 3]; 2]) {
        // Sutherland–Hodgman polygon clipping for a single plane.
        // We know a triangle clipped against a single plane can have
        // at most 4 vertices, so we use a small fixed-size buffer.
        let mut output = [tri[0]; 4];
        let mut out_len = 0usize;

        let mut prev = tri[2];
        let mut prev_inside = prev.clip_pos.w >= threshold;

        for &curr in tri.iter() {
            let curr_inside = curr.clip_pos.w >= threshold;

            match (prev_inside, curr_inside) {
                (true, true) => {
                    // prev in, curr in -> keep curr
                    output[out_len] = curr;
                    out_len += 1;
                }
                (true, false) => {
                    // prev in, curr out -> keep intersection
                    let inter = Self::intersect_near(&prev, &curr, threshold);
                    output[out_len] = inter;
                    out_len += 1;
                }
                (false, true) => {
                    // prev out, curr in -> keep intersection and curr
                    let inter = Self::intersect_near(&prev, &curr, threshold);
                    output[out_len] = inter;
                    out_len += 1;
                    output[out_len] = curr;
                    out_len += 1;
                }
                (false, false) => {
                    // both out -> keep nothing
                }
            }

            prev = curr;
            prev_inside = curr_inside;
        }

        let mut tris = [[tri[0]; 3]; 2];

        match out_len {
            0 => (0, tris),
            3 => {
                tris[0] = [output[0], output[1], output[2]];
                (1, tris)
            }
            4 => {
                tris[0] = [output[0], output[1], output[2]];
                tris[1] = [output[0], output[2], output[3]];
                (2, tris)
            }
            _ => {
                // Clipping against a single plane can only produce 0, 3 or 4 vertices
                // from an input triangle. Any other count indicates a bug.
                (0, tris)
            }
        }
    }
}
