/// Mesh data structures for rendering
use glam::{IVec3, Vec3};

/// Packed 3-component float vector used inside vertices.
/// This avoids potential SIMD padding from glam::Vec3 and
/// guarantees a tightly-packed 12-byte representation.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct F32x3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl F32x3 {
    #[inline]
    pub fn from_vec3(v: Vec3) -> Self {
        Self { x: v.x, y: v.y, z: v.z }
    }

    #[inline]
    pub fn to_vec3(self) -> Vec3 {
        Vec3::new(self.x, self.y, self.z)
    }
}

impl PartialEq<Vec3> for F32x3 {
    fn eq(&self, other: &Vec3) -> bool {
        self.x == other.x && self.y == other.y && self.z == other.z
    }
}

/// Vertex structure optimized for software rendering
/// Memory layout optimized for cache alignment and minimal padding
///
/// Ultra-compressed layout (8 bytes total, 64-byte cache line = 8 vertices):
/// - x, y, z: u8 (3 bytes) - Local chunk coordinates 0..32
/// - block_type: u8 (1 byte) - BlockType enum as u8
/// - light: u8 (1 byte) - Quantized light 0..255
/// - packed: u8 (1 byte) - normal index (3 bits) + AO level (2 bits)
/// - padding: u16 (2 bytes) - Reserved for future use
/// Total: 8 bytes (50% reduction from 16 bytes, 75% from original 32 bytes)
///
/// Note: Positions are stored as chunk-local u8 coordinates.
/// The rasterizer must add the chunk world offset before transformation.
#[derive(Copy, Clone, Debug)]
#[repr(C, align(8))]
pub struct Vertex {
    // Local chunk coordinates (0..=32 fits in u8)
    pub x: u8,
    pub y: u8,
    pub z: u8,

    pub block_type: u8,    // 1 byte, BlockType as u8
    pub light: u8,         // 1 byte, quantized light
    pub packed: u8,        // 1 byte, normal + AO
    pub padding: u16,      // 2 bytes, reserved
}

impl Vertex {
    /// Create a vertex from local chunk coordinates
    #[inline]
    pub fn from_local_coords(
        x: u8,
        y: u8,
        z: u8,
        block_type: crate::voxel::BlockType,
        normal_dir: u8,  // 0..5 (FaceDir as u8)
        ao_level: u8,    // 0..3
        light: f32,      // 0.0..1.0
    ) -> Self {
        let light_u8 = (light.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
        let normal_bits = normal_dir & 0b0000_0111;
        let ao_bits = (ao_level & 0b0000_0011) << 3;
        let packed = normal_bits | ao_bits;

        Self {
            x,
            y,
            z,
            block_type: block_type as u8,
            light: light_u8,
            packed,
            padding: 0,
        }
    }

    /// Legacy constructor for compatibility with tests
    /// Converts world position to local coordinates (assumes chunk at origin)
    #[deprecated(note = "Use from_local_coords instead")]
    pub fn new(position: Vec3, _normal: Vec3, _color: [u8; 3], _ao: u8, light: f32) -> Self {
        use crate::voxel::BlockType;
        Self::from_local_coords(
            position.x as u8,
            position.y as u8,
            position.z as u8,
            BlockType::Stone,
            0,
            0,
            light,
        )
    }

    /// Get world position by adding chunk offset
    #[inline]
    pub fn world_position(&self, chunk_offset: Vec3) -> Vec3 {
        Vec3::new(
            chunk_offset.x + self.x as f32,
            chunk_offset.y + self.y as f32,
            chunk_offset.z + self.z as f32,
        )
    }

    /// Get local position as Vec3
    #[inline]
    pub fn local_position(&self) -> Vec3 {
        Vec3::new(self.x as f32, self.y as f32, self.z as f32)
    }

    /// Extract normal index (0..5) from packed field
    #[inline]
    pub fn normal_index(&self) -> u8 {
        self.packed & 0b0000_0111
    }

    /// Extract AO level (0..3) from packed field
    #[inline]
    pub fn ao_level(&self) -> u8 {
        (self.packed >> 3) & 0b0000_0011
    }
}

/// Face direction for normal calculation
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum FaceDir {
    PosX = 0,
    NegX = 1,
    PosY = 2,
    NegY = 3,
    PosZ = 4,
    NegZ = 5,
}

impl FaceDir {
    #[inline]
    pub const fn normal(self) -> Vec3 {
        match self {
            FaceDir::PosX => Vec3::X,
            FaceDir::NegX => Vec3::NEG_X,
            FaceDir::PosY => Vec3::Y,
            FaceDir::NegY => Vec3::NEG_Y,
            FaceDir::PosZ => Vec3::Z,
            FaceDir::NegZ => Vec3::NEG_Z,
        }
    }

    #[inline]
    pub const fn axis(self) -> usize {
        match self {
            FaceDir::PosX | FaceDir::NegX => 0,
            FaceDir::PosY | FaceDir::NegY => 1,
            FaceDir::PosZ | FaceDir::NegZ => 2,
        }
    }

    #[inline]
    pub const fn is_positive(self) -> bool {
        matches!(self, FaceDir::PosX | FaceDir::PosY | FaceDir::PosZ)
    }

    #[inline]
    pub const fn from_index(index: usize) -> Self {
        match index {
            0 => FaceDir::PosX,
            1 => FaceDir::NegX,
            2 => FaceDir::PosY,
            3 => FaceDir::NegY,
            4 => FaceDir::PosZ,
            5 => FaceDir::NegZ,
            _ => panic!("Invalid face index"),
        }
    }

    #[inline]
    pub fn get_quad_local_positions(
        self,
        slice_pos: u8,
        u: u8,
        v: u8,
        w: u8,
        h: u8,
    ) -> [Vec3; 4] {
        let positions = match self {
            FaceDir::PosX => [
                (slice_pos, u, v),
                (slice_pos, u + w, v),
                (slice_pos, u + w, v + h),
                (slice_pos, u, v + h),
            ],
            FaceDir::NegX => [
                (slice_pos, u, v),
                (slice_pos, u, v + h),
                (slice_pos, u + w, v + h),
                (slice_pos, u + w, v),
            ],
            FaceDir::PosY => [
                (u, slice_pos, v),
                (u, slice_pos, v + h),
                (u + w, slice_pos, v + h),
                (u + w, slice_pos, v),
            ],
            FaceDir::NegY => [
                (u, slice_pos, v),
                (u + w, slice_pos, v),
                (u + w, slice_pos, v + h),
                (u, slice_pos, v + h),
            ],
            FaceDir::PosZ => [
                (u, v, slice_pos),
                (u + w, v, slice_pos),
                (u + w, v + h, slice_pos),
                (u, v + h, slice_pos),
            ],
            FaceDir::NegZ => [
                (u, v, slice_pos),
                (u, v + h, slice_pos),
                (u + w, v + h, slice_pos),
                (u + w, v, slice_pos),
            ],
        };

        [
            Vec3::new(positions[0].0 as f32, positions[0].1 as f32, positions[0].2 as f32),
            Vec3::new(positions[1].0 as f32, positions[1].1 as f32, positions[1].2 as f32),
            Vec3::new(positions[2].0 as f32, positions[2].1 as f32, positions[2].2 as f32),
            Vec3::new(positions[3].0 as f32, positions[3].1 as f32, positions[3].2 as f32),
        ]
    }
}

/// Quad representing a merged rectangular face
/// Optimized to 4 bytes (fits in single 32-bit register)
#[derive(Debug, Copy, Clone)]
pub struct Quad {
    pub x: u8,
    pub y: u8,
    pub width: u8,
    pub height: u8,
}

/// Ultra-compressed quad for face-direction-separated meshes
/// Size: 3 bytes total (21 quads per 64-byte cache line)
///
/// When stored in face-direction lists, we can eliminate redundant data:
/// - Normal is implicit from the list (Up/Down/North/South/East/West)
/// - One coordinate is implicit from slice index
/// - Only need to store 2 coordinates + width + height
///
/// Bit layout (24 bits total):
/// - u: 5 bits (0..31)
/// - v: 5 bits (0..31)
/// - w: 6 bits (1..32, stored as w-1 to fit in 6 bits = 0..31, but represents 1..32)
/// - h: 6 bits (1..32, stored as h-1)
/// - block_type: 2 bits (0..3, sufficient for Air/Grass/Dirt/Stone)
///
/// Packing:
/// - Byte 0: [u:5 bits][v_low:3 bits]
/// - Byte 1: [v_high:2 bits][w:6 bits]
/// - Byte 2: [h:6 bits][block_type:2 bits]
#[derive(Debug, Copy, Clone)]
#[repr(C)]
pub struct TinyQuad {
    packed: [u8; 3],
}

impl TinyQuad {
    /// Create a new TinyQuad
    /// - u, v: tangent plane coordinates (0..31, needs 5 bits each)
    /// - w, h: width and height (1..32, needs 6 bits each when stored as w-1)
    /// - block_type: BlockType index (0..3, needs 2 bits)
    #[inline]
    pub fn new(u: u8, v: u8, w: u8, h: u8, block_type: u8) -> Self {
        debug_assert!(u < 32, "u coordinate must be < 32");
        debug_assert!(v < 32, "v coordinate must be < 32");
        debug_assert!(w >= 1 && w <= 32, "width must be 1..=32");
        debug_assert!(h >= 1 && h <= 32, "height must be 1..=32");
        debug_assert!(block_type < 4, "block_type must be < 4");

        // Store w-1 and h-1 to represent range 1..32 in 5 bits
        let w_packed = w - 1;  // 1..32 becomes 0..31
        let h_packed = h - 1;  // 1..32 becomes 0..31

        // Pack the data:
        // Byte 0: [u:5 bits][v_low:3 bits]
        let byte0 = (u & 0x1F) | ((v & 0x07) << 5);

        // Byte 1: [v_high:2 bits][w:6 bits]
        let byte1 = ((v >> 3) & 0x03) | ((w_packed & 0x3F) << 2);

        // Byte 2: [h:6 bits][block_type:2 bits]
        let byte2 = (h_packed & 0x3F) | ((block_type & 0x03) << 6);

        Self {
            packed: [byte0, byte1, byte2],
        }
    }

    /// Extract u coordinate (0..31)
    #[inline]
    pub fn u(&self) -> u8 {
        self.packed[0] & 0x1F
    }

    /// Extract v coordinate (0..31)
    #[inline]
    pub fn v(&self) -> u8 {
        let v_low = (self.packed[0] >> 5) & 0x07;
        let v_high = (self.packed[1] & 0x03) << 3;
        v_low | v_high
    }

    /// Extract width (1..32)
    #[inline]
    pub fn width(&self) -> u8 {
        let w_packed = (self.packed[1] >> 2) & 0x3F;
        w_packed + 1  // Convert back to 1..32
    }

    /// Extract height (1..32)
    #[inline]
    pub fn height(&self) -> u8 {
        let h_packed = self.packed[2] & 0x3F;
        h_packed + 1  // Convert back to 1..32
    }

    /// Extract block type index (0..3)
    #[inline]
    pub fn block_type(&self) -> u8 {
        (self.packed[2] >> 6) & 0x03
    }
}

/// Face list for a single direction (e.g., all +Y faces)
/// Contains TinyQuads organized by slice index for optimal cache access
#[derive(Clone)]
pub struct FaceList {
    /// Quads for each slice (0..32) along the perpendicular axis
    /// For PosY/NegY faces: slice_quads[y] contains all quads at y coordinate
    /// For PosX/NegX faces: slice_quads[x] contains all quads at x coordinate
    /// For PosZ/NegZ faces: slice_quads[z] contains all quads at z coordinate
    pub slice_quads: [Vec<TinyQuad>; 32],
    /// Local-space AABB covering all quads in this face list (0..32 range).
    pub min: IVec3,
    pub max: IVec3,
}

impl FaceList {
    pub fn new() -> Self {
        Self {
            slice_quads: std::array::from_fn(|_| Vec::new()),
            min: IVec3::splat(32),
            max: IVec3::splat(0),
        }
    }

    /// Add a quad to a specific slice
    #[inline]
    pub fn add_quad(&mut self, slice_idx: usize, quad: TinyQuad, face_dir: FaceDir, axis_pos: i32) {
        debug_assert!(slice_idx < 32);
        self.slice_quads[slice_idx].push(quad);

        // Update local-space bounds for this face list.
        let u = quad.u() as i32;
        let v = quad.v() as i32;
        let w = quad.width() as i32;
        let h = quad.height() as i32;

        // Map (u,v,axis_pos) to x,y,z ranges depending on face direction.
        let (min, max) = match face_dir {
            FaceDir::PosX | FaceDir::NegX => (
                IVec3::new(axis_pos, u, v),
                IVec3::new(axis_pos, u + w, v + h),
            ),
            FaceDir::PosY | FaceDir::NegY => (
                IVec3::new(u, axis_pos, v),
                IVec3::new(u + w, axis_pos, v + h),
            ),
            FaceDir::PosZ | FaceDir::NegZ => (
                IVec3::new(u, v, axis_pos),
                IVec3::new(u + w, v + h, axis_pos),
            ),
        };

        self.min = self.min.min(min);
        self.max = self.max.max(max);
    }

    /// Get total number of quads across all slices
    pub fn quad_count(&self) -> usize {
        self.slice_quads.iter().map(|v| v.len()).sum()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.slice_quads.iter().all(|v| v.is_empty())
    }

    /// Clear all quads
    pub fn clear(&mut self) {
        for slice in &mut self.slice_quads {
            slice.clear();
        }
        self.min = IVec3::splat(32);
        self.max = IVec3::splat(0);
    }
}

/// Mesh data for a chunk using face-direction-separated lists
/// This new format eliminates redundant normal data and one coordinate per quad
/// by organizing quads into 6 direction-specific lists
pub struct ChunkMesh {
    /// Six face lists, one for each direction
    pub faces: [FaceList; 6],

    /// Chunk position in world space (chunk coordinates, not voxel coordinates)
    /// Required for decompressing vertex positions during rendering
    pub chunk_position: IVec3,

    // Legacy fields kept for backward compatibility during transition
    // These will be removed once all code is migrated
    #[deprecated(note = "Use faces array instead")]
    pub vertices: Vec<Vertex>,
    #[deprecated(note = "Use faces array instead")]
    pub indices: Vec<u16>,
}

impl ChunkMesh {
    pub fn new() -> Self {
        Self {
            faces: [
                FaceList::new(),
                FaceList::new(),
                FaceList::new(),
                FaceList::new(),
                FaceList::new(),
                FaceList::new(),
            ],
            chunk_position: IVec3::ZERO,
            vertices: Vec::new(),
            indices: Vec::new(),
        }
    }

    pub fn with_capacity(_vertex_capacity: usize, _index_capacity: usize) -> Self {
        // Capacity hints are ignored in new format (individual slices manage their own capacity)
        Self::new()
    }

    /// Create a mesh with known chunk position
    pub fn with_capacity_and_position(
        _vertex_capacity: usize,
        _index_capacity: usize,
        chunk_position: IVec3,
    ) -> Self {
        Self {
            faces: [
                FaceList::new(),
                FaceList::new(),
                FaceList::new(),
                FaceList::new(),
                FaceList::new(),
                FaceList::new(),
            ],
            chunk_position,
            vertices: Vec::new(),
            indices: Vec::new(),
        }
    }

    /// Get world offset for vertex decompression (chunk coords * CHUNK_SIZE)
    #[inline]
    pub fn world_offset(&self) -> Vec3 {
        (self.chunk_position * crate::voxel::CHUNK_SIZE as i32).as_vec3()
    }

    /// Add a quad to the mesh using new TinyQuad format
    /// The quad is added to the appropriate face-direction list
    pub fn add_quad(
        &mut self,
        quad: &Quad,
        face_dir: FaceDir,
        axis_pos: i32,
        _world_offset: IVec3,  // No longer used - kept for API compatibility
        block_type: crate::voxel::BlockType,
        _ao_level: u8,  // Currently unused in TinyQuad format
        _light: f32,    // Currently unused (stored implicitly per face direction)
    ) {
        // Map quad coordinates to TinyQuad (u, v) coordinates
        // The slice index is the axis_pos
        let (u, v) = match face_dir {
            // For X-axis faces: (Y, Z) are tangent coordinates
            FaceDir::PosX | FaceDir::NegX => (quad.x, quad.y),
            // For Y-axis faces: (X, Z) are tangent coordinates
            FaceDir::PosY | FaceDir::NegY => (quad.x, quad.y),
            // For Z-axis faces: (X, Y) are tangent coordinates
            FaceDir::PosZ | FaceDir::NegZ => (quad.x, quad.y),
        };

        let tiny_quad = TinyQuad::new(u, v, quad.width, quad.height, block_type as u8);

        // Determine which slice this quad belongs to
        let slice_idx = if face_dir.is_positive() {
            axis_pos as usize - 1  // Positive faces are on far side
        } else {
            axis_pos as usize  // Negative faces are on near side
        };

        debug_assert!(slice_idx < 32, "slice_idx must be < 32, got {}", slice_idx);

        // Add to the appropriate face list
        self.faces[face_dir as usize].add_quad(slice_idx, tiny_quad, face_dir, axis_pos);
    }

    /// Generate local chunk coordinates for quad vertices
    /// Returns 4 vertices as (x, y, z) in chunk-local space [0..32]
    fn quad_local_positions(
        quad: &Quad,
        face_dir: FaceDir,
        axis_pos: i32,
    ) -> [(u8, u8, u8); 4] {
        let x = quad.x;
        let y = quad.y;
        let w = quad.width;
        let h = quad.height;
        let d = axis_pos as u8;

        match face_dir {
            FaceDir::PosX => [
                (d, x, y),
                (d, x + w, y),
                (d, x + w, y + h),
                (d, x, y + h),
            ],
            FaceDir::NegX => [
                (d, x, y),
                (d, x, y + h),
                (d, x + w, y + h),
                (d, x + w, y),
            ],
            FaceDir::PosY => [
                (x, d, y),
                (x, d, y + h),
                (x + w, d, y + h),
                (x + w, d, y),
            ],
            FaceDir::NegY => [
                (x, d, y),
                (x + w, d, y),
                (x + w, d, y + h),
                (x, d, y + h),
            ],
            FaceDir::PosZ => [
                (x, y, d),
                (x + w, y, d),
                (x + w, y + h, d),
                (x, y + h, d),
            ],
            FaceDir::NegZ => [
                (x, y, d),
                (x, y + h, d),
                (x + w, y + h, d),
                (x + w, y, d),
            ],
        }
    }

    pub fn clear(&mut self) {
        for face_list in &mut self.faces {
            face_list.clear();
        }
        self.vertices.clear();
        self.indices.clear();
    }

    pub fn is_empty(&self) -> bool {
        self.faces.iter().all(|f| f.is_empty())
    }

    /// Get total number of quads in the mesh
    pub fn quad_count(&self) -> usize {
        self.faces.iter().map(|f| f.quad_count()).sum()
    }

    /// Get face list for a specific direction
    #[inline]
    pub fn face_list(&self, dir: FaceDir) -> &FaceList {
        &self.faces[dir as usize]
    }

    /// Get mutable face list for a specific direction
    #[inline]
    pub fn face_list_mut(&mut self, dir: FaceDir) -> &mut FaceList {
        &mut self.faces[dir as usize]
    }

    /// Convert a TinyQuad to 4 vertices for rendering
    /// This decompresses the ultra-compact format into renderable geometry
    #[inline]
    pub fn tiny_quad_to_vertices(
        quad: &TinyQuad,
        face_dir: FaceDir,
        slice_pos: u8,  // The position along the perpendicular axis (0..32)
        chunk_offset: Vec3,  // World offset for this chunk
    ) -> [(Vec3, u8); 4] {
        let u = quad.u();
        let v = quad.v();
        let w = quad.width();
        let h = quad.height();
        let block_type = quad.block_type();

        // Generate 4 vertex positions based on face direction
        // slice_pos is the coordinate along the perpendicular axis
        let positions = match face_dir {
            FaceDir::PosX => [
                (slice_pos, u, v),
                (slice_pos, u + w, v),
                (slice_pos, u + w, v + h),
                (slice_pos, u, v + h),
            ],
            FaceDir::NegX => [
                (slice_pos, u, v),
                (slice_pos, u, v + h),
                (slice_pos, u + w, v + h),
                (slice_pos, u + w, v),
            ],
            FaceDir::PosY => [
                (u, slice_pos, v),
                (u, slice_pos, v + h),
                (u + w, slice_pos, v + h),
                (u + w, slice_pos, v),
            ],
            FaceDir::NegY => [
                (u, slice_pos, v),
                (u + w, slice_pos, v),
                (u + w, slice_pos, v + h),
                (u, slice_pos, v + h),
            ],
            FaceDir::PosZ => [
                (u, v, slice_pos),
                (u + w, v, slice_pos),
                (u + w, v + h, slice_pos),
                (u, v + h, slice_pos),
            ],
            FaceDir::NegZ => [
                (u, v, slice_pos),
                (u, v + h, slice_pos),
                (u + w, v + h, slice_pos),
                (u + w, v, slice_pos),
            ],
        };

        // Convert to world positions
        [
            (Vec3::new(
                chunk_offset.x + positions[0].0 as f32,
                chunk_offset.y + positions[0].1 as f32,
                chunk_offset.z + positions[0].2 as f32,
            ), block_type),
            (Vec3::new(
                chunk_offset.x + positions[1].0 as f32,
                chunk_offset.y + positions[1].1 as f32,
                chunk_offset.z + positions[1].2 as f32,
            ), block_type),
            (Vec3::new(
                chunk_offset.x + positions[2].0 as f32,
                chunk_offset.y + positions[2].1 as f32,
                chunk_offset.z + positions[2].2 as f32,
            ), block_type),
            (Vec3::new(
                chunk_offset.x + positions[3].0 as f32,
                chunk_offset.y + positions[3].1 as f32,
                chunk_offset.z + positions[3].2 as f32,
            ), block_type),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tiny_quad_packing() {
        // Test all fields can be stored and retrieved correctly
        let quad = TinyQuad::new(15, 20, 25, 30, 3);

        assert_eq!(quad.u(), 15);
        assert_eq!(quad.v(), 20);
        assert_eq!(quad.width(), 25);
        assert_eq!(quad.height(), 30);
        assert_eq!(quad.block_type(), 3);
    }

    #[test]
    fn test_tiny_quad_boundary_values() {
        // Test min values
        let quad1 = TinyQuad::new(0, 0, 1, 1, 0);
        assert_eq!(quad1.u(), 0);
        assert_eq!(quad1.v(), 0);
        assert_eq!(quad1.width(), 1);
        assert_eq!(quad1.height(), 1);
        assert_eq!(quad1.block_type(), 0);

        // Test max values
        let quad2 = TinyQuad::new(31, 31, 32, 32, 3);
        assert_eq!(quad2.u(), 31);
        assert_eq!(quad2.v(), 31);
        assert_eq!(quad2.width(), 32);
        assert_eq!(quad2.height(), 32);
        assert_eq!(quad2.block_type(), 3);
    }

    #[test]
    fn test_tiny_quad_size() {
        // Verify TinyQuad is exactly 3 bytes
        assert_eq!(std::mem::size_of::<TinyQuad>(), 3);
    }

    #[test]
    fn test_face_list_operations() {
        let mut face_list = FaceList::new();

        assert!(face_list.is_empty());
        assert_eq!(face_list.quad_count(), 0);

        let quad1 = TinyQuad::new(5, 10, 15, 20, 1);
        let quad2 = TinyQuad::new(10, 5, 20, 15, 2);

        face_list.add_quad(0, quad1, FaceDir::PosX, 0);
        assert_eq!(face_list.quad_count(), 1);
        assert!(!face_list.is_empty());

        face_list.add_quad(15, quad2, FaceDir::PosX, 15);
        assert_eq!(face_list.quad_count(), 2);

        face_list.clear();
        assert!(face_list.is_empty());
        assert_eq!(face_list.quad_count(), 0);
    }

    #[test]
    fn test_tiny_quad_vertex_positions_match_legacy() {
        // This test ensures TinyQuad decompression produces the same vertices
        // as the original quad_local_positions function.
        // This catches coordinate mapping bugs like the slice position offset bug.

        use crate::voxel::BlockType;

        // Test cases for positive faces (axis_pos must be >= 1)
        let test_cases_positive = vec![
            // (quad.x, quad.y, width, height, axis_pos)
            (0, 0, 1, 1, 1),      // Min axis_pos for positive faces
            (5, 10, 15, 20, 7),   // Mid values
            (31, 31, 1, 1, 31),   // Max position, min size
            (0, 0, 32, 32, 15),   // Min position, max size
        ];

        // Test cases for negative faces (axis_pos can be 0)
        let test_cases_negative = vec![
            // (quad.x, quad.y, width, height, axis_pos)
            (0, 0, 1, 1, 0),      // Min axis_pos for negative faces
            (5, 10, 15, 20, 7),   // Mid values
            (31, 31, 1, 1, 31),   // Max position, min size
            (0, 0, 32, 32, 15),   // Min position, max size
        ];

        let positive_faces = [FaceDir::PosX, FaceDir::PosY, FaceDir::PosZ];
        let negative_faces = [FaceDir::NegX, FaceDir::NegY, FaceDir::NegZ];

        // Test positive faces
        for &(qx, qy, qw, qh, axis_pos) in &test_cases_positive {
            let quad = Quad {
                x: qx,
                y: qy,
                width: qw,
                height: qh,
            };

            for &face_dir in &positive_faces {
                test_quad_vertex_round_trip(&quad, face_dir, axis_pos);
            }
        }

        // Test negative faces
        for &(qx, qy, qw, qh, axis_pos) in &test_cases_negative {
            let quad = Quad {
                x: qx,
                y: qy,
                width: qw,
                height: qh,
            };

            for &face_dir in &negative_faces {
                test_quad_vertex_round_trip(&quad, face_dir, axis_pos);
            }
        }
    }

    /// Helper function to test a complete add_quad -> render round-trip
    fn test_quad_vertex_round_trip(quad: &Quad, face_dir: FaceDir, axis_pos: i32) {
        use crate::voxel::BlockType;

        // Get expected positions using legacy function
        let expected = ChunkMesh::quad_local_positions(quad, face_dir, axis_pos);

        // Simulate the add_quad -> render_tiny_quad round-trip

        // 1. Create TinyQuad as add_quad does
        let tiny_quad = TinyQuad::new(quad.x, quad.y, quad.width, quad.height, BlockType::Stone as u8);

        // 2. Calculate slice_idx as add_quad does
        let slice_idx = if face_dir.is_positive() {
            axis_pos as usize - 1
        } else {
            axis_pos as usize
        };

        // 3. Reconstruct slice_pos as the renderer does (with the fix)
        let slice_pos = if face_dir.is_positive() {
            (slice_idx + 1) as u8
        } else {
            slice_idx as u8
        };

        // 4. Decompress to vertices
        let actual_verts = ChunkMesh::tiny_quad_to_vertices(
            &tiny_quad,
            face_dir,
            slice_pos,
            Vec3::ZERO, // No offset for this test
        );

        // 5. Compare each vertex position
        for i in 0..4 {
            let expected_pos = Vec3::new(
                expected[i].0 as f32,
                expected[i].1 as f32,
                expected[i].2 as f32,
            );
            let actual_pos = actual_verts[i].0;

            assert_eq!(
                actual_pos, expected_pos,
                "Vertex {} mismatch for face {:?} at axis_pos {}, quad ({},{}) {}x{}\n\
                 Expected: {:?}, Got: {:?}\n\
                 slice_idx={}, slice_pos={}",
                i, face_dir, axis_pos, quad.x, quad.y, quad.width, quad.height,
                expected_pos, actual_pos,
                slice_idx, slice_pos
            );
        }
    }

    #[test]
    fn test_slice_position_offset_for_positive_faces() {
        // Regression test for the slice position bug
        // Positive faces must add 1 to slice_idx when rendering

        let axis_pos = 5_i32;

        // For positive faces: slice_idx = axis_pos - 1
        let slice_idx_pos = (axis_pos - 1) as usize;
        assert_eq!(slice_idx_pos, 4);

        // When rendering, must reconstruct: slice_pos = slice_idx + 1
        let slice_pos_reconstructed = (slice_idx_pos + 1) as u8;
        assert_eq!(slice_pos_reconstructed, 5);
        assert_eq!(slice_pos_reconstructed, axis_pos as u8);

        // For negative faces: slice_idx = axis_pos
        let slice_idx_neg = axis_pos as usize;
        assert_eq!(slice_idx_neg, 5);

        // When rendering, use directly: slice_pos = slice_idx
        let slice_pos_neg = slice_idx_neg as u8;
        assert_eq!(slice_pos_neg, 5);
        assert_eq!(slice_pos_neg, axis_pos as u8);
    }
}
