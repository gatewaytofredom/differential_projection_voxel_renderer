use super::mesh::{ChunkMesh, FaceDir, Quad};
use crate::voxel::block_type::BLOCK_TYPE_COUNT;
/// Optimized binary greedy meshing algorithm
/// Uses bitwise operations to represent 32x32 slices as u32 bitmasks
/// This allows processing up to 32 voxels simultaneously
use crate::voxel::{BlockData, BlockType, Chunk, CHUNK_SIZE, CHUNK_VOLUME};
use crate::count_call;
use crate::perf::FUNCTION_COUNTERS;
use glam::IVec3;
use std::collections::HashMap;

const CHUNK_SIZE_U32: u32 = CHUNK_SIZE as u32;
const BLOCK_TYPES: [BlockType; BLOCK_TYPE_COUNT] = BlockType::ALL;
type SliceMask = [u32; CHUNK_SIZE];

/// Index mapping chunk positions to chunk references for fast neighbour lookup.
type ChunkIndex<'a> = HashMap<(i32, i32, i32), &'a Chunk>;

struct ChunkNeighbors<'a> {
    pos_x: Option<&'a Chunk>,
    neg_x: Option<&'a Chunk>,
    pos_y: Option<&'a Chunk>,
    neg_y: Option<&'a Chunk>,
    pos_z: Option<&'a Chunk>,
    neg_z: Option<&'a Chunk>,
}

/// Workspace for meshing to eliminate per-slice allocations
/// Reusing this across slices eliminates ~192+ allocations per chunk
/// (6 faces × 32 slices × variable block types per slice)
pub struct MeshingWorkspace {
    /// Reusable quad buffer for greedy meshing (reused across all slices)
    pub quads: Vec<Quad>,
}

impl MeshingWorkspace {
    pub fn new() -> Self {
        Self {
            quads: Vec::with_capacity(64),
        }
    }

    /// Clear workspace for reuse (faster than dropping and reallocating)
    #[inline]
    pub fn clear(&mut self) {
        self.quads.clear();
    }
}

pub struct BinaryGreedyMesher;

impl BinaryGreedyMesher {
    /// Generate mesh from a single chunk without neighbour information.
    /// This retains the original behaviour where chunk borders are treated as air.
    pub fn mesh_chunk(chunk: &Chunk) -> Option<ChunkMesh> {
        let chunks: [&Chunk; 1] = [chunk];
        Self::mesh_chunk_in_world(chunk, &chunks)
    }

    /// Generate meshes for a collection of chunks.
    /// Returns meshes in the same order as input chunks, skipping uniform ones.
    pub fn mesh_world(chunks: &[&Chunk]) -> Vec<ChunkMesh> {
        if chunks.is_empty() {
            return Vec::new();
        }

        // Build an index for neighbour lookup once for the whole set.
        let mut index: ChunkIndex<'_> = HashMap::with_capacity(chunks.len());
        for &chunk in chunks {
            let pos = chunk.position;
            index.insert((pos.x, pos.y, pos.z), chunk);
        }

        chunks
            .iter()
            .filter_map(|chunk| Self::mesh_chunk_in_indexed_world(chunk, &index))
            .collect()
    }

    /// Generate mesh from a chunk using binary greedy meshing with access to neighbouring chunks.
    /// `all_chunks` should contain this chunk and any neighbours that might touch it.
    /// Returns None if chunk is uniform (optimization).
    pub fn mesh_chunk_in_world(chunk: &Chunk, all_chunks: &[&Chunk]) -> Option<ChunkMesh> {
        count_call!(FUNCTION_COUNTERS.mesh_chunk_calls);

        // Fast path for uniform chunks
        if chunk.is_uniform() {
            return None;
        }

        // Optimized capacity: terrain chunks average ~800 vertices, ~2400 indices
        // Using power-of-2 sizes for better allocator performance
        let mut mesh = ChunkMesh::with_capacity_and_position(1024, 3072, chunk.position);

        // Calculate world offset for this chunk (for backward compatibility with add_quad API)
        let chunk_world_offset = chunk.position * (CHUNK_SIZE as i32);

        // Resolve direct neighbours once for this chunk using linear search
        let neighbors = Self::build_neighbors(chunk, all_chunks);

        // Create workspace to eliminate allocations across all faces
        let mut workspace = MeshingWorkspace::new();

        // Process each face direction
        for face_dir in [
            FaceDir::PosX,
            FaceDir::NegX,
            FaceDir::PosY,
            FaceDir::NegY,
            FaceDir::PosZ,
            FaceDir::NegZ,
        ] {
            Self::mesh_face(chunk, &neighbors, face_dir, chunk_world_offset, &mut mesh, &mut workspace);
        }

        if mesh.is_empty() {
            None
        } else {
            Some(mesh)
        }
    }

    /// Generate mesh from a chunk using binary greedy meshing with access to neighbouring chunks,
    /// using a pre-built index for O(1) neighbour lookup. `index` should contain this chunk and
    /// any neighbours that might touch it.
    /// Returns None if chunk is uniform (optimization).
    pub fn mesh_chunk_in_indexed_world<'a>(
        chunk: &'a Chunk,
        index: &ChunkIndex<'a>,
    ) -> Option<ChunkMesh> {
        count_call!(FUNCTION_COUNTERS.mesh_chunk_calls);

        // Fast path for uniform chunks
        if chunk.is_uniform() {
            return None;
        }

        // Optimized capacity: terrain chunks average ~800 vertices, ~2400 indices
        // Using power-of-2 sizes for better allocator performance
        let mut mesh = ChunkMesh::with_capacity_and_position(1024, 3072, chunk.position);

        // Calculate world offset for this chunk (for backward compatibility with add_quad API)
        let chunk_world_offset = chunk.position * (CHUNK_SIZE as i32);

        // Resolve direct neighbours once for this chunk using the index
        let neighbors = Self::build_neighbors_indexed(chunk, index);

        // Create workspace to eliminate allocations across all faces
        let mut workspace = MeshingWorkspace::new();

        // Process each face direction
        for face_dir in [
            FaceDir::PosX,
            FaceDir::NegX,
            FaceDir::PosY,
            FaceDir::NegY,
            FaceDir::PosZ,
            FaceDir::NegZ,
        ] {
            Self::mesh_face(chunk, &neighbors, face_dir, chunk_world_offset, &mut mesh, &mut workspace);
        }

        if mesh.is_empty() {
            None
        } else {
            Some(mesh)
        }
    }

    /// Find neighbors using linear search - faster than HashMap for small chunk counts.
    /// With typical chunk counts (<27 for a 3x3x3 world), linear search with good cache
    /// locality outperforms HashMap due to avoided allocation and hashing overhead.
    #[inline]
    fn find_chunk_at_position<'a>(chunks: &[&'a Chunk], position: IVec3) -> Option<&'a Chunk> {
        count_call!(FUNCTION_COUNTERS.find_chunk_calls);
        // Linear search is O(n) but with excellent cache locality and no hashing overhead.
        // Benchmarking shows this is faster than HashMap for n < ~50 chunks.
        chunks.iter().find(|c| c.position == position).copied()
    }

    #[inline]
    fn build_neighbors<'a>(chunk: &'a Chunk, all_chunks: &[&'a Chunk]) -> ChunkNeighbors<'a> {
        let pos = chunk.position;
        ChunkNeighbors {
            pos_x: Self::find_chunk_at_position(all_chunks, pos + IVec3::X),
            neg_x: Self::find_chunk_at_position(all_chunks, pos - IVec3::X),
            pos_y: Self::find_chunk_at_position(all_chunks, pos + IVec3::Y),
            neg_y: Self::find_chunk_at_position(all_chunks, pos - IVec3::Y),
            pos_z: Self::find_chunk_at_position(all_chunks, pos + IVec3::Z),
            neg_z: Self::find_chunk_at_position(all_chunks, pos - IVec3::Z),
        }
    }

    #[inline]
    fn build_neighbors_indexed<'a>(
        chunk: &'a Chunk,
        index: &ChunkIndex<'a>,
    ) -> ChunkNeighbors<'a> {
        let pos = chunk.position;
        let key = |p: IVec3| (p.x, p.y, p.z);
        ChunkNeighbors {
            pos_x: index.get(&key(pos + IVec3::X)).copied(),
            neg_x: index.get(&key(pos - IVec3::X)).copied(),
            pos_y: index.get(&key(pos + IVec3::Y)).copied(),
            neg_y: index.get(&key(pos - IVec3::Y)).copied(),
            pos_z: index.get(&key(pos + IVec3::Z)).copied(),
            neg_z: index.get(&key(pos - IVec3::Z)).copied(),
        }
    }

    /// Mesh all faces in a single direction
    /// Uses workspace to eliminate allocations (reuses quad Vec across all slices)
    fn mesh_face(
        chunk: &Chunk,
        neighbors: &ChunkNeighbors<'_>,
        face_dir: FaceDir,
        world_offset: glam::IVec3,
        mesh: &mut ChunkMesh,
        workspace: &mut MeshingWorkspace,
    ) {
        let neighbor = match face_dir {
            FaceDir::PosX => neighbors.pos_x,
            FaceDir::NegX => neighbors.neg_x,
            FaceDir::PosY => neighbors.pos_y,
            FaceDir::NegY => neighbors.neg_y,
            FaceDir::PosZ => neighbors.pos_z,
            FaceDir::NegZ => neighbors.neg_z,
        };

        // Pre-compute lighting for this face direction
        let light = Self::compute_face_lighting(face_dir);

        // For each slice along the perpendicular axis
        for slice_idx in 0..CHUNK_SIZE {
            let (masks_by_type, used_block_types) =
                Self::generate_binary_masks(chunk, neighbor, face_dir, slice_idx);

            // For each block type in this slice, run greedy meshing separately
            for (idx, &block_type) in BLOCK_TYPES.iter().enumerate() {
                if !used_block_types[idx] {
                    continue;
                }

                // Reuse workspace quad buffer to eliminate allocation
                workspace.clear();
                Self::greedy_mesh_slice_into(&masks_by_type[idx], &mut workspace.quads);

                // For positive faces, the quad is on the far side of the voxel
                // For negative faces, the quad is on the near side
                // axis_pos represents where the FACE is, not where the voxel is
                let axis_pos = if face_dir.is_positive() {
                    slice_idx as i32 + 1
                } else {
                    slice_idx as i32
                };

                for quad in &workspace.quads {
                    // Pass BlockType directly instead of color
                    // AO level is 0 for now (can be enhanced later)
                    mesh.add_quad(quad, face_dir, axis_pos, world_offset, block_type, 0, light);
                }
            }
        }
    }

    /// Compute lighting value for a face based on its direction
    /// Uses the default shading configuration for consistency
    #[inline]
    fn compute_face_lighting(face_dir: FaceDir) -> f32 {
        // Default light direction: Vec3::new(0.4, 1.0, 0.3).normalize()
        // Precomputed normalized values
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

    /// Generate binary masks for a slice, grouped by block type.
    /// Each u32 represents a row of up to 32 voxels.
    fn generate_binary_masks(
        chunk: &Chunk,
        neighbor: Option<&Chunk>,
        face_dir: FaceDir,
        slice_idx: usize,
    ) -> ([SliceMask; BLOCK_TYPE_COUNT], [bool; BLOCK_TYPE_COUNT]) {
        count_call!(FUNCTION_COUNTERS.generate_binary_masks_calls);

        let mut masks = [[0u32; CHUNK_SIZE]; BLOCK_TYPE_COUNT];
        let mut used_types = [false; BLOCK_TYPE_COUNT];
        let is_positive = face_dir.is_positive();
        let axis = face_dir.axis();

        // Fast path: direct array access when chunk is varied
        // This eliminates enum branching from the inner loop (~1024 iterations)
        if let Some(chunk_blocks) = chunk.get_varied_blocks() {
            // Neighbor may be varied or uniform. We precompute both views:
            // - neighbor_blocks: direct array access for varied chunks
            // - neighbor_uniform_solid: fast path for uniform solid chunks
            let (neighbor_blocks, neighbor_uniform_solid) = match neighbor {
                Some(n) => {
                    let uniform_solid = n
                        .uniform_block_type()
                        .map(|t| t.is_solid())
                        .unwrap_or(false);
                    let varied_blocks = n.get_varied_blocks();
                    (varied_blocks, uniform_solid)
                }
                None => (None, false),
            };

            // Optimize memory access pattern: ensure innermost loop has unit stride
            // For axis 0 and 1, col increments X which is unit stride
            // For axis 2, col increments Y which is stride-32, so we swap row/col
            match axis {
                0 | 1 => {
                    // X-axis or Y-axis: col increments X (unit stride)
                    for row in 0..CHUNK_SIZE {
                        for col in 0..CHUNK_SIZE {
                            let (x, y, z) = Self::slice_to_chunk_coords(axis, slice_idx, row, col);
                            let index = (z * CHUNK_SIZE * CHUNK_SIZE) + (y * CHUNK_SIZE) + x;
                            let current = chunk_blocks[index];

                            if !current.is_solid() {
                                continue;
                            }

                            let has_neighbor = if is_positive {
                                Self::has_solid_neighbor_pos_direct(
                                    chunk_blocks,
                                    neighbor_blocks,
                                    neighbor_uniform_solid,
                                    x,
                                    y,
                                    z,
                                    axis,
                                )
                            } else {
                                Self::has_solid_neighbor_neg_direct(
                                    chunk_blocks,
                                    neighbor_blocks,
                                    neighbor_uniform_solid,
                                    x,
                                    y,
                                    z,
                                    axis,
                                )
                            };

                            if !has_neighbor {
                                let type_idx = current.block_type as usize;
                                debug_assert!(type_idx < BLOCK_TYPE_COUNT);
                                masks[type_idx][row] |= 1u32 << col;
                                used_types[type_idx] = true;
                            }
                        }
                    }
                }
                2 => {
                    // Z-axis: iterate X in outer, Y in inner for unit stride
                    // slice_to_chunk_coords(2, slice, row, col) = (row, col, slice)
                    // So for Z-axis: row=X, col=Y
                    for x in 0..CHUNK_SIZE {
                        for y in 0..CHUNK_SIZE {
                            let z = slice_idx;
                            let index = (z * CHUNK_SIZE * CHUNK_SIZE) + (y * CHUNK_SIZE) + x;
                            let current = chunk_blocks[index];

                            if !current.is_solid() {
                                continue;
                            }

                            let has_neighbor = if is_positive {
                                Self::has_solid_neighbor_pos_direct(
                                    chunk_blocks,
                                    neighbor_blocks,
                                    neighbor_uniform_solid,
                                    x,
                                    y,
                                    z,
                                    axis,
                                )
                            } else {
                                Self::has_solid_neighbor_neg_direct(
                                    chunk_blocks,
                                    neighbor_blocks,
                                    neighbor_uniform_solid,
                                    x,
                                    y,
                                    z,
                                    axis,
                                )
                            };

                            if !has_neighbor {
                                let type_idx = current.block_type as usize;
                                debug_assert!(type_idx < BLOCK_TYPE_COUNT);
                                // For Z-axis: row is X, col is Y (matches slice_to_chunk_coords)
                                masks[type_idx][x] |= 1u32 << y;
                                used_types[type_idx] = true;
                            }
                        }
                    }
                }
                _ => unreachable!(),
            }
        } else {
            // Fallback: original branching path for uniform chunks (shouldn't happen)
            for row in 0..CHUNK_SIZE {
                for col in 0..CHUNK_SIZE {
                    let (x, y, z) = Self::slice_to_chunk_coords(axis, slice_idx, row, col);
                    let current = chunk.get_block(x, y, z);

                    if !current.is_solid() {
                        continue;
                    }

                    let has_neighbor = if is_positive {
                        Self::has_solid_neighbor_pos(chunk, neighbor, x, y, z, axis)
                    } else {
                        Self::has_solid_neighbor_neg(chunk, neighbor, x, y, z, axis)
                    };

                    if !has_neighbor {
                        let index = current.block_type as usize;
                        debug_assert!(index < BLOCK_TYPE_COUNT);
                        masks[index][row] |= 1u32 << col;
                        used_types[index] = true;
                    }
                }
            }
        }

        (masks, used_types)
    }

    /// Map slice coordinates to chunk 3D coordinates
    /// For each axis, we iterate slices perpendicular to that axis
    /// row and col represent the 2D coordinates within each slice
    #[inline]
    fn slice_to_chunk_coords(
        axis: usize,
        slice: usize,
        row: usize,
        col: usize,
    ) -> (usize, usize, usize) {
        match axis {
            0 => (slice, row, col), // X axis: slice is X, (row, col) is (Y, Z)
            1 => (row, slice, col), // Y axis: slice is Y, (row, col) is (X, Z)
            2 => (row, col, slice), // Z axis: slice is Z, (row, col) is (X, Y)
            _ => unreachable!(),
        }
    }

    /// Check if there's a solid neighbor in positive direction (direct array access)
    /// Used when chunks are guaranteed to be Varied - eliminates enum branches
    #[inline]
    fn has_solid_neighbor_pos_direct(
        chunk_blocks: &[BlockData; CHUNK_VOLUME],
        neighbor_blocks: Option<&[BlockData; CHUNK_VOLUME]>,
        neighbor_uniform_solid: bool,
        x: usize,
        y: usize,
        z: usize,
        axis: usize,
    ) -> bool {
        count_call!(FUNCTION_COUNTERS.has_solid_neighbor_calls);
        match axis {
            // +X
            0 => {
                if x + 1 < CHUNK_SIZE {
                    let idx = (z * CHUNK_SIZE * CHUNK_SIZE) + (y * CHUNK_SIZE) + (x + 1);
                    chunk_blocks[idx].is_solid()
                } else if let Some(neighbor) = neighbor_blocks {
                    let idx = (z * CHUNK_SIZE * CHUNK_SIZE) + (y * CHUNK_SIZE);
                    neighbor[idx].is_solid()
                } else {
                    // Neighbor exists but is uniform: use precomputed solid flag
                    neighbor_uniform_solid
                }
            }
            // +Y
            1 => {
                if y + 1 < CHUNK_SIZE {
                    let idx = (z * CHUNK_SIZE * CHUNK_SIZE) + ((y + 1) * CHUNK_SIZE) + x;
                    chunk_blocks[idx].is_solid()
                } else if let Some(neighbor) = neighbor_blocks {
                    let idx = (z * CHUNK_SIZE * CHUNK_SIZE) + x;
                    neighbor[idx].is_solid()
                } else {
                    neighbor_uniform_solid
                }
            }
            // +Z
            2 => {
                if z + 1 < CHUNK_SIZE {
                    let idx = ((z + 1) * CHUNK_SIZE * CHUNK_SIZE) + (y * CHUNK_SIZE) + x;
                    chunk_blocks[idx].is_solid()
                } else if let Some(neighbor) = neighbor_blocks {
                    let idx = (y * CHUNK_SIZE) + x;
                    neighbor[idx].is_solid()
                } else {
                    neighbor_uniform_solid
                }
            }
            _ => false,
        }
    }

    /// Check if there's a solid neighbor in negative direction (direct array access)
    /// Used when chunks are guaranteed to be Varied - eliminates enum branches
    #[inline]
    fn has_solid_neighbor_neg_direct(
        chunk_blocks: &[BlockData; CHUNK_VOLUME],
        neighbor_blocks: Option<&[BlockData; CHUNK_VOLUME]>,
        neighbor_uniform_solid: bool,
        x: usize,
        y: usize,
        z: usize,
        axis: usize,
    ) -> bool {
        count_call!(FUNCTION_COUNTERS.has_solid_neighbor_calls);
        match axis {
            // -X
            0 => {
                if x > 0 {
                    let idx = (z * CHUNK_SIZE * CHUNK_SIZE) + (y * CHUNK_SIZE) + (x - 1);
                    chunk_blocks[idx].is_solid()
                } else if let Some(neighbor) = neighbor_blocks {
                    let idx =
                        (z * CHUNK_SIZE * CHUNK_SIZE) + (y * CHUNK_SIZE) + (CHUNK_SIZE - 1);
                    neighbor[idx].is_solid()
                } else {
                    neighbor_uniform_solid
                }
            }
            // -Y
            1 => {
                if y > 0 {
                    let idx = (z * CHUNK_SIZE * CHUNK_SIZE) + ((y - 1) * CHUNK_SIZE) + x;
                    chunk_blocks[idx].is_solid()
                } else if let Some(neighbor) = neighbor_blocks {
                    let idx =
                        (z * CHUNK_SIZE * CHUNK_SIZE) + ((CHUNK_SIZE - 1) * CHUNK_SIZE) + x;
                    neighbor[idx].is_solid()
                } else {
                    neighbor_uniform_solid
                }
            }
            // -Z
            2 => {
                if z > 0 {
                    let idx = ((z - 1) * CHUNK_SIZE * CHUNK_SIZE) + (y * CHUNK_SIZE) + x;
                    chunk_blocks[idx].is_solid()
                } else if let Some(neighbor) = neighbor_blocks {
                    let idx =
                        ((CHUNK_SIZE - 1) * CHUNK_SIZE * CHUNK_SIZE) + (y * CHUNK_SIZE) + x;
                    neighbor[idx].is_solid()
                } else {
                    neighbor_uniform_solid
                }
            }
            _ => false,
        }
    }

    /// Check if there's a solid neighbor in positive direction
    #[inline]
    fn has_solid_neighbor_pos(
        chunk: &Chunk,
        neighbor: Option<&Chunk>,
        x: usize,
        y: usize,
        z: usize,
        axis: usize,
    ) -> bool {
        count_call!(FUNCTION_COUNTERS.has_solid_neighbor_calls);
        match axis {
            // +X
            0 => {
                if x + 1 < CHUNK_SIZE {
                    chunk.get_block(x + 1, y, z).is_solid()
                } else if let Some(neighbor_chunk) = neighbor {
                    Self::chunk_block_is_solid(neighbor_chunk, 0, y, z)
                } else {
                    false
                }
            }
            // +Y
            1 => {
                if y + 1 < CHUNK_SIZE {
                    chunk.get_block(x, y + 1, z).is_solid()
                } else if let Some(neighbor_chunk) = neighbor {
                    Self::chunk_block_is_solid(neighbor_chunk, x, 0, z)
                } else {
                    false
                }
            }
            // +Z
            2 => {
                if z + 1 < CHUNK_SIZE {
                    chunk.get_block(x, y, z + 1).is_solid()
                } else if let Some(neighbor_chunk) = neighbor {
                    Self::chunk_block_is_solid(neighbor_chunk, x, y, 0)
                } else {
                    false
                }
            }
            _ => false,
        }
    }

    /// Check if there's a solid neighbor in negative direction
    #[inline]
    fn has_solid_neighbor_neg(
        chunk: &Chunk,
        neighbor: Option<&Chunk>,
        x: usize,
        y: usize,
        z: usize,
        axis: usize,
    ) -> bool {
        count_call!(FUNCTION_COUNTERS.has_solid_neighbor_calls);
        match axis {
            // -X
            0 => {
                if x > 0 {
                    chunk.get_block(x - 1, y, z).is_solid()
                } else if let Some(neighbor_chunk) = neighbor {
                    Self::chunk_block_is_solid(neighbor_chunk, CHUNK_SIZE - 1, y, z)
                } else {
                    false
                }
            }
            // -Y
            1 => {
                if y > 0 {
                    chunk.get_block(x, y - 1, z).is_solid()
                } else if let Some(neighbor_chunk) = neighbor {
                    Self::chunk_block_is_solid(neighbor_chunk, x, CHUNK_SIZE - 1, z)
                } else {
                    false
                }
            }
            // -Z
            2 => {
                if z > 0 {
                    chunk.get_block(x, y, z - 1).is_solid()
                } else if let Some(neighbor_chunk) = neighbor {
                    Self::chunk_block_is_solid(neighbor_chunk, x, y, CHUNK_SIZE - 1)
                } else {
                    false
                }
            }
            _ => false,
        }
    }

    #[inline]
    fn chunk_block_is_solid(chunk: &Chunk, x: usize, y: usize, z: usize) -> bool {
        match chunk.uniform_block_type() {
            Some(block_type) => block_type.is_solid(),
            None => chunk.get_block(x, y, z).is_solid(),
        }
    }

    /// Greedy mesh a binary slice to generate quads (legacy API - allocates Vec)
    /// This is the core optimization - uses bit manipulation for speed
    /// Note: Prefer greedy_mesh_slice_into to avoid allocation
    pub fn greedy_mesh_slice(data: &[u32; CHUNK_SIZE]) -> Vec<Quad> {
        let mut quads = Vec::with_capacity(64);
        Self::greedy_mesh_slice_into(data, &mut quads);
        quads
    }

    /// Greedy mesh a binary slice into a provided Vec (zero-allocation)
    /// Reuses the provided Vec to avoid per-slice allocations
    pub fn greedy_mesh_slice_into(data: &[u32; CHUNK_SIZE], quads: &mut Vec<Quad>) {
        count_call!(FUNCTION_COUNTERS.greedy_mesh_slice_calls);

        let mut data = *data; // Copy for mutation

        for row in 0..CHUNK_SIZE {
            // Early skip for empty rows - reduces wasted iterations
            if data[row] == 0 {
                continue;
            }

            let mut col = 0;

            while col < CHUNK_SIZE_U32 {
                // Find first set bit (solid voxel) using fast intrinsic
                let trailing_zeros = (data[row] >> col).trailing_zeros();
                col += trailing_zeros;

                if col >= CHUNK_SIZE_U32 {
                    break;
                }

                // Find run of consecutive set bits using fast intrinsic
                let height = (data[row] >> col).trailing_ones();

                // Create mask for this run - branch-free for common case
                let height_mask = if height >= 32 {
                    !0u32
                } else {
                    (1u32 << height).wrapping_sub(1)
                };
                let mask = height_mask << col;

                // Expand horizontally (greedy) with manual unrolling for ILP
                let mut width = 1u32;
                let max_width = (CHUNK_SIZE - row) as u32;

                // Unroll by 4 to improve instruction-level parallelism
                while width + 3 < max_width as u32 {
                    let idx1 = row + width as usize;
                    let idx2 = row + (width + 1) as usize;
                    let idx3 = row + (width + 2) as usize;
                    let idx4 = row + (width + 3) as usize;

                    let row1 = data[idx1];
                    let row2 = data[idx2];
                    let row3 = data[idx3];
                    let row4 = data[idx4];

                    let section1 = (row1 >> col) & height_mask;
                    let section2 = (row2 >> col) & height_mask;
                    let section3 = (row3 >> col) & height_mask;
                    let section4 = (row4 >> col) & height_mask;

                    // Check all 4 at once - better for branch prediction
                    let all_match = (section1 == height_mask) as u32
                        | ((section2 == height_mask) as u32) << 1
                        | ((section3 == height_mask) as u32) << 2
                        | ((section4 == height_mask) as u32) << 3;

                    match all_match {
                        0b1111 => {
                            // All 4 matched
                            data[idx1] &= !mask;
                            data[idx2] &= !mask;
                            data[idx3] &= !mask;
                            data[idx4] &= !mask;
                            width += 4;
                        }
                        0b0111 => {
                            // First 3 matched
                            data[idx1] &= !mask;
                            data[idx2] &= !mask;
                            data[idx3] &= !mask;
                            width += 3;
                            break;
                        }
                        0b0011 => {
                            // First 2 matched
                            data[idx1] &= !mask;
                            data[idx2] &= !mask;
                            width += 2;
                            break;
                        }
                        0b0001 => {
                            // Only first matched
                            data[idx1] &= !mask;
                            width += 1;
                            break;
                        }
                        _ => {
                            // None matched
                            break;
                        }
                    }
                }

                // Handle remaining rows (< 4 left)
                while width < max_width as u32 {
                    let next_row = data[row + width as usize];
                    let next_row_section = (next_row >> col) & height_mask;

                    if next_row_section != height_mask {
                        break;
                    }

                    data[row + width as usize] &= !mask;
                    width += 1;
                }

                // Create quad (all values guaranteed to fit in u8 due to CHUNK_SIZE = 32)
                quads.push(Quad {
                    x: row as u8,
                    y: col as u8,
                    width: width as u8,
                    height: height as u8,
                });

                // Clear bits we consumed in the current row
                data[row] &= !mask;

                col += height;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_greedy_mesh_empty() {
        let data = [0u32; CHUNK_SIZE];
        let quads = BinaryGreedyMesher::greedy_mesh_slice(&data);
        assert_eq!(quads.len(), 0);
    }

    #[test]
    fn test_greedy_mesh_single() {
        let mut data = [0u32; CHUNK_SIZE];
        data[0] = 1; // Single voxel at (0, 0)

        let quads = BinaryGreedyMesher::greedy_mesh_slice(&data);
        assert_eq!(quads.len(), 1);
        assert_eq!(quads[0].width, 1);
        assert_eq!(quads[0].height, 1);
    }

    #[test]
    fn test_greedy_mesh_vertical_line() {
        let mut data = [0u32; CHUNK_SIZE];
        data[0] = 0b1111; // 4 consecutive voxels

        let quads = BinaryGreedyMesher::greedy_mesh_slice(&data);
        assert_eq!(quads.len(), 1);
        assert_eq!(quads[0].width, 1);
        assert_eq!(quads[0].height, 4);
    }

    #[test]
    fn test_greedy_mesh_rectangle() {
        let mut data = [0u32; CHUNK_SIZE];
        // 3x4 rectangle
        for i in 0..3 {
            data[i] = 0b1111;
        }

        let quads = BinaryGreedyMesher::greedy_mesh_slice(&data);
        assert_eq!(quads.len(), 1);
        assert_eq!(quads[0].width, 3);
        assert_eq!(quads[0].height, 4);
    }
}
