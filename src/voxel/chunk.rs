/// Chunk data structure optimized for cache locality
/// Uses enum to handle uniform chunks efficiently (common case)
use super::{BlockData, BlockType};
use glam::IVec3;
use noise::{NoiseFn, Perlin};

pub const CHUNK_SIZE: usize = 32;
pub const CHUNK_SIZE_I32: i32 = CHUNK_SIZE as i32;
pub const CHUNK_VOLUME: usize = CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE;

/// Chunk storage optimized for common cases
/// Uniform chunks (all air/all solid) are stored as single value
/// This significantly reduces memory for sky and underground chunks
pub enum ChunkData {
    /// All voxels are the same type - only stores one value
    Uniform(BlockType),
    /// Heterogeneous voxels - stores full array
    /// Using Box to keep Chunk size small on stack
    Varied(Box<[BlockData; CHUNK_VOLUME]>),
}

pub struct Chunk {
    pub position: IVec3,
    pub data: ChunkData,
}

impl Chunk {
    /// Create a new chunk with all voxels set to the same type
    pub fn uniform(position: IVec3, block_type: BlockType) -> Self {
        Self {
            position,
            data: ChunkData::Uniform(block_type),
        }
    }

    /// Create a chunk with varied voxel data
    pub fn varied(position: IVec3, blocks: Box<[BlockData; CHUNK_VOLUME]>) -> Self {
        Self {
            position,
            data: ChunkData::Varied(blocks),
        }
    }

    /// Get block at local coordinates (0..CHUNK_SIZE)
    #[inline]
    pub fn get_block(&self, x: usize, y: usize, z: usize) -> BlockData {
        debug_assert!(x < CHUNK_SIZE && y < CHUNK_SIZE && z < CHUNK_SIZE);

        match &self.data {
            ChunkData::Uniform(block_type) => BlockData::new(*block_type),
            ChunkData::Varied(blocks) => {
                let index = (z * CHUNK_SIZE * CHUNK_SIZE) + (y * CHUNK_SIZE) + x;
                blocks[index]
            }
        }
    }

    /// Get block at local coordinates using linear index
    #[inline]
    pub fn get_block_index(&self, index: usize) -> BlockData {
        debug_assert!(index < CHUNK_VOLUME);

        match &self.data {
            ChunkData::Uniform(block_type) => BlockData::new(*block_type),
            ChunkData::Varied(blocks) => blocks[index],
        }
    }

    /// Check if all voxels are the same (for early exit optimizations)
    #[inline]
    pub fn is_uniform(&self) -> bool {
        matches!(self.data, ChunkData::Uniform(_))
    }

    /// Get direct access to varied block data for hot-path meshing
    /// Returns None if chunk is uniform (caller should have already filtered these out)
    /// This eliminates the enum match branch from tight loops
    #[inline]
    pub fn get_varied_blocks(&self) -> Option<&[BlockData; CHUNK_VOLUME]> {
        match &self.data {
            ChunkData::Varied(blocks) => Some(blocks),
            ChunkData::Uniform(_) => None,
        }
    }

    /// Returns the uniform block type if this chunk stores a single value
    #[inline]
    pub fn uniform_block_type(&self) -> Option<BlockType> {
        match &self.data {
            ChunkData::Uniform(block_type) => Some(*block_type),
            ChunkData::Varied(_) => None,
        }
    }

    /// Set block at local coordinates (0..CHUNK_SIZE)
    /// Converts uniform chunks to varied if necessary
    pub fn set_block(&mut self, x: usize, y: usize, z: usize, block: BlockData) {
        debug_assert!(x < CHUNK_SIZE && y < CHUNK_SIZE && z < CHUNK_SIZE);

        // If uniform, convert to varied first
        if let ChunkData::Uniform(uniform_type) = self.data {
            let blocks = Box::new([BlockData::new(uniform_type); CHUNK_VOLUME]);
            self.data = ChunkData::Varied(blocks);
        }

        // Now set the block
        if let ChunkData::Varied(ref mut blocks) = self.data {
            let index = (z * CHUNK_SIZE * CHUNK_SIZE) + (y * CHUNK_SIZE) + x;
            blocks[index] = block;
        }
    }

    /// Generate terrain using Perlin noise
    pub fn generate_terrain(position: IVec3) -> Self {
        let world_offset = position * CHUNK_SIZE_I32;

        // Reuse single Perlin instance across all height calculations
        let perlin = Perlin::new(12345);

        // Quick check for chunks that are definitely all air or all solid
        let (min_height, max_height) = Self::get_height_range(&perlin, world_offset);

        let chunk_min_y = world_offset.y;
        let chunk_max_y = world_offset.y + CHUNK_SIZE_I32;

        // All air above terrain
        if chunk_min_y > max_height {
            return Self::uniform(position, BlockType::Air);
        }

        // All solid below terrain (with some margin)
        if chunk_max_y < min_height - 10 {
            return Self::uniform(position, BlockType::Stone);
        }

        // Mixed chunk - generate full data
        let mut blocks = Box::new([BlockData::air(); CHUNK_VOLUME]);

        let stride_y = CHUNK_SIZE;
        let stride_z = CHUNK_SIZE * CHUNK_SIZE;

        for z in 0..CHUNK_SIZE {
            let world_z = world_offset.z + z as i32;
            let z_base = z * stride_z;
            for y in 0..CHUNK_SIZE {
                let world_y = world_offset.y + y as i32;
                let yz_base = z_base + y * stride_y;
                for x in 0..CHUNK_SIZE {
                    let world_x = world_offset.x + x as i32;

                    let height = Self::sample_terrain_height(&perlin, world_x, world_z);

                    let block_type = if world_y > height {
                        BlockType::Air
                    } else if world_y == height {
                        BlockType::Grass
                    } else if world_y > height - 3 {
                        BlockType::Dirt
                    } else {
                        BlockType::Stone
                    };

                    let index = yz_base + x;
                    blocks[index] = BlockData::new(block_type);
                }
            }
        }

        Self::varied(position, blocks)
    }

    #[inline]
    fn sample_terrain_height(perlin: &Perlin, x: i32, z: i32) -> i32 {
        let scale = 0.01;
        let noise_value = perlin.get([x as f64 * scale, z as f64 * scale]);
        (noise_value * 20.0) as i32
    }

    /// Create a test chunk that is fully solid (for testing visibility)
    pub fn generate_test_solid(position: IVec3) -> Self {
        let mut blocks = Box::new([BlockData::air(); CHUNK_VOLUME]);

        // Fill entire chunk with stone
        for i in 0..CHUNK_VOLUME {
            blocks[i] = BlockData::new(BlockType::Stone);
        }

        Self::varied(position, blocks)
    }

    /// Get min and max terrain height in a single pass - more cache efficient
    #[inline]
    fn get_height_range(perlin: &Perlin, world_offset: IVec3) -> (i32, i32) {
        let mut min = i32::MAX;
        let mut max = i32::MIN;

        for z in 0..CHUNK_SIZE {
            for x in 0..CHUNK_SIZE {
                let world_x = world_offset.x + x as i32;
                let world_z = world_offset.z + z as i32;
                let h = Self::sample_terrain_height(perlin, world_x, world_z);
                min = min.min(h);
                max = max.max(h);
            }
        }
        (min, max)
    }
}

/// Convert 3D coordinates to linear index
#[inline]
pub const fn coords_to_index(x: usize, y: usize, z: usize) -> usize {
    (z * CHUNK_SIZE * CHUNK_SIZE) + (y * CHUNK_SIZE) + x
}

/// Convert linear index to 3D coordinates
#[inline]
pub const fn index_to_coords(index: usize) -> (usize, usize, usize) {
    let z = index / (CHUNK_SIZE * CHUNK_SIZE);
    let remainder = index % (CHUNK_SIZE * CHUNK_SIZE);
    let y = remainder / CHUNK_SIZE;
    let x = remainder % CHUNK_SIZE;
    (x, y, z)
}
