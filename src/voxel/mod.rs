/// Core voxel data structures optimized for cache locality and performance
pub mod block_type;
pub mod chunk;

pub use block_type::BlockType;
pub use chunk::{Chunk, CHUNK_SIZE, CHUNK_VOLUME};

/// Compact block data - single byte for now, can expand later
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(transparent)]
pub struct BlockData {
    pub block_type: BlockType,
}

impl BlockData {
    #[inline]
    pub const fn new(block_type: BlockType) -> Self {
        Self { block_type }
    }

    #[inline]
    pub const fn is_solid(&self) -> bool {
        self.block_type.is_solid()
    }

    #[inline]
    pub const fn air() -> Self {
        Self {
            block_type: BlockType::Air,
        }
    }
}
