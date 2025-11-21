/// Block type enumeration
/// Using u8 representation for memory efficiency

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum BlockType {
    Air = 0,
    Grass = 1,
    Dirt = 2,
    Stone = 3,
}

pub const BLOCK_TYPE_COUNT: usize = 4;

// Lookup tables for block properties - eliminates branches in hot paths
const BLOCK_IS_SOLID_LUT: [bool; BLOCK_TYPE_COUNT] = [
    false, // Air
    true,  // Grass
    true,  // Dirt
    true,  // Stone
];

const BLOCK_COLORS_LUT: [[u8; 3]; BLOCK_TYPE_COUNT] = [
    [0, 0, 0],         // Air
    [34, 139, 34],     // Grass
    [139, 69, 19],     // Dirt
    [128, 128, 128],   // Stone
];

impl BlockType {
    pub const ALL: [BlockType; BLOCK_TYPE_COUNT] = [
        BlockType::Air,
        BlockType::Grass,
        BlockType::Dirt,
        BlockType::Stone,
    ];

    /// Fast lookup-table based solid check - no branches
    #[inline]
    pub const fn is_solid(self) -> bool {
        BLOCK_IS_SOLID_LUT[self as usize]
    }

    #[inline]
    pub const fn is_air(self) -> bool {
        matches!(self, BlockType::Air)
    }

    /// Fast lookup-table based color retrieval - no branches
    #[inline]
    pub const fn color(self) -> [u8; 3] {
        BLOCK_COLORS_LUT[self as usize]
    }

    /// Index into the texture atlas for this block type.
    /// Kept as a simple mapping to avoid branches in hot paths.
    #[inline]
    pub const fn texture_id(self) -> usize {
        match self {
            BlockType::Air => 0,
            BlockType::Grass => 1,
            BlockType::Dirt => 2,
            BlockType::Stone => 3,
        }
    }

    /// Convert from u8 to BlockType
    /// Returns Air for out-of-bounds values
    #[inline]
    pub const fn from_u8(value: u8) -> Self {
        match value {
            0 => BlockType::Air,
            1 => BlockType::Grass,
            2 => BlockType::Dirt,
            3 => BlockType::Stone,
            _ => BlockType::Air, // Default to Air for invalid values
        }
    }
}

impl Default for BlockType {
    fn default() -> Self {
        BlockType::Air
    }
}
