pub mod camera;
pub mod meshing;
pub mod perf;
pub mod rendering;
/// Voxel Engine - Optimization-first software rasterization engine
/// Built with compartmentalized benchmarkable components
pub mod voxel;
pub mod world;

pub use camera::{Camera, CameraController, Frustum};
pub use meshing::{BinaryGreedyMesher, ChunkMesh};
pub use perf::{CounterSnapshot, FunctionCounters, FUNCTION_COUNTERS};
pub use rendering::{Framebuffer, Rasterizer, ShadingConfig, OcclusionBuffer};
pub use voxel::{BlockData, BlockType, Chunk, CHUNK_SIZE, CHUNK_VOLUME};
pub use world::{World, WorldConfig};
