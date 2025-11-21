/// Meshing algorithms for converting voxel data to renderable geometry
/// Uses binary greedy meshing for optimal performance
pub mod binary_greedy;
pub mod face_packets;
pub mod mesh;

pub use binary_greedy::BinaryGreedyMesher;
pub use mesh::{ChunkMesh, FaceDir, FaceList, TinyQuad, Vertex};
