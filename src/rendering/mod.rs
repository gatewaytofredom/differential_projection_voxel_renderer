pub mod framebuffer;
/// Software rasterization pipeline
/// Optimized for performance with minimal branching
pub mod rasterizer;
pub mod shading;
pub mod texture;
pub mod occlusion;
pub mod simd_vertex;
pub mod culling;

// Hyper-Pipeline modules
pub mod differential_projection;
pub mod packet_pipeline;
pub mod span_walker;
pub mod macrotile;
pub mod hiz_buffer;
pub mod macrotile_renderer;

pub use framebuffer::Framebuffer;
pub use rasterizer::Rasterizer;
pub use shading::ShadingConfig;
pub use occlusion::OcclusionBuffer;
pub use culling::VisibleMesh;
pub use differential_projection::{FaceBasis, ProjectedPacket};
pub use packet_pipeline::PacketPipeline;
pub use span_walker::{SpanWalkerRasterizer, TrapezoidBatch};
pub use macrotile::{MacroTile, MacroTileBins, ThreadLocalBins, MeshId, MACROTILE_SIZE};
pub use hiz_buffer::{HiZBuffer, HIZ_BLOCK_SIZE};
pub use macrotile_renderer::{render_frame_macrotile, MacrotileRenderConfig};
