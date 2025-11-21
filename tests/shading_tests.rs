use glam::Vec3;
/// Tests for the shading configuration and helpers.
use voxel_engine::{self as ve, BlockType, ShadingConfig};
use ve::meshing::{Vertex, FaceDir};
use ve::meshing::mesh::F32x3;

#[test]
fn test_shading_brighter_when_facing_light() {
    let cfg = ShadingConfig::default();

    // Create vertices with different facing directions
    // PosY (up) is more lit than NegY (down) with default lighting
    let v_lit = Vertex::from_local_coords(
        0, 0, 0,
        BlockType::Stone,
        FaceDir::PosY as u8,  // Facing up towards light
        0,
        1.0
    );
    let v_dark = Vertex::from_local_coords(
        0, 0, 0,
        BlockType::Stone,
        FaceDir::NegY as u8,  // Facing down away from light
        0,
        1.0
    );

    let lit = cfg.vertex_light(&v_lit);
    let dark = cfg.vertex_light(&v_dark);

    assert!(
        lit > dark,
        "Vertex facing the light should be brighter ({} > {})",
        lit,
        dark
    );
}
