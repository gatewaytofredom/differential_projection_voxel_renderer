use glam::{IVec3, Vec3};
use std::collections::HashMap;
/// Integration tests for meshing correctness
/// These tests validate that the binary greedy mesher generates correct geometry
use voxel_engine::*;
use voxel_engine::meshing::{ChunkMesh, FaceDir, TinyQuad};

fn slice_pos_for(face_dir: FaceDir, slice_idx: usize) -> u8 {
    if face_dir.is_positive() {
        (slice_idx + 1) as u8
    } else {
        slice_idx as u8
    }
}

fn face_quads<'a>(
    mesh: &'a ChunkMesh,
    face_dir: FaceDir,
) -> impl Iterator<Item = (u8, &'a TinyQuad)> + 'a {
    mesh.face_list(face_dir)
        .slice_quads
        .iter()
        .enumerate()
        .flat_map(move |(slice_idx, quads)| {
            let slice_pos = slice_pos_for(face_dir, slice_idx);
            quads.iter().map(move |quad| (slice_pos, quad))
        })
}

fn quad_vertices(
    mesh: &ChunkMesh,
    face_dir: FaceDir,
    slice_pos: u8,
    quad: &TinyQuad,
) -> [(Vec3, u8); 4] {
    ChunkMesh::tiny_quad_to_vertices(quad, face_dir, slice_pos, mesh.world_offset())
}

fn all_face_quads<'a>(
    mesh: &'a ChunkMesh,
) -> impl Iterator<Item = (FaceDir, u8, &'a TinyQuad)> + 'a {
    [
        FaceDir::PosX,
        FaceDir::NegX,
        FaceDir::PosY,
        FaceDir::NegY,
        FaceDir::PosZ,
        FaceDir::NegZ,
    ]
    .into_iter()
    .flat_map(move |dir| face_quads(mesh, dir).map(move |(slice_pos, quad)| (dir, slice_pos, quad)))
}

#[test]
fn test_single_voxel_generates_six_faces() {
    // A single solid voxel surrounded by air should generate 6 faces (one per side)
    let mut chunk = Chunk::uniform(IVec3::ZERO, BlockType::Air);

    // Set a single voxel at the center
    chunk.set_block(16, 16, 16, BlockData::new(BlockType::Stone));

    let chunks = vec![chunk];
    let chunk_refs: Vec<&Chunk> = chunks.iter().collect();

    let mesh = BinaryGreedyMesher::mesh_chunk_in_world(chunks.first().unwrap(), &chunk_refs)
        .expect("Single voxel should generate a mesh");

    assert_eq!(mesh.quad_count(), 6, "Single voxel should emit 6 quads");

    for dir in [
        FaceDir::PosX,
        FaceDir::NegX,
        FaceDir::PosY,
        FaceDir::NegY,
        FaceDir::PosZ,
        FaceDir::NegZ,
    ] {
        let mut quads = face_quads(&mesh, dir);
        let (_slice_pos, quad) = quads.next().expect("Each face should have one quad");
        assert!(quads.next().is_none(), "Should only have one quad per face");
        assert_eq!(quad.width(), 1);
        assert_eq!(quad.height(), 1);
        assert_eq!(quad.block_type(), BlockType::Stone as u8);
    }
}

#[test]
fn test_face_positions_are_correct() {
    // Create a chunk with a single voxel at origin to test face positioning
    let mut chunk = Chunk::uniform(IVec3::ZERO, BlockType::Air);
    chunk.set_block(0, 0, 0, BlockData::new(BlockType::Stone));

    let chunks = vec![chunk];
    let chunk_refs: Vec<&Chunk> = chunks.iter().collect();

    let mesh = BinaryGreedyMesher::mesh_chunk_in_world(chunks.first().unwrap(), &chunk_refs)
        .expect("Should generate mesh");

    for dir in [
        FaceDir::PosX,
        FaceDir::NegX,
        FaceDir::PosY,
        FaceDir::NegY,
        FaceDir::PosZ,
        FaceDir::NegZ,
    ] {
        let (slice_pos, quad) =
            face_quads(&mesh, dir).next().expect("Should find one quad per face");

        let verts = quad_vertices(&mesh, dir, slice_pos, quad);
        let (axis, expected) = match dir {
            FaceDir::PosX => ('x', 1.0),
            FaceDir::NegX => ('x', 0.0),
            FaceDir::PosY => ('y', 1.0),
            FaceDir::NegY => ('y', 0.0),
            FaceDir::PosZ => ('z', 1.0),
            FaceDir::NegZ => ('z', 0.0),
        };

        for (pos, _) in verts.iter() {
            let value = match axis {
                'x' => pos.x,
                'y' => pos.y,
                'z' => pos.z,
                _ => unreachable!(),
            };
            assert!(
                (value - expected).abs() < 0.01,
                "{:?} face should lie on {}={}, got {}",
                dir,
                axis,
                expected,
                value
            );
        }
    }
}

#[test]
fn test_top_face_of_surface_voxel() {
    // Test that a voxel with air above it generates a top face at the correct position
    let mut chunk = Chunk::uniform(IVec3::ZERO, BlockType::Air);
    chunk.set_block(5, 10, 5, BlockData::new(BlockType::Grass));
    // Block above is air (default)

    let chunks = vec![chunk];
    let chunk_refs: Vec<&Chunk> = chunks.iter().collect();

    let mesh = BinaryGreedyMesher::mesh_chunk_in_world(chunks.first().unwrap(), &chunk_refs)
        .expect("Should generate mesh");

    let mut quads = face_quads(&mesh, FaceDir::PosY);
    let (slice_pos, quad) = quads.next().expect("Should have a top face");
    assert!(quads.next().is_none(), "Top face should be a single quad");

    let verts = quad_vertices(&mesh, FaceDir::PosY, slice_pos, quad);
    for (pos, _) in verts.iter() {
        assert!(
            (pos.y - 11.0).abs() < 0.01,
            "Top face should be at y=11.0, got y={}",
            pos.y
        );
    }
}

#[test]
fn test_bottom_face_of_floating_voxel() {
    // Test that a voxel with air below it generates a bottom face at the correct position
    let mut chunk = Chunk::uniform(IVec3::ZERO, BlockType::Air);
    chunk.set_block(5, 10, 5, BlockData::new(BlockType::Stone));

    let chunks = vec![chunk];
    let chunk_refs: Vec<&Chunk> = chunks.iter().collect();

    let mesh = BinaryGreedyMesher::mesh_chunk_in_world(chunks.first().unwrap(), &chunk_refs)
        .expect("Should generate mesh");

    let mut quads = face_quads(&mesh, FaceDir::NegY);
    let (slice_pos, quad) = quads.next().expect("Should have a bottom face");
    assert!(quads.next().is_none(), "Bottom face should be a single quad");

    let verts = quad_vertices(&mesh, FaceDir::NegY, slice_pos, quad);
    for (pos, _) in verts.iter() {
        assert!(
            (pos.y - 10.0).abs() < 0.01,
            "Bottom face should be at y=10.0, got y={}",
            pos.y
        );
    }
}

#[test]
fn test_internal_faces_are_culled() {
    // Two adjacent solid voxels should NOT have faces between them
    let mut chunk = Chunk::uniform(IVec3::ZERO, BlockType::Air);
    chunk.set_block(10, 10, 10, BlockData::new(BlockType::Stone));
    chunk.set_block(11, 10, 10, BlockData::new(BlockType::Stone)); // Adjacent in X

    let chunks = vec![chunk];
    let chunk_refs: Vec<&Chunk> = chunks.iter().collect();

    let mesh = BinaryGreedyMesher::mesh_chunk_in_world(chunks.first().unwrap(), &chunk_refs)
        .expect("Should generate mesh");

    assert_eq!(mesh.quad_count(), 6, "Merged prism should have 6 quads");

    for dir in [FaceDir::PosX, FaceDir::NegX] {
        for (slice_pos, quad) in face_quads(&mesh, dir) {
            let verts = quad_vertices(&mesh, dir, slice_pos, quad);
            for (pos, _) in verts.iter() {
                assert!(
                    (pos.x - 11.0).abs() > 0.01,
                    "Internal face at x=11 should be culled (found on {:?})",
                    dir
                );
            }
        }
    }
}

#[test]
fn test_world_offset_applied_correctly() {
    // Test that chunk position affects vertex world coordinates
    let chunk_pos = IVec3::new(1, 2, 3);
    let mut chunk = Chunk::uniform(chunk_pos, BlockType::Air);
    chunk.set_block(0, 0, 0, BlockData::new(BlockType::Stone));

    let chunks = vec![chunk];
    let chunk_refs: Vec<&Chunk> = chunks.iter().collect();

    let mesh = BinaryGreedyMesher::mesh_chunk_in_world(chunks.first().unwrap(), &chunk_refs)
        .expect("Should generate mesh");

    for (dir, slice_pos, quad) in all_face_quads(&mesh) {
        let verts = quad_vertices(&mesh, dir, slice_pos, quad);
        for (pos, _) in verts.iter() {
            assert!(
                pos.x >= 32.0 && pos.x <= 33.0,
                "X should be in range [32,33], got {}",
                pos.x
            );
            assert!(
                pos.y >= 64.0 && pos.y <= 65.0,
                "Y should be in range [64,65], got {}",
                pos.y
            );
            assert!(
                pos.z >= 96.0 && pos.z <= 97.0,
                "Z should be in range [96,97], got {}",
                pos.z
            );
        }
    }
}

#[test]
fn test_greedy_meshing_merges_quads() {
    // A 2x2 plane of voxels should merge into a single quad
    let mut chunk = Chunk::uniform(IVec3::ZERO, BlockType::Air);

    // Create a 2x2 flat surface
    for x in 0..2 {
        for z in 0..2 {
            chunk.set_block(x, 0, z, BlockData::new(BlockType::Stone));
        }
    }

    let chunks = vec![chunk];
    let chunk_refs: Vec<&Chunk> = chunks.iter().collect();

    let mesh = BinaryGreedyMesher::mesh_chunk_in_world(chunks.first().unwrap(), &chunk_refs)
        .expect("Should generate mesh");

    let mut quads = face_quads(&mesh, FaceDir::PosY);
    let (_slice_pos, quad) = quads
        .next()
        .expect("2x2 surface should merge into a single top quad");
    assert!(quads.next().is_none(), "Top surface should be one quad");
    assert_eq!(quad.width(), 2);
    assert_eq!(quad.height(), 2);
}

#[test]
fn test_uniform_air_chunk_generates_no_mesh() {
    let chunk = Chunk::uniform(IVec3::ZERO, BlockType::Air);
    let chunks = vec![chunk];
    let chunk_refs: Vec<&Chunk> = chunks.iter().collect();
    let mesh = BinaryGreedyMesher::mesh_chunk_in_world(chunks.first().unwrap(), &chunk_refs);

    assert!(mesh.is_none(), "All-air chunk should return None");
}

#[test]
fn test_uniform_solid_chunk_generates_no_mesh() {
    // Uniform chunks (all same block type) are optimized out
    // This is correct behavior - no internal or external faces to render
    // since we don't have neighbor chunk data
    let chunk = Chunk::uniform(IVec3::ZERO, BlockType::Stone);
    let chunks = vec![chunk];
    let chunk_refs: Vec<&Chunk> = chunks.iter().collect();
    let mesh = BinaryGreedyMesher::mesh_chunk_in_world(chunks.first().unwrap(), &chunk_refs);

    // Uniform chunks return None as an optimization
    assert!(
        mesh.is_none(),
        "Uniform solid chunk should return None (optimization)"
    );
}

#[test]
fn test_vertex_normals_match_face_direction() {
    let mut chunk = Chunk::uniform(IVec3::ZERO, BlockType::Air);
    chunk.set_block(10, 10, 10, BlockData::new(BlockType::Stone));

    let chunks = vec![chunk];
    let chunk_refs: Vec<&Chunk> = chunks.iter().collect();

    let mesh = BinaryGreedyMesher::mesh_chunk_in_world(chunks.first().unwrap(), &chunk_refs)
        .expect("Should generate mesh");

    for dir in [
        FaceDir::PosX,
        FaceDir::NegX,
        FaceDir::PosY,
        FaceDir::NegY,
        FaceDir::PosZ,
        FaceDir::NegZ,
    ] {
        let mut quads = face_quads(&mesh, dir);
        let (slice_pos, quad) = quads
            .next()
            .expect("Each face should have a single quad for isolated voxel");
        assert!(quads.next().is_none(), "One quad per face expected");

        let verts = quad_vertices(&mesh, dir, slice_pos, quad);
        let tri_normal = (verts[1].0 - verts[0].0)
            .cross(verts[2].0 - verts[0].0)
            .normalize();
        let expected = dir.normal();
        let dot = tri_normal.dot(expected);
        assert!(
            dot > 0.9,
            "Triangle normal {:?} should align with {:?} for {:?} (dot={})",
            tri_normal,
            expected,
            dir,
            dot
        );
    }
}

#[test]
fn test_color_matches_block_type() {
    let mut chunk = Chunk::uniform(IVec3::ZERO, BlockType::Air);
    chunk.set_block(0, 0, 0, BlockData::new(BlockType::Grass));

    let chunks = vec![chunk];
    let chunk_refs: Vec<&Chunk> = chunks.iter().collect();

    let mesh = BinaryGreedyMesher::mesh_chunk_in_world(chunks.first().unwrap(), &chunk_refs)
        .expect("Should generate mesh");

    let grass_block_type = BlockType::Grass as u8;

    for (_dir, _slice, quad) in all_face_quads(&mesh) {
        assert_eq!(
            quad.block_type(),
            grass_block_type,
            "Quad block_type should match Grass"
        );
    }
}

#[test]
fn test_stacked_voxels_face_positions() {
    // Test a vertical stack of 3 voxels to ensure faces are at correct Y positions
    let mut chunk = Chunk::uniform(IVec3::ZERO, BlockType::Air);
    chunk.set_block(5, 5, 5, BlockData::new(BlockType::Stone)); // Bottom
    chunk.set_block(5, 6, 5, BlockData::new(BlockType::Stone)); // Middle
    chunk.set_block(5, 7, 5, BlockData::new(BlockType::Stone)); // Top

    let chunks = vec![chunk];
    let chunk_refs: Vec<&Chunk> = chunks.iter().collect();

    let mesh = BinaryGreedyMesher::mesh_chunk_in_world(chunks.first().unwrap(), &chunk_refs)
        .expect("Should generate mesh");

    let mut neg_y_quads = face_quads(&mesh, FaceDir::NegY);
    let (neg_slice, neg_quad) = neg_y_quads
        .next()
        .expect("Stack should have a bottom face");
    assert!(neg_y_quads.next().is_none(), "Only one bottom quad expected");
    let neg_verts = quad_vertices(&mesh, FaceDir::NegY, neg_slice, neg_quad);
    for (pos, _) in neg_verts.iter() {
        assert!(
            (pos.y - 5.0).abs() < 0.01,
            "Bottom face should be at y=5.0, got y={}",
            pos.y
        );
    }

    let mut pos_y_quads = face_quads(&mesh, FaceDir::PosY);
    let (pos_slice, pos_quad) = pos_y_quads
        .next()
        .expect("Stack should have a top face");
    assert!(pos_y_quads.next().is_none(), "Only one top quad expected");
    let pos_verts = quad_vertices(&mesh, FaceDir::PosY, pos_slice, pos_quad);
    for (pos, _) in pos_verts.iter() {
        assert!(
            (pos.y - 8.0).abs() < 0.01,
            "Top face should be at y=8.0, got y={}",
            pos.y
        );
    }
}

#[test]
fn test_greedy_meshing_respects_block_types() {
    // Stack three voxels with different block types and ensure side faces
    // are not merged across types (each type should produce its own quad)
    let mut chunk = Chunk::uniform(IVec3::ZERO, BlockType::Air);
    chunk.set_block(0, 0, 0, BlockData::new(BlockType::Grass));
    chunk.set_block(0, 1, 0, BlockData::new(BlockType::Dirt));
    chunk.set_block(0, 2, 0, BlockData::new(BlockType::Stone));

    let chunks = vec![chunk];
    let chunk_refs: Vec<&Chunk> = chunks.iter().collect();

    let mesh = BinaryGreedyMesher::mesh_chunk_in_world(chunks.first().unwrap(), &chunk_refs)
        .expect("Should generate mesh");

    assert_eq!(
        mesh.face_list(FaceDir::NegX).quad_count(),
        3,
        "Each block type should keep its own -X quad"
    );

    let mut neg_x_counts: HashMap<u8, usize> = HashMap::new();
    for (_slice_pos, quad) in face_quads(&mesh, FaceDir::NegX) {
        *neg_x_counts.entry(quad.block_type()).or_insert(0) += 1;
    }

    let grass_type = BlockType::Grass as u8;
    let dirt_type = BlockType::Dirt as u8;
    let stone_type = BlockType::Stone as u8;

    // Expect one quad (4 vertices) per block type on the -X side
    assert_eq!(
        neg_x_counts.get(&grass_type),
        Some(&1usize),
        "Grass side face should have one quad"
    );
    assert_eq!(
        neg_x_counts.get(&dirt_type),
        Some(&1usize),
        "Dirt side face should have one quad"
    );
    assert_eq!(
        neg_x_counts.get(&stone_type),
        Some(&1usize),
        "Stone side face should have one quad"
    );

    // And no other unexpected colors
    assert_eq!(
        neg_x_counts.len(),
        3,
        "Should only have faces for Grass, Dirt, and Stone"
    );
}

#[test]
fn test_quad_winding_matches_face_normal() {
    // Verify that the generated triangles for each face direction
    // have their geometric normal aligned with FaceDir::normal().
    use voxel_engine::meshing::mesh::Quad as MeshQuad;

    let quad = MeshQuad {
        x: 0,
        y: 0,
        width: 1,
        height: 1,
    };

    for &face_dir in &[
        FaceDir::PosX,
        FaceDir::NegX,
        FaceDir::PosY,
        FaceDir::NegY,
        FaceDir::PosZ,
        FaceDir::NegZ,
    ] {
        let mut mesh = ChunkMesh::new();
        let axis_pos = if face_dir.is_positive() { 1 } else { 0 };
        mesh.add_quad(
            &quad,
            face_dir,
            axis_pos,
            IVec3::ZERO,
            BlockType::Stone,
            0,
            1.0,
        );

        let mut quads = face_quads(&mesh, face_dir);
        let (slice_pos, tiny_quad) = quads
            .next()
            .expect("add_quad should populate face list");
        assert!(quads.next().is_none(), "Should only store one quad");

        let verts = quad_vertices(&mesh, face_dir, slice_pos, tiny_quad);
        let tri_normal = (verts[1].0 - verts[0].0)
            .cross(verts[2].0 - verts[0].0)
            .normalize();
        let expected = face_dir.normal();

        let dot = tri_normal.dot(expected);
        assert!(
            dot > 0.9,
            "Triangle normal {:?} should align with face normal {:?} for {:?} (dot={})",
            tri_normal,
            expected,
            face_dir,
            dot
        );
    }
}

#[test]
fn test_internal_faces_between_adjacent_chunks_are_culled() {
    // Two voxels in neighbouring chunks sharing a face should not render
    // faces at the chunk boundary when neighbour information is available.
    let mut left = Chunk::uniform(IVec3::ZERO, BlockType::Air);
    let mut right = Chunk::uniform(IVec3::new(1, 0, 0), BlockType::Air);

    // Place one stone voxel at the positive X edge of the left chunk
    // and one at the negative X edge of the right chunk.
    left.set_block(CHUNK_SIZE - 1, 16, 16, BlockData::new(BlockType::Stone));
    right.set_block(0, 16, 16, BlockData::new(BlockType::Stone));

    let mut chunks = Vec::new();
    chunks.push(left);
    chunks.push(right);

    let chunk_refs: Vec<&Chunk> = chunks.iter().collect();

    let mesh_left = BinaryGreedyMesher::mesh_chunk_in_world(&chunks[0], &chunk_refs)
        .expect("Left chunk should generate a mesh");
    let mesh_right = BinaryGreedyMesher::mesh_chunk_in_world(&chunks[1], &chunk_refs)
        .expect("Right chunk should generate a mesh");

    assert_eq!(
        mesh_left.face_list(FaceDir::PosX).quad_count(),
        0,
        "Left chunk should not emit +X faces on shared boundary"
    );
    assert_eq!(
        mesh_right.face_list(FaceDir::NegX).quad_count(),
        0,
        "Right chunk should not emit -X faces on shared boundary"
    );
}

#[test]
fn test_stale_mesh_keeps_boundary_faces_when_neighbor_added() {
    // Scenario:
    // 1. Mesh a chunk while its neighbour does not exist yet → boundary faces are emitted.
    // 2. Later, add the neighbour and mesh only that new chunk.
    // If we keep using the old mesh for the original chunk without re-meshing it,
    // those boundary faces become "ghost" internal faces between solid blocks.

    // Left and right chunks sharing a boundary along +X / -X.
    let mut left = Chunk::uniform(IVec3::ZERO, BlockType::Air);
    let mut right = Chunk::uniform(IVec3::new(1, 0, 0), BlockType::Air);

    // Place two different solid blocks touching at the chunk boundary.
    left.set_block(
        CHUNK_SIZE - 1,
        16,
        16,
        BlockData::new(BlockType::Grass),
    );
    right.set_block(0, 16, 16, BlockData::new(BlockType::Dirt));

    // Step 1: mesh the left chunk without its neighbour present.
    let world_left_only: Vec<&Chunk> = vec![&left];
    let left_mesh_stale = BinaryGreedyMesher::mesh_chunk_in_world(&left, &world_left_only)
        .expect("Left chunk alone should generate a mesh");

    let stale_boundary_quads = left_mesh_stale.face_list(FaceDir::PosX).quad_count();
    assert!(
        stale_boundary_quads > 0,
        "Without neighbour chunk, left boundary should render +X faces"
    );

    // Step 2: mesh both chunks with neighbour information available.
    let world_with_neighbors: Vec<&Chunk> = vec![&left, &right];
    let left_mesh_fresh =
        BinaryGreedyMesher::mesh_chunk_in_world(&left, &world_with_neighbors)
            .expect("Left chunk with neighbour should generate a mesh");
    let right_mesh_fresh =
        BinaryGreedyMesher::mesh_chunk_in_world(&right, &world_with_neighbors)
            .expect("Right chunk with neighbour should generate a mesh");

    assert_eq!(
        left_mesh_fresh.face_list(FaceDir::PosX).quad_count(),
        0,
        "With neighbour present, left chunk should not emit +X faces on shared boundary"
    );
    assert_eq!(
        right_mesh_fresh.face_list(FaceDir::NegX).quad_count(),
        0,
        "With neighbour present, right chunk should not emit -X faces on shared boundary"
    );

    // Using the stale mesh alongside the fresh neighbour would therefore render
    // extra internal faces that should no longer exist.
    assert!(
        stale_boundary_quads
            > left_mesh_fresh
                .face_list(FaceDir::PosX)
                .quad_count(),
        "Stale mesh retains boundary faces that a fresh mesh correctly culls once the neighbour exists"
    );
}
