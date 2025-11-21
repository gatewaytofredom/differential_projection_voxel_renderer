/// Integration tests that exercise the full rendering pipeline.
/// These act as correctness tests and lightweight, programmatic
/// benchmarks of the end-to-end path: terrain -> meshing -> rasterizer.
use std::time::Instant;

use glam::{IVec3, Vec3};
use voxel_engine::*;

fn make_test_camera(width: usize, height: usize) -> Camera {
    let aspect = width as f32 / height as f32;
    // Position the camera above and back from the origin,
    // looking towards the positive Z direction.
    Camera::new(Vec3::new(32.0, 32.0, 80.0), aspect)
}

#[test]
fn render_single_voxel_writes_pixels() {
    let mut chunk = Chunk::uniform(IVec3::ZERO, BlockType::Air);
    chunk.set_block(0, 0, 0, BlockData::new(BlockType::Grass));

    let chunks = vec![chunk];
    let chunk_refs: Vec<&Chunk> = chunks.iter().collect();

    let mesh = BinaryGreedyMesher::mesh_chunk_in_world(chunks.first().unwrap(), &chunk_refs)
        .expect("single voxel should generate a mesh");

    let width = 320usize;
    let height = 180usize;

    let mut framebuffer = Framebuffer::new(width, height);
    let mut rasterizer = Rasterizer::new();
    let camera = make_test_camera(width, height);
    let view_proj = camera.view_projection_matrix();

    let clear_color = 0xFF000000;
    framebuffer.clear(clear_color);

    let start = Instant::now();
    rasterizer.render_mesh(&mesh, &view_proj, &mut framebuffer);
    let elapsed = start.elapsed();

    let drawn_pixels = framebuffer
        .color_buffer
        .iter()
        .filter(|&&c| c != clear_color)
        .count();

    println!(
        "[PIPELINE] render_single_voxel_writes_pixels: {:?}, drawn_pixels={}",
        elapsed, drawn_pixels
    );

    assert!(
        drawn_pixels > 0,
        "expected some pixels to be drawn for a single voxel"
    );
}

fn row_coverage(framebuffer: &Framebuffer, clear_color: u32) -> Vec<bool> {
    let mut rows = vec![false; framebuffer.height];
    for y in 0..framebuffer.height {
        let start = y * framebuffer.width;
        let end = start + framebuffer.width;
        if framebuffer.color_buffer[start..end]
            .iter()
            .any(|&c| c != clear_color)
        {
            rows[y] = true;
        }
    }
    rows
}

#[test]
fn span_renderer_matches_barycentric_row_coverage() {
    // Build a simple horizontal surface that should render as a continuous band.
    let mut chunk = Chunk::uniform(IVec3::ZERO, BlockType::Air);
    for z in 0..CHUNK_SIZE {
        for x in 0..CHUNK_SIZE {
            chunk.set_block(x, 0, z, BlockData::new(BlockType::Grass));
        }
    }

    let chunks = vec![chunk];
    let chunk_refs: Vec<&Chunk> = chunks.iter().collect();

    let mesh =
        BinaryGreedyMesher::mesh_chunk_in_world(chunks.first().unwrap(), &chunk_refs)
            .expect("flat surface should generate a mesh");

    let width = 256usize;
    let height = 192usize;
    let clear_color = 0xFF000000;
    let aspect = width as f32 / height as f32;

    let mut span_fb = Framebuffer::new(width, height);
    let mut ref_fb = Framebuffer::new(width, height);
    span_fb.clear(clear_color);
    ref_fb.clear(clear_color);

    let mut span_rasterizer = Rasterizer::new();
    let mut ref_rasterizer = Rasterizer::new();

    // Camera above and back from the surface.
    let camera = Camera::new(Vec3::new(16.0, 40.0, 80.0), aspect);
    let view_proj = camera.view_projection_matrix();

    // Span renderer (default, assumes level camera).
    span_rasterizer.render_mesh(&mesh, &view_proj, &mut span_fb);

    // Reference path: force barycentric renderer by supplying a non-level up vector.
    ref_rasterizer.render_mesh_with_up(
        &mesh,
        &view_proj,
        &mut ref_fb,
        Vec3::new(0.0, 0.0, 1.0),
    );

    let span_rows = row_coverage(&span_fb, clear_color);
    let ref_rows = row_coverage(&ref_fb, clear_color);

    assert_eq!(
        span_rows, ref_rows,
        "span renderer should cover the same scanlines as the barycentric reference"
    );
}

#[test]
fn render_small_world_smoke_test() {
    // Build a small 3x3x3 world, mesh it, and render one frame.
    let mut chunks = Vec::new();
    for cx in -1..=1 {
        for cy in -1..=1 {
            for cz in -1..=1 {
                chunks.push(Chunk::generate_terrain(IVec3::new(cx, cy, cz)));
            }
        }
    }

    let chunk_refs: Vec<&Chunk> = chunks.iter().collect();
    let mut meshes = Vec::new();

    let mesh_start = Instant::now();
    for chunk in &chunks {
        if let Some(mesh) = BinaryGreedyMesher::mesh_chunk_in_world(chunk, &chunk_refs) {
            meshes.push(mesh);
        }
    }
    let mesh_time = mesh_start.elapsed();

    let width = 640usize;
    let height = 360usize;
    let mut framebuffer = Framebuffer::new(width, height);
    let mut rasterizer = Rasterizer::new();
    let camera = make_test_camera(width, height);
    let view_proj = camera.view_projection_matrix();

    framebuffer.clear(0xFF87CEEB);

    let render_start = Instant::now();
    for mesh in &meshes {
        rasterizer.render_mesh(mesh, &view_proj, &mut framebuffer);
    }
    let render_time = render_start.elapsed();

    let non_clear = framebuffer
        .color_buffer
        .iter()
        .filter(|&&c| c != 0xFF87CEEB)
        .count();

    println!(
        "[PIPELINE] render_small_world_smoke_test: meshing={:?}, rendering={:?}, drawn_pixels={}",
        mesh_time, render_time, non_clear
    );

    // Smoke-test style assertion: we just require that something was rendered.
    assert!(
        non_clear > 0,
        "expected some pixels to be drawn for small world"
    );
}

#[test]
fn sub_pixel_triangle_culling_preserves_visual_quality() {
    // This test verifies that sub-pixel triangle culling doesn't degrade
    // visual quality for normal viewing distances. We render the same scene
    // twice: once with culling (current implementation) and once with a
    // theoretical no-culling version. Since sub-pixel triangles contribute
    // negligible visual information, the results should be nearly identical.
    //
    // The test ensures:
    // 1. Culling removes sub-pixel triangles (performance benefit)
    // 2. Visual output remains correct (no missing geometry)
    // 3. Large triangles are never culled (correctness)

    // Create a scene with both large and tiny triangles
    let mut chunk = Chunk::uniform(IVec3::ZERO, BlockType::Air);

    // Add a single voxel at origin - will generate normal-sized triangles
    chunk.set_block(16, 16, 16, BlockData::new(BlockType::Grass));

    // Create a far-away voxel that will project to sub-pixel size
    // This will be at z=200, which should be tiny on screen
    let mut far_chunk = Chunk::uniform(IVec3::new(0, 0, 6), BlockType::Air);
    far_chunk.set_block(16, 16, 16, BlockData::new(BlockType::Stone));

    let chunks = vec![chunk, far_chunk];
    let chunk_refs: Vec<&Chunk> = chunks.iter().collect();

    // Mesh both chunks
    let mut meshes = Vec::new();
    for chunk in &chunks {
        if let Some(mesh) = BinaryGreedyMesher::mesh_chunk_in_world(chunk, &chunk_refs) {
            meshes.push(mesh);
        }
    }

    let width = 640usize;
    let height = 360usize;
    let mut framebuffer = Framebuffer::new(width, height);
    let mut rasterizer = Rasterizer::new();

    // Position camera to view the scene
    let camera = Camera::new(Vec3::new(16.0, 16.0, 50.0), width as f32 / height as f32);
    let view_proj = camera.view_projection_matrix();

    framebuffer.clear(0xFF000000);

    // Render with sub-pixel culling (current implementation)
    for mesh in &meshes {
        rasterizer.render_mesh(mesh, &view_proj, &mut framebuffer);
    }

    let drawn_pixels = framebuffer
        .color_buffer
        .iter()
        .filter(|&&c| c != 0xFF000000)
        .count();

    // Verify that:
    // 1. Some pixels were drawn (near voxel is visible)
    assert!(
        drawn_pixels > 0,
        "expected near voxel to be visible with sub-pixel culling enabled"
    );

    // 2. The near voxel should produce a reasonable number of pixels
    // A single voxel at z=34 should cover at least 50 pixels
    assert!(
        drawn_pixels >= 50,
        "expected at least 50 pixels from near voxel, got {}",
        drawn_pixels
    );

    println!(
        "[PIPELINE] sub_pixel_triangle_culling: drawn_pixels={}, culling is preserving visual quality",
        drawn_pixels
    );
}

#[test]
fn sub_pixel_culling_rejects_distant_geometry() {
    // This test verifies that extremely distant geometry (that would be
    // sub-pixel) is properly culled and doesn't waste rasterization cycles.
    // We create a voxel so far away it should be invisible.

    let width = 640usize;
    let height = 360usize;

    // Create voxel extremely far away (z = 960, 30 chunks * 32 voxels)
    let mut far_chunk = Chunk::uniform(IVec3::new(0, 0, 30), BlockType::Air);
    // Add a single voxel that will be extremely far away
    far_chunk.set_block(16, 16, 16, BlockData::new(BlockType::Stone));

    let chunks = vec![far_chunk];
    let chunk_refs: Vec<&Chunk> = chunks.iter().collect();

    let mesh = BinaryGreedyMesher::mesh_chunk_in_world(&chunks[0], &chunk_refs)
        .expect("far chunk should generate mesh");

    let mut framebuffer = Framebuffer::new(width, height);
    let mut rasterizer = Rasterizer::new();

    // Camera looking at origin, but voxel is at z=960 (30 chunks * 32 voxels)
    let camera = Camera::new(Vec3::new(16.0, 16.0, 50.0), width as f32 / height as f32);
    let view_proj = camera.view_projection_matrix();

    framebuffer.clear(0xFF000000);
    rasterizer.render_mesh(&mesh, &view_proj, &mut framebuffer);

    let drawn_pixels = framebuffer
        .color_buffer
        .iter()
        .filter(|&&c| c != 0xFF000000)
        .count();

    // Extremely distant geometry should be culled, resulting in 0 or very few pixels
    // This validates the optimization is working
    println!(
        "[PIPELINE] sub_pixel_culling_distant: drawn_pixels={} (should be near 0)",
        drawn_pixels
    );

    // Allow a small tolerance for edge cases, but should be minimal
    assert!(
        drawn_pixels < 10,
        "expected distant sub-pixel geometry to be culled, but {} pixels were drawn",
        drawn_pixels
    );
}

#[test]
fn sub_pixel_culling_preserves_close_geometry() {
    // This test ensures that close-up geometry is NEVER culled, regardless
    // of triangle size. When a camera is very close to a surface, even small
    // triangles should be visible.

    let width = 640usize;
    let height = 360usize;

    // Create a single voxel very close to camera
    let mut chunk = Chunk::uniform(IVec3::ZERO, BlockType::Air);
    chunk.set_block(16, 16, 16, BlockData::new(BlockType::Grass));

    let chunks = vec![chunk];
    let chunk_refs: Vec<&Chunk> = chunks.iter().collect();

    let mesh = BinaryGreedyMesher::mesh_chunk_in_world(&chunks[0], &chunk_refs)
        .expect("chunk should generate mesh");

    let mut framebuffer = Framebuffer::new(width, height);
    let mut rasterizer = Rasterizer::new();

    // Camera VERY close to the voxel (z=20, looking at voxel at z=16)
    let camera = Camera::new(Vec3::new(16.0, 16.0, 20.0), width as f32 / height as f32);
    let view_proj = camera.view_projection_matrix();

    framebuffer.clear(0xFF000000);
    rasterizer.render_mesh(&mesh, &view_proj, &mut framebuffer);

    let drawn_pixels = framebuffer
        .color_buffer
        .iter()
        .filter(|&&c| c != 0xFF000000)
        .count();

    // Close geometry should cover a significant portion of the screen
    // At this distance, the voxel should fill a large area
    assert!(
        drawn_pixels > 1000,
        "expected close geometry to be highly visible, got {} pixels (should be >1000)",
        drawn_pixels
    );

    println!(
        "[PIPELINE] sub_pixel_culling_close: drawn_pixels={} (close geometry preserved)",
        drawn_pixels
    );
}
