/// Comprehensive test suite for horizon-based occlusion culling
///
/// This test suite validates that horizon culling does not produce false positives
/// (culling chunks that are actually visible) by comparing against ground truth
/// visibility determined by actual rasterization.

use glam::{Vec2, Vec3};
use voxel_engine::{
    camera::Camera,
    voxel::{Chunk, CHUNK_SIZE},
    rendering::{Framebuffer, Rasterizer},
    meshing::BinaryGreedyMesher,
};

const SCREEN_WIDTH: usize = 1280;
const SCREEN_HEIGHT: usize = 720;

/// Represents a test chunk with position and metadata
#[derive(Debug, Clone)]
struct TestChunk {
    position: Vec3,
    center: Vec3,
    /// Whether this chunk should be visible (ground truth)
    should_be_visible: Option<bool>,
}

impl TestChunk {
    fn new(chunk_x: i32, chunk_y: i32, chunk_z: i32) -> Self {
        let position = Vec3::new(
            (chunk_x * CHUNK_SIZE as i32) as f32,
            (chunk_y * CHUNK_SIZE as i32) as f32,
            (chunk_z * CHUNK_SIZE as i32) as f32,
        );
        let half_size = CHUNK_SIZE as f32 * 0.5;
        let center = position + Vec3::splat(half_size);

        Self {
            position,
            center,
            should_be_visible: None,
        }
    }
}

/// Simulates the horizon culling algorithm from main.rs
fn apply_horizon_culling(
    camera_pos: Vec3,
    chunks: &[TestChunk],
    horizon_bins: usize,
    horizon_margin: f32,
) -> Vec<bool> {
    let mut horizon = vec![f32::NEG_INFINITY; horizon_bins];
    let mut visibility = vec![true; chunks.len()];

    // Sort by distance (simulating front-to-back ordering)
    let mut indices: Vec<usize> = (0..chunks.len()).collect();
    indices.sort_by(|&a, &b| {
        let dist_a = (chunks[a].center - camera_pos).length_squared();
        let dist_b = (chunks[b].center - camera_pos).length_squared();
        dist_a.partial_cmp(&dist_b).unwrap_or(std::cmp::Ordering::Equal)
    });

    for &i in &indices {
        let chunk = &chunks[i];
        let to_center = chunk.center - camera_pos;
        let xz = Vec2::new(to_center.x, to_center.z);
        let dist_xz = xz.length();

        // Skip extremely close chunks
        if dist_xz < 1e-3 {
            continue;
        }

        let angle = xz.y.atan2(xz.x); // [-pi, pi]
        let bin_f = (angle + std::f32::consts::PI)
            / (2.0 * std::f32::consts::PI)
            * horizon_bins as f32;
        let mut bin = bin_f.floor() as isize;
        if bin < 0 {
            bin += horizon_bins as isize;
        }
        let bin = (bin as usize) % horizon_bins;

        // Use the top of the chunk for conservative culling
        let half_height = CHUNK_SIZE as f32 * 0.5;
        let top_y = chunk.center.y + half_height;
        let height = top_y - camera_pos.y;
        let slope = height / dist_xz.max(1e-3);

        // Cull if this chunk lies clearly below the horizon
        // IMPORTANT: Only cull for positive slopes (terrain above camera).
        // Negative slopes (terrain below camera) should not be horizon-culled.
        let should_cull = slope >= 0.0 && slope + horizon_margin < horizon[bin];

        if should_cull {
            visibility[i] = false;
        } else {
            if slope > horizon[bin] {
                horizon[bin] = slope;
            }
        }
    }

    visibility
}

/// Ground truth: Check if chunk is actually visible by rasterizing it
fn is_chunk_visible_ground_truth(
    camera: &Camera,
    chunk_center: Vec3,
    framebuffer: &mut Framebuffer,
    rasterizer: &mut Rasterizer,
) -> bool {
    // Clear buffers
    framebuffer.clear(0xFF000000); // Black

    // Create a simple test chunk (solid cube)
    let chunk_pos = (chunk_center - Vec3::splat(CHUNK_SIZE as f32 * 0.5)) / CHUNK_SIZE as f32;
    let chunk = Chunk::generate_test_solid(chunk_pos.as_ivec3());

    // Mesh the chunk
    let all_chunks = vec![chunk];
    let chunk_refs: Vec<&Chunk> = all_chunks.iter().collect();
    if let Some(mesh) = BinaryGreedyMesher::mesh_chunk_in_world(&all_chunks[0], &chunk_refs) {
        let view_proj = camera.view_projection_matrix();

        // Count pixels before rendering
        let pixels_before: usize = framebuffer
            .color_buffer
            .iter()
            .filter(|&&pixel| pixel != 0xFF000000)
            .count();

        // Render the mesh
        rasterizer.render_mesh(&mesh, &view_proj, framebuffer);

        // Count pixels after rendering
        let pixels_after: usize = framebuffer
            .color_buffer
            .iter()
            .filter(|&&pixel| pixel != 0xFF000000)
            .count();

        // If any pixels were drawn, chunk is visible
        pixels_after > pixels_before
    } else {
        false // Uniform chunk or no geometry
    }
}

/// Check if a chunk's AABB is in the view frustum
fn is_chunk_in_frustum(camera: &Camera, chunk_center: Vec3) -> bool {
    let frustum = camera.extract_frustum();
    let half_size = CHUNK_SIZE as f32 * 0.5;
    let min = chunk_center - Vec3::splat(half_size);
    let max = chunk_center + Vec3::splat(half_size);
    frustum.intersects_aabb(min, max)
}

// ============================================================================
// TEST SCENARIOS
// ============================================================================

#[test]
fn test_horizon_culling_flat_terrain() {
    // Scenario: All chunks at same height, none should be horizon-culled
    let camera_pos = Vec3::new(0.0, 100.0, 0.0);
    let mut camera = Camera::new(camera_pos, SCREEN_WIDTH as f32 / SCREEN_HEIGHT as f32);

    let mut chunks = vec![];
    for x in -5..=5 {
        for z in -5..=5 {
            chunks.push(TestChunk::new(x, 0, z));
        }
    }

    let visibility = apply_horizon_culling(camera_pos, &chunks, 64, 0.05);

    let culled_count = visibility.iter().filter(|&&v| !v).count();

    // On flat terrain, horizon culling should not cull chunks in front of camera
    // (some behind camera may be culled, which is correct)
    println!("Flat terrain: {}/{} chunks culled", culled_count, chunks.len());

    // Verify that chunks in front of camera are not culled
    for (i, chunk) in chunks.iter().enumerate() {
        if (chunk.center - camera_pos).z > 0.0 {
            // Chunk is in front
            let in_frustum = is_chunk_in_frustum(&camera, chunk.center);
            if in_frustum && !visibility[i] {
                panic!(
                    "FALSE POSITIVE: Chunk at {:?} in frustum but horizon-culled!",
                    chunk.center
                );
            }
        }
    }
}

#[test]
fn test_horizon_culling_hill_scenario() {
    // Scenario: Camera on ground, hill in front, chunks behind hill
    // Camera at origin looking along +Z axis
    let camera_pos = Vec3::new(0.0, 10.0, 0.0);
    let mut camera = Camera::new(camera_pos, SCREEN_WIDTH as f32 / SCREEN_HEIGHT as f32); // Looking along +Z

    let mut chunks = vec![
        // Ground level in front of camera (should be visible)
        TestChunk::new(0, 0, 1),
        TestChunk::new(0, 0, 2),

        // Hill (elevated chunks)
        TestChunk::new(0, 2, 3), // Y=2 (elevated)
        TestChunk::new(0, 3, 3), // Y=3 (top of hill)

        // Behind hill at ground level (may be culled - this is correct)
        TestChunk::new(0, 0, 4),
        TestChunk::new(0, 0, 5),

        // Behind hill but elevated (should be visible)
        TestChunk::new(0, 4, 5), // Y=4 (peaks over hill)
    ];

    let visibility = apply_horizon_culling(camera_pos, &chunks, 64, 0.05);

    // Verify near chunks are not culled
    assert!(visibility[0], "Near ground chunk should not be culled");
    assert!(visibility[1], "Near ground chunk should not be culled");

    // Hill itself should be visible
    assert!(visibility[2], "Hill chunk should not be culled");
    assert!(visibility[3], "Hill top should not be culled");

    // Elevated chunk behind hill should be visible (peaks over horizon)
    assert!(
        visibility[6],
        "Elevated chunk behind hill should be visible (peaks over horizon)"
    );

    println!("Hill scenario: {}/{} chunks visible",
             visibility.iter().filter(|&&v| v).count(),
             chunks.len());
}

#[test]
fn test_horizon_culling_edge_case_camera_rotation() {
    // Test edge case: Camera rotation causes different bins to be used
    let camera_pos = Vec3::new(0.0, 50.0, 0.0);

    let chunk = TestChunk::new(5, 0, 5);

    // Test at different camera yaw angles
    for yaw_deg in [0.0f32, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0] {
        let yaw = yaw_deg.to_radians();
        let mut camera = Camera::new(camera_pos, SCREEN_WIDTH as f32 / SCREEN_HEIGHT as f32);
        camera.yaw = yaw;

        let visibility = apply_horizon_culling(camera_pos, &[chunk.clone()], 64, 0.05);
        let in_frustum = is_chunk_in_frustum(&camera, chunk.center);

        if in_frustum && !visibility[0] {
            panic!(
                "FALSE POSITIVE at yaw {:.0}°: Chunk in frustum but horizon-culled!",
                yaw_deg
            );
        }
    }
}

#[test]
fn test_horizon_culling_bin_boundary() {
    // Test edge case: Chunks near bin boundaries
    // This tests for off-by-one errors in bin calculation
    let camera_pos = Vec3::new(0.0, 50.0, 0.0);

    // Create chunks at precise angular positions to test bin boundaries
    let horizon_bins = 64;
    let degrees_per_bin = 360.0 / horizon_bins as f32;

    for bin in 0..horizon_bins {
        let angle_deg = bin as f32 * degrees_per_bin;
        let angle_rad = angle_deg.to_radians();

        // Place chunk at this angle
        let distance = 100.0;
        let x = distance * angle_rad.cos();
        let z = distance * angle_rad.sin();

        let chunk_pos = camera_pos + Vec3::new(x, 0.0, z);
        let chunk = TestChunk {
            position: chunk_pos,
            center: chunk_pos,
            should_be_visible: None,
        };

        let visibility = apply_horizon_culling(camera_pos, &[chunk], horizon_bins, 0.05);

        // First chunk in each bin should be visible (establishes horizon)
        assert!(
            visibility[0],
            "First chunk at bin boundary {} (angle {:.1}°) should not be culled",
            bin, angle_deg
        );
    }
}

#[test]
fn test_horizon_culling_negative_slope() {
    // Test chunks below camera (negative slope)
    let camera_pos = Vec3::new(0.0, 100.0, 0.0); // Camera high up

    let chunks = vec![
        TestChunk::new(0, 0, 5),   // Ground level, in front
        TestChunk::new(0, -5, 10), // Below ground level, farther away
    ];

    let visibility = apply_horizon_culling(camera_pos, &chunks, 64, 0.05);

    // Both should be visible (negative slopes don't cull unless below existing horizon)
    assert!(visibility[0], "Ground level chunk should be visible");
    assert!(visibility[1], "Below-ground chunk should be visible (negative slope)");
}

#[test]
fn test_horizon_culling_very_close_chunks() {
    // Test edge case: Chunks very close to camera (dist_xz < 1e-3)
    let camera_pos = Vec3::new(16.0, 16.0, 16.0); // Center of chunk (0,0,0)

    let chunks = vec![
        TestChunk::new(0, 0, 0), // Camera is inside this chunk
        TestChunk::new(1, 0, 0), // Adjacent chunk
    ];

    let visibility = apply_horizon_culling(camera_pos, &chunks, 64, 0.05);

    // Very close chunks should not be culled
    assert!(visibility[0], "Chunk containing camera should not be culled");
    assert!(visibility[1], "Adjacent chunk should not be culled");
}

#[test]
fn test_horizon_culling_concave_terrain() {
    // Test valley scenario: Camera on hill, valley below, hill on other side
    let camera_pos = Vec3::new(0.0, 100.0, 0.0);

    let chunks = vec![
        // Near hill (camera position)
        TestChunk::new(0, 3, 1),

        // Valley (lower elevation)
        TestChunk::new(0, 0, 3),
        TestChunk::new(0, 0, 4),

        // Far hill (same height as camera)
        TestChunk::new(0, 3, 6),
        TestChunk::new(0, 4, 6), // Peak of far hill
    ];

    let visibility = apply_horizon_culling(camera_pos, &chunks, 64, 0.05);

    // Near hill should be visible
    assert!(visibility[0], "Near hill should be visible");

    // Valley should be visible (below horizon but in view)
    assert!(visibility[1], "Valley chunk 1 should be visible");
    assert!(visibility[2], "Valley chunk 2 should be visible");

    // Far hill peak should be visible (same height as camera)
    assert!(visibility[4], "Far hill peak should be visible");
}

// ============================================================================
// GROUND TRUTH VALIDATION TESTS
// ============================================================================

#[test]
#[ignore] // Expensive test - run manually with: cargo test --test horizon_culling_tests -- --ignored
fn test_horizon_culling_vs_ground_truth() {
    // Compare horizon culling results against actual rasterization
    let mut framebuffer = Framebuffer::new(SCREEN_WIDTH, SCREEN_HEIGHT);
    let mut rasterizer = Rasterizer::new();
    rasterizer.backface_culling = true;

    // Camera on ground looking forward
    let camera_pos = Vec3::new(0.0, 50.0, 0.0);
    let camera = Camera::new(camera_pos, SCREEN_WIDTH as f32 / SCREEN_HEIGHT as f32);

    // Create test scenario: Hill blocking distant chunks
    let test_cases = vec![
        (TestChunk::new(0, 0, 2), "Near ground chunk"),
        (TestChunk::new(0, 2, 4), "Hill"),
        (TestChunk::new(0, 0, 6), "Behind hill (ground)"),
        (TestChunk::new(0, 3, 6), "Behind hill (elevated)"),
    ];

    for (chunk, description) in &test_cases {
        // Get ground truth
        let actually_visible = is_chunk_visible_ground_truth(
            &camera,
            chunk.center,
            &mut framebuffer,
            &mut rasterizer,
        );

        // Get horizon culling result
        let horizon_visible = apply_horizon_culling(camera_pos, &[chunk.clone()], 64, 0.05)[0];

        // Check for false positives (horizon says culled, but actually visible)
        if actually_visible && !horizon_visible {
            panic!(
                "FALSE POSITIVE: {} at {:?} is actually visible but horizon-culled!",
                description, chunk.center
            );
        }

        println!(
            "{}: actually_visible={}, horizon_visible={}",
            description, actually_visible, horizon_visible
        );
    }
}

// ============================================================================
// DIAGNOSTIC TESTS - IDENTIFY BUG PATTERNS
// ============================================================================

#[test]
fn test_bin_calculation_consistency() {
    // Test that bin calculation is consistent and doesn't have edge cases
    let _camera_pos = Vec3::ZERO;

    for angle_deg in 0..360 {
        let angle_rad = (angle_deg as f32).to_radians();
        let distance = 100.0;

        let x = distance * angle_rad.cos();
        let z = distance * angle_rad.sin();

        let to_center = Vec3::new(x, 0.0, z);
        let xz = Vec2::new(to_center.x, to_center.z);
        let computed_angle = xz.y.atan2(xz.x);

        // Verify atan2 returns value in [-pi, pi]
        assert!(
            computed_angle >= -std::f32::consts::PI && computed_angle <= std::f32::consts::PI,
            "atan2 out of range: {} at angle {}°",
            computed_angle,
            angle_deg
        );

        // Verify bin calculation doesn't overflow
        let horizon_bins = 64;
        let bin_f = (computed_angle + std::f32::consts::PI)
            / (2.0 * std::f32::consts::PI)
            * horizon_bins as f32;

        assert!(
            bin_f >= 0.0 && bin_f < horizon_bins as f32,
            "bin_f out of range: {} at angle {}°",
            bin_f,
            angle_deg
        );

        let mut bin = bin_f.floor() as isize;
        if bin < 0 {
            bin += horizon_bins as isize;
        }
        let bin = (bin as usize) % horizon_bins;

        assert!(
            bin < horizon_bins,
            "Final bin out of range: {} at angle {}°",
            bin,
            angle_deg
        );
    }
}

#[test]
fn test_slope_calculation_edge_cases() {
    // Test slope calculation for edge cases
    let camera_pos = Vec3::new(0.0, 50.0, 0.0);

    let test_cases = vec![
        ("Same height", Vec3::new(100.0, 50.0, 100.0)),
        ("Above camera", Vec3::new(100.0, 150.0, 100.0)),
        ("Below camera", Vec3::new(100.0, -50.0, 100.0)),
        ("Very close", Vec3::new(1.0, 50.0, 1.0)),
        ("Very far", Vec3::new(1000.0, 50.0, 1000.0)),
    ];

    for (description, chunk_center) in test_cases {
        let to_center = chunk_center - camera_pos;
        let xz = Vec2::new(to_center.x, to_center.z);
        let dist_xz = xz.length();

        let half_height = CHUNK_SIZE as f32 * 0.5;
        let top_y = chunk_center.y + half_height;
        let height = top_y - camera_pos.y;
        let slope = height / dist_xz.max(1e-3);

        // Verify slope is finite and reasonable
        assert!(
            slope.is_finite(),
            "Slope is not finite for {}: {}",
            description,
            slope
        );

        println!("{}: slope={:.4}, dist_xz={:.2}", description, slope, dist_xz);
    }
}

#[test]
fn test_margin_effect() {
    // Test how margin affects culling decisions
    let camera_pos = Vec3::new(0.0, 50.0, 0.0);

    let chunks = vec![
        TestChunk::new(0, 2, 3), // Hill at z=3
        TestChunk::new(0, 2, 6), // Same height at z=6 (farther)
    ];

    for margin in [0.0, 0.01, 0.05, 0.1, 0.2] {
        let visibility = apply_horizon_culling(camera_pos, &chunks, 64, margin);

        println!(
            "Margin {:.2}: near={}, far={}",
            margin, visibility[0], visibility[1]
        );

        // Near chunk should always be visible
        assert!(
            visibility[0],
            "Near chunk culled with margin {}",
            margin
        );
    }
}

// ============================================================================
// HELPER FUNCTIONS FOR DEBUGGING
// ============================================================================

/// Print detailed diagnostic information about horizon culling decision
#[allow(dead_code)]
fn debug_horizon_culling(camera_pos: Vec3, chunk: &TestChunk, horizon_bins: usize) {
    let to_center = chunk.center - camera_pos;
    let xz = Vec2::new(to_center.x, to_center.z);
    let dist_xz = xz.length();

    let angle = xz.y.atan2(xz.x);
    let angle_deg = angle.to_degrees();

    let bin_f = (angle + std::f32::consts::PI)
        / (2.0 * std::f32::consts::PI)
        * horizon_bins as f32;
    let mut bin = bin_f.floor() as isize;
    if bin < 0 {
        bin += horizon_bins as isize;
    }
    let bin = (bin as usize) % horizon_bins;

    let half_height = CHUNK_SIZE as f32 * 0.5;
    let top_y = chunk.center.y + half_height;
    let height = top_y - camera_pos.y;
    let slope = height / dist_xz.max(1e-3);

    println!("=== Horizon Culling Debug ===");
    println!("Chunk center: {:?}", chunk.center);
    println!("Camera pos: {:?}", camera_pos);
    println!("To center: {:?}", to_center);
    println!("XZ distance: {:.2}", dist_xz);
    println!("Angle: {:.2}° (rad: {:.4})", angle_deg, angle);
    println!("Bin (float): {:.4}", bin_f);
    println!("Bin (final): {}", bin);
    println!("Top Y: {:.2}", top_y);
    println!("Height: {:.2}", height);
    println!("Slope: {:.6}", slope);
}
