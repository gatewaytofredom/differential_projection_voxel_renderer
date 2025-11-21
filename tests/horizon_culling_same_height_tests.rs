/// Critical tests for horizon culling when camera is at or below terrain height
/// This tests the specific bug: chunks incorrectly culled when camera.y <= terrain.y

use glam::{Vec2, Vec3};
use voxel_engine::{
    camera::Camera,
    voxel::CHUNK_SIZE,
};

const SCREEN_WIDTH: usize = 1280;
const SCREEN_HEIGHT: usize = 720;

#[derive(Debug, Clone)]
struct TestChunk {
    center: Vec3,
    chunk_coords: (i32, i32, i32),
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
            center,
            chunk_coords: (chunk_x, chunk_y, chunk_z),
        }
    }
}

fn apply_horizon_culling(
    camera_pos: Vec3,
    chunks: &[TestChunk],
    horizon_bins: usize,
    horizon_margin: f32,
) -> Vec<(bool, f32, f32)> {
    let mut horizon = vec![f32::NEG_INFINITY; horizon_bins];
    let mut results = Vec::new();

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

        if dist_xz < 1e-3 {
            results.push((true, 0.0, f32::NEG_INFINITY));
            continue;
        }

        let angle = xz.y.atan2(xz.x);
        let bin_f = (angle + std::f32::consts::PI)
            / (2.0 * std::f32::consts::PI)
            * horizon_bins as f32;
        let mut bin = bin_f.floor() as isize;
        if bin < 0 {
            bin += horizon_bins as isize;
        }
        let bin = (bin as usize) % horizon_bins;

        // CRITICAL: This is where the bug happens
        let half_height = CHUNK_SIZE as f32 * 0.5;
        let top_y = chunk.center.y + half_height;  // Top of chunk
        let height = top_y - camera_pos.y;
        let slope = height / dist_xz.max(1e-3);

        let should_cull = slope >= 0.0 && slope + horizon_margin < horizon[bin];

        results.push((!should_cull, slope, horizon[bin]));

        if !should_cull {
            if slope > horizon[bin] {
                horizon[bin] = slope;
            }
        }
    }

    results
}

#[test]
fn test_camera_at_terrain_height() {
    // CRITICAL TEST: Camera at exactly the same Y as terrain
    // This is when the bug appears in the screenshot

    let terrain_y = 2; // Chunk at Y=2 means center at Y=48, top at Y=64
    let chunk_center_y = (terrain_y * CHUNK_SIZE as i32) as f32 + 16.0; // = 80.0
    let chunk_top_y = chunk_center_y + 16.0; // = 96.0

    // Camera at the TOP of the chunk (where player would be standing on terrain)
    let camera_y = chunk_top_y;
    let camera_pos = Vec3::new(0.0, camera_y, 0.0);

    let chunks = vec![
        TestChunk::new(0, terrain_y, 3),  // In front, same height
        TestChunk::new(1, terrain_y, 3),  // Adjacent, same height
        TestChunk::new(0, terrain_y, 4),  // Farther, same height
    ];

    let results = apply_horizon_culling(camera_pos, &chunks, 64, 0.05);

    for (i, (visible, slope, horizon_slope)) in results.iter().enumerate() {
        println!(
            "Chunk {:?}: visible={}, slope={:.6}, horizon={:.6}",
            chunks[i].chunk_coords, visible, slope, horizon_slope
        );

        // When camera is at terrain height, slope should be very small but NOT culled
        assert!(
            *visible,
            "CRITICAL BUG: Chunk {:?} culled when camera at terrain height!\n\
             Camera Y: {:.1}, Chunk top Y: {:.1}\n\
             Slope: {:.6}, Horizon: {:.6}\n\
             This causes the visible gaps in your screenshot!",
            chunks[i].chunk_coords,
            camera_y,
            chunks[i].center.y + 16.0,
            slope,
            horizon_slope
        );
    }
}

#[test]
fn test_camera_below_terrain_height() {
    // Camera slightly below the top of terrain
    let terrain_y = 2;
    let chunk_center_y = (terrain_y * CHUNK_SIZE as i32) as f32 + 16.0;
    let chunk_top_y = chunk_center_y + 16.0;

    // Camera 10 units below the top of terrain
    let camera_y = chunk_top_y - 10.0;
    let camera_pos = Vec3::new(0.0, camera_y, 0.0);

    let chunks = vec![
        TestChunk::new(0, terrain_y, 3),
        TestChunk::new(1, terrain_y, 3),
        TestChunk::new(2, terrain_y, 3),
    ];

    let results = apply_horizon_culling(camera_pos, &chunks, 64, 0.05);

    for (i, (visible, slope, horizon_slope)) in results.iter().enumerate() {
        println!(
            "Camera below terrain - Chunk {:?}: visible={}, slope={:.6}, horizon={:.6}",
            chunks[i].chunk_coords, visible, slope, horizon_slope
        );

        // All should be visible - we're below the terrain looking at it
        assert!(
            *visible,
            "Chunk {:?} incorrectly culled when camera below terrain!\n\
             Slope: {:.6} (should be small positive but not culled)",
            chunks[i].chunk_coords, slope
        );
    }
}

#[test]
fn test_near_chunk_establishes_bad_horizon() {
    // This replicates the exact bug scenario:
    // 1. Near chunk at same height establishes a horizon
    // 2. Far chunks at same height get culled incorrectly

    let terrain_y = 2;
    let chunk_top_y = (terrain_y * CHUNK_SIZE as i32) as f32 + 32.0;
    let camera_y = chunk_top_y; // At terrain height
    let camera_pos = Vec3::new(0.0, camera_y, 0.0);

    let chunks = vec![
        TestChunk::new(0, terrain_y, 2),  // NEAR chunk (z=2)
        TestChunk::new(0, terrain_y, 5),  // FAR chunk (z=5) - same height!
    ];

    let results = apply_horizon_culling(camera_pos, &chunks, 64, 0.05);

    let (near_visible, near_slope, _) = results[0];
    let (far_visible, far_slope, far_horizon) = results[1];

    println!("\n=== BUG REPRODUCTION ===");
    println!("Camera at terrain height: Y = {:.1}", camera_y);
    println!("Near chunk: slope={:.6}, visible={}", near_slope, near_visible);
    println!("Far chunk: slope={:.6}, horizon={:.6}, visible={}", far_slope, far_horizon, far_visible);

    // The bug: near chunk sets a small positive slope as horizon
    // Far chunk has similar small positive slope, gets culled!
    if !far_visible {
        panic!(
            "BUG REPRODUCED! Far chunk incorrectly culled!\n\
             Near chunk established horizon at slope={:.6}\n\
             Far chunk has slope={:.6}, margin=0.05\n\
             Check: {:.6} + 0.05 = {:.6} < {:.6} = {}\n\
             \n\
             This is the bug causing gaps in your screenshot!\n\
             When camera is at terrain height, all terrain at that height\n\
             has small positive slopes (~0.0 to 0.1) which incorrectly cull each other.",
            near_slope,
            far_slope,
            far_slope, far_slope + 0.05, far_horizon,
            far_slope + 0.05 < far_horizon
        );
    }
}

#[test]
fn test_slope_precision_at_same_height() {
    // Test the precise slope values when camera is at terrain height
    let terrain_y = 0;
    let chunk_center_y = 16.0; // Center of Y=0 chunk
    let chunk_top_y = 32.0;    // Top of Y=0 chunk

    let camera_y = chunk_top_y;
    let camera_pos = Vec3::new(0.0, camera_y, 0.0);

    // Calculate expected slopes for chunks at different distances
    for z in 2..8 {
        let chunk = TestChunk::new(0, terrain_y, z);
        let dist_xz = ((chunk.center.z - camera_pos.z).powi(2)).sqrt();

        // Expected slope: (top_y - camera_y) / dist_xz
        let expected_slope = (chunk_top_y - camera_y) / dist_xz;

        println!(
            "Distance z={}: dist_xz={:.1}, expected_slope={:.6}",
            z, dist_xz, expected_slope
        );

        // When camera is at terrain height, slope should be exactly 0.0
        assert!(
            expected_slope.abs() < 1e-6,
            "Slope should be ~0 when camera at terrain height, got {:.6}",
            expected_slope
        );
    }
}

#[test]
fn test_margin_effect_at_same_height() {
    // Test how margin affects culling when slopes are near-zero
    let terrain_y = 0;
    let camera_y = 32.0; // At top of terrain
    let camera_pos = Vec3::new(0.0, camera_y, 0.0);

    let chunks = vec![
        TestChunk::new(0, terrain_y, 2),
        TestChunk::new(0, terrain_y, 5),
    ];

    for &margin in &[0.0, 0.01, 0.05, 0.1] {
        let results = apply_horizon_culling(camera_pos, &chunks, 64, margin);

        println!("\nMargin {:.2}:", margin);
        for (i, (visible, slope, horizon_slope)) in results.iter().enumerate() {
            println!(
                "  Chunk z={}: slope={:.6}, horizon={:.6}, visible={}",
                chunks[i].chunk_coords.2, slope, horizon_slope, visible
            );
        }

        // With any margin, near-zero slopes should not cull each other
        assert!(
            results.iter().all(|(v, _, _)| *v),
            "With margin {:.2}, chunks at same height should not cull each other",
            margin
        );
    }
}

#[test]
fn test_realistic_player_on_ground() {
    // Realistic scenario: Player standing on ground looking across flat terrain
    // This is the EXACT scenario from your screenshot

    let terrain_height_chunks = 0; // Ground level
    let player_eye_height = 48.0; // Player standing on ground (chunk top + eye offset)

    let camera_pos = Vec3::new(0.0, player_eye_height, 0.0);

    // Flat terrain extending outward
    let mut chunks = Vec::new();
    for z in 1..12 {
        for x in -5..=5 {
            chunks.push(TestChunk::new(x, terrain_height_chunks, z));
        }
    }

    let results = apply_horizon_culling(camera_pos, &chunks, 64, 0.05);

    let mut culled_chunks = Vec::new();
    for (i, (visible, slope, horizon_slope)) in results.iter().enumerate() {
        if !*visible {
            culled_chunks.push((chunks[i].chunk_coords, *slope, *horizon_slope));
        }
    }

    if !culled_chunks.is_empty() {
        println!("\n=== CULLED CHUNKS (BUG!) ===");
        for (coords, slope, horizon) in &culled_chunks {
            println!(
                "Chunk {:?}: slope={:.6}, horizon={:.6}, diff={:.6}",
                coords, slope, horizon, slope - horizon
            );
        }

        panic!(
            "{} chunks incorrectly culled when player on flat ground!\n\
             This creates the visible gaps in your screenshot.\n\
             Camera Y: {:.1} (player eye height)\n\
             Terrain top Y: 32.0\n\
             Player is only {:.1} units above terrain.",
            culled_chunks.len(),
            camera_pos.y,
            camera_pos.y - 32.0
        );
    }
}

#[test]
fn test_solution_use_center_not_top() {
    // Test potential fix: use chunk CENTER instead of TOP for slope calculation
    let terrain_y = 0;
    let chunk_center_y = 16.0;
    let chunk_top_y = 32.0;

    let camera_y = chunk_top_y; // At terrain top
    let camera_pos = Vec3::new(0.0, camera_y, 0.0);

    let chunks = vec![
        TestChunk::new(0, terrain_y, 2),
        TestChunk::new(0, terrain_y, 5),
    ];

    println!("\n=== TESTING SOLUTION ===");

    // Current implementation (uses TOP)
    for chunk in &chunks {
        let dist_xz = (chunk.center.z - camera_pos.z).abs();
        let slope_using_top = (chunk_top_y - camera_y) / dist_xz;
        println!(
            "Chunk z={}, using TOP: slope={:.6}",
            chunk.chunk_coords.2, slope_using_top
        );
    }

    // Proposed fix (use CENTER)
    for chunk in &chunks {
        let dist_xz = (chunk.center.z - camera_pos.z).abs();
        let slope_using_center = (chunk_center_y - camera_y) / dist_xz;
        println!(
            "Chunk z={}, using CENTER: slope={:.6}",
            chunk.chunk_coords.2, slope_using_center
        );
    }

    // Using center gives negative slopes (camera above center)
    // These won't be culled by our fix (slope < 0.0 always kept)
}
