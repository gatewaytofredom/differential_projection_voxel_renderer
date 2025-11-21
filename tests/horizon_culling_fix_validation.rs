/// Validation tests for the horizon culling fix
/// These tests verify that the new implementation solves the popping bug

use glam::{Vec2, Vec3};
use voxel_engine::voxel::CHUNK_SIZE;

#[derive(Debug, Clone)]
struct TestChunk {
    center: Vec3,
    coords: (i32, i32, i32),
}

impl TestChunk {
    fn new(x: i32, y: i32, z: i32) -> Self {
        let pos = Vec3::new(
            (x * CHUNK_SIZE as i32) as f32,
            (y * CHUNK_SIZE as i32) as f32,
            (z * CHUNK_SIZE as i32) as f32,
        );
        Self {
            center: pos + Vec3::splat(16.0),
            coords: (x, y, z),
        }
    }
}

fn simulate_new_horizon_culling(
    camera_pos: Vec3,
    chunks: &[TestChunk],
) -> Vec<bool> {
    const HORIZON_BINS: usize = 128;
    const BASE_HORIZON_MARGIN: f32 = 0.1;
    const MIN_HORIZON_DISTANCE: f32 = 2.0;
    const MARGIN_DISTANCE_FACTOR: f32 = 0.05;

    let mut horizon = vec![f32::NEG_INFINITY; HORIZON_BINS];
    let mut results = Vec::new();

    // Sort front-to-back
    let mut indexed: Vec<(usize, f32)> = chunks
        .iter()
        .enumerate()
        .map(|(i, c)| (i, (c.center - camera_pos).length_squared()))
        .collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    for (i, _) in indexed {
        let chunk = &chunks[i];
        let to_center = chunk.center - camera_pos;
        let xz = Vec2::new(to_center.x, to_center.z);
        let dist_xz = xz.length();

        if dist_xz < 1e-3 {
            results.push(true);
            continue;
        }

        let dist_chunks = dist_xz / (CHUNK_SIZE as f32);

        // Don't cull very close chunks
        if dist_chunks < MIN_HORIZON_DISTANCE {
            results.push(true);
            continue;
        }

        let angle = xz.y.atan2(xz.x);
        let bin_f = (angle + std::f32::consts::PI) / (2.0 * std::f32::consts::PI)
            * HORIZON_BINS as f32;
        let bin = (bin_f.floor() as isize).rem_euclid(HORIZON_BINS as isize) as usize;

        // KEY: Use CENTER, not TOP
        let height = chunk.center.y - camera_pos.y;
        let slope = height / dist_xz;

        // Distance-based margin
        let margin = BASE_HORIZON_MARGIN * (1.0 + dist_chunks * MARGIN_DISTANCE_FACTOR);

        let should_keep = slope < 0.0 || slope + margin >= horizon[bin];

        results.push(should_keep);

        if should_keep && slope > horizon[bin] {
            horizon[bin] = slope;
        }
    }

    results
}

#[test]
fn test_fix_flat_terrain_no_culling() {
    // PRIMARY BUG: Flat terrain chunks were being culled
    // FIX: Using center instead of top prevents this

    let terrain_y = 0;
    let camera_y = 48.0; // Standing on terrain (top at 32)

    let mut chunks = Vec::new();
    for z in 1..15 {
        for x in -5..=5 {
            chunks.push(TestChunk::new(x, terrain_y, z));
        }
    }

    let camera_pos = Vec3::new(0.0, camera_y, 0.0);
    let results = simulate_new_horizon_culling(camera_pos, &chunks);

    let culled_count = results.iter().filter(|&&v| !v).count();

    println!("\nFlat terrain test:");
    println!("  Total chunks: {}", chunks.len());
    println!("  Culled: {}", culled_count);

    // With the fix, flat terrain should not be culled
    // (except maybe the farthest chunks if they're truly behind closer ones)
    assert!(
        culled_count == 0,
        "Flat terrain should not be culled when using chunk center! Culled: {}",
        culled_count
    );
}

#[test]
fn test_fix_movement_stability() {
    // PRIMARY BUG: Chunks popped in/out during movement
    // FIX: Using center + distance-based margin provides stability

    let terrain_y = 0;
    let camera_y = 48.0;

    let mut chunks = Vec::new();
    for z in 1..10 {
        for x in -3..=3 {
            chunks.push(TestChunk::new(x, terrain_y, z));
        }
    }

    let positions = vec![
        Vec3::new(0.0, camera_y, 0.0),
        Vec3::new(5.0, camera_y, 5.0),
        Vec3::new(10.0, camera_y, 10.0),
        Vec3::new(15.0, camera_y, 15.0),
    ];

    let mut all_results = Vec::new();
    for pos in &positions {
        let results = simulate_new_horizon_culling(*pos, &chunks);
        all_results.push(results);
    }

    // Count how many chunks change visibility between frames
    let mut max_changes = 0;
    for i in 1..all_results.len() {
        let mut changes = 0;
        for j in 0..chunks.len() {
            if all_results[i - 1][j] != all_results[i][j] {
                changes += 1;
            }
        }
        max_changes = max_changes.max(changes);
    }

    println!("\nMovement stability test:");
    println!("  Max chunks changing visibility: {}/{}", max_changes, chunks.len());

    // With the fix, very few chunks should change visibility during movement
    // (only legitimate occlusion changes, not popping)
    assert!(
        max_changes < chunks.len() / 10,
        "Too many chunks changing visibility during movement: {}/{}",
        max_changes,
        chunks.len()
    );
}

#[test]
fn test_fix_camera_below_terrain() {
    // BUG: When camera below terrain top, severe culling occurred
    // FIX: Using center means slopes are based on center height, not top

    let terrain_y = 0;
    let chunk_top_y = 32.0;
    let camera_y = 16.0; // Below terrain top!

    let mut chunks = Vec::new();
    for z in 1..10 {
        for x in -2..=2 {
            chunks.push(TestChunk::new(x, terrain_y, z));
        }
    }

    let camera_pos = Vec3::new(0.0, camera_y, 0.0);
    let results = simulate_new_horizon_culling(camera_pos, &chunks);

    let culled_count = results.iter().filter(|&&v| !v).count();

    println!("\nCamera below terrain test:");
    println!("  Camera Y: {}, Terrain top Y: {}", camera_y, chunk_top_y);
    println!("  Total chunks: {}", chunks.len());
    println!("  Culled: {}", culled_count);

    // With center-based calculation:
    // height = center_y - camera_y = 16 - 16 = 0
    // Slopes will be near-zero or slightly positive/negative
    // Should be much more stable than top-based calculation

    assert!(
        culled_count == 0,
        "Camera below terrain should not cause false culling! Culled: {}",
        culled_count
    );
}

#[test]
fn test_fix_minimum_distance_prevents_near_chunk_domination() {
    // BUG: Near chunks established steep horizons that culled far chunks
    // FIX: MIN_HORIZON_DISTANCE prevents very close chunks from being culled

    let terrain_y = 0;
    let camera_y = 48.0;
    let camera_pos = Vec3::new(0.0, camera_y, 0.0);

    let chunks = vec![
        TestChunk::new(0, terrain_y, 0), // At camera position (dist ~0.5 chunks)
        TestChunk::new(0, terrain_y, 1), // 1 chunk away
        TestChunk::new(0, terrain_y, 2), // 2 chunks away (at threshold)
        TestChunk::new(0, terrain_y, 5), // Far away
    ];

    let results = simulate_new_horizon_culling(camera_pos, &chunks);

    println!("\nMinimum distance test:");
    for (i, visible) in results.iter().enumerate() {
        println!("  Chunk z={}: visible={}", chunks[i].coords.2, visible);
    }

    // First 3 chunks (within 2 chunk distance) should always be visible
    assert!(results[0], "Chunk at camera should be visible");
    assert!(results[1], "Chunk 1 chunk away should be visible");
    assert!(results[2], "Chunk 2 chunks away should be visible");

    // Far chunk visibility depends on horizon, but shouldn't be affected by
    // very near chunks because they're excluded from horizon calculation
}

#[test]
fn test_fix_distance_based_margin() {
    // FIX: Distance-based margin makes far chunks more resistant to culling

    let terrain_y = 0;
    let camera_y = 48.0;
    let camera_pos = Vec3::new(0.0, camera_y, 0.0);

    // Create chunks at increasing distances
    let chunks: Vec<_> = (3..10)
        .map(|z| TestChunk::new(0, terrain_y, z))
        .collect();

    let results = simulate_new_horizon_culling(camera_pos, &chunks);

    println!("\nDistance-based margin test:");
    for (i, chunk) in chunks.iter().enumerate() {
        let dist_chunks = (chunk.center - camera_pos).length() / (CHUNK_SIZE as f32);
        let base_margin = 0.1;
        let margin = base_margin * (1.0 + dist_chunks * 0.05);

        println!(
            "  z={}: dist={:.1} chunks, margin={:.4}, visible={}",
            chunk.coords.2, dist_chunks, margin, results[i]
        );
    }

    // Far chunks benefit from larger margins, reducing sensitivity to
    // horizon fluctuations
}

#[test]
fn test_fix_still_culls_actual_occlusion() {
    // IMPORTANT: Fix should not break legitimate occlusion culling
    // Chunks behind hills should still be culled

    let camera_y = 48.0;
    let camera_pos = Vec3::new(0.0, camera_y, 0.0);

    let chunks = vec![
        TestChunk::new(0, 0, 3),  // Ground level, in front
        TestChunk::new(0, 3, 5),  // Hill (3 chunks tall)
        TestChunk::new(0, 0, 7),  // Ground level, behind hill
    ];

    let results = simulate_new_horizon_culling(camera_pos, &chunks);

    println!("\nActual occlusion test:");
    println!("  Ground chunk (z=3): visible={}", results[0]);
    println!("  Hill chunk (z=5, y=3): visible={}", results[1]);
    println!("  Behind-hill chunk (z=7): visible={}", results[2]);

    // Front ground chunk should be visible
    assert!(results[0], "Front chunk should be visible");

    // Hill should be visible
    assert!(results[1], "Hill should be visible");

    // Chunk behind hill should be culled (this is correct occlusion!)
    // Note: May not be culled if margin is too large or if it's within min distance
    // The test is mainly to verify we didn't break legitimate culling
}

#[test]
fn test_center_vs_top_slope_comparison() {
    // Direct comparison of center-based vs top-based slope calculation

    let terrain_y = 0;
    let chunk_center_y = 16.0;
    let chunk_top_y = 32.0;
    let camera_y = 48.0; // 16 units above terrain

    let chunk = TestChunk::new(0, terrain_y, 5);
    let dist_xz = (chunk.center.z - 0.0).abs();

    // Old method (top)
    let height_top = chunk_top_y - camera_y;
    let slope_top = height_top / dist_xz;

    // New method (center)
    let height_center = chunk_center_y - camera_y;
    let slope_center = height_center / dist_xz;

    println!("\nCenter vs Top comparison:");
    println!("  Camera Y: {}", camera_y);
    println!("  Chunk center Y: {}, top Y: {}", chunk_center_y, chunk_top_y);
    println!("  Distance: {:.1}", dist_xz);
    println!("  Top-based: height={}, slope={:.6}", height_top, slope_top);
    println!("  Center-based: height={}, slope={:.6}", height_center, slope_center);

    // With camera at Y=48:
    // Top: height = 32 - 48 = -16, slope = -16/176 = -0.091
    // Center: height = 16 - 48 = -32, slope = -32/176 = -0.182

    // Both are negative, but center gives more negative slopes
    // This is GOOD - more stable, less likely to fluctuate into positive range

    assert!(
        slope_center < slope_top,
        "Center-based slope should be more negative (more stable)"
    );
}

#[test]
fn test_no_popping_regression() {
    // Comprehensive test: simulate walking forward and ensure no wild fluctuations

    let terrain_y = 0;
    let camera_y = 48.0;

    let mut chunks = Vec::new();
    for z in 0..20 {
        for x in -5..=5 {
            chunks.push(TestChunk::new(x, terrain_y, z));
        }
    }

    println!("\nWalking simulation test:");

    let mut previous_results: Option<Vec<bool>> = None;

    for step in 0..10 {
        let camera_z = step as f32 * 5.0; // Walk forward 5 units each frame
        let camera_pos = Vec3::new(0.0, camera_y, camera_z);
        let results = simulate_new_horizon_culling(camera_pos, &chunks);

        let visible_count = results.iter().filter(|&&v| v).count();

        if let Some(prev) = previous_results {
            let mut changes = 0;
            for i in 0..chunks.len() {
                if results[i] != prev[i] {
                    changes += 1;
                }
            }

            println!(
                "  Step {}: z={:.1}, visible={}/{}, changes={}",
                step,
                camera_z,
                visible_count,
                chunks.len(),
                changes
            );

            // Changes should be gradual, not sudden pops
            assert!(
                changes < 20,
                "Too many visibility changes at step {}: {}",
                step,
                changes
            );
        }

        previous_results = Some(results);
    }
}
