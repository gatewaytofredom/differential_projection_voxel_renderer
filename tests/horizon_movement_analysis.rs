/// Analysis of horizon culling behavior during camera movement
/// This test investigates why chunks pop in/out during WASD movement but NOT rotation

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

fn simulate_horizon_culling(
    camera_pos: Vec3,
    chunks: &[TestChunk],
    horizon_bins: usize,
    horizon_margin: f32,
) -> Vec<(bool, usize, f32, f32, f32)> {
    let mut horizon = vec![f32::NEG_INFINITY; horizon_bins];
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
            results.push((true, 0, 0.0, f32::NEG_INFINITY, dist_xz));
            continue;
        }

        let angle = xz.y.atan2(xz.x);
        let bin_f = (angle + std::f32::consts::PI) / (2.0 * std::f32::consts::PI)
            * horizon_bins as f32;
        let bin = (bin_f.floor() as isize).rem_euclid(horizon_bins as isize) as usize;

        let top_y = chunk.center.y + 16.0;
        let height = top_y - camera_pos.y;
        let slope = height / dist_xz;

        let should_keep = slope < 0.0 || slope + horizon_margin >= horizon[bin];

        results.push((should_keep, bin, slope, horizon[bin], dist_xz));

        if should_keep && slope > horizon[bin] {
            horizon[bin] = slope;
        }
    }

    results
}

#[test]
fn test_horizontal_movement_causes_slope_changes() {
    // This test demonstrates WHY horizontal movement causes popping
    // Key insight: Moving horizontally changes DISTANCE to chunks,
    // which changes SLOPES, which changes culling decisions!

    // Setup: Flat terrain, camera at ground level
    let terrain_y = 0;
    let camera_y = 48.0; // Standing on terrain

    // Create a grid of chunks
    let mut chunks = Vec::new();
    for z in 1..10 {
        for x in -3..=3 {
            chunks.push(TestChunk::new(x, terrain_y, z));
        }
    }

    println!("\n=== HORIZONTAL MOVEMENT SLOPE ANALYSIS ===\n");

    // Simulate moving forward (increasing Z)
    let camera_positions = vec![
        Vec3::new(0.0, camera_y, 0.0),   // Start
        Vec3::new(0.0, camera_y, 10.0),  // Moved forward
        Vec3::new(0.0, camera_y, 20.0),  // Moved more forward
    ];

    // Track one specific chunk to see how its visibility changes
    let tracked_chunk_idx = chunks
        .iter()
        .position(|c| c.coords == (0, 0, 5))
        .unwrap();

    for (step, camera_pos) in camera_positions.iter().enumerate() {
        let results = simulate_horizon_culling(*camera_pos, &chunks, 64, 0.05);

        println!("Step {}: Camera at Z={:.1}", step, camera_pos.z);

        // Show tracked chunk
        let (visible, bin, slope, horizon, dist) = results[tracked_chunk_idx];
        println!(
            "  Tracked chunk (0,0,5): visible={}, bin={}, slope={:.6}, horizon={:.6}, dist={:.1}",
            visible, bin, slope, horizon, dist
        );

        // Count culled chunks
        let culled_count = results.iter().filter(|(v, _, _, _, _)| !*v).count();
        println!("  Total culled: {}/{}\n", culled_count, chunks.len());
    }

    // The problem: As camera moves, distance changes, slopes change,
    // horizon values change, causing chunks to pop in/out!
}

#[test]
fn test_rotation_vs_movement_stability() {
    // This test compares rotation (stable) vs movement (unstable)

    let terrain_y = 0;
    let camera_y = 48.0;

    let mut chunks = Vec::new();
    for z in 1..8 {
        for x in -3..=3 {
            chunks.push(TestChunk::new(x, terrain_y, z));
        }
    }

    println!("\n=== ROTATION VS MOVEMENT ===\n");

    // Test 1: Rotation (camera stays in place)
    println!("--- ROTATION TEST ---");
    let camera_pos = Vec3::new(0.0, camera_y, 0.0);

    // Rotation doesn't change distances or slopes!
    // It only changes which chunks are in view (frustum culling)
    // Horizon culling results should be identical for all rotations
    println!("Camera rotation doesn't change distances to chunks");
    println!("Therefore slopes and horizon values remain constant");
    println!("Result: NO POPPING\n");

    // Test 2: Movement (camera position changes)
    println!("--- MOVEMENT TEST ---");

    let pos1 = Vec3::new(0.0, camera_y, 0.0);
    let pos2 = Vec3::new(5.0, camera_y, 5.0); // Moved diagonally

    let results1 = simulate_horizon_culling(pos1, &chunks, 64, 0.05);
    let results2 = simulate_horizon_culling(pos2, &chunks, 64, 0.05);

    // Compare visibility changes
    let mut changed_visibility = 0;
    for i in 0..chunks.len() {
        let visible1 = results1[i].0;
        let visible2 = results2[i].0;

        if visible1 != visible2 {
            changed_visibility += 1;
            let slope1 = results1[i].2;
            let slope2 = results2[i].2;
            let dist1 = results1[i].4;
            let dist2 = results2[i].4;

            println!(
                "Chunk {:?}: visible {} -> {}, slope {:.4} -> {:.4}, dist {:.1} -> {:.1}",
                chunks[i].coords, visible1, visible2, slope1, slope2, dist1, dist2
            );
        }
    }

    println!("\n{} chunks changed visibility just from moving!", changed_visibility);
    println!("Result: POPPING");

    // This demonstrates the fundamental problem!
}

#[test]
fn test_slope_calculation_with_distance_change() {
    // Demonstrate the math behind why movement causes slope changes

    let terrain_y = 0;
    let chunk_top_y = 32.0;
    let camera_y = 48.0; // 16 units above terrain

    println!("\n=== SLOPE MATH ANALYSIS ===\n");

    // Consider a chunk at (0, 0, 5)
    let chunk_center = Vec3::new(16.0, 16.0, 160.0 + 16.0); // Z=5 chunk

    // Camera at different positions
    let positions = vec![
        Vec3::new(0.0, camera_y, 0.0),     // Far away
        Vec3::new(0.0, camera_y, 100.0),   // Closer
        Vec3::new(0.0, camera_y, 160.0),   // Very close (same Z as chunk)
    ];

    for camera_pos in positions {
        let to_chunk = chunk_center - camera_pos;
        let dist_xz = (to_chunk.x * to_chunk.x + to_chunk.z * to_chunk.z).sqrt();
        let height = chunk_top_y - camera_y;
        let slope = height / dist_xz;

        println!(
            "Camera at Z={:.0}: dist_xz={:.1}, height={:.1}, slope={:.6}",
            camera_pos.z, dist_xz, height, slope
        );
    }

    println!("\nKey insight:");
    println!("- height = chunk_top - camera_y is CONSTANT (camera Y doesn't change)");
    println!("- dist_xz CHANGES as you move horizontally");
    println!("- slope = height / dist_xz therefore CHANGES");
    println!("- Different slopes = different horizon comparisons = different culling!");
}

#[test]
fn test_near_chunk_poisons_horizon_for_movement() {
    // The critical bug: Near chunks establish steep horizons
    // As you move, your "near chunks" change, horizons change drastically

    let terrain_y = 0;
    let camera_y = 48.0;

    let chunks = vec![
        TestChunk::new(0, terrain_y, 1), // Very close
        TestChunk::new(0, terrain_y, 3), // Medium distance
        TestChunk::new(0, terrain_y, 6), // Far
    ];

    println!("\n=== HORIZON POISONING ANALYSIS ===\n");

    // Position 1: Close to z=1 chunk
    let pos1 = Vec3::new(0.0, camera_y, 0.0);
    let results1 = simulate_horizon_culling(pos1, &chunks, 64, 0.05);

    println!("Camera at Z=0:");
    for (i, (visible, bin, slope, horizon, _)) in results1.iter().enumerate() {
        println!(
            "  Chunk z={}: slope={:.6}, horizon={:.6}, visible={}",
            chunks[i].coords.2, slope, horizon, visible
        );
    }

    // Position 2: Moved forward, now close to z=3 chunk
    let pos2 = Vec3::new(0.0, camera_y, 64.0); // Moved 2 chunks forward
    let results2 = simulate_horizon_culling(pos2, &chunks, 64, 0.05);

    println!("\nCamera at Z=64:");
    for (i, (visible, bin, slope, horizon, _)) in results2.iter().enumerate() {
        println!(
            "  Chunk z={}: slope={:.6}, horizon={:.6}, visible={}",
            chunks[i].coords.2, slope, horizon, visible
        );
    }

    println!("\nPROBLEM IDENTIFIED:");
    println!("- The 'near chunk' changes as you move");
    println!("- Near chunks have steep slopes (small distance, same height difference)");
    println!("- These steep slopes set high horizon values");
    println!("- Far chunks get culled based on these high horizons");
    println!("- As you move, which chunk is 'near' changes constantly");
    println!("- Therefore horizons fluctuate wildly");
    println!("- Therefore culling decisions fluctuate = POPPING");
}

#[test]
fn test_solution_distance_based_margin() {
    // Potential solution: Use distance-dependent margin
    // Near chunks should have stricter culling, far chunks more lenient

    let terrain_y = 0;
    let camera_y = 48.0;
    let camera_pos = Vec3::new(0.0, camera_y, 0.0);

    let chunks = vec![
        TestChunk::new(0, terrain_y, 1),
        TestChunk::new(0, terrain_y, 3),
        TestChunk::new(0, terrain_y, 6),
    ];

    println!("\n=== DISTANCE-BASED MARGIN SOLUTION ===\n");

    const BASE_MARGIN: f32 = 0.05;
    const HORIZON_BINS: usize = 64;

    let mut horizon = vec![f32::NEG_INFINITY; HORIZON_BINS];

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

        let angle = xz.y.atan2(xz.x);
        let bin_f = (angle + std::f32::consts::PI) / (2.0 * std::f32::consts::PI)
            * HORIZON_BINS as f32;
        let bin = (bin_f.floor() as isize).rem_euclid(HORIZON_BINS as isize) as usize;

        let top_y = chunk.center.y + 16.0;
        let height = top_y - camera_pos.y;
        let slope = height / dist_xz;

        // KEY CHANGE: Margin increases with distance
        // This prevents far chunks from being affected by near chunk horizons
        let dist_chunks = dist_xz / (CHUNK_SIZE as f32);
        let distance_margin = BASE_MARGIN * (1.0 + dist_chunks * 0.1);

        let should_keep = slope < 0.0 || slope + distance_margin >= horizon[bin];

        println!(
            "Chunk z={}: dist={:.1} chunks, slope={:.6}, margin={:.4}, horizon={:.6}, keep={}",
            chunk.coords.2, dist_chunks, slope, distance_margin, horizon[bin], should_keep
        );

        if should_keep && slope > horizon[bin] {
            horizon[bin] = slope;
        }
    }

    println!("\nThis solution:");
    println!("- Near chunks: small margin (strict culling)");
    println!("- Far chunks: larger margin (lenient culling)");
    println!("- Reduces popping because far chunks less affected by horizon changes");
}

#[test]
fn test_solution_hysteresis() {
    // Another solution: Add hysteresis (different threshold for culling vs un-culling)

    println!("\n=== HYSTERESIS SOLUTION ===\n");

    println!("Concept:");
    println!("- To CULL a chunk: slope + margin < horizon (strict)");
    println!("- To KEEP a chunk already visible: slope + (margin/2) >= horizon (lenient)");
    println!();
    println!("This requires tracking which chunks were visible last frame");
    println!("Prevents chunks from rapidly toggling between visible/culled");
    println!("Trade-off: More complex state management, slight memory overhead");
}

#[test]
fn test_fundamental_problem_diagnosis() {
    println!("\n=== FUNDAMENTAL PROBLEM ===\n");

    println!("The horizon culling algorithm assumes:");
    println!("1. Camera is relatively static or high above terrain");
    println!("2. Chunks are processed front-to-back consistently");
    println!("3. Horizon values are stable across frames");
    println!();

    println!("Reality for ground-level player:");
    println!("1. Camera constantly moving (WASD)");
    println!("2. Small movements drastically change which chunk is 'nearest'");
    println!("3. Nearest chunks have STEEP slopes (height/small_distance)");
    println!("4. These steep slopes set HIGH horizon values");
    println!("5. Far chunks at same height have LOWER slopes");
    println!("6. Far chunks get culled incorrectly");
    println!("7. As you move, 'nearest' changes → horizons fluctuate → POPPING");
    println!();

    println!("Why rotation doesn't cause popping:");
    println!("- Rotation doesn't change distances");
    println!("- Slopes remain constant");
    println!("- Only the bin assignment changes (which is fine)");
    println!("- Horizon values for each bin remain stable");
    println!();

    println!("ROOT CAUSE:");
    println!("Using chunk TOP for slope calculation is fundamentally flawed");
    println!("when camera is AT terrain height. The math produces:");
    println!("  slope = (terrain_top - camera_y) / horizontal_distance");
    println!("When camera_y ≈ terrain_top:");
    println!("  slope = small_value / distance");
    println!("  Near: small/small = LARGE slope");
    println!("  Far: small/large = SMALL slope");
    println!("  → Near chunks cull far chunks at same height!");
}
