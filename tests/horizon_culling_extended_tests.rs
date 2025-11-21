/// Extended test suite for horizon culling to diagnose visible terrain gaps
/// These tests focus on the specific artifacts seen in actual rendering

use glam::{Vec2, Vec3};
use voxel_engine::{
    camera::Camera,
    voxel::{Chunk, CHUNK_SIZE},
};

const SCREEN_WIDTH: usize = 1280;
const SCREEN_HEIGHT: usize = 720;

#[derive(Debug, Clone)]
struct TestChunk {
    position: Vec3,
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
            position,
            center,
            chunk_coords: (chunk_x, chunk_y, chunk_z),
        }
    }
}

/// Simulates horizon culling with detailed diagnostics
fn apply_horizon_culling_with_diagnostics(
    camera_pos: Vec3,
    chunks: &[TestChunk],
    horizon_bins: usize,
    horizon_margin: f32,
) -> (Vec<bool>, Vec<HorizonDiagnostic>) {
    let mut horizon = vec![f32::NEG_INFINITY; horizon_bins];
    let mut visibility = vec![true; chunks.len()];
    let mut diagnostics = Vec::new();

    // Sort by distance
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

        let half_height = CHUNK_SIZE as f32 * 0.5;
        let top_y = chunk.center.y + half_height;
        let height = top_y - camera_pos.y;
        let slope = height / dist_xz.max(1e-3);

        let should_cull = slope >= 0.0 && slope + horizon_margin < horizon[bin];

        diagnostics.push(HorizonDiagnostic {
            chunk_coords: chunk.chunk_coords,
            bin,
            slope,
            horizon_slope: horizon[bin],
            margin: horizon_margin,
            culled: should_cull,
            dist_xz,
            angle_deg: angle.to_degrees(),
        });

        if should_cull {
            visibility[i] = false;
        } else {
            if slope > horizon[bin] {
                horizon[bin] = slope;
            }
        }
    }

    (visibility, diagnostics)
}

#[derive(Debug)]
struct HorizonDiagnostic {
    chunk_coords: (i32, i32, i32),
    bin: usize,
    slope: f32,
    horizon_slope: f32,
    margin: f32,
    culled: bool,
    dist_xz: f32,
    angle_deg: f32,
}

// ============================================================================
// TESTS FOR VISIBLE TERRAIN GAPS (Like in screenshot)
// ============================================================================

#[test]
fn test_adjacent_chunks_same_height() {
    // Test that adjacent chunks at the same height don't cull each other
    let camera_pos = Vec3::new(0.0, 100.0, 0.0);

    // Create a row of adjacent chunks at ground level
    let chunks: Vec<TestChunk> = (0..10)
        .map(|x| TestChunk::new(x, 0, 5))
        .collect();

    let (visibility, diagnostics) = apply_horizon_culling_with_diagnostics(
        camera_pos,
        &chunks,
        64,
        0.05,
    );

    // Check for incorrectly culled chunks
    for (i, (chunk, &visible)) in chunks.iter().zip(visibility.iter()).enumerate() {
        if !visible {
            let diag = &diagnostics[i];
            panic!(
                "Adjacent chunk at {:?} incorrectly culled!\n\
                 Slope: {:.4}, Horizon: {:.4}, Margin: {:.4}\n\
                 Bin: {}, Angle: {:.1}°\n\
                 This would create visible gaps in terrain!",
                chunk.chunk_coords,
                diag.slope,
                diag.horizon_slope,
                diag.margin,
                diag.bin,
                diag.angle_deg
            );
        }
    }
}

#[test]
fn test_hillside_chunks_at_different_angles() {
    // Simulates chunks on a hillside viewed from different camera positions
    // This replicates the scenario in the screenshot
    let camera_pos = Vec3::new(0.0, 80.0, 0.0);

    let mut chunks = vec![];

    // Create chunks forming a hillside
    for z in 1..8 {
        for x in -3..=3 {
            // Height varies with distance to simulate terrain
            let height = if z < 4 { 0 } else { (z - 3) };
            chunks.push(TestChunk::new(x, height, z));
        }
    }

    let (visibility, diagnostics) = apply_horizon_culling_with_diagnostics(
        camera_pos,
        &chunks,
        64,
        0.05,
    );

    // Verify no adjacent chunks are culled when they should be visible
    for (i, chunk) in chunks.iter().enumerate() {
        if !visibility[i] {
            let diag = &diagnostics[i];

            // Check if any adjacent chunk is visible (would create gap)
            let adjacent_visible = chunks.iter().enumerate().any(|(j, other)| {
                if !visibility[j] {
                    return false;
                }

                let dx = (chunk.chunk_coords.0 - other.chunk_coords.0).abs();
                let dy = (chunk.chunk_coords.1 - other.chunk_coords.1).abs();
                let dz = (chunk.chunk_coords.2 - other.chunk_coords.2).abs();

                // Check if adjacent (within 1 chunk)
                dx <= 1 && dy <= 1 && dz <= 1 && (dx + dy + dz) > 0
            });

            if adjacent_visible {
                println!(
                    "WARNING: Chunk {:?} culled but has visible adjacent chunks!\n\
                     Slope: {:.4}, Horizon: {:.4}, Bin: {}, Angle: {:.1}°",
                    chunk.chunk_coords,
                    diag.slope,
                    diag.horizon_slope,
                    diag.bin,
                    diag.angle_deg
                );
            }
        }
    }
}

#[test]
fn test_camera_pitch_downward() {
    // Test looking down at terrain (common in screenshot scenario)
    let camera_pos = Vec3::new(0.0, 150.0, 0.0);

    // Camera looking downward
    let mut camera = Camera::new(camera_pos, SCREEN_WIDTH as f32 / SCREEN_HEIGHT as f32);
    camera.pitch = -0.5; // Looking down ~28 degrees

    let chunks: Vec<TestChunk> = (0..5)
        .flat_map(|z| (0..5).map(move |x| TestChunk::new(x, 0, z)))
        .collect();

    let (visibility, _) = apply_horizon_culling_with_diagnostics(
        camera_pos,
        &chunks,
        64,
        0.05,
    );

    let visible_count = visibility.iter().filter(|&&v| v).count();

    // When looking down, most terrain below should be visible
    assert!(
        visible_count >= 20,
        "When looking down, expected at least 20/25 chunks visible, got {}",
        visible_count
    );
}

#[test]
fn test_bin_boundary_adjacent_chunks() {
    // Test chunks that fall on bin boundaries
    // This can cause issues where adjacent chunks use different bins
    let camera_pos = Vec3::ZERO;
    let horizon_bins = 64;
    let degrees_per_bin = 360.0 / horizon_bins as f32;

    for bin_idx in 0..horizon_bins {
        let angle_deg = bin_idx as f32 * degrees_per_bin;

        // Place chunks just before and after bin boundary
        let angle1 = (angle_deg - 0.5).to_radians();
        let angle2 = (angle_deg + 0.5).to_radians();

        let distance = 100.0;

        let chunks = vec![
            TestChunk {
                position: Vec3::ZERO,
                center: Vec3::new(
                    distance * angle1.cos(),
                    50.0,
                    distance * angle1.sin(),
                ),
                chunk_coords: (0, 0, 0),
            },
            TestChunk {
                position: Vec3::ZERO,
                center: Vec3::new(
                    distance * angle2.cos(),
                    50.0,
                    distance * angle2.sin(),
                ),
                chunk_coords: (0, 0, 1),
            },
        ];

        let (visibility, diagnostics) = apply_horizon_culling_with_diagnostics(
            camera_pos,
            &chunks,
            horizon_bins,
            0.05,
        );

        // Both chunks should be visible (neither should cull the other)
        if !visibility[0] || !visibility[1] {
            panic!(
                "Bin boundary issue at bin {}: chunks at {:.1}° and {:.1}° have visibility {:?}\n\
                 Diagnostics: {:?}",
                bin_idx,
                angle_deg - 0.5,
                angle_deg + 0.5,
                visibility,
                diagnostics
            );
        }
    }
}

#[test]
fn test_slope_calculation_precision() {
    // Test that slope calculations are consistent for chunks at similar positions
    let camera_pos = Vec3::new(0.0, 100.0, 0.0);

    // Create chunks that are very close in position
    let chunks = vec![
        TestChunk::new(5, 0, 5),
        TestChunk::new(5, 0, 6), // Adjacent in Z
        TestChunk::new(6, 0, 5), // Adjacent in X
    ];

    let (visibility, diagnostics) = apply_horizon_culling_with_diagnostics(
        camera_pos,
        &chunks,
        64,
        0.05,
    );

    // Check slope consistency
    for (i, diag) in diagnostics.iter().enumerate() {
        println!(
            "Chunk {:?}: slope={:.6}, bin={}, culled={}",
            chunks[i].chunk_coords,
            diag.slope,
            diag.bin,
            diag.culled
        );
    }

    // All should have similar slopes and none should be culled
    let slopes: Vec<f32> = diagnostics.iter().map(|d| d.slope).collect();
    let max_slope = slopes.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let min_slope = slopes.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    let slope_variation = max_slope - min_slope;

    assert!(
        slope_variation < 0.1,
        "Slope variation too large for adjacent chunks: {:.6}",
        slope_variation
    );

    assert!(
        visibility.iter().all(|&v| v),
        "Some adjacent chunks were incorrectly culled"
    );
}

#[test]
fn test_terrain_with_small_elevation_changes() {
    // Test gently sloping terrain (like in the screenshot)
    let camera_pos = Vec3::new(0.0, 100.0, 0.0);

    let mut chunks = vec![];
    for z in 0..10 {
        for x in -5..=5 {
            // Gentle slope: each Z increases height slightly
            let height = -(z / 3); // Slight downward slope
            chunks.push(TestChunk::new(x, height, z));
        }
    }

    let (visibility, diagnostics) = apply_horizon_culling_with_diagnostics(
        camera_pos,
        &chunks,
        64,
        0.05,
    );

    // Check for gaps in continuous slope
    for z in 1..9 {
        for x in -4..=4 {
            let idx = chunks.iter().position(|c| {
                c.chunk_coords.0 == x && c.chunk_coords.2 == z
            }).unwrap();

            if !visibility[idx] {
                // Check if surrounded by visible chunks (would create hole)
                let neighbors = [
                    (x-1, z), (x+1, z), (x, z-1), (x, z+1)
                ];

                let visible_neighbors = neighbors.iter().filter(|&&(nx, nz)| {
                    chunks.iter().position(|c| {
                        c.chunk_coords.0 == nx && c.chunk_coords.2 == nz
                    }).map_or(false, |ni| visibility[ni])
                }).count();

                if visible_neighbors >= 3 {
                    let diag = &diagnostics[idx];
                    panic!(
                        "Chunk ({}, {}, {}) culled but surrounded by {} visible neighbors!\n\
                         This creates a visible hole in terrain.\n\
                         Slope: {:.4}, Horizon: {:.4}, Margin: {:.4}, Bin: {}",
                        x, chunks[idx].chunk_coords.1, z,
                        visible_neighbors,
                        diag.slope,
                        diag.horizon_slope,
                        diag.margin,
                        diag.bin
                    );
                }
            }
        }
    }
}

#[test]
fn test_margin_too_aggressive() {
    // Test if the margin is causing false positives
    let camera_pos = Vec3::new(0.0, 100.0, 0.0);

    let chunks = vec![
        TestChunk::new(0, 0, 5),  // Establishes horizon
        TestChunk::new(1, 0, 6),  // Slightly farther, slightly offset
    ];

    // Test with different margins
    for &margin in &[0.0, 0.01, 0.05, 0.1, 0.2] {
        let (visibility, diagnostics) = apply_horizon_culling_with_diagnostics(
            camera_pos,
            &chunks,
            64,
            margin,
        );

        println!("\nMargin {:.2}:", margin);
        for (i, diag) in diagnostics.iter().enumerate() {
            println!(
                "  Chunk {:?}: slope={:.4}, horizon={:.4}, culled={}, slope_diff={:.4}",
                chunks[i].chunk_coords,
                diag.slope,
                diag.horizon_slope,
                diag.culled,
                diag.slope - diag.horizon_slope
            );
        }

        // With small or zero margin, adjacent chunks at same height should never cull
        if margin <= 0.05 {
            assert!(
                visibility.iter().all(|&v| v),
                "With margin {:.2}, adjacent flat chunks incorrectly culled",
                margin
            );
        }
    }
}

#[test]
fn test_horizon_update_order() {
    // Test that processing order doesn't cause issues
    let camera_pos = Vec3::new(0.0, 100.0, 0.0);

    // Create chunks at same angle but different distances
    let chunks = vec![
        TestChunk::new(0, 0, 3),   // Near
        TestChunk::new(0, 0, 5),   // Medium
        TestChunk::new(0, 0, 7),   // Far
        TestChunk::new(0, 1, 7),   // Far but elevated
    ];

    let (visibility, diagnostics) = apply_horizon_culling_with_diagnostics(
        camera_pos,
        &chunks,
        64,
        0.05,
    );

    println!("\nProcessing order test:");
    for (i, diag) in diagnostics.iter().enumerate() {
        println!(
            "Chunk {:?}: dist={:.1}, slope={:.4}, horizon={:.4}, culled={}",
            chunks[i].chunk_coords,
            diag.dist_xz,
            diag.slope,
            diag.horizon_slope,
            diag.culled
        );
    }

    // Near chunks should always be visible
    assert!(visibility[0], "Nearest chunk should never be culled");

    // Elevated far chunk should be visible
    assert!(
        visibility[3],
        "Elevated chunk should be visible even if at same angle"
    );
}

#[test]
fn test_realistic_camera_scenario() {
    // Simulate the exact scenario from the screenshot:
    // Camera at moderate height looking across rolling terrain

    let camera_pos = Vec3::new(0.0, 80.0, 0.0);
    let mut camera = Camera::new(camera_pos, SCREEN_WIDTH as f32 / SCREEN_HEIGHT as f32);
    camera.pitch = -0.2; // Slightly downward
    camera.yaw = 0.0;

    // Generate terrain similar to screenshot
    let mut chunks = vec![];
    for z in 0..12 {
        for x in -6..=6 {
            // Simulated rolling hills
            let base_height = -2;
            let height_variation = if z >= 3 && z <= 6 { 1 } else { 0 };
            let height = base_height + height_variation;

            chunks.push(TestChunk::new(x, height, z));
        }
    }

    let (visibility, diagnostics) = apply_horizon_culling_with_diagnostics(
        camera_pos,
        &chunks,
        64,
        0.05,
    );

    let total_chunks = chunks.len();
    let visible_chunks = visibility.iter().filter(|&&v| v).count();
    let culled_chunks = total_chunks - visible_chunks;

    println!("\nRealistic scenario:");
    println!("Total chunks: {}", total_chunks);
    println!("Visible: {}", visible_chunks);
    println!("Culled: {} ({:.1}%)", culled_chunks, 100.0 * culled_chunks as f32 / total_chunks as f32);

    // Check for holes
    let mut holes_found = 0;
    for (i, chunk) in chunks.iter().enumerate() {
        if !visibility[i] {
            let (x, _, z) = chunk.chunk_coords;

            // Count visible neighbors
            let neighbors = [
                (x-1, z), (x+1, z), (x, z-1), (x, z+1)
            ];

            let visible_neighbors = neighbors.iter().filter(|&&(nx, nz)| {
                chunks.iter().enumerate().any(|(ni, nc)| {
                    nc.chunk_coords.0 == nx && nc.chunk_coords.2 == nz && visibility[ni]
                })
            }).count();

            if visible_neighbors >= 3 {
                holes_found += 1;
                let diag = &diagnostics[i];
                println!(
                    "HOLE: Chunk {:?} culled with {} visible neighbors (bin {}, slope {:.4})",
                    chunk.chunk_coords,
                    visible_neighbors,
                    diag.bin,
                    diag.slope
                );
            }
        }
    }

    assert_eq!(
        holes_found, 0,
        "Found {} holes in terrain (chunks culled but surrounded by visible chunks)",
        holes_found
    );
}
