/// Exact reproduction of the bug from the screenshot
/// Tests with HORIZON_BINS=128 and HORIZON_MARGIN=0.1 as currently set

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
) -> Vec<(bool, usize, f32, f32)> {
    const HORIZON_BINS: usize = 128;
    const HORIZON_MARGIN: f32 = 0.1;

    let mut horizon = vec![f32::NEG_INFINITY; HORIZON_BINS];
    let mut results = Vec::new();

    // Sort front-to-back
    let mut indexed: Vec<(usize, f32)> = chunks.iter().enumerate()
        .map(|(i, c)| (i, (c.center - camera_pos).length_squared()))
        .collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    for (i, _) in indexed {
        let chunk = &chunks[i];
        let to_center = chunk.center - camera_pos;
        let xz = Vec2::new(to_center.x, to_center.z);
        let dist_xz = xz.length();

        if dist_xz < 1e-3 {
            results.push((true, 0, 0.0, f32::NEG_INFINITY));
            continue;
        }

        let angle = xz.y.atan2(xz.x);
        let bin_f = (angle + std::f32::consts::PI) / (2.0 * std::f32::consts::PI) * HORIZON_BINS as f32;
        let bin = (bin_f.floor() as isize).rem_euclid(HORIZON_BINS as isize) as usize;

        let top_y = chunk.center.y + 16.0;
        let height = top_y - camera_pos.y;
        let slope = height / dist_xz;

        let should_keep = slope < 0.0 || slope + HORIZON_MARGIN >= horizon[bin];

        results.push((should_keep, bin, slope, horizon[bin]));

        if should_keep && slope > horizon[bin] {
            horizon[bin] = slope;
        }
    }

    results
}

#[test]
fn test_exact_screenshot_scenario() {
    // Based on screenshot: camera appears to be at or near ground level
    // looking across relatively flat terrain with some hills

    let camera_y = 48.0; // Standing on Y=0 terrain (top at 32) + eye height
    let camera_pos = Vec3::new(50.0, camera_y, 50.0);

    // Create a grid of flat terrain
    let mut chunks = Vec::new();
    for z in 0..15 {
        for x in 0..15 {
            chunks.push(TestChunk::new(x, 0, z)); // All at ground level
        }
    }

    let results = simulate_horizon_culling(camera_pos, &chunks);

    let mut culled = Vec::new();
    for (i, (visible, bin, slope, horizon)) in results.iter().enumerate() {
        if !visible {
            culled.push((chunks[i].coords, *bin, *slope, *horizon));
        }
    }

    if !culled.is_empty() {
        println!("\n=== BUG FOUND ===");
        println!("Camera at ({:.1}, {:.1}, {:.1})", camera_pos.x, camera_pos.y, camera_pos.z);
        println!("{} chunks incorrectly culled:\n", culled.len());

        for (coords, bin, slope, horizon) in &culled {
            println!(
                "Chunk {:?}: bin={}, slope={:.6}, horizon={:.6}, margin=0.1",
                coords, bin, slope, horizon
            );
            println!("  Check: {:.6} + 0.1 = {:.6} >= {:.6}? {}",
                     slope, slope + 0.1, horizon, slope + 0.1 >= *horizon);
        }

        panic!("\nThese chunks should be visible but were culled!");
    }
}

#[test]
fn test_camera_slightly_above_terrain() {
    // Camera just a bit above terrain - common walking scenario
    let terrain_top_y = 32.0;
    let camera_y = terrain_top_y + 5.0;  // 5 units above terrain
    let camera_pos = Vec3::new(0.0, camera_y, 0.0);

    let mut chunks = Vec::new();
    for z in 1..10 {
        for x in -3..=3 {
            chunks.push(TestChunk::new(x, 0, z));
        }
    }

    let results = simulate_horizon_culling(camera_pos, &chunks);

    let culled_count = results.iter().filter(|(v, _, _, _)| !v).count();

    println!("\nCamera {:.1} units above terrain:", camera_y - terrain_top_y);
    println!("Total chunks: {}", chunks.len());
    println!("Culled: {}", culled_count);

    // With camera only 5 units above, slopes are very small
    // Let's see what's happening
    for (i, (visible, bin, slope, horizon)) in results.iter().enumerate() {
        let chunk = &chunks[i];
        if chunk.coords.0 == 0 {  // Print chunks along Z axis
            println!(
                "Z={}: bin={}, slope={:.6}, horizon={:.6}, visible={}",
                chunk.coords.2, bin, slope, horizon, visible
            );
        }
    }

    assert_eq!(culled_count, 0, "No chunks should be culled when camera slightly above flat terrain");
}

#[test]
fn test_camera_height_sweep() {
    // Test various camera heights to find where culling breaks
    let mut chunks = Vec::new();
    for z in 0..8 {
        for x in -2..=2 {
            chunks.push(TestChunk::new(x, 0, z));
        }
    }

    // Sweep camera from below terrain to above
    for camera_y in [16.0, 24.0, 32.0, 40.0, 48.0, 64.0, 96.0] {
        let camera_pos = Vec3::new(0.0, camera_y, 0.0);
        let results = simulate_horizon_culling(camera_pos, &chunks);

        let culled_count = results.iter().filter(|(v, _, _, _)| !v).count();

        println!("\nCamera Y={:.0} (terrain top=32):", camera_y);
        println!("  Culled: {}/{}", culled_count, chunks.len());

        // Show a few examples
        for (i, (visible, bin, slope, horizon)) in results.iter().enumerate().take(5) {
            if !visible {
                println!("    Chunk {:?}: bin={}, slope={:.6}, horizon={:.6}",
                         chunks[i].coords, bin, slope, horizon);
            }
        }

        if culled_count > chunks.len() / 2 {
            panic!("Too many chunks culled at camera Y={:.0}!", camera_y);
        }
    }
}
