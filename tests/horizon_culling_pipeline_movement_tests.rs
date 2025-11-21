/// End-to-end horizon culling test that exercises the full
/// world → meshing → horizon culling → rasterization pipeline
/// while moving the camera through the world.
///
/// The test renders each frame twice:
/// - Once with horizon culling disabled (baseline)
/// - Once with horizon culling enabled (test path)
///
/// For each frame we compare the resulting framebuffers and assert
/// that horizon culling never turns a non-clear pixel into a clear
/// (sky) pixel. Legitimate occlusion should only remove geometry that
/// is fully hidden behind nearer terrain, which does not change the
/// nearest visible surface at any pixel.

use glam::{IVec3, Vec2, Vec3};
use std::collections::HashMap;
use voxel_engine::*;

#[derive(Copy, Clone)]
struct VisibleMesh<'a> {
    mesh: &'a ChunkMesh,
    center: Vec3,
    distance_sq: f32,
}

/// Horizon culling implementation used for this test. It mirrors the
/// in-engine logic but is kept local so we can compare against a
/// baseline that has horizon culling disabled.
fn apply_horizon_culling(camera_pos: Vec3, meshes: &mut Vec<VisibleMesh<'_>>) {
    use std::cmp::Ordering;

    const HORIZON_BINS: usize = 128;
    const BASE_HORIZON_MARGIN: f32 = 0.1;
    const MIN_HORIZON_DISTANCE: f32 = 2.0;
    const MARGIN_DISTANCE_FACTOR: f32 = 0.05;

    // Front-to-back order so nearer chunks establish the horizon.
    meshes.sort_by(|a, b| {
        a.distance_sq
            .partial_cmp(&b.distance_sq)
            .unwrap_or(Ordering::Equal)
    });

    let half_size = CHUNK_SIZE as f32 * 0.5;

    let mut horizon = [f32::NEG_INFINITY; HORIZON_BINS];
    let mut write = 0usize;

    for i in 0..meshes.len() {
        let info = meshes[i];

        let to_center = info.center - camera_pos;
        let xz = Vec2::new(to_center.x, to_center.z);
        let dist_xz = xz.length();

        if dist_xz < 1e-3 {
            meshes[write] = info;
            write += 1;
            continue;
        }

        let dist_chunks = dist_xz / (CHUNK_SIZE as f32);

        // Don't horizon-cull very close chunks – they are always kept
        // and do not participate in horizon building.
        if dist_chunks < MIN_HORIZON_DISTANCE {
            meshes[write] = info;
            write += 1;
            continue;
        }

        let angle = xz.y.atan2(xz.x); // [-pi, pi]
        let bin_f = (angle + std::f32::consts::PI)
            / (2.0 * std::f32::consts::PI)
            * HORIZON_BINS as f32;
        let mut bin = bin_f.floor() as isize;
        if bin < 0 {
            bin += HORIZON_BINS as isize;
        }
        let bin = (bin as usize) % HORIZON_BINS;

        // Approximate vertical extent of this chunk along this ray.
        let inv_dist = 1.0 / dist_xz.max(1e-3);
        let height_center = info.center.y - camera_pos.y;
        let slope_center = height_center * inv_dist;
        let slope_extent = half_size * inv_dist;
        let slope_top = slope_center + slope_extent;

        // Entire chunk below the camera – never horizon-cull, since it
        // cannot occlude anything above eye level.
        if slope_top <= 0.0 {
            meshes[write] = info;
            write += 1;
            continue;
        }

        let margin = BASE_HORIZON_MARGIN * (1.0 + dist_chunks * MARGIN_DISTANCE_FACTOR);
        let current_horizon = horizon[bin];

        // Only cull when even the top of the chunk lies well below the
        // established horizon in this bin.
        if current_horizon.is_finite() && slope_top + margin < current_horizon {
            continue;
        }

        // Keep and update horizon with this chunk's top slope.
        meshes[write] = info;
        write += 1;

        if slope_top > horizon[bin] {
            horizon[bin] = slope_top;
        }
    }

    meshes.truncate(write);
}

fn collect_visible_meshes<'a>(
    world: &'a World,
    mesh_cache: &'a HashMap<IVec3, Option<ChunkMesh>>,
    camera: &Camera,
) -> Vec<VisibleMesh<'a>> {
    let frustum = camera.extract_frustum();
    let visible_chunks = world.get_visible_chunks_frustum(camera.position, Some(&frustum));

    let mut visible_meshes = Vec::with_capacity(visible_chunks.len());

    for chunk in &visible_chunks {
        if let Some(Some(mesh)) = mesh_cache.get(&chunk.position) {
            let min = (chunk.position * CHUNK_SIZE as i32).as_vec3();
            let max = min + Vec3::splat(CHUNK_SIZE as f32);
            let center = (min + max) * 0.5;
            let distance_sq = (center - camera.position).length_squared();
            visible_meshes.push(VisibleMesh {
                mesh,
                center,
                distance_sq,
            });
        }
    }

    visible_meshes
}

fn render_world_frame(
    world: &World,
    mesh_cache: &HashMap<IVec3, Option<ChunkMesh>>,
    camera: &Camera,
    framebuffer: &mut Framebuffer,
    rasterizer: &mut Rasterizer,
    use_horizon_culling: bool,
    clear_color: u32,
) {
    framebuffer.clear(clear_color);

    let mut visible_meshes = collect_visible_meshes(world, mesh_cache, camera);

    if use_horizon_culling {
        apply_horizon_culling(camera.position, &mut visible_meshes);
    } else {
        // Baseline: sort front-to-back only.
        visible_meshes.sort_by(|a, b| {
            a.distance_sq
                .partial_cmp(&b.distance_sq)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    let view_proj = camera.view_projection_matrix();
    let mut slices = framebuffer.split_into_stripes(1);
    if let Some(slice) = slices.first_mut() {
        for info in &visible_meshes {
            rasterizer.render_mesh_into_slice(info.mesh, &view_proj, slice);
        }
    }
}

#[test]
fn horizon_culling_does_not_remove_visible_pixels_during_movement() {
    let width = 640usize;
    let height = 360usize;
    let aspect = width as f32 / height as f32;
    let clear_color = 0xFF87CEEB; // Sky blue

    // World setup: moderately large region around origin to exercise
    // real terrain shapes while staying deterministic.
    let mut world = World::new(WorldConfig {
        view_distance: 8,
        frustum_culling: true,
        max_chunks_per_frame: 1024,
    });
    world.generate_region(IVec3::new(-8, -2, -8), IVec3::new(8, 2, 8));

    // Pre-mesh all chunks once using the indexed meshing path to mirror
    // the main application.
    let all_chunks = world.get_all_chunks();
    let mut index: HashMap<(i32, i32, i32), &Chunk> = HashMap::with_capacity(all_chunks.len());
    for chunk in &all_chunks {
        let pos = chunk.position;
        index.insert((pos.x, pos.y, pos.z), *chunk);
    }

    let mut mesh_cache: HashMap<IVec3, Option<ChunkMesh>> = HashMap::new();
    for chunk in &all_chunks {
        let mesh = BinaryGreedyMesher::mesh_chunk_in_indexed_world(chunk, &index);
        mesh_cache.insert(chunk.position, mesh);
    }

    let mut framebuffer_base = Framebuffer::new(width, height);
    let mut framebuffer_horizon = Framebuffer::new(width, height);
    let mut rasterizer_base = Rasterizer::new();
    let mut rasterizer_horizon = Rasterizer::new();

    // Simulate a simple movement path roughly along the +X/+Z diagonal,
    // keeping camera orientation fixed so only world-space movement
    // affects visibility.
    let camera_y = 32.0;
    let positions = [
        Vec3::new(0.0, camera_y, 80.0),
        Vec3::new(8.0, camera_y, 72.0),
        Vec3::new(16.0, camera_y, 64.0),
        Vec3::new(24.0, camera_y, 56.0),
        Vec3::new(32.0, camera_y, 48.0),
    ];

    for pos in &positions {
        let mut camera = Camera::new(*pos, aspect);
        // Default yaw/pitch in Camera::new already looks towards -Z,
        // which matches our world layout.

        render_world_frame(
            &world,
            &mesh_cache,
            &camera,
            &mut framebuffer_base,
            &mut rasterizer_base,
            false,
            clear_color,
        );

        render_world_frame(
            &world,
            &mesh_cache,
            &camera,
            &mut framebuffer_horizon,
            &mut rasterizer_horizon,
            true,
            clear_color,
        );

        let base = framebuffer_base.color_buffer_slice();
        let hz = framebuffer_horizon.color_buffer_slice();

        // Count pixels that were visible in the baseline but became
        // clear (sky) when horizon culling was applied.
        let mut missing_pixels = 0usize;
        for (c_base, c_hz) in base.iter().zip(hz.iter()) {
            if *c_base != clear_color && *c_hz == clear_color {
                missing_pixels += 1;
            }
        }

        assert!(
            missing_pixels == 0,
            "horizon culling removed {} pixels that were visible without culling at camera pos {:?}",
            missing_pixels,
            pos
        );
    }
}

