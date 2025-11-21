use glam::{Vec2, Vec3};

use crate::voxel::CHUNK_SIZE;
use crate::meshing::ChunkMesh;
use std::cmp::Ordering;

/// A mesh that has passed frustum culling and has computed distance information.
#[derive(Copy, Clone)]
pub struct VisibleMesh<'a> {
    pub mesh: &'a ChunkMesh,
    pub center: Vec3,
    pub distance_sq: f32,
}

/// Configurable parameters for horizon culling.
pub struct HorizonCullingConfig {
    /// Number of angular bins around the camera (360 degrees).
    pub bins: usize,
    /// Base margin used to damp popping when slopes are close.
    pub base_margin: f32,
    /// Extra margin per chunk of distance (stabilizes far geometry).
    pub margin_dist_factor: f32,
    /// Minimum distance (in chunks) before horizon culling is considered.
    pub min_dist_chunks: f32,
}

impl Default for HorizonCullingConfig {
    fn default() -> Self {
        Self {
            bins: 128,
            base_margin: 0.1,
            margin_dist_factor: 0.05,
            min_dist_chunks: 2.0,
        }
    }
}

/// Apply horizon culling to a set of visible meshes.
/// Sorts front-to-back and removes meshes hidden behind previously-seen terrain.
pub fn apply_horizon_culling(
    camera_pos: Vec3,
    meshes: &mut Vec<VisibleMesh<'_>>,
    config: &HorizonCullingConfig,
) {
    // Stable front-to-back ordering for consistent results.
    meshes.sort_by(|a, b| {
        a.distance_sq
            .partial_cmp(&b.distance_sq)
            .unwrap_or(Ordering::Equal)
    });

    if meshes.is_empty() {
        return;
    }

    let mut horizon = vec![f32::NEG_INFINITY; config.bins];
    let mut write_idx = 0usize;

    let chunk_size = CHUNK_SIZE as f32;
    let half_chunk = chunk_size * 0.5;

    for i in 0..meshes.len() {
        let mesh = meshes[i];

        let to_center = mesh.center - camera_pos;
        let xz = Vec2::new(to_center.x, to_center.z);
        let dist_xz = xz.length();

        // Too close or degenerate distances are always kept.
        if dist_xz < 1e-3 {
            meshes[write_idx] = mesh;
            write_idx += 1;
            continue;
        }

        let dist_chunks = dist_xz / chunk_size;

        // Keep very close chunks; they do not build or respect horizon.
        if dist_chunks < config.min_dist_chunks {
            meshes[write_idx] = mesh;
            write_idx += 1;
            continue;
        }

        // Angular bin around the camera (wrap to 0..bins).
        let angle = xz.y.atan2(xz.x);
        let bin_f =
            (angle + std::f32::consts::PI) / (2.0 * std::f32::consts::PI) * config.bins as f32;
        let mut bin = bin_f.floor() as isize;
        if bin < 0 {
            bin += config.bins as isize;
        }
        let bin = (bin as usize) % config.bins;

        // Center slope gives stable classification (prevents popping at terrain height).
        let height = mesh.center.y - camera_pos.y;
        let slope = height / dist_xz;

        let margin = config.base_margin * (1.0 + dist_chunks * config.margin_dist_factor);
        let current_horizon = horizon[bin];

        // Cull only when clearly below the established horizon.
        let should_cull = slope >= 0.0 && (slope + margin) < current_horizon;

        if !should_cull {
            // Keep mesh.
            meshes[write_idx] = mesh;
            write_idx += 1;

            // Update horizon using the top of the chunk as the occluder.
            let top_slope = (mesh.center.y + half_chunk - camera_pos.y) / dist_xz;
            if top_slope > current_horizon {
                horizon[bin] = top_slope;
            }
        }
    }

    meshes.truncate(write_idx);
}
