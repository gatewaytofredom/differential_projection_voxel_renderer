/// World management system with view distance culling
/// Handles chunk generation, caching, and visibility determination
use crate::camera::Frustum;
use crate::voxel::{Chunk, CHUNK_SIZE};
use glam::{IVec3, Vec3};
use std::collections::HashMap;

/// World configuration parameters
#[derive(Debug, Clone)]
pub struct WorldConfig {
    /// View distance in chunks (radius from camera position)
    pub view_distance: i32,
    /// Whether to enable frustum culling (in addition to distance culling)
    pub frustum_culling: bool,
    /// Maximum number of chunks to generate per frame (for async generation)
    pub max_chunks_per_frame: usize,
}

impl Default for WorldConfig {
    fn default() -> Self {
        Self {
            view_distance: 8,
            frustum_culling: true,
            max_chunks_per_frame: 4,
        }
    }
}

/// Manages the voxel world with dynamic chunk loading
pub struct World {
    /// All generated chunks, indexed by chunk position
    chunks: HashMap<IVec3, Chunk>,
    /// Configuration
    config: WorldConfig,
    /// Last camera chunk position for change detection
    last_camera_chunk: Option<IVec3>,
}

impl World {
    pub fn new(config: WorldConfig) -> Self {
        Self {
            chunks: HashMap::new(),
            config,
            last_camera_chunk: None,
        }
    }

    /// Get chunk at position, generating if needed
    pub fn get_or_generate_chunk(&mut self, chunk_pos: IVec3) -> &Chunk {
        self.chunks
            .entry(chunk_pos)
            .or_insert_with(|| Chunk::generate_terrain(chunk_pos))
    }

    /// Update world based on camera position
    /// Returns true if chunks changed (needs remeshing)
    pub fn update(&mut self, camera_position: Vec3) -> bool {
        let camera_chunk = world_to_chunk_pos(camera_position);

        self.last_camera_chunk = Some(camera_chunk);

        // Generate chunks within view distance
        let mut chunks_generated = 0;
        let view_distance = self.config.view_distance;

        for cx in (camera_chunk.x - view_distance)..=(camera_chunk.x + view_distance) {
            for cy in (camera_chunk.y - view_distance)..=(camera_chunk.y + view_distance) {
                for cz in (camera_chunk.z - view_distance)..=(camera_chunk.z + view_distance) {
                    let chunk_pos = IVec3::new(cx, cy, cz);

                    // Distance check (spherical view distance)
                    let dist_sq = (chunk_pos - camera_chunk).length_squared() as f32;
                    if dist_sq > (view_distance * view_distance) as f32 {
                        continue;
                    }

                    // Generate if not exists
                    if !self.chunks.contains_key(&chunk_pos) {
                        self.chunks
                            .insert(chunk_pos, Chunk::generate_terrain(chunk_pos));
                        chunks_generated += 1;

                        // Limit chunks generated per frame
                        if chunks_generated >= self.config.max_chunks_per_frame {
                            return true;
                        }
                    }
                }
            }
        }

        // Unload chunks outside view distance (with hysteresis)
        let unload_distance = view_distance + 2;
        self.chunks.retain(|&pos, _| {
            let dist_sq = (pos - camera_chunk).length_squared() as f32;
            dist_sq <= (unload_distance * unload_distance) as f32
        });

        chunks_generated > 0
    }

    /// Get all chunks within view distance of camera (no frustum culling)
    pub fn get_visible_chunks(&self, camera_position: Vec3) -> Vec<&Chunk> {
        let camera_chunk = world_to_chunk_pos(camera_position);
        let view_distance_sq = (self.config.view_distance * self.config.view_distance) as f32;

        self.chunks
            .values()
            .filter(|chunk| {
                let dist_sq = (chunk.position - camera_chunk).length_squared() as f32;
                dist_sq <= view_distance_sq
            })
            .collect()
    }

    /// Get all chunks visible from camera with optional frustum culling
    /// This is the recommended method for rendering
    pub fn get_visible_chunks_frustum(
        &self,
        camera_position: Vec3,
        frustum: Option<&Frustum>,
    ) -> Vec<&Chunk> {
        let camera_chunk = world_to_chunk_pos(camera_position);
        let view_distance_sq = (self.config.view_distance * self.config.view_distance) as f32;

        self.chunks
            .values()
            .filter(|chunk| {
                // Distance culling (sphere)
                let dist_sq = (chunk.position - camera_chunk).length_squared() as f32;
                if dist_sq > view_distance_sq {
                    return false;
                }

                // Frustum culling (if enabled and provided)
                if self.config.frustum_culling {
                    if let Some(frustum) = frustum {
                        let (min, max) = chunk_bounds(chunk.position);
                        return frustum.intersects_aabb(min, max);
                    }
                }

                true
            })
            .collect()
    }

    /// Get all chunks (for benchmarking/testing)
    pub fn get_all_chunks(&self) -> Vec<&Chunk> {
        self.chunks.values().collect()
    }

    /// Get chunk count
    pub fn chunk_count(&self) -> usize {
        self.chunks.len()
    }

    /// Pre-generate a region of chunks (for testing/benchmarking)
    pub fn generate_region(&mut self, min: IVec3, max: IVec3) {
        for cx in min.x..=max.x {
            for cy in min.y..=max.y {
                for cz in min.z..=max.z {
                    let chunk_pos = IVec3::new(cx, cy, cz);
                    self.chunks
                        .entry(chunk_pos)
                        .or_insert_with(|| Chunk::generate_terrain(chunk_pos));
                }
            }
        }
    }

    /// Returns true if a chunk exists at the given position.
    pub fn contains_chunk(&self, position: IVec3) -> bool {
        self.chunks.contains_key(&position)
    }

    /// Get configuration
    pub fn config(&self) -> &WorldConfig {
        &self.config
    }

    /// Update view distance at runtime (in chunks).
    pub fn set_view_distance(&mut self, view_distance: i32) {
        self.config.view_distance = view_distance.max(1);
    }

    /// Current view distance in chunks.
    pub fn view_distance(&self) -> i32 {
        self.config.view_distance
    }

    /// Clear all chunks
    pub fn clear(&mut self) {
        self.chunks.clear();
        self.last_camera_chunk = None;
    }
}

/// Convert world position to chunk position
#[inline]
pub fn world_to_chunk_pos(world_pos: Vec3) -> IVec3 {
    IVec3::new(
        (world_pos.x / CHUNK_SIZE as f32).floor() as i32,
        (world_pos.y / CHUNK_SIZE as f32).floor() as i32,
        (world_pos.z / CHUNK_SIZE as f32).floor() as i32,
    )
}

/// Get chunk bounds in world space
#[inline]
pub fn chunk_bounds(chunk_pos: IVec3) -> (Vec3, Vec3) {
    let min = (chunk_pos * CHUNK_SIZE as i32).as_vec3();
    let max = min + Vec3::splat(CHUNK_SIZE as f32);
    (min, max)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn test_world_to_chunk_pos() {
        assert_eq!(world_to_chunk_pos(Vec3::ZERO), IVec3::ZERO);
        assert_eq!(world_to_chunk_pos(Vec3::new(16.0, 16.0, 16.0)), IVec3::ZERO);
        assert_eq!(
            world_to_chunk_pos(Vec3::new(32.0, 32.0, 32.0)),
            IVec3::new(1, 1, 1)
        );
        assert_eq!(
            world_to_chunk_pos(Vec3::new(-1.0, -1.0, -1.0)),
            IVec3::new(-1, -1, -1)
        );
    }

    #[test]
    fn test_world_generation() {
        let mut world = World::new(WorldConfig {
            view_distance: 2,
            ..Default::default()
        });

        world.generate_region(IVec3::ZERO, IVec3::new(2, 2, 2));
        assert_eq!(world.chunk_count(), 27); // 3x3x3
    }

    #[test]
    fn test_view_distance_culling() {
        let mut world = World::new(WorldConfig {
            view_distance: 1,
            ..Default::default()
        });

        // Generate larger region
        world.generate_region(IVec3::new(-5, -5, -5), IVec3::new(5, 5, 5));
        let total_chunks = world.chunk_count();

        // Get visible chunks from center
        let visible = world.get_visible_chunks(Vec3::ZERO);

        // With view distance 1, we should see ~7 chunks (center + 6 neighbors in sphere)
        assert!(visible.len() < total_chunks);
        assert!(visible.len() >= 7);

        // Increasing view distance should not reduce visible chunk count.
        let before = visible.len();
        world.set_view_distance(2);
        let visible_expanded = world.get_visible_chunks(Vec3::ZERO);
        assert!(
            visible_expanded.len() >= before,
            "increasing view distance should not reduce visible chunk count"
        );
    }

    #[test]
    fn test_update_streams_chunks_until_view_filled() {
        let config = WorldConfig {
            view_distance: 2,
            frustum_culling: false,
            max_chunks_per_frame: 3,
        };

        let mut world = World::new(config);
        let camera_pos = Vec3::ZERO;
        let camera_chunk = world_to_chunk_pos(camera_pos);
        let view_distance = world.config().view_distance;

        // Compute the set of chunk positions that should exist within view distance.
        let mut expected = HashSet::new();
        for cx in (camera_chunk.x - view_distance)..=(camera_chunk.x + view_distance) {
            for cy in (camera_chunk.y - view_distance)..=(camera_chunk.y + view_distance) {
                for cz in (camera_chunk.z - view_distance)..=(camera_chunk.z + view_distance) {
                    let pos = IVec3::new(cx, cy, cz);
                    let dist_sq = (pos - camera_chunk).length_squared() as f32;
                    if dist_sq <= (view_distance * view_distance) as f32 {
                        expected.insert((pos.x, pos.y, pos.z));
                    }
                }
            }
        }

        let max_per_frame = world.config().max_chunks_per_frame;
        assert!(max_per_frame > 0, "max_chunks_per_frame must be > 0 for streaming");

        // Call update enough times to allow streaming to fill the view sphere.
        let iterations = (expected.len() + max_per_frame - 1) / max_per_frame + 1;
        for _ in 0..iterations {
            world.update(camera_pos);
        }

        let actual: HashSet<(i32, i32, i32)> = world
            .get_all_chunks()
            .iter()
            .map(|chunk| {
                let pos = chunk.position;
                (pos.x, pos.y, pos.z)
            })
            .collect();

        assert_eq!(
            actual, expected,
            "world.update should eventually generate all chunks within view distance"
        );
    }
}
