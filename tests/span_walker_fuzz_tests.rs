//! Differential Fuzzing Tests for the SpanWalkerRasterizer
// ... (rest of the file content from the previous turn, but with the test logic updated)
use voxel_engine::camera::Camera;
use voxel_engine::meshing::face_packets::ChunkFacePackets;
use voxel_engine::meshing::{BinaryGreedyMesher, ChunkMesh, FaceDir};
use voxel_engine::rendering::{Framebuffer, PacketPipeline, SpanWalkerRasterizer};
use voxel_engine::voxel::{BlockData, BlockType, Chunk, CHUNK_SIZE};
use glam::{vec3, IVec3, Mat4, Vec2, Vec3, Vec4};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

const SCREEN_WIDTH: usize = 128;
const SCREEN_HEIGHT: usize = 128;

// --- Helper Functions for the new "Oracle" Rasterizer ---

/// A simple vertex for our simple rasterizer.
#[derive(Copy, Clone)]
struct SimpleVertex {
    pos: Vec4, // Clip space
}

fn ndc_to_screen(ndc: Vec3) -> Vec2 {
    Vec2::new(
        (ndc.x + 1.0) * 0.5 * SCREEN_WIDTH as f32,
        (1.0 - ndc.y) * 0.5 * SCREEN_HEIGHT as f32,
    )
}

fn edge_function(a: Vec2, b: Vec2, c: Vec2) -> f32 {
    (c.x - a.x) * (b.y - a.y) - (c.y - a.y) * (b.x - a.x)
}

/// Simple, slow, but correct barycentric triangle rasterizer. This is our "Oracle".
fn rasterize_triangle_oracle(
    v0: SimpleVertex,
    v1: SimpleVertex,
    v2: SimpleVertex,
    color: u32,
    fb: &mut Framebuffer,
) {
    if color == 0 { return; } // Skip air blocks

    // Perspective divide
    let p0_ndc = v0.pos / v0.pos.w;
    let p1_ndc = v1.pos / v1.pos.w;
    let p2_ndc = v2.pos / v2.pos.w;

    // Viewport transform
    let p0_screen = ndc_to_screen(p0_ndc.truncate());
    let p1_screen = ndc_to_screen(p1_ndc.truncate());
    let p2_screen = ndc_to_screen(p2_ndc.truncate());

    // Bounding box
    let min_x = p0_screen.x.min(p1_screen.x).min(p2_screen.x).floor() as i32;
    let max_x = p0_screen.x.max(p1_screen.x).max(p2_screen.x).ceil() as i32;
    let min_y = p0_screen.y.min(p1_screen.y).min(p2_screen.y).floor() as i32;
    let max_y = p0_screen.y.max(p1_screen.y).max(p2_screen.y).ceil() as i32;

    // Clip to framebuffer
    let min_x = min_x.max(0);
    let max_x = max_x.min(SCREEN_WIDTH as i32 - 1);
    let min_y = min_y.max(0);
    let max_y = max_y.min(SCREEN_HEIGHT as i32 - 1);

    let area = edge_function(p0_screen, p1_screen, p2_screen);
    if area <= 0.0 { return; } // Backface or degenerate

    for y in min_y..=max_y {
        for x in min_x..=max_x {
            let p = Vec2::new(x as f32 + 0.5, y as f32 + 0.5);
            let mut w0 = edge_function(p1_screen, p2_screen, p);
            let mut w1 = edge_function(p2_screen, p0_screen, p);
            let mut w2 = edge_function(p0_screen, p1_screen, p);

            if w0 >= 0.0 && w1 >= 0.0 && w2 >= 0.0 {
                w0 /= area;
                w1 /= area;
                w2 /= area;

                let depth = w0 * p0_ndc.z + w1 * p1_ndc.z + w2 * p2_ndc.z;
                fb.set_pixel(x as usize, y as usize, color, depth);
            }
        }
    }
}
// --- End Oracle Helpers ---

fn setup_scene(rng: &mut ChaCha8Rng) -> (Chunk, Camera, Mat4) {
    let chunk_pos = IVec3::ZERO;
    let mut chunk = Chunk::uniform(chunk_pos, BlockType::Air);
    for y in 0..CHUNK_SIZE {
        for z in 0..CHUNK_SIZE {
            for x in 0..CHUNK_SIZE {
                let height = (x as f32 / CHUNK_SIZE as f32 * 10.0).sin() * 2.0
                    + (z as f32 / CHUNK_SIZE as f32 * 10.0).cos() * 2.0
                    + 8.0;
                if (y as f32) < height {
                    let block_type = BlockType::from_u8(rng.gen_range(1..=3));
                    chunk.set_block(x, y, z, BlockData::new(block_type));
                }
            }
        }
    }
    let camera_pos = vec3(
        CHUNK_SIZE as f32 * 0.5,
        CHUNK_SIZE as f32 * 1.5,
        CHUNK_SIZE as f32 * 0.5,
    );
    let mut camera = Camera::new(camera_pos, SCREEN_WIDTH as f32 / SCREEN_HEIGHT as f32);
    camera.look_at(vec3(16.0, 8.0, 16.0), Vec3::Y);
    let view_proj = camera.view_projection_matrix();
    (chunk, camera, view_proj)
}

fn render_golden_image(mesh: &ChunkMesh, view_proj: &Mat4) -> Framebuffer {
    let mut framebuffer = Framebuffer::new(SCREEN_WIDTH, SCREEN_HEIGHT);
    let chunk_offset = mesh.chunk_position.as_vec3();

    for face_dir_idx in 0..6 {
        let face_dir = FaceDir::from_index(face_dir_idx);
        let face_list = &mesh.faces[face_dir_idx];

        for (slice_idx, quads) in face_list.slice_quads.iter().enumerate() {
             let slice_pos = if face_dir.is_positive() {
                    (slice_idx + 1) as u8
                } else {
                    slice_idx as u8
                };

            for quad in quads {
                let u = quad.u();
                let v = quad.v();
                let w = quad.width();
                let h = quad.height();
                
                let local_positions = face_dir.get_quad_local_positions(slice_pos, u, v, w, h);
                
                let mut clip_verts = [SimpleVertex { pos: Vec4::ZERO }; 4];
                for i in 0..4 {
                    let world_pos = chunk_offset + local_positions[i];
                    clip_verts[i] = SimpleVertex { pos: *view_proj * world_pos.extend(1.0) };
                }

                let color = BlockType::from_u8(quad.block_type()).color();
                let color_u32 = 0xFF000000 | ((color[0] as u32) << 16) | ((color[1] as u32) << 8) | (color[2] as u32);
                
                // Triangle 1
                rasterize_triangle_oracle(clip_verts[0], clip_verts[1], clip_verts[2], color_u32, &mut framebuffer);
                // Triangle 2
                rasterize_triangle_oracle(clip_verts[0], clip_verts[2], clip_verts[3], color_u32, &mut framebuffer);
            }
        }
    }
    framebuffer
}

fn render_test_image(mesh: &ChunkMesh, view_proj: &Mat4, _camera: &Camera) -> Framebuffer {
    let mut framebuffer = Framebuffer::new(SCREEN_WIDTH, SCREEN_HEIGHT);
    let mut packet_pipeline = PacketPipeline::new();

    let face_packets = ChunkFacePackets::from_chunk_mesh(mesh);
    let chunk_pos = mesh.chunk_position;
    packet_pipeline.process_chunk_packets(&face_packets, chunk_pos, view_proj);

    let span_walker = SpanWalkerRasterizer::new(SCREEN_WIDTH as u32, SCREEN_HEIGHT as u32);
    let mut slice = framebuffer.as_full_slice_mut();

    for packet in packet_pipeline.projected_packets() {
        span_walker.rasterize_projected_packet(packet, &mut slice);
    }
    framebuffer
}

#[test]
fn test_span_walker_vs_triangle_rasterizer_fuzz() {
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let (chunk, camera, view_proj) = setup_scene(&mut rng);

    let neighbors: Vec<&Chunk> = vec![&chunk];
    let mesh = BinaryGreedyMesher::mesh_chunk_in_world(&chunk, &neighbors)
        .expect("meshing should succeed");

    let fb_golden = render_golden_image(&mesh, &view_proj);
    let fb_test = render_test_image(&mesh, &view_proj, &camera);

    let golden_colors = fb_golden.color_buffer_slice();
    let test_colors = fb_test.color_buffer_slice();
    let golden_depth = &fb_golden.depth_buffer;
    let test_depth = &fb_test.depth_buffer;

    let mut mismatches = 0;
    for y in 0..SCREEN_HEIGHT {
        for x in 0..SCREEN_WIDTH {
            let idx = y * SCREEN_WIDTH + x;
            if golden_colors[idx] != test_colors[idx] || (golden_depth[idx] - test_depth[idx]).abs() > 1e-5 {
                if mismatches < 10 {
                    println!(
                        "Mismatch at ({}, {}): Golden(C:{:08X}, D:{:.5}) != Test(C:{:08X}, D:{:.5})",
                        x, y, golden_colors[idx], golden_depth[idx], test_colors[idx], test_depth[idx]
                    );
                }
                mismatches += 1;
            }
        }
    }
    if mismatches > 0 {
        panic!("{} pixel mismatches found between golden and test images.", mismatches);
    }
}