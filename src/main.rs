/// Main application entry point
/// Handles window creation, input, and render loop
use glam::{IVec3, Vec2, Vec3};
use mimalloc::MiMalloc;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;
use std::collections::HashMap;
use std::num::NonZeroU32;
use std::sync::Arc;
use std::time::Instant;
use voxel_engine::*;
use winit::{
    event::*,
    event_loop::{ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::WindowBuilder,
};

fn main() {
    println!("=== Voxel Engine - Software Rasterizer ===");
    println!("Controls:");
    println!("  WASD - Move camera");
    println!("  Space/Shift - Up/Down");
    println!("  Mouse - Look around");
    println!("  ESC - Exit");
    println!();

    // Create event loop and window
    let event_loop = EventLoop::new().unwrap();
    let window = Arc::new(
        WindowBuilder::new()
            .with_title("Voxel Engine")
            .with_inner_size(winit::dpi::LogicalSize::new(1280, 720))
            .build(&event_loop)
            .unwrap(),
    );

    // Initialize software rendering context
    let context = softbuffer::Context::new(window.clone()).unwrap();
    let mut surface = softbuffer::Surface::new(&context, window.clone()).unwrap();

    let window_size = window.inner_size();
    let mut framebuffer =
        Framebuffer::new(window_size.width as usize, window_size.height as usize);
    let mut occlusion_buffer =
        OcclusionBuffer::new(window_size.width as usize, window_size.height as usize, 128, 72);

    // Initialize camera
    let aspect_ratio = window_size.width as f32 / window_size.height as f32;
    let mut camera = Camera::new(Vec3::new(0.0, 10.0, 20.0), aspect_ratio);
    let mut camera_controller = CameraController::new();

    // Initialize rasterizer
    let mut rasterizer = Rasterizer::new();

    // Initialize world with larger view distance
    println!("Initializing world system...");
    let world_config = WorldConfig {
        view_distance: 12, // Much larger world!
        frustum_culling: true,
        max_chunks_per_frame: 16,
    };
    let mut world = World::new(world_config);

    // Pre-generate initial region around spawn
    println!("Generating initial world region...");
    let world_gen_start = Instant::now();
    world.generate_region(IVec3::new(-6, -2, -6), IVec3::new(6, 2, 6));
    println!(
        "World generation: {:.2}ms ({} chunks)",
        world_gen_start.elapsed().as_millis(),
        world.chunk_count()
    );

    // Initial mesh generation with per-chunk cache
    println!("Meshing initial chunks...");
    let mesh_start = Instant::now();
    let frustum = camera.extract_frustum();
    let visible_chunks = world.get_visible_chunks_frustum(camera.position, Some(&frustum));
    let mut mesh_cache: HashMap<IVec3, Option<ChunkMesh>> = HashMap::new();
    if !visible_chunks.is_empty() {
        // Build a small index for neighbour lookup in the initial region.
        let mut index: HashMap<(i32, i32, i32), &Chunk> = HashMap::with_capacity(visible_chunks.len());
        for chunk in &visible_chunks {
            let pos = chunk.position;
            index.insert((pos.x, pos.y, pos.z), *chunk);
        }

        for chunk in &visible_chunks {
            let mesh = BinaryGreedyMesher::mesh_chunk_in_indexed_world(chunk, &index);
            mesh_cache.insert(chunk.position, mesh);
        }
    }
    let initial_mesh_count = mesh_cache.values().filter(|m| m.is_some()).count();
    println!(
        "Initial meshing time: {:.2}ms",
        mesh_start.elapsed().as_millis()
    );
    println!("Generated {} meshes\n", initial_mesh_count);

    // Timing
    let mut last_frame = Instant::now();
    let mut frame_count = 0u32;
    let mut fps_timer = Instant::now();

    // Mouse state
    let mut mouse_captured = false;
    let mut last_mouse_pos: Option<(f64, f64)> = None;

    // Rendering toggles
    let mut occlusion_culling_enabled = false;

    // Persistent tile bins to avoid reallocation every frame
    let mut stripe_bins: Vec<Vec<usize>> = Vec::new();

    event_loop
        .run(move |event, elwt| {
            elwt.set_control_flow(ControlFlow::Poll);

            match event {
                Event::WindowEvent { event, .. } => match event {
                    WindowEvent::CloseRequested => {
                        elwt.exit();
                    }
                    WindowEvent::Resized(new_size) => {
                        framebuffer.resize(new_size.width as usize, new_size.height as usize);
                        occlusion_buffer.resize(new_size.width as usize, new_size.height as usize);
                        camera.set_aspect_ratio(new_size.width as f32 / new_size.height as f32);
                    }
                    WindowEvent::KeyboardInput { event, .. } => {
                        let pressed = event.state == ElementState::Pressed;

                        if let PhysicalKey::Code(keycode) = event.physical_key {
                            match keycode {
                                KeyCode::KeyW => camera_controller.forward_pressed = pressed,
                                KeyCode::KeyS => camera_controller.backward_pressed = pressed,
                                KeyCode::KeyA => camera_controller.left_pressed = pressed,
                                KeyCode::KeyD => camera_controller.right_pressed = pressed,
                                KeyCode::Space => camera_controller.up_pressed = pressed,
                                KeyCode::ShiftLeft => camera_controller.down_pressed = pressed,
                                // Toggle shading on/off to reduce per-pixel work
                                KeyCode::KeyF if pressed => {
                                    rasterizer.enable_shading = !rasterizer.enable_shading;
                                    println!(
                                        "Shading: {}",
                                        if rasterizer.enable_shading {
                                            "ON"
                                        } else {
                                            "OFF (flat colors)"
                                        }
                                    );
                                }
                                // Toggle chunk-level occlusion culling
                                KeyCode::KeyO if pressed => {
                                    occlusion_culling_enabled = !occlusion_culling_enabled;
                                    println!(
                                        "Occlusion culling: {}",
                                        if occlusion_culling_enabled {
                                            "ON"
                                        } else {
                                            "OFF"
                                        }
                                    );
                                }
                                // Adjust view distance at runtime to control mesh count
                                KeyCode::Digit1 if pressed => {
                                    world.set_view_distance(6);
                                    println!("View distance set to 6 chunks");
                                }
                                KeyCode::Digit2 if pressed => {
                                    world.set_view_distance(8);
                                    println!("View distance set to 8 chunks");
                                }
                                KeyCode::Digit3 if pressed => {
                                    world.set_view_distance(12);
                                    println!("View distance set to 12 chunks");
                                }
                                KeyCode::Escape if pressed => {
                                    if mouse_captured {
                                        mouse_captured = false;
                                        let _ = window.set_cursor_visible(true);
                                    } else {
                                        elwt.exit();
                                    }
                                }
                                _ => {}
                            }
                        }
                    }
                    WindowEvent::MouseInput { state, button, .. } => {
                        if button == MouseButton::Left && state == ElementState::Pressed {
                            mouse_captured = true;
                            let _ = window.set_cursor_visible(false);
                        }
                    }
                    WindowEvent::CursorMoved { position, .. } => {
                        if mouse_captured {
                            if let Some(last_pos) = last_mouse_pos {
                                let delta_x = position.x - last_pos.0;
                                let delta_y = position.y - last_pos.1;
                                camera.rotate(delta_x as f32, delta_y as f32);
                            }
                            last_mouse_pos = Some((position.x, position.y));
                        }
                    }
                    WindowEvent::RedrawRequested => {
                        // Calculate delta time
                        let now = Instant::now();
                        let dt = (now - last_frame).as_secs_f32();
                        last_frame = now;

                        // Update camera
                        camera_controller.update_camera(&mut camera, dt);

                        // Update world (generate/unload chunks based on camera position)
                        world.update(camera.position);

                        // Determine visible chunks for current camera + frustum
                        let frustum = camera.extract_frustum();
                        let visible_chunks =
                            world.get_visible_chunks_frustum(camera.position, Some(&frustum));

                        // Identify chunks that need meshing or re-meshing (including neighbours).
                        let mut chunks_to_mesh = Vec::new();
                        let all_chunks = world.get_all_chunks();

                        // 1. Find newly visible chunks that have no mesh yet.
                        for chunk in &visible_chunks {
                            if !mesh_cache.contains_key(&chunk.position) {
                                chunks_to_mesh.push(chunk.position);

                                // 2. If we are generating a mesh for a new chunk,
                                // also re-mesh its neighbours that already have meshes,
                                // because they may have exposed "air faces" that are now internal.
                                let neighbors = [
                                    IVec3::X,
                                    IVec3::NEG_X,
                                    IVec3::Y,
                                    IVec3::NEG_Y,
                                    IVec3::Z,
                                    IVec3::NEG_Z,
                                ];

                                for offset in neighbors {
                                    let neighbor_pos = chunk.position + offset;
                                    if world.contains_chunk(neighbor_pos)
                                        && mesh_cache.contains_key(&neighbor_pos)
                                    {
                                        chunks_to_mesh.push(neighbor_pos);
                                    }
                                }
                            }
                        }

                        // 3. Re-mesh all identified chunks using an indexed neighbour lookup.
                        if !chunks_to_mesh.is_empty() {
                            // Deduplicate positions (sort by coordinates).
                            chunks_to_mesh.sort_by_key(|pos| (pos.x, pos.y, pos.z));
                            chunks_to_mesh.dedup();

                            // Build index for neighbour lookup once.
                            let mut index: HashMap<(i32, i32, i32), &Chunk> =
                                HashMap::with_capacity(all_chunks.len());
                            for chunk in &all_chunks {
                                let pos = chunk.position;
                                index.insert((pos.x, pos.y, pos.z), *chunk);
                            }

                            for pos in chunks_to_mesh {
                                if let Some(chunk) = index.get(&(pos.x, pos.y, pos.z)) {
                                    let mesh =
                                        BinaryGreedyMesher::mesh_chunk_in_indexed_world(chunk, &index);
                                    mesh_cache.insert(pos, mesh);
                                }
                            }
                        }

                        // Drop meshes for chunks that have been unloaded from the world
                        mesh_cache.retain(|pos, _| world.contains_chunk(*pos));

                        // Collect visible meshes with spatial information for culling/sorting
                        let mut visible_meshes = Vec::with_capacity(visible_chunks.len());
                        for chunk in &visible_chunks {
                            if let Some(Some(mesh)) = mesh_cache.get(&chunk.position) {
                                let min = (chunk.position * CHUNK_SIZE as i32).as_vec3();
                                let max = min + Vec3::splat(CHUNK_SIZE as f32);
                                let center = (min + max) * 0.5;
                                let distance_sq =
                                    (center - camera.position).length_squared();
                                visible_meshes.push(VisibleMesh {
                                    mesh,
                                    center,
                                    distance_sq,
                                });
                            }
                        }

                        // Sort visible meshes front-to-back (occlusion handled via depth buffer)
                        apply_horizon_culling_and_sort(camera.position, &mut visible_meshes);
                        // Render frame with chunk-level occlusion.
                        let rendered_meshes = render_frame(
                            &mut framebuffer,
                            &mut rasterizer,
                            &camera,
                            &visible_meshes,
                            &mut occlusion_buffer,
                            occlusion_culling_enabled,
                            &mut stripe_bins,
                        );

                        // Copy framebuffer to window
                        surface
                            .resize(
                                NonZeroU32::new(framebuffer.width as u32).unwrap(),
                                NonZeroU32::new(framebuffer.height as u32).unwrap(),
                            )
                            .unwrap();

                        let mut buffer = surface.buffer_mut().unwrap();
                        buffer.copy_from_slice(framebuffer.color_buffer_slice());
                        buffer.present().unwrap();

                        // FPS counter with additional stats
                        frame_count += 1;
                        if fps_timer.elapsed().as_secs() >= 1 {
                            println!(
                                "FPS: {} | Chunks: {} | Meshes: {}",
                                frame_count,
                                world.chunk_count(),
                                rendered_meshes
                            );
                            frame_count = 0;
                            fps_timer = Instant::now();
                        }
                    }
                    _ => {}
                },
                Event::AboutToWait => {
                    window.request_redraw();
                }
                _ => {}
            }
        })
        .unwrap();
}

#[derive(Copy, Clone)]
struct VisibleMesh<'a> {
    mesh: &'a ChunkMesh,
    center: Vec3,
    distance_sq: f32,
}

#[derive(Copy, Clone)]
struct ProjectedMesh<'a> {
    info: VisibleMesh<'a>,
    rect_min_x: i32,
    rect_min_y: i32,
    rect_max_x: i32,
    rect_max_y: i32,
    near_depth: f32,
    use_occlusion: bool,
}

/// Sort meshes front-to-back. Any occlusion is handled later via the
/// screen-space occlusion buffer in `render_frame`.
fn apply_horizon_culling_and_sort(_camera_pos: Vec3, meshes: &mut Vec<VisibleMesh<'_>>) {
    use std::cmp::Ordering;

    // Always sort from near to far so the rasterizer gets front-to-back ordering.
    meshes.sort_by(|a, b| {
        a.distance_sq
            .partial_cmp(&b.distance_sq)
            .unwrap_or(Ordering::Equal)
    });
}

fn render_frame(
    framebuffer: &mut Framebuffer,
    rasterizer: &mut Rasterizer,
    camera: &Camera,
    visible_meshes: &[VisibleMesh<'_>],
    occlusion: &mut OcclusionBuffer,
    enable_occlusion_culling: bool,
    stripe_bins: &mut Vec<Vec<usize>>,
) -> usize {
    use rayon::prelude::*;

    let frame_start = Instant::now();

    // Clear buffers
    framebuffer.clear(0xFF87CEEB); // Sky blue
    occlusion.clear();

    // Get view-projection matrix
    let view_proj = camera.view_projection_matrix();

    let half_size = CHUNK_SIZE as f32 * 0.5;

    let width = framebuffer.width as f32;
    let height = framebuffer.height as f32;

    // --- 1. PROJECTION PASS ---
    let mut projected: Vec<ProjectedMesh<'_>> = visible_meshes
        .par_iter()
        .filter_map(|info| {
            let center = info.center;
            let min = center - Vec3::splat(half_size);
            let max = center + Vec3::splat(half_size);

            let corners = [
                Vec3::new(min.x, min.y, min.z),
                Vec3::new(max.x, min.y, min.z),
                Vec3::new(min.x, max.y, min.z),
                Vec3::new(max.x, max.y, min.z),
                Vec3::new(min.x, min.y, max.z),
                Vec3::new(max.x, min.y, max.z),
                Vec3::new(min.x, max.y, max.z),
                Vec3::new(max.x, max.y, max.z),
            ];

            let mut rect_min_x = i32::MAX;
            let mut rect_min_y = i32::MAX;
            let mut rect_max_x = i32::MIN;
            let mut rect_max_y = i32::MIN;
            let mut near_depth = f32::INFINITY;

            // Detect if any corner lands behind the near plane; projection becomes unstable.
            let mut any_corner_behind = false;

            for corner in &corners {
                let clip = view_proj * corner.extend(1.0);

                if clip.w <= 0.001 {
                    any_corner_behind = true;
                }

                if clip.w > 0.001 {
                    let ndc = clip / clip.w;
                    near_depth = near_depth.min(ndc.z);

                    let sx = (ndc.x + 1.0) * 0.5 * width;
                    let sy = (1.0 - ndc.y) * 0.5 * height;

                    rect_min_x = rect_min_x.min(sx.floor() as i32);
                    rect_max_x = rect_max_x.max(sx.ceil() as i32);
                    rect_min_y = rect_min_y.min(sy.floor() as i32);
                    rect_max_y = rect_max_y.max(sy.ceil() as i32);
                }
            }

            if any_corner_behind {
                rect_min_x = 0;
                rect_min_y = 0;
                rect_max_x = width as i32 - 1;
                rect_max_y = height as i32 - 1;
                near_depth = 0.0;
            } else {
                if near_depth.is_infinite() || near_depth > 1.0 {
                    return None;
                }

                rect_min_x = rect_min_x.max(0);
                rect_min_y = rect_min_y.max(0);
                rect_max_x = rect_max_x.min(width as i32 - 1);
                rect_max_y = rect_max_y.min(height as i32 - 1);

                if rect_min_x > rect_max_x || rect_min_y > rect_max_y {
                    return None;
                }
            }

            const OCCLUSION_MIN_DISTANCE_CHUNKS: f32 = 2.0;
            let occlusion_min_distance_sq =
                (CHUNK_SIZE as f32 * OCCLUSION_MIN_DISTANCE_CHUNKS).powi(2);
            let use_occlusion = enable_occlusion_culling
                && info.distance_sq >= occlusion_min_distance_sq;

            Some(ProjectedMesh {
                info: *info,
                rect_min_x,
                rect_min_y,
                rect_max_x,
                rect_max_y,
                near_depth,
                use_occlusion,
            })
        })
        .collect();

    // Sort front-to-back
    use std::cmp::Ordering;
    projected.sort_by(|a, b| {
        a.near_depth
            .partial_cmp(&b.near_depth)
            .unwrap_or(Ordering::Equal)
    });

    // --- 2. OCCLUSION PASS ---
    let mut survivors: Vec<ProjectedMesh<'_>> = Vec::with_capacity(projected.len());

    for proj in &projected {
        if proj.use_occlusion
            && occlusion.is_occluded(
                proj.rect_min_x,
                proj.rect_min_y,
                proj.rect_max_x,
                proj.rect_max_y,
                proj.near_depth,
            )
        {
            continue;
        }

        if enable_occlusion_culling {
            occlusion.mark_rect(
                proj.rect_min_x,
                proj.rect_min_y,
                proj.rect_max_x,
                proj.rect_max_y,
                proj.near_depth,
            );
        }
        survivors.push(*proj);
    }

    // --- 3. STRIPE BINNING PASS ---
    // Use 1D Y-axis binning instead of 2D tiles to reduce setup redundancy
    // A mesh spanning horizontally might cover 10 tiles but only 1-2 stripes
    let thread_count = rayon::current_num_threads();
    let stripe_count = thread_count * 4; // Over-subscribe for load balancing
    let height = framebuffer.height;
    let stripe_h = (height + stripe_count - 1) / stripe_count;

    // Resize and clear bins without deallocating inner Vec capacities
    let required_bins = stripe_count;
    if stripe_bins.len() < required_bins {
        stripe_bins.resize(required_bins, Vec::new());
    }
    for bin in stripe_bins.iter_mut().take(required_bins) {
        bin.clear();
    }

    // Simple 1D binning by Y coordinate
    for (mesh_idx, proj) in survivors.iter().enumerate() {
        let start_stripe = (proj.rect_min_y as usize) / stripe_h;
        let end_stripe = (proj.rect_max_y as usize) / stripe_h;

        // Clamp to valid range
        let start = start_stripe.min(stripe_count - 1);
        let end = end_stripe.min(stripe_count - 1);

        for s in start..=end {
            stripe_bins[s].push(mesh_idx);
        }
    }

    // --- 4. PARALLEL STRIPE RENDERING ---
    let enable_shading = rasterizer.enable_shading;
    let backface_culling = rasterizer.backface_culling;
    let shading_config = rasterizer.shading;
    let atlas = rasterizer.atlas.clone();

    // Split framebuffer into horizontal stripes
    let slices = framebuffer.split_into_stripes(stripe_count);

    // CRITICAL OPTIMIZATION:
    // 1. Zip slices with their bins
    // 2. Filter out empty bins on the main thread (cheap)
    // 3. Collect into a "dense" list of actual work
    let work_items: Vec<_> = slices
        .into_iter()
        .zip(stripe_bins.iter())
        .filter(|(_, bin)| !bin.is_empty())
        .collect();

    // OPTIMIZATION: Fine-grained parallelism with resource reuse
    // Stripes reduce setup redundancy compared to tiles (mesh processed 1-2x instead of 10-20x)
    // for_each_init ensures we only create one Rasterizer per thread
    work_items.into_par_iter().for_each_init(
        // Init: Create Rasterizer ONCE per thread
        || {
            let mut r = Rasterizer::new_with_atlas(atlas.clone());
            r.enable_shading = enable_shading;
            r.backface_culling = backface_culling;
            r.shading = shading_config;
            r
        },
        // Work: Process a SINGLE stripe (Rayon work-stealing handles load balancing)
        |local_rasterizer, (mut slice, bin)| {
            for &mesh_idx in bin {
                let mesh_data = &survivors[mesh_idx];
                local_rasterizer.render_mesh_into_slice(mesh_data.info.mesh, &view_proj, &mut slice);
            }
        },
    );

    let frame_time = frame_start.elapsed();
    if frame_time.as_millis() > 16 {
        println!(
            "[WARN] Frame time: {:.2}ms (> 16ms)",
            frame_time.as_millis()
        );
    }

    survivors.len()
}
