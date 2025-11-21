// Reproduction tests for SpanWalkerRasterizer bugs
//
// Bug 1: Fractional start_y causes quads to be skipped
// Bug 2: Vertical gaps between quads cause early loop termination

use voxel_engine::rendering::framebuffer::Framebuffer;
use voxel_engine::rendering::span_walker::SpanWalkerRasterizer;
use voxel_engine::rendering::differential_projection::ProjectedPacket;

#[test]
fn test_fractional_start_y_bug() {
    // Bug 1: If a quad's top edge is at fractional Y (e.g., 10.9),
    // the rasterizer starts at floor(10.9) = 10, but update_active_mask(10.0)
    // sees 10.0 < 10.9 and doesn't activate the quad.
    // The loop exits immediately, skipping the quad entirely.

    let width = 200;
    let height = 200;
    let mut fb = Framebuffer::new(width, height);

    // Create a packet with a quad that has fractional Y coordinates
    let mut projected = ProjectedPacket::new();
    projected.count = 1;

    // This quad should cover scanlines 11-14 (from Y=10.9 to Y=15.0)
    // In NDC, map to screen space such that we get fractional coordinates
    // NDC -0.892 maps to screen Y ~10.9 in 200-pixel viewport
    // NDC -0.85 maps to screen Y ~15.0
    projected.screen_x_min[0] = -0.5;
    projected.screen_y_min[0] = -0.892; // Fractional screen Y
    projected.screen_x_max[0] = 0.5;
    projected.screen_y_max[0] = -0.85;
    projected.depth_near[0] = 0.5;
    projected.block_type[0] = 1;
    projected.visibility_mask = 1;

    let span_walker = SpanWalkerRasterizer::new(width as u32, height as u32);
    let mut slice = fb.as_full_slice_mut();
    span_walker.rasterize_projected_packet(&projected, &mut slice);

    // Count filled pixels - should be non-zero
    let filled_pixels = fb.color_buffer_slice().iter().filter(|&&p| p != 0).count();

    assert!(
        filled_pixels > 0,
        "Bug reproduced: Quad with fractional start_y was skipped. Expected pixels, got {}",
        filled_pixels
    );
}

#[test]
fn test_vertical_gap_bug() {
    // Bug 2: Two quads separated vertically.
    // Quad A: Y 10..15
    // Quad B: Y 20..25
    // Loop runs 10..14. At Y=15, quad A finishes, mask becomes 0.
    // Loop terminates. Quad B (starting at 20) is never reached.

    let width = 200;
    let height = 200;
    let mut fb = Framebuffer::new(width, height);

    let mut projected = ProjectedPacket::new();
    projected.count = 2;

    // Quad A: top of screen (Y ~10-15)
    projected.screen_x_min[0] = -0.5;
    projected.screen_y_min[0] = -0.9;  // Y ~10
    projected.screen_x_max[0] = 0.5;
    projected.screen_y_max[0] = -0.85; // Y ~15
    projected.depth_near[0] = 0.5;
    projected.block_type[0] = 1;

    // Quad B: separated vertically (Y ~20-25)
    projected.screen_x_min[1] = -0.5;
    projected.screen_y_min[1] = -0.8;  // Y ~20
    projected.screen_x_max[1] = 0.5;
    projected.screen_y_max[1] = -0.75; // Y ~25
    projected.depth_near[1] = 0.5;
    projected.block_type[1] = 2;

    projected.visibility_mask = 0b11; // Both visible

    let span_walker = SpanWalkerRasterizer::new(width as u32, height as u32);
    let mut slice = fb.as_full_slice_mut();
    span_walker.rasterize_projected_packet(&projected, &mut slice);

    // Count filled pixels - both quads should render
    let filled_pixels = fb.color_buffer_slice().iter().filter(|&&p| p != 0).count();

    // Each quad should fill roughly 100x5 = 500 pixels
    // Total ~1000 pixels (allowing tolerance)
    assert!(
        filled_pixels >= 500,
        "Bug reproduced: Second quad with vertical gap was skipped. Expected ~1000 pixels, got {}",
        filled_pixels
    );
}

#[test]
fn test_combined_fractional_and_gap() {
    // Combined test: Multiple quads with fractional coordinates AND gaps
    let width = 200;
    let height = 200;
    let mut fb = Framebuffer::new(width, height);

    let mut projected = ProjectedPacket::new();
    projected.count = 3;

    // Three quads with different vertical positions and fractional coordinates
    for i in 0..3 {
        let y_offset = i as f32 * 0.15; // Create vertical separation
        projected.screen_x_min[i] = -0.4;
        projected.screen_y_min[i] = -0.9 + y_offset;
        projected.screen_x_max[i] = 0.4;
        projected.screen_y_max[i] = -0.87 + y_offset;
        projected.depth_near[i] = 0.5;
        projected.block_type[i] = (i + 1) as u8;
    }

    projected.visibility_mask = 0b111; // All visible

    let span_walker = SpanWalkerRasterizer::new(width as u32, height as u32);
    let mut slice = fb.as_full_slice_mut();
    span_walker.rasterize_projected_packet(&projected, &mut slice);

    let filled_pixels = fb.color_buffer_slice().iter().filter(|&&p| p != 0).count();

    assert!(
        filled_pixels >= 100,
        "Bug reproduced: Quads with fractional Y and gaps were skipped. Expected pixels, got {}",
        filled_pixels
    );
}
