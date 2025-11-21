// Reproduction test for triangle rasterizer horizontal gap bug
//
// This test demonstrates the sub-pixel precision bug in render_triangle_span_from_clip
// where triangles with vertical bounds that fall between integer scanlines are skipped.

#[test]
fn test_scanline_range_calculation() {
    // This test directly validates the scanline range calculation logic

    // Case 1: Triangle spanning Y 10.1 to 10.9
    let min_y = 10.1f32;
    let max_y = 10.9f32;

    // Buggy logic (OLD)
    let start_buggy = min_y.ceil() as i32;   // 11
    let end_buggy = max_y.floor() as i32;    // 10
    // Range 11..=10 is empty - BUG!

    // Correct logic (NEW) - pixel center sampling
    // We want all y where min_y <= y + 0.5 <= max_y
    let start_correct = (min_y - 0.5).ceil() as i32;  // ceil(9.6) = 10
    let end_correct = (max_y - 0.5).floor() as i32;   // floor(10.4) = 10
    // Range 10..=10 contains scanline 10 - CORRECT!

    assert!(
        start_correct <= end_correct,
        "Correct logic should produce a valid range for Y [10.1, 10.9]"
    );

    assert!(
        start_buggy > end_buggy,
        "Buggy logic produces an invalid empty range for Y [10.1, 10.9]"
    );

    // Verify pixel center is inside the triangle bounds
    let pixel_center_y = start_correct as f32 + 0.5;  // 10.5
    assert!(
        pixel_center_y >= min_y && pixel_center_y <= max_y,
        "Pixel center {} should be inside triangle bounds [{}, {}]",
        pixel_center_y, min_y, max_y
    );

    // Case 2: Triangle spanning Y 10.0 to 11.0 (exact integer bounds)
    let min_y2 = 10.0f32;
    let max_y2 = 11.0f32;

    let start2 = (min_y2 - 0.5).ceil() as i32;  // ceil(9.5) = 10
    let end2 = (max_y2 - 0.5).floor() as i32;   // floor(10.5) = 10
    // Should cover scanline 10 (pixel center at 10.5)

    assert_eq!(start2, 10, "Should start at scanline 10");
    assert_eq!(end2, 10, "Should end at scanline 10");

    // Case 3: Triangle spanning Y 10.0 to 10.4 (less than 1 pixel)
    let min_y3 = 10.0f32;
    let max_y3 = 10.4f32;

    let start3 = (min_y3 - 0.5).ceil() as i32;  // ceil(9.5) = 10
    let end3 = (max_y3 - 0.5).floor() as i32;   // floor(9.9) = 9
    // Range 10..=9 is empty - correctly skip (pixel center 10.5 is outside [10.0, 10.4])

    assert!(
        start3 > end3,
        "Triangle not covering any pixel center should produce empty range"
    );

    // Case 4: Triangle spanning Y 10.6 to 11.0
    let min_y4 = 10.6f32;
    let max_y4 = 11.0f32;

    let start4 = (min_y4 - 0.5).ceil() as i32;  // ceil(10.1) = 11
    let end4 = (max_y4 - 0.5).floor() as i32;   // floor(10.5) = 10
    // Range 11..=10 is empty - correctly skip (pixel center 10.5 is outside [10.6, 11.0])

    assert!(
        start4 > end4,
        "Triangle not covering any pixel center should produce empty range"
    );
}

#[test]
fn test_pixel_center_coverage() {
    // Verify that the pixel-center sampling convention is correct

    // A pixel at integer coordinate Y covers the range [Y, Y+1)
    // and has its center at Y + 0.5

    let pixel_y = 10;
    let pixel_center = pixel_y as f32 + 0.5;  // 10.5

    // Triangle must include pixel center to be drawn
    let triangle_min = 10.1f32;
    let triangle_max = 10.9f32;

    // Check coverage
    let is_covered = pixel_center >= triangle_min && pixel_center <= triangle_max;
    assert!(is_covered, "Pixel center 10.5 should be covered by triangle [10.1, 10.9]");

    // Calculate scanline range with correct formula
    let y_start = (triangle_min - 0.5).ceil() as i32;
    let y_end = (triangle_max - 0.5).floor() as i32;

    assert_eq!(y_start, 10, "Scanline range should include pixel 10");
    assert_eq!(y_end, 10, "Scanline range should include pixel 10");
    assert!(y_start <= y_end, "Range should be valid");
}
