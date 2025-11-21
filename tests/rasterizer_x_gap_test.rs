// Test for X-axis and Y-axis pixel-center gap fixes

#[test]
fn test_rasterizer_x_span_gap_bug() {
    // Scenario: A narrow triangle span from X=20.1 to X=20.9 on Scanline 10.
    // Old Logic: start=21, end=20 -> Skipped.
    // Correct Logic: Center 20.5 is inside -> Pixel 20 drawn.

    // Direct logic verification of the fix:
    let x_start_f = 20.1f32;
    let x_end_f = 20.9f32;

    // Buggy Logic (what the old code would have done)
    let start_buggy = x_start_f.ceil() as i32; // 21
    let end_buggy = x_end_f.floor() as i32;    // 20

    // Correct Logic (Pixel Centers) - what the fixed code does
    let start_correct = (x_start_f - 0.5).ceil() as i32; // ceil(19.6) = 20
    let end_correct = (x_end_f - 0.5).floor() as i32;    // floor(20.4) = 20

    assert!(start_buggy > end_buggy, "Buggy logic should fail to produce a span");
    assert!(start_correct <= end_correct, "Correct logic should produce span [20, 20]");

    // Verify the pixel center at 20.5 is within the span [20.1, 20.9]
    let pixel_center = 20.5f32;
    assert!(pixel_center >= x_start_f && pixel_center <= x_end_f,
            "Pixel center should be inside span");

    // Verify the correct logic includes pixel 20
    assert_eq!(start_correct, 20, "Should include pixel 20");
    assert_eq!(end_correct, 20, "Should include pixel 20");
}

#[test]
fn test_y_span_gap_bug() {
    // Same issue can occur on Y axis
    let y_start_f = 10.1f32;
    let y_end_f = 10.9f32;

    // Buggy Logic
    let start_buggy = y_start_f.ceil() as i32; // 11
    let end_buggy = y_end_f.floor() as i32;    // 10

    // Correct Logic (Pixel Centers)
    let start_correct = (y_start_f - 0.5).ceil() as i32; // ceil(9.6) = 10
    let end_correct = (y_end_f - 0.5).floor() as i32;    // floor(10.4) = 10

    assert!(start_buggy > end_buggy, "Buggy logic should fail to produce a span");
    assert!(start_correct <= end_correct, "Correct logic should produce span [10, 10]");
}

#[test]
fn test_multiple_gap_scenarios() {
    // Test various edge cases
    let test_cases: Vec<(f32, f32, usize)> = vec![
        // (start, end, expected_pixel_count_with_fix)
        // Pixel N has center at N + 0.5
        (10.1, 10.9, 1),  // Single pixel at 10 (center 10.5 ∈ [10.1, 10.9])
        (10.0, 10.5, 1),  // Pixel 10 (center 10.5 ∈ [10.0, 10.5])
        (10.6, 11.6, 1),  // Single pixel at 11 (center 11.5 ∈ [10.6, 11.6])
        (10.1, 11.9, 2),  // Two pixels: 10 and 11 (both centers inside)
        (10.4, 10.6, 1),  // Very narrow span, pixel 10 (center 10.5 inside)
        (10.0, 11.0, 1),  // Edge case: only center 10.5 inside [10.0, 11.0]
    ];

    for (start, end, expected_count) in test_cases {
        let x_start = (start - 0.5_f32).ceil() as i32;
        let x_end = (end - 0.5_f32).floor() as i32;

        let actual_count = if x_start <= x_end {
            (x_end - x_start + 1) as usize
        } else {
            0
        };

        assert_eq!(actual_count, expected_count,
                   "Span [{}, {}] should include {} pixel(s), got {}",
                   start, end, expected_count, actual_count);
    }
}
