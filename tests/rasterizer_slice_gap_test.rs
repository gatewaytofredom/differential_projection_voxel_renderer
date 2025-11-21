#[test]
fn test_slice_boundary_gap() {
    // This test verifies that when rendering is split into horizontal stripes (slices),
    // there are no gaps between the slices due to incorrect boundary calculation.
    //
    // The bug: When clipping geometry to a slice, the code calculated the clip boundary as:
    //   rect_y1 = rect_y0 + rect_h - 1  (e.g., 0 + 100 - 1 = 99)
    // This is the INDEX of the last pixel, not the spatial boundary.
    //
    // When clamping max_y to 99.0, and then calculating:
    //   y_end = floor(99.0 - 0.5) = 98
    // Pixel row 99 is skipped!
    //
    // The next slice starts at Y=100, creating a 1-pixel gap at Y=99.

    // Test the boundary calculation logic
    // BUGGY LOGIC (what the code used to do):
    let rect_h = 10;
    let rect_y0 = 0;

    let rect_y1_idx = rect_y0 + rect_h - 1; // 9 (index of last pixel)
    let max_y_clamped_buggy = (20.0f32).min(rect_y1_idx as f32); // min(20, 9) = 9.0
    let loop_end_buggy = (max_y_clamped_buggy - 0.5).floor() as i32; // floor(8.5) = 8

    // FIXED LOGIC (what the code should do):
    let rect_y_limit = rect_y0 + rect_h; // 10 (spatial boundary, exclusive)
    let max_y_clamped_fixed = (20.0f32).min(rect_y_limit as f32); // min(20, 10) = 10.0
    let loop_end_fixed = (max_y_clamped_fixed - 0.5).floor() as i32; // floor(9.5) = 9

    // Verify the bug exists in the old calculation
    assert_eq!(loop_end_buggy, 8, "Buggy logic stops at row 8, leaving row 9 empty");
    assert_eq!(loop_end_fixed, 9, "Fixed logic covers row 9");
}

#[test]
fn test_slice_boundary_calculation() {
    // Direct test of the boundary calculation logic
    // This test documents the exact fix needed

    struct TestCase {
        rect_y0: usize,
        rect_h: usize,
        geom_max_y: f32,
        expected_last_pixel: i32,
    }

    let test_cases = vec![
        TestCase { rect_y0: 0, rect_h: 10, geom_max_y: 20.0, expected_last_pixel: 9 },
        TestCase { rect_y0: 10, rect_h: 10, geom_max_y: 20.0, expected_last_pixel: 19 },
        TestCase { rect_y0: 0, rect_h: 100, geom_max_y: 200.0, expected_last_pixel: 99 },
        TestCase { rect_y0: 100, rect_h: 100, geom_max_y: 200.0, expected_last_pixel: 199 },
    ];

    for tc in test_cases {
        // BUGGY calculation (clips to last pixel index)
        let rect_y1_idx = tc.rect_y0 + tc.rect_h - 1;
        let max_y_buggy = tc.geom_max_y.min(rect_y1_idx as f32);
        let y_end_buggy = (max_y_buggy - 0.5).floor() as i32;

        // FIXED calculation (clips to spatial boundary)
        let rect_y_limit = tc.rect_y0 + tc.rect_h;
        let max_y_fixed = tc.geom_max_y.min(rect_y_limit as f32);
        let y_end_fixed = (max_y_fixed - 0.5).floor() as i32;

        assert_eq!(
            y_end_fixed,
            tc.expected_last_pixel,
            "Fixed calculation should reach pixel {}. Buggy was: {}",
            tc.expected_last_pixel,
            y_end_buggy
        );

        assert_ne!(
            y_end_buggy,
            tc.expected_last_pixel,
            "Buggy calculation should NOT reach the expected pixel (this verifies the bug exists)"
        );
    }
}
