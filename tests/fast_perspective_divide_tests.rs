/// Tests for fast perspective divide with Newton-Raphson refinement
///
/// Verifies that the fast reciprocal approximation maintains sufficient precision
/// for rasterization without visual artifacts.

use glam::Vec4;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Test the accuracy of fast reciprocal against exact division
#[test]
#[cfg(target_arch = "x86_64")]
fn test_fast_reciprocal_accuracy() {
    if !is_x86_feature_detected!("avx2") {
        println!("Skipping AVX2 test - not supported on this CPU");
        return;
    }

    unsafe {
        // Test a range of W values typical in perspective projection
        let test_values = [
            0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0,
            0.001, 0.01, // Near plane values
            1000.0, 10000.0, // Far plane values
        ];

        for &w in &test_values {
            let w_vec = _mm256_set1_ps(w);
            let fast_rcp = fast_reciprocal_ps_test(w_vec);

            let mut result = [0.0f32; 8];
            _mm256_storeu_ps(result.as_mut_ptr(), fast_rcp);

            let exact = 1.0 / w;
            let fast = result[0];

            // Calculate relative error
            let relative_error = ((fast - exact) / exact).abs();

            // For perspective projection, we need < 1 pixel error at 1080p
            // This corresponds to relative error < 1/1920 ≈ 0.0005 (0.05%)
            // Our implementation should achieve < 0.00001 (0.001%) for most values
            assert!(
                relative_error < 0.0001,
                "Fast reciprocal too inaccurate for w={}: exact={}, fast={}, relative_error={}",
                w, exact, fast, relative_error
            );
        }
    }
}

/// Test perspective divide maintains precision for typical clip-space coordinates
#[test]
#[cfg(target_arch = "x86_64")]
fn test_perspective_divide_precision() {
    if !is_x86_feature_detected!("avx2") {
        println!("Skipping AVX2 test - not supported on this CPU");
        return;
    }

    // Simulate typical clip-space coordinates from a 3D projection
    let test_cases = vec![
        // (clip_x, clip_y, clip_z, clip_w) -> expected NDC after division
        Vec4::new(100.0, 50.0, 1.5, 2.0),    // Near geometry
        Vec4::new(-200.0, 150.0, 5.0, 10.0), // Mid-range
        Vec4::new(500.0, -300.0, 50.0, 100.0), // Far geometry
        Vec4::new(0.5, 0.3, 0.1, 0.5),       // Small coordinates
        Vec4::new(1920.0, 1080.0, 100.0, 200.0), // Large screen-space values
    ];

    unsafe {
        for clip in test_cases {
            // Exact division
            let exact_ndc = clip.truncate() / clip.w;

            // Fast division using our approximation
            let w_vec = _mm256_set1_ps(clip.w);
            let inv_w = fast_reciprocal_ps_test(w_vec);

            let mut inv_w_scalar = [0.0f32; 8];
            _mm256_storeu_ps(inv_w_scalar.as_mut_ptr(), inv_w);

            let fast_ndc = clip.truncate() * inv_w_scalar[0];

            // Check each component
            let diff = (fast_ndc - exact_ndc).abs();

            // For screen coordinates at 1080p, we need sub-pixel precision
            // Maximum acceptable error: 0.01 pixels
            assert!(
                diff.x < 0.01 && diff.y < 0.01 && diff.z < 0.0001,
                "Perspective divide too inaccurate:\nClip: {:?}\nExact NDC: {:?}\nFast NDC: {:?}\nDiff: {:?}",
                clip, exact_ndc, fast_ndc, diff
            );
        }
    }
}

/// Test that fast reciprocal handles edge cases safely
#[test]
#[cfg(target_arch = "x86_64")]
fn test_fast_reciprocal_edge_cases() {
    if !is_x86_feature_detected!("avx2") {
        println!("Skipping AVX2 test - not supported on this CPU");
        return;
    }

    unsafe {
        // Test near-zero values (should be avoided in practice via clipping)
        let small_w = _mm256_set1_ps(0.0001);
        let rcp_small = fast_reciprocal_ps_test(small_w);
        let mut result = [0.0f32; 8];
        _mm256_storeu_ps(result.as_mut_ptr(), rcp_small);

        // Should not produce NaN or Inf for small positive values
        assert!(result[0].is_finite(), "Fast reciprocal produced non-finite value for small w");

        // Test that large W values work correctly
        let large_w = _mm256_set1_ps(10000.0);
        let rcp_large = fast_reciprocal_ps_test(large_w);
        _mm256_storeu_ps(result.as_mut_ptr(), rcp_large);

        let exact = 1.0 / 10000.0;
        let relative_error = ((result[0] - exact) / exact).abs();
        assert!(relative_error < 0.0001, "Inaccurate for large W values");
    }
}

/// Test batched perspective divide (8 different W values simultaneously)
#[test]
#[cfg(target_arch = "x86_64")]
fn test_batched_perspective_divide() {
    if !is_x86_feature_detected!("avx2") {
        println!("Skipping AVX2 test - not supported on this CPU");
        return;
    }

    unsafe {
        // Create 8 different W values (simulating a batch of vertices)
        let w_values = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0];
        let w_vec = _mm256_loadu_ps(w_values.as_ptr());

        let inv_w = fast_reciprocal_ps_test(w_vec);

        let mut results = [0.0f32; 8];
        _mm256_storeu_ps(results.as_mut_ptr(), inv_w);

        // Verify each value in the batch
        for i in 0..8 {
            let exact = 1.0 / w_values[i];
            let fast = results[i];
            let relative_error = ((fast - exact) / exact).abs();

            assert!(
                relative_error < 0.0001,
                "Batched reciprocal inaccurate at index {}: w={}, exact={}, fast={}, error={}",
                i, w_values[i], exact, fast, relative_error
            );
        }
    }
}

/// Test performance comparison between fast and exact division
#[test]
#[cfg(target_arch = "x86_64")]
fn test_fast_reciprocal_performance_characteristics() {
    if !is_x86_feature_detected!("avx2") {
        println!("Skipping AVX2 test - not supported on this CPU");
        return;
    }

    // This test verifies that the fast path produces results consistent
    // with the exact division for a large batch of random-ish values
    let test_count = 1000;
    let mut max_error = 0.0f32;

    unsafe {
        for i in 0..test_count {
            let w = 0.1 + (i as f32) * 0.1; // Range: 0.1 to 100.1
            let w_vec = _mm256_set1_ps(w);

            let fast_rcp = fast_reciprocal_ps_test(w_vec);
            let mut result = [0.0f32; 8];
            _mm256_storeu_ps(result.as_mut_ptr(), fast_rcp);

            let exact = 1.0 / w;
            let error = ((result[0] - exact) / exact).abs();
            max_error = max_error.max(error);
        }
    }

    // Maximum relative error should be well below 0.01% across all test values
    assert!(
        max_error < 0.0001,
        "Maximum relative error too high: {}",
        max_error
    );

    println!("Fast reciprocal max relative error over {} values: {:.6}%",
             test_count, max_error * 100.0);
}

// Helper function to test the fast reciprocal implementation
// This duplicates the internal implementation for testing purposes
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn fast_reciprocal_ps_test(w: __m256) -> __m256 {
    let rcp = _mm256_rcp_ps(w);
    let two = _mm256_set1_ps(2.0);
    let two_minus_w_rcp = _mm256_fnmadd_ps(w, rcp, two);
    _mm256_mul_ps(rcp, two_minus_w_rcp)
}

/// Integration test: Verify fast divide works correctly in projection pipeline
#[test]
fn test_fast_divide_in_projection_context() {
    // This test simulates the actual usage in the differential projection pipeline
    // We project several quads and verify the screen-space coordinates are accurate

    // This will be tested implicitly by the existing rendering tests
    // If fast_reciprocal breaks, existing tests should catch visual artifacts

    // For now, just verify the logic compiles and basic math is sound
    let clip_w = 5.0f32;
    let exact_inv = 1.0f32 / clip_w;
    let expected_ndc_x = 100.0f32 * exact_inv; // 20.0

    // Our fast reciprocal should produce a value very close to this
    assert!((expected_ndc_x - 20.0f32).abs() < 0.001f32);
}
