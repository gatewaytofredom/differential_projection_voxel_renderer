/// Basic, configurable shading utilities.
/// Kept separate from the rasterizer so lighting models
/// can evolve independently of the rasterization pipeline.
use crate::meshing::Vertex;
use glam::Vec3;

/// Simple directional + ambient shading configuration.
#[derive(Copy, Clone, Debug)]
pub struct ShadingConfig {
    /// Direction the light is coming from (world space).
    /// Typically from above and slightly from one side.
    pub light_dir: Vec3,
    /// Constant ambient term added to all fragments.
    pub ambient: f32,
    /// Strength of the directional (Lambert) term.
    pub diffuse: f32,
    /// If true, modulate lighting by the vertex AO value.
    pub use_ao: bool,
}

impl Default for ShadingConfig {
    fn default() -> Self {
        Self {
            // Slightly from +X/+Z and above
            light_dir: Vec3::new(0.4, 1.0, 0.3).normalize(),
            ambient: 0.35,
            diffuse: 0.65,
            use_ao: true,
        }
    }
}

impl ShadingConfig {
    /// Compute a scalar light intensity for a single vertex.
    /// Returns a value in [0, 1].
    /// Note: This function is deprecated as lighting is now pre-computed in the mesher.
    #[inline]
    pub fn vertex_light(&self, vertex: &Vertex) -> f32 {
        // Extract normal from packed field
        let normal_index = vertex.normal_index();
        let normal = match normal_index {
            0 => Vec3::X,        // PosX
            1 => Vec3::NEG_X,    // NegX
            2 => Vec3::Y,        // PosY
            3 => Vec3::NEG_Y,    // NegY
            4 => Vec3::Z,        // PosZ
            5 => Vec3::NEG_Z,    // NegZ
            _ => Vec3::Y,        // Default fallback
        };

        let l = self.light_dir; // already normalized in Default
        let lambert = normal.dot(l).max(0.0);
        let mut light = self.ambient + self.diffuse * lambert;

        if self.use_ao {
            let ao_level = vertex.ao_level();
            let ao_factor = match ao_level {
                0 => 1.0,
                1 => 0.8,
                2 => 0.6,
                _ => 0.4,
            };
            light *= ao_factor;
        }

        light.clamp(0.0, 1.0)
    }

    /// Apply a light factor to an RGB color and pack into u32.
    /// Optimized for instruction-level parallelism and reduced conversions.
    #[inline]
    pub fn shade_color(&self, base: [u8; 3], light: f32) -> u32 {
        // Light is already clamped in vertex generation for pre-computed values
        // Use integer arithmetic for better performance
        let light_u8 = (light * 255.0) as u32;

        // Compute all channels in parallel (CPU can execute these simultaneously)
        // Using u32 arithmetic avoids multiple type conversions
        let r = ((base[0] as u32 * light_u8) >> 8).min(255);
        let g = ((base[1] as u32 * light_u8) >> 8).min(255);
        let b = ((base[2] as u32 * light_u8) >> 8).min(255);

        // Pack directly without function call (better inlining)
        0xFF000000 | (r << 16) | (g << 8) | b
    }

    /// Apply a light factor to an ARGB32 color.
    /// This is useful for textured fragments where the base color is already packed.
    #[inline]
    pub fn shade_color_u32(&self, base: u32, light: f32) -> u32 {
        // Unpack
        let r = (base >> 16) & 0xFF;
        let g = (base >> 8) & 0xFF;
        let b = base & 0xFF;

        // Convert light to fixed point 8.8 for faster multiply
        let light_fp = (light * 256.0) as u32;

        // Shade
        let r_lit = (r * light_fp) >> 8;
        let g_lit = (g * light_fp) >> 8;
        let b_lit = (b * light_fp) >> 8;

        // Clamp
        let r_final = r_lit.min(255);
        let g_final = g_lit.min(255);
        let b_final = b_lit.min(255);

        0xFF000000 | (r_final << 16) | (g_final << 8) | b_final
    }
}
