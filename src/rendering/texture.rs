/// Micro-texture implementation optimized for cache locality.
/// Aligned to 64 bytes; palette stored as ARGB32 for zero-cost sampling.
#[repr(C, align(64))]
#[derive(Copy, Clone, Debug)]
pub struct MicroTexture {
    // Palette stored as ARGB32 for zero-cost sampling
    // 16 colors * 4 bytes = 64 bytes
    pub palette: [u32; 16],

    // Texture indices (8x8, 4 bits per pixel) packed 2 pixels per byte.
    // High nibble = even x, Low nibble = odd x
    pub indices: [u8; 32],
}

impl MicroTexture {
    /// Sample the texture at coordinates (u, v).
    /// Inputs are wrapped to 0..7 (tiling).
    #[inline(always)]
    pub fn sample(&self, u: u8, v: u8) -> u32 {
        // Wrap coordinates to 0..7
        let x = u & 7;
        let y = v & 7;

        // Calculate linear pixel index (0..63)
        let pixel_idx = (y << 3) | x;

        // Find byte index (pixel_idx / 2)
        let byte_idx = pixel_idx >> 1;
        // Safety: byte_idx is always 0..31
        let byte = unsafe { *self.indices.get_unchecked(byte_idx as usize) };

        // Extract 4-bit palette index
        // If pixel index is even (0, 2...), use high nibble. Else low.
        let palette_idx = if (pixel_idx & 1) == 0 { (byte >> 4) & 0xF } else { byte & 0xF };

        // Palette already holds ARGB32, just return it.
        unsafe { *self.palette.get_unchecked(palette_idx as usize) }
    }
}

#[inline(always)]
fn rgb565_to_argb32(c: u16) -> u32 {
    let r = (c >> 11) & 0x1F;
    let g = (c >> 5) & 0x3F;
    let b = c & 0x1F;

    // Expand to 8-bit (bit replication for better accuracy)
    let r8 = (r << 3) | (r >> 2);
    let g8 = (g << 2) | (g >> 4);
    let b8 = (b << 3) | (b >> 2);

    0xFF000000 | ((r8 as u32) << 16) | ((g8 as u32) << 8) | (b8 as u32)
}

/// Texture Atlas holding all block textures
pub struct TextureAtlas {
    pub textures: Vec<MicroTexture>,
}

impl Default for TextureAtlas {
    fn default() -> Self {
        let mut textures = Vec::new();

        // 0: Air / Unused (Magenta checkerboard for debug)
        textures.push(create_checkerboard(0xF81F, 0x0000));

        // 1: Grass (Green noise)
        textures.push(create_noise(0x03E0, 0x02E0));

        // 2: Dirt (Brown noise)
        // R=139(17), G=69(17), B=19(2) -> RGB565 approx 0x8A22
        textures.push(create_noise(0x8A22, 0x71C2));

        // 3: Stone (Grey noise)
        textures.push(create_noise(0x8410, 0x73AE));

        Self { textures }
    }
}

fn create_checkerboard(c1: u16, c2: u16) -> MicroTexture {
    let mut palette = [0u32; 16];
    palette[0] = rgb565_to_argb32(c1);
    palette[1] = rgb565_to_argb32(c2);

    let mut indices = [0u8; 32];
    for i in 0..64 {
        let x = i % 8;
        let y = i / 8;
        let color_idx = ((x + y) % 2) as u8; // 0 or 1

        let byte_idx = i / 2;
        if i % 2 == 0 {
            indices[byte_idx] |= color_idx << 4;
        } else {
            indices[byte_idx] |= color_idx;
        }
    }

    MicroTexture { palette, indices }
}

fn create_noise(base: u16, dark: u16) -> MicroTexture {
    let mut palette = [0u32; 16];
    // Fill palette with gradients between base and dark
    for i in 0..16 {
        if i % 2 == 0 {
            palette[i] = rgb565_to_argb32(base);
        } else {
            palette[i] = rgb565_to_argb32(dark);
        }
    }

    // Simple pseudo-random indices
    let mut indices = [0u8; 32];
    let mut seed: u32 = 12345;
    for i in 0..32 {
        seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
        indices[i] = (seed >> 16) as u8;
    }

    MicroTexture { palette, indices }
}
