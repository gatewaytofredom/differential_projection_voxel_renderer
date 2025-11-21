//! Face-aligned, structure-of-arrays packet layout for the Hyper-Pipeline.
//! Each packet stores up to 32 quads on a single face direction using SoA
//! layout to make SIMD projection trivial (load-and-go).
use super::mesh::FaceDir;
use crate::voxel::BlockType;

/// Number of quads per packet. Chosen to align with AVX2 width (8 lanes) and
/// to fit cleanly in cache when processed in a hot loop.
pub const PACKET_CAPACITY: usize = 32;

/// A single SoA packet containing up to 32 quads for one face direction.
/// All arrays are 32-byte aligned for efficient SIMD loads.
#[repr(C, align(32))]
#[derive(Clone, Debug)]
pub struct FacePacket32 {
    pub len: u8,
    pub u_min: [u8; PACKET_CAPACITY],
    pub v_min: [u8; PACKET_CAPACITY],
    pub u_len: [u8; PACKET_CAPACITY],
    pub v_len: [u8; PACKET_CAPACITY],
    /// Axis position of the face (slice index in the perpendicular axis).
    pub axis_pos: [u8; PACKET_CAPACITY],
    /// Block type for each quad (material id / palette index).
    pub block_type: [u8; PACKET_CAPACITY],
}

impl FacePacket32 {
    pub fn new() -> Self {
        Self {
            len: 0,
            u_min: [0; PACKET_CAPACITY],
            v_min: [0; PACKET_CAPACITY],
            u_len: [0; PACKET_CAPACITY],
            v_len: [0; PACKET_CAPACITY],
            axis_pos: [0; PACKET_CAPACITY],
            block_type: [0; PACKET_CAPACITY],
        }
    }

    #[inline]
    fn push(
        &mut self,
        u: u8,
        v: u8,
        w: u8,
        width: u8,
        height: u8,
        block: BlockType,
    ) {
        debug_assert!((self.len as usize) < PACKET_CAPACITY);
        let i = self.len as usize;
        self.u_min[i] = u;
        self.v_min[i] = v;
        self.u_len[i] = width;
        self.v_len[i] = height;
        self.axis_pos[i] = w;
        self.block_type[i] = block as u8;
        self.len += 1;
    }

    #[inline]
    pub fn is_full(&self) -> bool {
        self.len as usize >= PACKET_CAPACITY
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// Builder that batches quads into packets for one face direction.
struct FacePacketBuilder {
    packets: Vec<FacePacket32>,
    current: FacePacket32,
}

impl FacePacketBuilder {
    fn new() -> Self {
        Self {
            packets: Vec::new(),
            current: FacePacket32::new(),
        }
    }

    fn push(
        &mut self,
        u: u8,
        v: u8,
        w: u8,
        width: u8,
        height: u8,
        block: BlockType,
    ) {
        if self.current.is_full() {
            let full = std::mem::replace(&mut self.current, FacePacket32::new());
            self.packets.push(full);
        }
        self.current.push(u, v, w, width, height, block);
    }

    fn finish(mut self) -> Vec<FacePacket32> {
        if !self.current.is_empty() {
            self.packets.push(self.current);
        }
        self.packets
    }
}

/// SoA packet stream for all six face directions of a chunk.
#[derive(Default)]
pub struct ChunkFacePackets {
    pub faces: [Vec<FacePacket32>; 6],
}

impl ChunkFacePackets {
    pub fn new() -> Self {
        Self::default()
    }

    /// Build packets from an existing chunk mesh (TinyQuad-based face lists).
    pub fn from_chunk_mesh(mesh: &crate::meshing::ChunkMesh) -> Self {
        let mut builders = [
            FacePacketBuilder::new(),
            FacePacketBuilder::new(),
            FacePacketBuilder::new(),
            FacePacketBuilder::new(),
            FacePacketBuilder::new(),
            FacePacketBuilder::new(),
        ];

        // Each face_dir has a FaceList with slice_quads storing TinyQuads.
        for &face_dir in &[
            FaceDir::PosX,
            FaceDir::NegX,
            FaceDir::PosY,
            FaceDir::NegY,
            FaceDir::PosZ,
            FaceDir::NegZ,
        ] {
            let face_idx = face_dir as usize;
            let face_list = mesh.face_list(face_dir);

            for (slice_idx, quads) in face_list.slice_quads.iter().enumerate() {
                if quads.is_empty() {
                    continue;
                }
                // Convert slice index back to actual axis position (match add_quad logic).
                let axis_pos = if face_dir.is_positive() {
                    (slice_idx + 1) as u8
                } else {
                    slice_idx as u8
                };

                for quad in quads {
                    builders[face_idx].push(
                        quad.u(),
                        quad.v(),
                        axis_pos,
                        quad.width(),
                        quad.height(),
                        BlockType::from_u8(quad.block_type()),
                    );
                }
            }
        }

        let mut faces: [Vec<FacePacket32>; 6] = Default::default();
        for (i, builder) in builders.into_iter().enumerate() {
            faces[i] = builder.finish();
        }

        ChunkFacePackets { faces }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voxel::{BlockData, Chunk};
    use glam::IVec3;

    #[test]
    fn single_voxel_generates_single_entry_packets() {
        let mut chunk = Chunk::uniform(IVec3::ZERO, BlockType::Air);
        chunk.set_block(16, 16, 16, BlockData::new(BlockType::Stone));
        let chunks = vec![chunk];
        let chunk_refs: Vec<&Chunk> = chunks.iter().collect();
        let mesh =
            crate::meshing::BinaryGreedyMesher::mesh_chunk_in_world(&chunks[0], &chunk_refs)
                .expect("mesh expected");

        let packets = ChunkFacePackets::from_chunk_mesh(&mesh);

        for (dir_idx, face_packets) in packets.faces.iter().enumerate() {
            assert_eq!(
                face_packets.len(),
                1,
                "face {} should have exactly one packet",
                dir_idx
            );
            let packet = &face_packets[0];
            assert_eq!(packet.len, 1, "packet should contain one quad");
            assert_eq!(packet.block_type[0], BlockType::Stone as u8);
        }
    }

    #[test]
    fn packets_flush_after_capacity() {
        // Force >32 quads on one face to ensure packet splitting works.
        let dir = FaceDir::PosY;
        let mut builder = FacePacketBuilder::new();
        for i in 0..(PACKET_CAPACITY * 2 + 5) {
            let u = (i % 32) as u8;
            let v = (i / 32) as u8;
            builder.push(u, v, 1, 1, 1, BlockType::Grass);
        }
        let packets = builder.finish();
        assert_eq!(packets.len(), 3);
        assert_eq!(packets[0].len, PACKET_CAPACITY as u8);
        assert_eq!(packets[1].len, PACKET_CAPACITY as u8);
        assert_eq!(packets[2].len, 5);
        // Check contents are filled sequentially
        assert_eq!(packets[0].u_min[0], 0);
        assert_eq!(packets[1].u_min[0], 0);
        assert_eq!(packets[2].u_min[0], 0);
        let _ = dir; // silence unused warning in case we add dir-based logic later
    }
}
