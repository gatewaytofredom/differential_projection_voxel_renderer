/// Camera system with FPS-style controls
/// Mouse look and WASD movement
use glam::{Mat4, Quat, Vec3, Vec4};

pub struct Camera {
    pub position: Vec3,
    pub yaw: f32,   // Rotation around Y axis (radians)
    pub pitch: f32, // Rotation around X axis (radians)
    pub fov: f32,
    pub near: f32,
    pub far: f32,
    pub aspect_ratio: f32,

    // Movement state
    pub move_speed: f32,
    pub mouse_sensitivity: f32,
}

impl Camera {
    pub fn new(position: Vec3, aspect_ratio: f32) -> Self {
        Self {
            position,
            yaw: 0.0,
            pitch: 0.0,
            fov: 70.0f32.to_radians(),
            near: 0.1,
            far: 1000.0,
            aspect_ratio,
            move_speed: 10.0,
            mouse_sensitivity: 0.002,
        }
    }

    /// Update camera orientation to look at a specific target point.
    pub fn look_at(&mut self, target: Vec3, up: Vec3) {
        let view_matrix = Mat4::look_at_rh(self.position, target, up);
        let rotation_quat = Quat::from_mat4(&view_matrix.inverse());
        let (pitch, yaw, _roll) = rotation_quat.to_euler(glam::EulerRot::YXZ);
        self.yaw = yaw;
        self.pitch = pitch;
    }

    /// Get view matrix
    pub fn view_matrix(&self) -> Mat4 {
        let rotation = self.rotation_quat();
        let forward = rotation * Vec3::NEG_Z;
        let target = self.position + forward;
        let up = rotation * Vec3::Y;

        Mat4::look_at_rh(self.position, target, up)
    }

    /// Get projection matrix
    pub fn projection_matrix(&self) -> Mat4 {
        Mat4::perspective_rh(self.fov, self.aspect_ratio, self.near, self.far)
    }

    /// Get combined view-projection matrix
    pub fn view_projection_matrix(&self) -> Mat4 {
        self.projection_matrix() * self.view_matrix()
    }

    /// Get forward direction vector
    pub fn forward(&self) -> Vec3 {
        self.rotation_quat() * Vec3::NEG_Z
    }

    /// Get right direction vector
    pub fn right(&self) -> Vec3 {
        self.rotation_quat() * Vec3::X
    }

    /// Get up direction vector
    pub fn up(&self) -> Vec3 {
        self.rotation_quat() * Vec3::Y
    }

    /// Get rotation quaternion
    fn rotation_quat(&self) -> Quat {
        Quat::from_rotation_y(self.yaw) * Quat::from_rotation_x(self.pitch)
    }

    /// Update camera orientation from mouse delta
    pub fn rotate(&mut self, mouse_delta_x: f32, mouse_delta_y: f32) {
        self.yaw += mouse_delta_x * self.mouse_sensitivity;
        self.pitch -= mouse_delta_y * self.mouse_sensitivity;

        // Clamp pitch to prevent gimbal lock
        const MAX_PITCH: f32 = std::f32::consts::FRAC_PI_2 - 0.01;
        self.pitch = self.pitch.clamp(-MAX_PITCH, MAX_PITCH);
    }

    /// Move camera in local space
    pub fn move_local(&mut self, forward: f32, right: f32, up: f32, dt: f32) {
        let move_vec = self.forward() * forward + self.right() * right + Vec3::Y * up;
        self.position += move_vec * self.move_speed * dt;
    }

    /// Update aspect ratio (call when window resizes)
    pub fn set_aspect_ratio(&mut self, aspect_ratio: f32) {
        self.aspect_ratio = aspect_ratio;
    }

    /// Extract frustum planes from the view-projection matrix
    /// Returns a Frustum for AABB culling
    pub fn extract_frustum(&self) -> Frustum {
        Frustum::from_view_projection(&self.view_projection_matrix())
    }
}

/// View frustum represented as 6 planes for AABB culling
/// Planes are stored in Hessian normal form: ax + by + cz + d = 0
/// where (a,b,c) is the outward-facing normal
#[derive(Debug, Clone, Copy)]
pub struct Frustum {
    /// 6 planes: left, right, bottom, top, near, far
    pub planes: [Vec4; 6],
}

impl Frustum {
    /// Extract frustum planes from a view-projection matrix
    /// Using Gribb-Hartmann method (fast extraction from MVP)
    pub fn from_view_projection(vp: &Mat4) -> Self {
        // Extract rows from the matrix. In glam's column-major Mat4,
        // rows still correspond to the plane extraction used by the
        // standard Gribb-Hartmann method.
        let row0 = vp.row(0);
        let row1 = vp.row(1);
        let row2 = vp.row(2);
        let row3 = vp.row(3);

        // Extract and normalize planes
        let mut planes = [Vec4::ZERO; 6];

        // Left plane: row3 + row0
        planes[0] = Self::normalize_plane(row3 + row0);
        // Right plane: row3 - row0
        planes[1] = Self::normalize_plane(row3 - row0);
        // Bottom plane: row3 + row1
        planes[2] = Self::normalize_plane(row3 + row1);
        // Top plane: row3 - row1
        planes[3] = Self::normalize_plane(row3 - row1);
        // Near plane: row3 + row2
        planes[4] = Self::normalize_plane(row3 + row2);
        // Far plane: row3 - row2
        planes[5] = Self::normalize_plane(row3 - row2);

        Self { planes }
    }

    /// Normalize a plane equation
    #[inline]
    fn normalize_plane(plane: Vec4) -> Vec4 {
        let normal_length = plane.truncate().length();
        if normal_length > 0.0001 {
            plane / normal_length
        } else {
            plane
        }
    }

    /// Test if an AABB intersects the frustum
    /// Returns true if the box is at least partially inside
    pub fn intersects_aabb(&self, min: Vec3, max: Vec3) -> bool {
        // For each plane, check if the AABB is completely outside
        for plane in &self.planes {
            // Get the "positive vertex" - the corner furthest along the plane normal
            let p_vertex = Vec3::new(
                if plane.x > 0.0 { max.x } else { min.x },
                if plane.y > 0.0 { max.y } else { min.y },
                if plane.z > 0.0 { max.z } else { min.z },
            );

            // If the positive vertex is outside this plane, the whole box is outside
            if plane.x * p_vertex.x + plane.y * p_vertex.y + plane.z * p_vertex.z + plane.w < 0.0
            {
                return false;
            }
        }

        // Box is at least partially inside all planes
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frustum_culls_box_behind_camera() {
        let camera = Camera::new(Vec3::ZERO, 16.0 / 9.0);
        let frustum = camera.extract_frustum();

        // In front of the camera (looking towards -Z)
        let front_min = Vec3::new(-1.0, -1.0, -10.0);
        let front_max = Vec3::new(1.0, 1.0, -8.0);

        // Behind the camera
        let back_min = Vec3::new(-1.0, -1.0, 8.0);
        let back_max = Vec3::new(1.0, 1.0, 10.0);

        assert!(
            frustum.intersects_aabb(front_min, front_max),
            "box in front of camera should be inside frustum"
        );
        assert!(
            !frustum.intersects_aabb(back_min, back_max),
            "box behind camera should be outside frustum"
        );
    }
}

/// Camera controller - handles input state
pub struct CameraController {
    pub forward_pressed: bool,
    pub backward_pressed: bool,
    pub left_pressed: bool,
    pub right_pressed: bool,
    pub up_pressed: bool,
    pub down_pressed: bool,
}

impl CameraController {
    pub fn new() -> Self {
        Self {
            forward_pressed: false,
            backward_pressed: false,
            left_pressed: false,
            right_pressed: false,
            up_pressed: false,
            down_pressed: false,
        }
    }

    /// Update camera based on controller state
    pub fn update_camera(&self, camera: &mut Camera, dt: f32) {
        let mut forward = 0.0;
        let mut right = 0.0;
        let mut up = 0.0;

        if self.forward_pressed {
            forward += 1.0;
        }
        if self.backward_pressed {
            forward -= 1.0;
        }
        if self.right_pressed {
            right += 1.0;
        }
        if self.left_pressed {
            right -= 1.0;
        }
        if self.up_pressed {
            up += 1.0;
        }
        if self.down_pressed {
            up -= 1.0;
        }

        camera.move_local(forward, right, up, dt);
    }
}
