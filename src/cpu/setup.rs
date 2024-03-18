use glam::{Vec3, UVec2, Vec2, Mat4, Vec4, Vec4Swizzles};
use image::{ImageBuffer, Rgba};
use crate::consts::*;
use super::sdf::scene_sdf;

fn cam_dir(fov: f32, size: Vec2, frag_coord: Vec2) -> Vec3 {
    let xy = frag_coord - (size / 2.0);
    let z = size.y / (fov.to_radians()/2.0).tan();

    Vec3::new(xy.x, -xy.y, -z).normalize()
}

pub(crate) fn lookat_matrix(cam_pos: Vec3, center: Vec3, up: Vec3) -> Mat4 {
    let f = (center - cam_pos).normalize();
    let s = f.cross(up).normalize();
    let u = s.cross(f);

    Mat4::from_cols(
    Vec4::new(s.x, s.y, s.z, 0.0), 
    Vec4::new(u.x, u.y, u.z, 0.0),
    Vec4::new(-f.x, -f.y, -f.z, 0.0), 
    Vec4::new(0.0, 0.0, 0.0, 1.0)
    )
}

fn reflect(i: Vec3, n: Vec3) -> Vec3 {
    i - n.dot(i) * 2.0 * n
}

pub fn render(cam_pos: Vec3, march_dir: Vec3, start: f32, end: f32) -> Vec3 {
    let mut depth = start;

    for _ in 0..MAX_MARCHING_STEPS {
        let obj_info = scene_sdf(cam_pos + march_dir * depth);

        if obj_info.sdf < EPSILON {
            return phong_illumination(
                obj_info.ambient, 
                obj_info.diffuse, 
                obj_info.specular, 
                obj_info.shininess, 
                cam_pos + march_dir * depth, 
                cam_pos
            )
        }

        depth += obj_info.sdf;

        if depth >= end - EPSILON {
            return Vec3::new(170.0 / 255.0, 150.0 / 255.0, 1.0)
        }
    }

    Vec3::new(170.0 / 255.0, 150.0 / 255.0, 1.0)
}

fn estimate_normal(p: Vec3) -> Vec3 {
    (Vec3::new(
        scene_sdf(Vec3::new(p.x + EPSILON, p.y, p.z)).sdf - scene_sdf(Vec3::new(p.x - EPSILON, p.y, p.z)).sdf,
        scene_sdf(Vec3::new(p.x, p.y + EPSILON, p.z)).sdf - scene_sdf(Vec3::new(p.x, p.y - EPSILON, p.z)).sdf,
        scene_sdf(Vec3::new(p.x, p.y, p.z + EPSILON)).sdf - scene_sdf(Vec3::new(p.x, p.y, p.z - EPSILON)).sdf
    )).normalize()
}

fn phong_contrib_for_light(diffuse: Vec3, specular: Vec3, alpha: f32, p: Vec3, cam_pos: Vec3, light_pos: Vec3, light_intensity: Vec3) -> Vec3 {
    let normal = estimate_normal(p);
    let light = (light_pos - p).normalize();
    let view = (cam_pos - p).normalize();
    let r = reflect(-light, normal).normalize();

    let dotln = light.dot(normal);
    let dotrv = r.dot(view).clamp(0.0, 1.0);

    if dotln < 0.0 {
        return Vec3::new(0.0, 0.0, 0.0)
    }

    let mut res = light_intensity * diffuse * dotln + specular * dotrv.powf(alpha);

    let sha_start = p + normal * 0.1;

    let light_dir = (light_pos - sha_start).normalize();

    let mut depth = MIN_DIST;
    for _ in 0..MAX_MARCHING_STEPS {
        let obj_info = scene_sdf(sha_start + light_dir * depth);

        if obj_info.sdf < 0.01 {
            res *= 0.0;

            break;
        }

        depth += obj_info.sdf;

        if depth >= MAX_DIST - EPSILON {
            break;
        }
    }
    res
}

pub fn phong_illumination(ambient: Vec3, diffuse: Vec3, specular: Vec3, alpha: f32, p: Vec3, cam_pos: Vec3) -> Vec3 {
    let ambient_light = Vec3::new(1.0, 1.0, 1.0) * 0.2; 

    let mut color = ambient * ambient_light;

    let light_pos = Vec3::new(0.0, 4.0, 3.0);

    let light_intensity = Vec3::new(0.7, 0.7, 0.7);

    color += phong_contrib_for_light(diffuse, specular, alpha, p, cam_pos, light_pos, light_intensity);

    color
}

pub fn launch_par(cam_pos: Vec3, resolution: UVec2, fov: f32, view_vector: Vec3) -> ImageBuffer<Rgba<u8>, Vec<u8>> {
    let view_to_world = lookat_matrix(
        cam_pos, 
        view_vector, 
        Vec3::new(0.0, 1.0, 0.0)
    );

    ImageBuffer::from_par_fn(resolution.x, resolution.y, |x, y| {
        let view_dir = cam_dir(
            fov, 
            Vec2::new(resolution.x as f32, resolution.y as f32), 
            Vec2::new(x as f32, y as f32)
        );

        let world_dir = (view_to_world * Vec4::new(view_dir.x, view_dir.y, view_dir.z, 0.0)).xyz();

        let color = render(cam_pos, world_dir, MIN_DIST, MAX_DIST);

        Rgba([(color.x * 255.0) as u8, (color.y * 255.0) as u8, (color.z * 255.0) as u8, 255u8])
    })
}

pub fn launch_linear(cam_pos: Vec3, resolution: UVec2, fov:f32, view_vector: Vec3) -> ImageBuffer<Rgba<u8>, Vec<u8>> {
    let view_to_world = lookat_matrix(
        cam_pos, 
        view_vector, 
        Vec3::new(0.0, 1.0, 0.0)
    );

    ImageBuffer::from_fn(resolution.x, resolution.y, |x, y| {
        let view_dir = cam_dir(
            fov, 
            Vec2::new(resolution.x as f32, resolution.y as f32), 
            Vec2::new(x as f32, y as f32)
        );

        let world_dir = (view_to_world * Vec4::new(view_dir.x, view_dir.y, view_dir.z, 0.0)).xyz();

        let color = render(cam_pos, world_dir, MIN_DIST, MAX_DIST);

        Rgba([(color.x * 255.0) as u8, (color.y * 255.0) as u8, (color.z * 255.0) as u8, 255u8])
    })
}