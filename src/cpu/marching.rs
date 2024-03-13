use glam::Vec3;
use crate::consts::*;
use super::sdf::scene_sdf;


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

    if dotrv < 0.0 {
        return light_intensity * diffuse * dotln
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