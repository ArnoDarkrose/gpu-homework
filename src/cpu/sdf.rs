use glam::{Vec3, Vec3Swizzles};

pub struct ObjectInfo {
    pub sdf: f32,
    pub ambient: Vec3,
    pub diffuse: Vec3,
    pub specular: Vec3,
    pub shininess: f32
}

fn sphere_sdf(point: Vec3) -> f32 {
    point.length() - 1.0
}

fn union_sdf(first: ObjectInfo, second: ObjectInfo) -> ObjectInfo {
    if first.sdf < second.sdf {first} else {second}
}

pub fn scene_sdf(p: Vec3) -> ObjectInfo {
    union_sdf (
        ObjectInfo {
            sdf: menger_sdf(p - Vec3::new(0.0, 1.0, 0.0)),
            ambient: Vec3::new(0.5, 0.5, 0.5),
            diffuse: Vec3::new(0.5, 0.5, 0.5),
            specular: Vec3::new(0.1, 0.1, 0.1),
            shininess: 15.0
        }
    ,

        union_sdf(
            ObjectInfo {
                sdf: sphere_sdf(p - Vec3::new(3.0, 1.0, 0.0)),
                ambient: Vec3::new(0.5, 0.2, 0.2),
                diffuse: Vec3::new(0.6, 0.2, 0.2),
                specular: Vec3::new(2.0, 2.0,2.0),
                shininess: 15.0
            }, 
            ObjectInfo {
                sdf: plain_sdf(p - Vec3::new(0.0, 0.0, 0.0)),
                ambient: Vec3::new(1.1, 1.2, 1.0),
                diffuse: Vec3::new(0.6, 0.1, 0.65),
                specular: Vec3::new(1.0, 1.0, 1.0),
                shininess: 20.0
            }
        )
    )
}

fn cube_sdf(p: Vec3, b: Vec3) -> f32 {
    let d = p.abs() - b;

    let inside_dist = d.y.max(d.z).max(d.x).min(0.0);

    let outside_dist = d.max(Vec3::new(0.0, 0.0, 0.0)).length();

    inside_dist + outside_dist
}

fn mod_vec3_f32(x: Vec3, y: f32) -> Vec3 {
    x - (y * (x/y).floor())
}
pub fn menger_sdf(p: Vec3) -> f32 {
    let mut d = cube_sdf(p, Vec3::new(1.0, 1.0, 1.0));

    let mut s = 1.0_f32;
    
    for _ in 0..5 {
        let a = mod_vec3_f32(p * s, 2.0) - Vec3::new(1.0, 1.0, 1.0);

        s *= 3.0;

        let r = Vec3::new(1.0, 1.0, 1.0) - a.abs() * 3.0; 

        let c = cross_sdf(r)/s;

        d = d.max(c);
    }

    d
}

fn cross_sdf(p: Vec3) -> f32 {
    let da =  cube_sdf(p, Vec3::new(f32::INFINITY, 1.0, 1.0));
    let db = cube_sdf(p.yzx(), Vec3::new(1.0, f32::INFINITY, 1.0));
    let dc = cube_sdf(p.zxy(), Vec3::new(1.0, 1.0, f32::INFINITY));

    db.min(dc).min(da)
}

fn plain_sdf(p: Vec3) -> f32 {
    p.y
}