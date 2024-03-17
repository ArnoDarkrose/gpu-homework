use image::{ImageBuffer, Rgba};
use glam::{Vec3, UVec2};

pub mod cpu;
pub mod gpu;
pub mod consts;

trait ParseCoord {
    fn parse_coord(self) -> Vec3;
}

impl ParseCoord for String {
    fn parse_coord(self) -> Vec3 {
        let coords:Vec<_> = 
            self
                .split(',')
                .map(|coord| coord.trim().parse().expect("failed to parse coordinates"))
                .collect();

        Vec3::from_slice(&coords)
    }
}

fn main() {
    let arguments = std::env::args();
    let arguments = arguments::parse(arguments).expect("failed to parse command line arguments");

    let cam_pos = arguments.get::<String>("cam").unwrap_or("0.0, 1.5, 10.0".to_string()).parse_coord();
    
    let resolution = UVec2::new(
        arguments.get::<u32>("width").unwrap_or(1980),
        arguments.get::<u32>("height").unwrap_or(1080)
    );

    let fov = arguments.get::<f32>("fov").unwrap_or(45.0);
    let view_vector = arguments.get::<String>("view").unwrap_or("0.0, 0.0, 0.0".to_string()).parse_coord();

    let((buf, future), cpu_img) = rayon::join(
        || gpu::setup::launch(cam_pos, resolution, fov, view_vector), 
        || cpu::setup::launch_par(cam_pos, resolution, fov, view_vector)
    );

    println!("Saving cpu_image...");
    cpu_img.save("cpu_image.png").expect("failed to save image");

    future.wait(None).unwrap();

    let buffer_content = buf.read().expect("failed to read buffer");
    let gpu_img = ImageBuffer::<Rgba<u8>, _>::from_raw(resolution.x, resolution.y, &buffer_content[..]).unwrap();
    
    println!("Saving gpu image");
    gpu_img.save("gpu_image.png").expect("failed to save image");
}
