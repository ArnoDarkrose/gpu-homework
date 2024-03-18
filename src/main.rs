use image::{ImageBuffer, Rgba};
use glam::{Vec3, UVec2};
use std::time::{self, Duration};

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

    let single_tread_start = time::Instant::now();
    let cpu_img_single_threaded = cpu::setup::launch_linear(cam_pos, resolution, fov, view_vector);
    let one_thread_time = time::Instant::now().duration_since(single_tread_start);

    println!("CPU single thread time: {:.2?}", one_thread_time);

    cpu_img_single_threaded.save("cpu_img_single_treaded.png").expect("failed to save image");

    let multi_thread_start = time::Instant::now();
    let cpu_img_multi_threaded = cpu::setup::launch_par(cam_pos, resolution, fov, view_vector);
    let multi_thread_time = time::Instant::now().duration_since(multi_thread_start);

    println!("CPU multi thread time: {:.2?}", multi_thread_time);

    cpu_img_multi_threaded.save("cpu_img_multi_threade.png").expect("failed to save image");

    let (buf, future, query_pool, device) = gpu::setup::launch(cam_pos, resolution, fov, view_vector);
    future.wait(None).unwrap();

    let mut timestamps:[u32; 3] = [0; 3];
    let query_res = query_pool
    .get_results(
        0..3, 
        &mut timestamps, 
        vulkano::query::QueryResultFlags::WAIT)
    .expect("failed to get query results");

    if !query_res {
        panic!{"failed to get query results"};
    }

    let gpu_time_with_copy = Duration::from_nanos(((timestamps[2] - timestamps[0]) as f32 * device.physical_device().properties().timestamp_period) as u64);
    let gpu_time_without_copy = Duration::from_nanos(((timestamps[1] - timestamps[0]) as f32 * device.physical_device().properties().timestamp_period) as u64);

    println!("GPU time without copying: {:.2?}", gpu_time_without_copy);
    println!("GPU time with copying: {:.2?}", gpu_time_with_copy);
    println!("Copy time: {:.2?}", gpu_time_with_copy - gpu_time_without_copy);

    let buffer_content = buf.read().expect("failed to read buffer");
    let gpu_img = ImageBuffer::<Rgba<u8>, _>::from_raw(resolution.x, resolution.y, &buffer_content[..]).unwrap();
    
    gpu_img.save("gpu_image.png").expect("failed to save image");
}
