use vulkano::{
    library::VulkanLibrary,
    instance::{Instance, InstanceCreateInfo},
    device:: {
        QueueFlags, Device, DeviceCreateInfo, QueueCreateInfo,
        physical::PhysicalDeviceType
    },
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
    buffer::{Buffer, BufferCreateInfo, BufferUsage},
    image:: {
        Image, ImageCreateInfo, ImageType, ImageUsage,
        view::ImageView
    },
    command_buffer::{
        ClearColorImageInfo, AutoCommandBufferBuilder,
        allocator::{StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo},
        CommandBufferUsage, PrimaryAutoCommandBuffer
    },
    format::{Format, ClearColorValue},
    pipeline::{
        ComputePipeline, Pipeline, PipelineBindPoint, PipelineLayout, PipelineShaderStageCreateInfo,
        compute::ComputePipelineCreateInfo,
        layout::PipelineDescriptorSetLayoutCreateInfo
    },
    descriptor_set::{
        PersistentDescriptorSet, WriteDescriptorSet,
        allocator::StandardDescriptorSetAllocator
    }
};
use image::{Rgb, RgbImage};
use glam::{Mat4, UVec2, Vec2, Vec3, Vec4, Vec4Swizzles};
use std::sync::{Arc, Mutex};
use rayon::prelude::*;

pub mod cpu;
pub mod gpu;
pub mod consts;

use cpu::marching::render;
use consts::*;

fn cam_dir(fov: f32, size: Vec2, frag_coord: Vec2) -> Vec3 {
    let xy = frag_coord - (size / 2.0);
    let z = size.y / (fov.to_radians()/2.0).tan();

    Vec3::new(xy.x, -xy.y, -z).normalize()
}

fn lookat_matrix(cam_pos: Vec3, center: Vec3, up: Vec3) -> Mat4 {
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

    let library = VulkanLibrary::new().expect("no vulkan library");
    let instance = Instance::new(
        library, 
        InstanceCreateInfo::default()
    ).expect("failed to create instance");

    let physical = instance
        .enumerate_physical_devices()
        .expect("failed to enumerate physical devices")
        .min_by_key(|p| match p.properties().device_type {
            PhysicalDeviceType::DiscreteGpu => 0,
            PhysicalDeviceType::IntegratedGpu => 1,
            PhysicalDeviceType::VirtualGpu => 2,
            PhysicalDeviceType::Cpu => 3,
            _ => 5
        })
        .expect("no devices available");

    let queue_family_index = physical.clone()
        .queue_family_properties()
        .iter()
        .position(|properties|  properties.queue_flags.contains(QueueFlags::GRAPHICS))
        .expect("failed to create queue") as u32;

    let (device, mut queue) = Device::new(
        physical.clone(),
        DeviceCreateInfo {
            queue_create_infos: vec![QueueCreateInfo {
                queue_family_index,
                ..Default::default()
            }],
            enabled_extensions: physical.supported_extensions().to_owned(),
            ..Default::default()
        }
    ).expect("failed to create device");

    let queue = queue.next().unwrap();

    let mem_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));

    let image = Image::new(
        mem_allocator.clone(),
        ImageCreateInfo {
            image_type: ImageType::Dim2d,
            format: Format::R8G8B8_UNORM,
            extent: [resolution.x, resolution.y, 1],
            usage: ImageUsage::TRANSFER_SRC | ImageUsage::TRANSFER_DST,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
            ..Default::default()
        }
    ).expect("failed to create image buffer");

    let shader = gpu::shaders::cs::load(device.clone()).expect("failed to create shader module");
    let cs = shader.entry_point("main").unwrap();
    let stage = PipelineShaderStageCreateInfo::new(cs);
    let layout = PipelineLayout::new(
        device.clone(),
        PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
            .into_pipeline_layout_create_info(device.clone())
            .unwrap()
    ).unwrap();

    let compute_pipeline = ComputePipeline::new(
        device.clone(),
        None,
        ComputePipelineCreateInfo::stage_layout(stage, layout)
    ).expect("failed to create compute pipeline");

    let view = ImageView::new_default(image.clone()).unwrap();

    let descriptor_set_alloca

    let command_buffer_allocator = StandardCommandBufferAllocator::new(
            device.clone(),
            Default::default()
    );

    let mut builder = AutoCommandBufferBuilder::primary(
        &command_buffer_allocator,
        queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit
    ).expect("failed to create command buffer builder");

    builder
        .clear_color_image(ClearColorImageInfo {
            clear_value: ClearColorValue::Float([0.0, 0.0, 1.0, 1.0]),
            ..ClearColorImageInfo::image(image.clone())
        }).expect("failed to apple clear_color_image");
    
    let command_buffer = builder.build().expect("failed to build command buffer");

    
    /*
    let pixel_count = Arc::new(Mutex::new(0));
    

    let img_buf: Vec<_> = (0..resolution.x * resolution.y)
        .into_par_iter()
        .map( |pixel_num| {
            let x = pixel_num / resolution.y;
            let y = pixel_num % resolution.y;

            let view_dir = cam_dir(
                fov, 
                Vec2::new(resolution.x as f32, resolution.y as f32), 
                Vec2::new(x as f32, y as f32)
            );

            let view_to_world = lookat_matrix(
                cam_pos, 
                view_vector, 
                Vec3::new(0.0, 1.0, 0.0)
            );
            let world_dir = (view_to_world * Vec4::new(view_dir.x, view_dir.y, view_dir.z, 0.0)).xyz();

            let color = render(cam_pos, world_dir, MIN_DIST, MAX_DIST);


            let mut pixel_count = pixel_count.lock().unwrap();
            *pixel_count += 1;
            if *pixel_count % (resolution.x * resolution.y / 10) == 0 {
                println!("Rendered {}%", *pixel_count * 100 / (resolution.x * resolution.y))
            }
            
            (x, y, Rgb([
                (color.x * 255.0) as u8, 
                (color.y * 255.0) as u8, 
                (color.z * 255.0) as u8,
            ]))

            //[(color.x * 255.0) as u8, (color.y * 255.0) as u8, (color.z * 255.0) as u8]
        })
        .collect();

    println!("Saving image...");

    let mut img = RgbImage::new(resolution.x, resolution.y);

    for (x, y, pixel) in img_buf {
        img.put_pixel(x, y, pixel);
    }

    //let img = RgbImage::from_vec(resolution[0], resolution[1], img_buf).unwrap();

    img.save("image.png").unwrap();
    */
}
