use glam::{UVec2, Mat4, Vec3};
use std::sync::Arc;

use vulkano::{
    library::VulkanLibrary,
    instance::{Instance, InstanceCreateInfo},
    device:: {
        QueueFlags, Device, DeviceCreateInfo, QueueCreateInfo,
        physical::{PhysicalDeviceType, PhysicalDevice}, Queue
    },
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
    buffer::{Buffer, BufferCreateInfo, BufferUsage, subbuffer::Subbuffer},
    image:: {
        Image, ImageCreateInfo, ImageType, ImageUsage,
        view::ImageView
    },
    command_buffer::{
        AutoCommandBufferBuilder,
        allocator::StandardCommandBufferAllocator,
        CommandBufferUsage, CopyImageToBufferInfo, CommandBufferExecFuture
    },
    format::Format,
    pipeline::{
        ComputePipeline, Pipeline, PipelineBindPoint, PipelineLayout, PipelineShaderStageCreateInfo,
        compute::ComputePipelineCreateInfo,
        layout::PipelineDescriptorSetLayoutCreateInfo
    },
    descriptor_set::{
        PersistentDescriptorSet, WriteDescriptorSet,
        allocator::StandardDescriptorSetAllocator
    },
    sync::{self, GpuFuture, future::{FenceSignalFuture, NowFuture}},
};

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
struct LaunchInfo {
    cam_pos: Vec3,
    fov: f32,
    lookat: Mat4
}

//unsafe impl bytemuck::Pod for Uniform {}

fn select_physical(instance: Arc<Instance>) -> Arc<PhysicalDevice> {
    instance
        .enumerate_physical_devices()
        .expect("failed to enumerate physical devices")
        .min_by_key(|p| match p.properties().device_type {
            PhysicalDeviceType::DiscreteGpu => 0,
            PhysicalDeviceType::IntegratedGpu => 1,
            PhysicalDeviceType::VirtualGpu => 2,
            PhysicalDeviceType::Cpu => 3,
            _ => 5
        })
        .expect("no devices available")
}

fn select_queue_family(physical: Arc<PhysicalDevice>) -> u32 {
    physical
        .queue_family_properties()
        .iter()
        .position(|properties|  properties.queue_flags.contains(QueueFlags::GRAPHICS))
        .expect("failed to create queue") as u32
}

fn create_device(physical: Arc<PhysicalDevice>, queue_family_index: u32) -> (Arc<Device>, impl ExactSizeIterator<Item = Arc<Queue>>) {
    Device::new(
        physical,
        DeviceCreateInfo {
            queue_create_infos: vec![QueueCreateInfo {
                queue_family_index,
                ..Default::default()
            }],
            ..Default::default()
        }
    ).expect("failed to create device")
}

pub fn launch(cam_pos: Vec3, resolution: UVec2, fov: f32, view_vector: Vec3) -> (Subbuffer<[u8]>, FenceSignalFuture<CommandBufferExecFuture<NowFuture>>) {
    let library = VulkanLibrary::new().expect("no vulkan library");
    let instance = Instance::new(
        library, 
        InstanceCreateInfo::default()
    ).expect("failed to create instance");

    let physical = select_physical(instance);

    let (device, mut queue) = create_device(physical.clone(), select_queue_family(physical.clone()));
    let queue = queue.next().unwrap();

    let mem_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));

    let image = Image::new(
        mem_allocator.clone(),
        ImageCreateInfo {
            image_type: ImageType::Dim2d,
            format: Format::R8G8B8A8_UNORM,
            extent: [resolution.x, resolution.y, 1],
            usage: ImageUsage::TRANSFER_SRC | ImageUsage::STORAGE,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
            ..Default::default()
        }
    ).expect("failed to create image buffer");

    let shader = super::shaders::cs::load(device.clone()).expect("failed to create shader module");
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

    let descriptor_set_allocator = StandardDescriptorSetAllocator::new(device.clone(), Default::default());

    let input_info = LaunchInfo {fov, cam_pos, lookat: crate::cpu::setup::lookat_matrix(cam_pos, view_vector, Vec3::new(0.0, 1.0, 0.0))};
    let launch_info_buffer = Buffer::new_slice::<u8>(
        mem_allocator.clone(), 
        BufferCreateInfo {
            usage: BufferUsage::UNIFORM_BUFFER,
            ..Default::default()
        }, 
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_RANDOM_ACCESS,
            ..Default::default()
        }, 
        bytemuck::bytes_of(&input_info).len() as u64
    ).unwrap();

    launch_info_buffer.write().unwrap().swap_with_slice(bytemuck::bytes_of_mut(&mut input_info.clone()));

    let layout = compute_pipeline.layout().set_layouts().first().unwrap();    
    let set = PersistentDescriptorSet::new(
        &descriptor_set_allocator,
        layout.clone(),
        [WriteDescriptorSet::image_view(0, view), WriteDescriptorSet::buffer(1, launch_info_buffer)],
        []
    ).unwrap();

    let buf = Buffer::from_iter(
        mem_allocator.clone(), 
        BufferCreateInfo {
            usage: BufferUsage::TRANSFER_DST,
            ..Default::default()
        }, 
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_HOST | MemoryTypeFilter::HOST_RANDOM_ACCESS,
            ..Default::default()
        }, 
        (0..resolution.x * resolution.y * 4).map(|_| 0u8)
    ).expect("failed to create buffer");

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
        .bind_pipeline_compute(compute_pipeline.clone())
        .unwrap()
        .bind_descriptor_sets(
            PipelineBindPoint::Compute,
            compute_pipeline.layout().clone(),
            0,
        set
        )
        .unwrap()
        .dispatch([resolution.x / 8, resolution.y / 8, 1])
        .unwrap()
        .copy_image_to_buffer(CopyImageToBufferInfo::image_buffer(image, buf.clone()))
        .unwrap();
    
    let command_buffer = builder.build().expect("failed to build command buffer");

    let future = sync::now(device)
        .then_execute(queue, command_buffer)
        .unwrap()
        .then_signal_fence_and_flush()
        .unwrap();
    
    (buf, future)
}
