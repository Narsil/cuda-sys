#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::{c_void, CStr, CString};
    use std::mem::size_of;
    use std::ptr::{null, null_mut};

    #[derive(Debug)]
    pub struct CudaError(String);

    #[must_use]
    pub fn checkCudaErrors(err: CUresult) -> Result<(), CudaError> {
        if err == cudaError_enum_CUDA_SUCCESS {
            Ok(())
        } else {
            unsafe {
                let mut str_ptr = null();

                cuGetErrorString(err, &mut str_ptr);
                let string = CStr::from_ptr(str_ptr).to_str().unwrap().to_string();
                // panic!("{string:?}");
                // println!("Here {string:?} {:?}", string.as_ptr());
                // // println!("Here {ptr:?} ",);
                let err = CudaError(string);
                Err(err)
            }
        }
    }

    #[test]
    fn ptx_usage() {
        const TENSOR_DIM: usize = 1024;
        unsafe {
            // CUdevice    device;
            // CUmodule    cudaModule;
            // CUcontext   context;
            // CUfunction  function;
            // CUlinkState linker;
            // int         devCount;
            let device: CUdevice = 0;
            let mut cudaModule: CUmodule = null_mut();
            let mut context: CUcontext = null_mut();
            let mut function: CUfunction = null_mut();
            let mut devCount = 0;

            // // CUDA initialization
            // checkCudaErrors(cuInit(0));
            // checkCudaErrors(cuDeviceGetCount(&devCount));
            // checkCudaErrors(cuDeviceGet(&device, 0));
            checkCudaErrors(cuInit(0)).unwrap();
            checkCudaErrors(cuDeviceGetCount(&mut devCount)).unwrap();

            // char name[128];
            // checkCudaErrors(cuDeviceGetName(name, 128, device));
            // std::cout << "Using CUDA Device [0]: " << name << "\n";
            let mut name = [0i8; 128];
            checkCudaErrors(cuDeviceGetName(&mut name as *mut i8, 128, device)).unwrap();
            println!(
                "Using CUDA Device [0]: {}",
                String::from_utf8(name.iter().map(|i| *i as u8).collect::<Vec<_>>()).unwrap()
            );

            // int devMajor, devMinor;
            // checkCudaErrors(cuDeviceComputeCapability(&devMajor, &devMinor, device));
            // std::cout << "Device Compute Capability: "
            //           << devMajor << "." << devMinor << "\n";
            // if (devMajor < 2) {
            //   std::cerr << "ERROR: Device 0 is not SM 2.0 or greater\n";
            //   return 1;
            // }
            let mut devMajor = 0;
            let mut devMinor = 0;
            checkCudaErrors(cuDeviceComputeCapability(
                &mut devMajor,
                &mut devMinor,
                device,
            ))
            .unwrap();
            println!("Device Compute Capability: {devMajor}.{devMinor}");
            if devMajor < 2 {
                panic!("ERROR: Device 0 is not SM 2.0 or greater");
            }

            let ptx_code = include_str!("kernel.ptx");
            let ptx_code = CString::new(ptx_code).unwrap();

            // Create driver context
            checkCudaErrors(cuCtxCreate_v2(&mut context, 0, device)).unwrap();

            // Create module for object
            checkCudaErrors(cuModuleLoadData(
                &mut cudaModule,
                ptx_code.as_ptr() as *const c_void,
            ))
            .unwrap();

            // Get kernel function
            let kernel_name = CString::new("kernel").unwrap();
            checkCudaErrors(cuModuleGetFunction(
                &mut function,
                cudaModule,
                kernel_name.as_ptr(),
            ))
            .unwrap();

            // // Device data
            let mut devBufferA: CUdeviceptr = 0;
            let mut devBufferB: CUdeviceptr = 0;
            let mut devBufferC: CUdeviceptr = 0;

            checkCudaErrors(cuMemAlloc_v2(
                &mut devBufferA,
                size_of::<f32>() * TENSOR_DIM,
            ))
            .unwrap();
            checkCudaErrors(cuMemAlloc_v2(
                &mut devBufferB,
                size_of::<f32>() * TENSOR_DIM,
            ))
            .unwrap();
            checkCudaErrors(cuMemAlloc_v2(
                &mut devBufferC,
                size_of::<f32>() * TENSOR_DIM,
            ))
            .unwrap();

            let mut hostA = [0.0f32; TENSOR_DIM];
            let mut hostB = [0.0f32; TENSOR_DIM];
            let mut hostC = [0.0f32; TENSOR_DIM];

            // // Populate input
            for i in 0..TENSOR_DIM {
                hostA[i] = i as f32;
                hostB[i] = (2 * i) as f32;
                hostC[i] = 0.0;
            }

            checkCudaErrors(cuMemcpyHtoD_v2(
                devBufferA,
                &hostA[0] as *const f32 as *const c_void,
                size_of::<f32>() * TENSOR_DIM,
            ))
            .unwrap();
            checkCudaErrors(cuMemcpyHtoD_v2(
                devBufferB,
                &hostB[0] as *const f32 as *const c_void,
                size_of::<f32>() * TENSOR_DIM,
            ))
            .unwrap();

            let blockSizeX = TENSOR_DIM as u32;
            let blockSizeY = 1;
            let blockSizeZ = 1;
            let gridSizeX = 1;
            let gridSizeY = 1;
            let gridSizeZ = 1;

            // // Kernel parameters
            // void *KernelParams[] = { &devBufferA, &devBufferB, &devBufferC };
            let mut kernel_params = [
                &mut devBufferA as *mut u64 as *mut c_void,
                &mut devBufferB as *mut u64 as *mut c_void,
                &mut devBufferC as *mut u64 as *mut c_void,
            ];

            println!("Launching kernel");

            // Kernel launch
            checkCudaErrors(cuLaunchKernel(
                function,
                gridSizeX,
                gridSizeY,
                gridSizeZ,
                blockSizeX,
                blockSizeY,
                blockSizeZ,
                0,
                null_mut(),
                &mut kernel_params as *mut [*mut c_void; 3] as *mut *mut c_void,
                null_mut(),
            ))
            .unwrap();

            // Retrieve device data
            checkCudaErrors(cuMemcpyDtoH_v2(
                &mut hostC[0] as *mut f32 as *mut c_void,
                devBufferC,
                size_of::<f32>() * TENSOR_DIM,
            ))
            .unwrap();

            let mut string = String::from("Result:\n");
            for i in 0..TENSOR_DIM {
                string.push_str(&format!("{} + {} = {}\n", hostA[i], hostB[i], hostC[i]));
            }
            println!("{}", string);

            // Clean up after ourselves
            checkCudaErrors(cuMemFree_v2(devBufferA)).unwrap();
            checkCudaErrors(cuMemFree_v2(devBufferB)).unwrap();
            checkCudaErrors(cuMemFree_v2(devBufferC)).unwrap();
            checkCudaErrors(cuModuleUnload(cudaModule)).unwrap();
            checkCudaErrors(cuCtxDestroy_v2(context)).unwrap();

            // return 0;
        }
    }
}
