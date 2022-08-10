#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::raw::c_void;

    unsafe fn cudacheck(err: cudaError_t) {
        // let cudaError_t err = cmd;
        if err != cudaError_cudaSuccess {
            panic!(
                "Failed: Cuda error '{:?}'\n",
                std::ffi::CStr::from_ptr(cudaGetErrorString(err))
            );
        }
    }

    unsafe fn ncclcheck(err: ncclResult_t) {
        // let cudaError_t err = cmd;
        if err != ncclResult_t_ncclSuccess {
            panic!(
                "Failed: NCCL error '{:?}'\n",
                std::ffi::CStr::from_ptr(ncclGetErrorString(err))
            );
        }
    }

    #[test]
    fn test_nccl_example() {
        unsafe {
            // ncclComm_t comms[4];
            let n_dev = 4;
            let mut comms: Vec<ncclComm_t> = vec![std::ptr::null_mut(); n_dev];

            // //managing 4 devices
            // int nDev = 4;
            // int size = 32*1024*1024;
            let size = 32 * 1024 * 1024;
            // int devs[4] = { 0, 1, 2, 3 };
            let devs: Vec<_> = (0..n_dev as i32).collect();

            // //allocating and initializing device buffers
            // float** sendbuff = (float**)malloc(nDev * sizeof(float*));
            let sendbuff: *mut *mut f32 =
                libc::malloc(std::mem::size_of::<*mut f32>() * n_dev) as *mut *mut f32;
            // float** recvbuff = (float**)malloc(nDev * sizeof(float*));
            let recvbuff: *mut *mut f32 =
                libc::malloc(std::mem::size_of::<*mut f32>() * n_dev) as *mut *mut f32;
            // cudaStream_t* s = (cudaStream_t*)malloc(sizeof(cudaStream_t)*nDev);
            let s = libc::malloc(std::mem::size_of::<cudaStream_t>() * n_dev) as *mut cudaStream_t;

            // for (int i = 0; i < nDev; ++i) {
            for i in 0..n_dev as i32 {
                //   CUDACHECK(cudaSetDevice(i));
                cudacheck(cudaSetDevice(i));
                //   CUDACHECK(cudaMalloc(sendbuff + i, size * sizeof(float)));
                cudacheck(cudaMalloc(
                    sendbuff.offset(i as isize) as *mut *mut c_void,
                    size * std::mem::size_of::<f32>() as u64,
                ));
                // //   CUDACHECK(cudaMalloc(recvbuff + i, size * sizeof(float)));
                cudacheck(cudaMalloc(
                    recvbuff.offset(i as isize) as *mut *mut c_void,
                    size * std::mem::size_of::<f32>() as u64,
                ));
                // //   CUDACHECK(cudaMemset(sendbuff[i], 1, size * sizeof(float)));
                println!("Pointer {:?}", sendbuff.offset(i as isize) as *mut c_void);
                cudacheck(cudaMemset(
                    (*sendbuff.offset(i as isize)) as *mut c_void,
                    1i32,
                    (size * std::mem::size_of::<f32>() as u64) as crate::size_t,
                ));
                // //   CUDACHECK(cudaMemset(recvbuff[i], 0, size * sizeof(float)));
                // cudacheck(cudaMemset(recvbuff[i], 0, size * sizeof(float)));
                cudacheck(cudaMemset(
                    (*recvbuff.offset(i as isize)) as *mut c_void,
                    1i32,
                    (size * std::mem::size_of::<f32>() as u64) as crate::size_t,
                ));
                // //   CUDACHECK(cudaStreamCreate(s+i));
                cudacheck(cudaStreamCreate(s.offset(i as isize)));
                // // }
            }

            // //initializing NCCL
            // NCCLCHECK(ncclCommInitAll(comms, nDev, devs));
            ncclcheck(ncclCommInitAll(
                comms.as_mut_ptr(),
                n_dev as i32,
                devs.as_ptr(),
            ));

            //  //calling NCCL communication API. Group API is required when using
            //  //multiple devices per thread
            // NCCLCHECK(ncclGroupStart());
            // for (int i = 0; i < nDev; ++i)
            //   NCCLCHECK(ncclAllReduce((const void*)sendbuff[i], (void*)recvbuff[i], size, ncclFloat, ncclSum,
            //       comms[i], s[i]));
            // NCCLCHECK(ncclGroupEnd());

            // //synchronizing on CUDA streams to wait for completion of NCCL operation
            // for (int i = 0; i < nDev; ++i) {
            //   CUDACHECK(cudaSetDevice(i));
            //   CUDACHECK(cudaStreamSynchronize(s[i]));
            // }

            // //free device buffers
            // for (int i = 0; i < nDev; ++i) {
            //   CUDACHECK(cudaSetDevice(i));
            //   CUDACHECK(cudaFree(sendbuff[i]));
            //   CUDACHECK(cudaFree(recvbuff[i]));
            // }

            // //finalizing NCCL
            // for(int i = 0; i < nDev; ++i)
            //     ncclCommDestroy(comms[i]);

            // printf("Success \n");
            // return 0;
        }
    }
}
