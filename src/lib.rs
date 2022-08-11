#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::raw::c_void;
    use thiserror::Error;

    #[derive(Error, Debug)]
    struct CudaError(String);
    impl std::fmt::Display for CudaError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

    unsafe fn cudacheck(err: cudaError_t) -> Result<(), CudaError> {
        if err != cudaError_cudaSuccess {
            Err(CudaError(format!(
                "Failed: Cuda error '{:?}'\n",
                std::ffi::CStr::from_ptr(cudaGetErrorString(err))
            )))
        } else {
            Ok(())
        }
    }

    #[derive(Error, Debug)]
    struct NcclError(String);

    impl std::fmt::Display for NcclError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

    unsafe fn ncclcheck(err: ncclResult_t) -> Result<(), NcclError> {
        // let cudaError_t err = cmd;
        if err != ncclResult_t_ncclSuccess {
            Err(NcclError(format!(
                "Failed: NCCL error '{:?}'\n",
                std::ffi::CStr::from_ptr(ncclGetErrorString(err))
            )))
        } else {
            Ok(())
        }
    }

    #[test]
    fn test_nccl_example() {
        unsafe {
            // ncclComm_t comms[4];
            let n_dev = 16;
            let mut comms: Vec<ncclComm_t> = vec![std::ptr::null_mut(); n_dev];

            // //managing 4 devices
            // int nDev = 4;
            // int size = 32*1024*1024;
            let size = 32 * 1024 * 1024;
            // int devs[4] = { 0, 1, 2, 3 };
            let devs: Vec<_> = (0..n_dev as i32).collect();

            // //allocating and initializing device buffers
            // float** sendbuff = (float**)malloc(nDev * sizeof(float*));
            let mut sendbuff: Vec<*mut f32> = vec![std::ptr::null_mut(); n_dev];
            // float** recvbuff = (float**)malloc(nDev * sizeof(float*));
            let mut recvbuff: Vec<*mut f32> = vec![std::ptr::null_mut(); n_dev];
            // cudaStream_t* s = (cudaStream_t*)malloc(sizeof(cudaStream_t)*nDev);
            let mut s: Vec<cudaStream_t> = vec![std::ptr::null_mut(); n_dev];

            // for (int i = 0; i < nDev; ++i) {
            for i in 0..n_dev {
                //   CUDACHECK(cudaSetDevice(i));
                cudacheck(cudaSetDevice(i as i32)).unwrap();
                //   CUDACHECK(cudaMalloc(sendbuff + i, size * sizeof(float)));
                cudacheck(cudaMalloc(
                    &mut sendbuff[i] as *mut *mut f32 as *mut *mut c_void,
                    size * std::mem::size_of::<f32>() as u64,
                ))
                .unwrap();
                // //   CUDACHECK(cudaMalloc(recvbuff + i, size * sizeof(float)));
                cudacheck(cudaMalloc(
                    &mut recvbuff[i] as *mut *mut f32 as *mut *mut c_void,
                    size * std::mem::size_of::<f32>() as u64,
                ))
                .unwrap();
                // //   CUDACHECK(cudaMemset(sendbuff[i], 1, size * sizeof(float)));
                cudacheck(cudaMemset(
                    sendbuff[i] as *mut c_void,
                    1,
                    (size * std::mem::size_of::<f32>() as u64) as size_t,
                ))
                .unwrap();
                // //   CUDACHECK(cudaMemset(recvbuff[i], 0, size * sizeof(float)));
                // cudacheck(cudaMemset(recvbuff[i], 0, size * sizeof(float))).unwrap();
                cudacheck(cudaMemset(
                    recvbuff[i] as *mut c_void,
                    0,
                    (size * std::mem::size_of::<f32>() as u64) as size_t,
                ))
                .unwrap();
                // //   CUDACHECK(cudaStreamCreate(s+i));
                cudacheck(cudaStreamCreate(&mut s[i])).unwrap();
                // // }
            }

            // //initializing NCCL
            // NCCLCHECK(ncclCommInitAll(comms, nDev, devs));
            ncclcheck(ncclCommInitAll(
                comms.as_mut_ptr(),
                n_dev as i32,
                devs.as_ptr(),
            ))
            .unwrap();

            //  //calling NCCL communication API. Group API is required when using
            //  //multiple devices per thread
            // NCCLCHECK(ncclGroupStart());
            for _ in 0..10 {
                let start = std::time::Instant::now();
                ncclcheck(ncclGroupStart()).unwrap();
                // for (int i = 0; i < nDev; ++i)
                for i in 0..n_dev {
                    // ncclcheck(ncclAllReduce((const void*)sendbuff[i], (void*)recvbuff[i], size, ncclFloat, ncclSum,
                    //     comms[i], s[i]));
                    ncclcheck(ncclAllReduce(
                        sendbuff[i] as *const c_void,
                        recvbuff[i] as *mut c_void,
                        size,
                        ncclDataType_t_ncclFloat,
                        ncclRedOp_t_ncclSum,
                        comms[i],
                        s[i],
                    ))
                    .unwrap();
                }
                // NCCLCHECK(ncclGroupEnd());
                ncclcheck(ncclGroupEnd()).unwrap();

                // //synchronizing on CUDA streams to wait for completion of NCCL operation
                // for (int i = 0; i < nDev; ++i) {
                for i in 0..n_dev {
                    //   CUDACHECK(cudaSetDevice(i));
                    cudacheck(cudaSetDevice(i as i32)).unwrap();
                    //   CUDACHECK(cudaStreamSynchronize(s[i]));
                    cudacheck(cudaStreamSynchronize(s[i])).unwrap();
                    // }
                }
                let elapsed = start.elapsed();
                println!("Elapsed {:?}", elapsed);
            }

            //free device buffers
            // for (int i = 0; i < nDev; ++i) {
            for i in 0..n_dev {
                //   CUDACHECK(cudaSetDevice(i));
                cudacheck(cudaSetDevice(i as i32)).unwrap();
                //   CUDACHECK(cudaFree(sendbuff[i]));
                cudacheck(cudaFree(sendbuff[i] as *mut c_void)).unwrap();
                //   CUDACHECK(cudaFree(recvbuff[i]));
                cudacheck(cudaFree(recvbuff[i] as *mut c_void)).unwrap();
                // }
            }

            // //finalizing NCCL
            // for(int i = 0; i < nDev; ++i)
            for i in 0..n_dev {
                //     ncclCommDestroy(comms[i]);
                ncclCommDestroy(comms[i]);
            }

            // printf("Success \n");
            // return 0;
        }
    }

    #[derive(Debug, Error)]
    enum ThreadError{
        #[error("Cuda error")]
        CudaError(#[from] CudaError),
        #[error("Nccl error")]
        NcclError(#[from] NcclError),
    }

    unsafe fn thread_inner(ranks: i32, rank: i32, unique_id: ncclUniqueId) -> Result<Vec<std::time::Duration>, ThreadError>{
            let size = 32 * 1024 * 1024;
                    let mut comm: ncclComm_t = std::ptr::null_mut();
                    let mut sendbuff = std::ptr::null_mut();
                    let mut recvbuff = std::ptr::null_mut();

                    cudacheck(cudaSetDevice(rank))?;
                    ncclcheck(ncclCommInitRank(&mut comm, ranks, unique_id, rank))?;
                    cudacheck(cudaMalloc(
                        &mut sendbuff as *mut *mut f32 as *mut *mut c_void,
                        size * std::mem::size_of::<f32>() as u64,
                    ))
                    ?;
                    cudacheck(cudaMalloc(
                        &mut recvbuff as *mut *mut f32 as *mut *mut c_void,
                        size * std::mem::size_of::<f32>() as u64,
                    ))
                    ?;
                    cudacheck(cudaMemset(
                        sendbuff as *mut c_void,
                        1,
                        (size * std::mem::size_of::<f32>() as u64) as size_t,
                    ))
                    ?;
                    cudacheck(cudaMemset(
                        recvbuff as *mut c_void,
                        0,
                        (size * std::mem::size_of::<f32>() as u64) as size_t,
                    ))
                    ?;

                    let mut stream = std::ptr::null_mut();
                    cudacheck(cudaStreamCreate(&mut stream))?;
                    
                    let mut timings = vec![];
                    for _ in 0..10{
                    let start = std::time::Instant::now();
                    ncclcheck(ncclAllReduce(
                        sendbuff as *const c_void,
                        recvbuff as *mut c_void,
                        size,
                        ncclDataType_t_ncclFloat,
                        ncclRedOp_t_ncclSum,
                        comm,
                        stream,
                    ))
                    ?;
                    cudacheck(cudaStreamSynchronize(stream))?;
                    let elapsed = start.elapsed();
                    timings.push(elapsed);
                    }

                    cudacheck(cudaFree(sendbuff as *mut c_void))?;
                    cudacheck(cudaFree(recvbuff as *mut c_void))?;
                    ncclCommDestroy(comm);
                    Ok( timings)
    }

    #[test]
    fn test_nccl_thread() {
        unsafe {
            let n_dev = 16;
            let (s, r) = std::sync::mpsc::channel();

            //initializing NCCL
            let internal: [i8; 128] = [0; 128];
            let mut unique_id = ncclUniqueId { internal };
            ncclcheck(ncclGetUniqueId(&mut unique_id)).unwrap();

            // for (int i = 0; i < nDev; ++i) {
            for i in 0..n_dev {
                let s = s.clone();
                let unique_id = unique_id.clone();
                std::thread::spawn(move || {
                    let out = thread_inner(n_dev as i32, i as i32, unique_id);
                    s.send((out, i)).unwrap();


                });
            }

            for _ in 0..n_dev {
                let (out, rank) = r.recv().unwrap();
                let out = out.unwrap();
                println!("Received {:?} {:?}", rank, out);
            }
        }
    }
}
