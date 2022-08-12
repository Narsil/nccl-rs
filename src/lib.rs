use nccl_sys::*;
use tch::{kind::Kind, Tensor};
use thiserror::Error;

#[derive(Error, Debug)]
pub struct CudaError(String);
impl std::fmt::Display for CudaError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

unsafe fn cudacheck(err: cudaError_t) -> Result<(), CudaError> {
    if err != cudaError_cudaSuccess {
        Err(CudaError(format!(
            "{:?}",
            std::ffi::CStr::from_ptr(cudaGetErrorString(err))
        )))
    } else {
        Ok(())
    }
}

#[derive(Error, Debug)]
pub struct NcclError(String);

impl std::fmt::Display for NcclError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

unsafe fn ncclcheck(err: ncclResult_t) -> Result<(), NcclError> {
    // let cudaError_t err = cmd;
    if err != ncclResult_t_ncclSuccess {
        Err(NcclError(format!(
            "{:?}",
            std::ffi::CStr::from_ptr(ncclGetErrorString(err))
        )))
    } else {
        Ok(())
    }
}

#[derive(Debug, Error)]
pub enum ThreadGroupError {
    #[error("Cuda error {0}")]
    CudaError(#[from] CudaError),
    #[error("Nccl error {0}, use NCCL_DEBUG=INFO to get more information")]
    NcclError(#[from] NcclError),
}

pub struct ThreadGroup {
    // ranks: i32,
    // rank: i32,
    comm: ncclComm_t,
    stream: *mut CUstream_st,
}

fn kind_to_nccl(kind: Kind) -> ncclDataType_t {
    match kind {
        Kind::Half => ncclDataType_t_ncclFloat16,
        Kind::Float => ncclDataType_t_ncclFloat,
        Kind::Double => ncclDataType_t_ncclFloat64,
        Kind::Int8 => ncclDataType_t_ncclInt8,
        // Kind::Int16 => ncclDataType_t_ncclInt16,
        Kind::Int => ncclDataType_t_ncclInt32,
        Kind::Uint8 => ncclDataType_t_ncclChar,
        // Kind::Uint16 => ncclDataType_t_ncclUint16,
        // Kind::Uint32 => ncclDataType_t_ncclUint32,
        // Kind::Uint64 => ncclDataType_t_ncclUint64,
        _ => todo!(),
    }
}

impl ThreadGroup {
    pub fn new(ranks: i32, rank: i32, unique_id: ncclUniqueId) -> Result<Self, ThreadGroupError> {
        let mut comm: ncclComm_t = std::ptr::null_mut();
        let mut stream = std::ptr::null_mut();
        unsafe {
            cudacheck(cudaSetDevice(rank))?;
            ncclcheck(ncclCommInitRank(&mut comm, ranks, unique_id, rank))?;
            cudacheck(cudaStreamCreate(&mut stream))?;
            cudacheck(cudaStreamSynchronize(stream))?;
        }
        Ok(Self {
            // ranks,
            // rank,
            comm,
            stream,
        })
    }

    pub fn new_id() -> Result<ncclUniqueId, ThreadGroupError> {
        let internal: [i8; 128] = [0; 128];
        let mut unique_id = ncclUniqueId { internal };
        unsafe {
            ncclcheck(ncclGetUniqueId(&mut unique_id))?;
        }
        Ok(unique_id)
    }

    pub fn all_reduce(&self, x: Tensor) -> Result<Tensor, ThreadGroupError> {
        let size: i64 = x.size().into_iter().product();
        let nccl_type = kind_to_nccl(x.kind());
        unsafe {
            ncclcheck(ncclAllReduce(
                x.data_ptr(),
                x.data_ptr(),
                size as u64,
                nccl_type,
                ncclRedOp_t_ncclSum,
                self.comm,
                self.stream,
            ))?;
        }
        Ok(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::{kind::Kind, Cuda, Device};

    #[test]
    fn simple_test() {
        let world_size = Cuda::device_count() as i32;
        let id = ThreadGroup::new_id().unwrap();
        for rank in 0..world_size {
            std::thread::spawn(move || {
                let out = Tensor::ones(
                    &[32, 1024, 1024],
                    (Kind::Float, Device::Cuda(rank as usize)),
                );
                let group = ThreadGroup::new(world_size, rank, id).unwrap();
                let out = group.all_reduce(out).unwrap();
                let values: Vec<_> = Vec::<f64>::from(out).into_iter().take(5).collect();

                assert_eq!(values, vec![world_size as f64; 5]);
            });
        }
    }

    #[test]
    fn simple_test_2_gpus() {
        let world_size = Cuda::device_count() as i32;
        if world_size < 2 {
            return;
        }
        // We force the number of the world so we can statically check the result
        let world_size = 2;
        let id = ThreadGroup::new_id().unwrap();
        for rank in 0..world_size {
            std::thread::spawn(move || {
                let out = Tensor::ones(
                    &[32, 1024, 1024],
                    (Kind::Float, Device::Cuda(rank as usize)),
                );
                let group = ThreadGroup::new(world_size, rank, id).unwrap();
                let out = group.all_reduce(out).unwrap();
                let values: Vec<_> = Vec::<f64>::from(out).into_iter().take(5).collect();

                assert_eq!(values, vec![2.0; 5]);
            });
        }
    }
}
