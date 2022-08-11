import torch
import datetime
import os

def initialize_torch_distributed():
    rank = int(os.getenv('RANK', '0'))
    world_size = int(os.getenv("WORLD_SIZE", '1'))

    if torch.cuda.is_available():
        # initialized `torch.distributed`
        # Set the device id.
        assert world_size <= torch.cuda.device_count(), "Each process is one gpu"
        device = rank % torch.cuda.device_count()
        torch.cuda.set_device(device)
        backend = "nccl"
    else:
        backend = "gloo"


    # Call the init process.
    init_method = 'tcp://'
    master_ip = os.getenv('MASTER_ADDR', 'localhost')
    master_port = os.getenv('MASTER_PORT', '6000')
    init_method += master_ip + ':' + master_port
    torch.distributed.init_process_group(
        backend=backend,
        world_size=world_size,
        rank=rank,
        init_method=init_method)

    return torch.distributed.distributed_c10d._get_default_group()

def print_rank(string):
    group = torch.distributed.distributed_c10d._get_default_group()
    print(f"[R {group.rank()}]: {string}")

def main():
    process_group = initialize_torch_distributed()
    tp_rank = process_group.rank()
    tp_world_size = process_group.size()

    out = torch.ones((32, 1024, 1024)).to(torch.float).to(f'cuda:{tp_rank}')

    for i in range(4):
        torch.cuda.synchronize()
        start = datetime.datetime.now()
        torch.distributed.all_reduce(out, group=process_group, async_op=True)
        stop = datetime.datetime.now() - start
        torch.cuda.synchronize()

        if tp_rank == 0:
            print(out[:2, :2, :2])
            print_rank(f"{stop}")

if __name__ == "__main__":
    main()
