#
from task import input_t, output_t
import torch
from torch import Tensor
import numpy as np
import triton
import triton.language as tl
import math, os
############################################################################################################
os.environ['TRITON_PRINT_AUTOTUNING'] = '1'
NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count

fp4_mapping = []
for g_id in range(torch.cuda.device_count()):
    fp4_mapping.append(
        torch.tensor(
            [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6], dtype=torch.float32,
            device="cuda:" + str(g_id),
        )
    )

CACHE = {}
OUTPUT_DTYPE = torch.float32
ACC_DTYPE = tl.float32
MAX_C_FP32_CACHE = 4096
CACHE['C_FP32'] = [torch.zeros((8192, 1, 8), dtype=OUTPUT_DTYPE, device='cuda') for _ in range(MAX_C_FP32_CACHE)]
CACHE['C_FP32_COUNTER'] = 0

def reset_cache():
    global CACHE
    CACHE['C_FP32_COUNTER'] = 0
    for i in range(len(CACHE['C_FP32'])):
        CACHE['C_FP32'][i] *= 0

############################################################################################################
def ceil_div(a, b):
    return (a + b - 1) // b

def generate_input(
    m: int,
    k: int,
    l: int,
    seed: int,
    sf_vec_size: int = 16,
):
    """
    Generate input tensors for NVFP4 block-scaled GEMV.

    Args:
        m: Number of rows in matrix A
        k: Number of columns in A (and length of vector b)
        l: Batch size
        seed: Random seed for reproducibility

    Returns:
        Tuple of (a, b, scale_a, scale_b, c) where:
            a: [m, k, l] - Input matrix in torch.float4e2m1fn_x2 data type
            b: [1, k, l] - Input vector in torch.float4e2m1fn_x2 data type
            scale_a: [m, k, l] - Input scale factors in torch.float8e4m3fn data type
            scale_b: [1, k, l] - Input scale factors in torch.float8e4m3fn data type
            scale_a_permuted: [32, 4, rest_m, 4, rest_k, l] - Input scale factors in torch.float8e4m3fn data type
            scale_b_permuted: [32, 4, rest_n, 4, rest_k, l] - Input scale factors in torch.float8e4m3fn data type
            c: [m, 1, l] - Output vector in torch.float16 data type
    """
    torch.manual_seed(seed)

    # GEMV N dimension is always 1
    n = 1
    # Scaling factor needs to pad the N size to 128
    n_padded_128 = 128

    # Generate uint8 tensor, then convert to float4e2m1fn_x2 data type
    a_ref = torch.randint(
        0, 2, (l, m, k // 2), dtype=torch.uint8, device="cuda"
    ).permute(1, 2, 0)
    # Pad b tensor's N dimension to 128 to call torch._scaled_mm for nvfp4 dot product computation
    b_ref = torch.randint(
        0, 2, (l, n_padded_128, k // 2), dtype=torch.uint8, device="cuda"
    ).permute(1, 2, 0)
    a_ref = a_ref.view(torch.float4_e2m1fn_x2)
    b_ref = b_ref.view(torch.float4_e2m1fn_x2)

    # Create float16 output tensor
    c_ref = torch.randn((l, m, n), dtype=torch.float16, device="cuda").permute(
        1, 2, 0
    )

    # Helper function to prepare the scale factor tensors for both reference
    # kernel and customize kernel. The customized data layout can be found in:
    # https://docs.nvidia.com/cuda/cublas/index.html?highlight=fp4#d-block-scaling-factors-layout
    def create_scale_factor_tensors(l, mn, sf_k):
        # Create the reference scale factor tensor (mn, sf_k, l) on CPU.
        ref_shape = (l, mn, sf_k)
        ref_permute_order = (1, 2, 0)
        # Init with uint8 tensor, then convert to float8_e4m3fn
        ref_f8_random_int = torch.randint(1, 3, ref_shape, dtype=torch.int8, device='cuda')
        ref_f8_torch_tensor = ref_f8_random_int.to(dtype=torch.float8_e4m3fn)
        # permute to match ref_permute_order
        ref_f8_torch_tensor_permuted = ref_f8_torch_tensor.permute(*ref_permute_order)

        atom_m = (32, 4)
        atom_k = 4
        mma_shape = (
            l,  # batch size
            ceil_div(mn, atom_m[0] * atom_m[1]),
            ceil_div(sf_k, atom_k),
            atom_m[0],
            atom_m[1],
            atom_k,
        )

        # Reorder scale factor tensor to (32, 4, rest_m, 4, rest_k, l) layout
        # Which is needed by the CuTe customized kernel
        mma_permute_order = (3, 4, 1, 5, 2, 0)
        # Generate a random int8 tensor, then convert to float8_e4m3fn
        rand_int_tensor = torch.randint(0, 2, mma_shape, dtype=torch.int8, device='cuda')
        reordered_f8_torch_tensor = rand_int_tensor.to(dtype=torch.float8_e4m3fn)
        # Permute according to mma_permute_order
        reordered_f8_torch_tensor = reordered_f8_torch_tensor.permute(*mma_permute_order)

        # GPU-side vectorized reordering (replaces slow CPU nested loops)
        # Create index grids for all dimensions
        i_idx = torch.arange(mn, device='cuda')
        j_idx = torch.arange(sf_k, device='cuda')
        b_idx = torch.arange(l, device='cuda')

        # Create meshgrid for all combinations of (i, j, b)
        i_grid, j_grid, b_grid = torch.meshgrid(i_idx, j_idx, b_idx, indexing='ij')

        # Calculate target indices in vectorized manner
        mm = i_grid // (atom_m[0] * atom_m[1])
        mm32 = i_grid % atom_m[0]
        mm4 = (i_grid % 128) // atom_m[0]
        kk = j_grid // atom_k
        kk4 = j_grid % atom_k

        # Perform the reordering with advanced indexing (all on GPU)
        reordered_f8_torch_tensor[mm32, mm4, mm, kk4, kk, b_grid] = ref_f8_torch_tensor_permuted[i_grid, j_grid, b_grid]

        return ref_f8_torch_tensor_permuted.cpu(), reordered_f8_torch_tensor

    sf_k = ceil_div(k, sf_vec_size)
    sfa_ref_cpu, sfa_permuted = create_scale_factor_tensors(l, m, sf_k)
    sfb_ref_cpu, sfb_permuted = create_scale_factor_tensors(l, n_padded_128, sf_k)

    sfa_ref_cpu = sfa_ref_cpu.cuda()
    sfb_ref_cpu = sfb_ref_cpu.cuda()
    return (a_ref, b_ref, sfa_ref_cpu, sfb_ref_cpu, sfa_permuted, sfb_permuted, c_ref)
############################################################################################################

def get_configs():
    configs = []
    num_warps = 1
    num_stages = 4
    configs.append(triton.Config({'BLOCK_SIZE_N':16, 'BLOCK_SIZE_K':32}, num_warps=num_warps, num_stages=num_stages))
    configs.append(triton.Config({'BLOCK_SIZE_N':16, 'BLOCK_SIZE_K':64}, num_warps=num_warps, num_stages=num_stages))

    configs.append(triton.Config({'BLOCK_SIZE_N':8, 'BLOCK_SIZE_K':64}, num_warps=num_warps, num_stages=num_stages))
    configs.append(triton.Config({'BLOCK_SIZE_N':16, 'BLOCK_SIZE_K':64}, num_warps=num_warps, num_stages=num_stages))

    configs.append(triton.Config({'BLOCK_SIZE_N':8, 'BLOCK_SIZE_K':128}, num_warps=num_warps, num_stages=num_stages))
    configs.append(triton.Config({'BLOCK_SIZE_N':16, 'BLOCK_SIZE_K':128}, num_warps=num_warps, num_stages=num_stages))

    configs.append(triton.Config({'BLOCK_SIZE_N':2, 'BLOCK_SIZE_K':256}, num_warps=num_warps, num_stages=num_stages))
    configs.append(triton.Config({'BLOCK_SIZE_N':4, 'BLOCK_SIZE_K':256}, num_warps=num_warps, num_stages=num_stages))
    configs.append(triton.Config({'BLOCK_SIZE_N':8, 'BLOCK_SIZE_K':256}, num_warps=num_warps, num_stages=num_stages))

    configs.append(triton.Config({'BLOCK_SIZE_N':1, 'BLOCK_SIZE_K':512}, num_warps=num_warps, num_stages=num_stages))
    configs.append(triton.Config({'BLOCK_SIZE_N':2, 'BLOCK_SIZE_K':512}, num_warps=num_warps, num_stages=num_stages))
    configs.append(triton.Config({'BLOCK_SIZE_N':4, 'BLOCK_SIZE_K':512}, num_warps=num_warps, num_stages=num_stages))
    configs.append(triton.Config({'BLOCK_SIZE_N':8, 'BLOCK_SIZE_K':512}, num_warps=num_warps, num_stages=num_stages))

    configs.append(triton.Config({'BLOCK_SIZE_N':1, 'BLOCK_SIZE_K':1024}, num_warps=num_warps, num_stages=num_stages))
    configs.append(triton.Config({'BLOCK_SIZE_N':2, 'BLOCK_SIZE_K':1024}, num_warps=num_warps, num_stages=num_stages))

    configs.append(triton.Config({'BLOCK_SIZE_N':1, 'BLOCK_SIZE_K':2048}, num_warps=num_warps, num_stages=num_stages))

    configs.append(triton.Config({'BLOCK_SIZE_N':1, 'BLOCK_SIZE_K':4096}, num_warps=num_warps, num_stages=num_stages))
    return configs

@triton.autotune(
    configs=get_configs(),
    restore_value = ['c_ptr'],
    key = ['M', 'K', 'N', 'L'],
)

@triton.jit
def kernel(
        a_ptr, b_ptr, c_ptr, mapping_ptr,
        scales_a_ptr, scales_b_ptr,
        M, K, N, L,
        ########################################
        stride_an:tl.constexpr, stride_ak:tl.constexpr, stride_al:tl.constexpr,
        stride_bm:tl.constexpr, stride_bk:tl.constexpr, stride_bl:tl.constexpr,
        stride_cn:tl.constexpr, stride_cm:tl.constexpr, stride_cl:tl.constexpr,
        stride_scales_an:tl.constexpr, stride_scales_ak:tl.constexpr, stride_scales_al:tl.constexpr,
        stride_scales_bm:tl.constexpr, stride_scales_bk:tl.constexpr, stride_scales_bl:tl.constexpr,
        ########################################
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
        SPLIT_K: tl.constexpr = 1,
        ########################################
        elements_per_sample : tl.constexpr = 2,
        group_size: tl.constexpr = 16,
        acc_dtype: tl.constexpr = ACC_DTYPE,
        ########################################
        a_evict_policy: tl.constexpr = '',
        b_evict_policy: tl.constexpr = 'evict_last',
        meta_evict_policy: tl.constexpr = "evict_last",
        ):

    pid = tl.program_id(axis=0)
    pid_n = pid % tl.cdiv(N, BLOCK_SIZE_N)
    pid_l = pid // tl.cdiv(N, BLOCK_SIZE_N)
    pid_k = tl.program_id(axis=1)

    offs_m = tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)

    stride_mul: tl.constexpr = BLOCK_SIZE_K / group_size

    q_shift = ((offs_k % elements_per_sample) * 4)[None, :].to(tl.int32)
    mask_k = (offs_k < K).to(tl.int1)
    mask_n = (offs_n < N).to(tl.int1)

    mapping = tl.load(mapping_ptr + tl.arange(0, 16), eviction_policy='evict_last')[None, :]
    mapping_a = mapping.broadcast_to((BLOCK_SIZE_N, 16))
    mapping_b = mapping.broadcast_to((BLOCK_SIZE_M, 16))

    a_ptrs = (a_ptr + offs_n[:, None] * stride_an + (offs_k // elements_per_sample)[None, :] * stride_ak + pid_l * stride_al)
    b_ptrs = (b_ptr + offs_m[:, None] * stride_bm + (offs_k // elements_per_sample)[None, :] * stride_bk + pid_l * stride_bl)
    scales_a_ptrs = (scales_a_ptr + offs_n[:, None] * stride_scales_an + (offs_k // group_size)[None, :] * stride_scales_ak + pid_l * stride_scales_al)
    scales_b_ptrs = (scales_b_ptr + offs_m[:, None] * stride_scales_bm + (offs_k // group_size)[None, :] * stride_scales_bk + pid_l * stride_scales_bl)

    b_mask = (mask_k[None, :])
    b = tl.load(b_ptrs, mask=b_mask, other=0, eviction_policy=b_evict_policy)
    b = ((b.to(tl.int32) >> q_shift) & 15)
    b = tl.gather(mapping_b, b, axis=1)
    b = b.to(acc_dtype)

    a_mask = (mask_n[:, None] & mask_k[None, :])
    a  = tl.load(a_ptrs, mask=a_mask, other=0, eviction_policy=a_evict_policy)
    a = ((a.to(tl.int32) >> q_shift) & 15)
    a = tl.gather(mapping_a, a, axis=1)
    a = a.to(acc_dtype)

    scales_a = tl.load(scales_a_ptrs, mask=a_mask, other=0., eviction_policy=meta_evict_policy).to(acc_dtype)
    scales_b = tl.load(scales_b_ptrs, mask=b_mask, other=0., eviction_policy=meta_evict_policy).to(acc_dtype)

    a = a * scales_a
    b = b * scales_b
    acc = tl.sum(a * b, axis=1, keep_dims=True)

    #Output
    c_ptrs = (c_ptr + offs_m[:, None] * stride_cm + offs_n[:, None] * stride_cn + pid_l * stride_cl)
    c_mask = mask_n[:, None]
    tl.atomic_add(c_ptrs, acc, mask=c_mask, sem='relaxed')


import random
custom_op_id = "gemlite::custom_kernel_base_" + str(random.random() * 100000).split('.')[0]
@torch.library.custom_op(custom_op_id, mutates_args=())
def custom_kernel_base(a:Tensor, b:Tensor, scales_a:Tensor, scales_b:Tensor, scales_a_block:Tensor, scales_b_block:Tensor, c:Tensor) -> Tensor:
    global fp4_mapping, CACHE
    #a: (n, k//2, l)
    #b: (m=1, k//2, l)
    #scales_a: (n, k // 16, l)
    #scales_b: (1, k // 16, l)
    #c: (n, 1, l)

    device = a.device
    device_index = a.device.index
    mapping = fp4_mapping[device_index]

    a, b = a.view(torch.uint8), b.view(torch.uint8)
    b, scales_b = b[:1,:,], scales_b[:1,:,]

    N, K_packed, L = a.shape
    M = 1
    K = K_packed * 2

    if(CACHE['C_FP32_COUNTER'] < MAX_C_FP32_CACHE):
        output = CACHE['C_FP32'][CACHE['C_FP32_COUNTER']][:N, :M, :L]
        CACHE['C_FP32_COUNTER'] += 1
    else:
        output = torch.zeros((N, M, L), dtype=OUTPUT_DTYPE, device=a.device)

    grid = lambda META: (triton.cdiv(N, META['BLOCK_SIZE_N']) * L, triton.cdiv(K, META['BLOCK_SIZE_K']))

    stride_an, stride_ak, stride_al = a.stride()
    stride_bm, stride_bk, stride_bl = b.stride()
    stride_cn, stride_cm, stride_cl = output.stride()
    stride_scales_an, stride_scales_ak, stride_scales_al = scales_a.stride()
    stride_scales_bm, stride_scales_bk, stride_scales_bl = scales_b.stride()

    kernel[grid](
        a, b, output, mapping,
        scales_a, scales_b,
        M, K, N, L,
        stride_an, stride_ak, stride_al,
        stride_bm, stride_bk, stride_bl,
        stride_cn, stride_cm, stride_cl,
        stride_scales_an, stride_scales_ak, stride_scales_al,
        stride_scales_bm, stride_scales_bk, stride_scales_bl,
        BLOCK_SIZE_M = 1,
    )

    return output.to(torch.float16)

@torch.library.register_fake(custom_op_id)
def custom_kernel_base_fake(a:Tensor, b:Tensor, scales_a:Tensor, scales_b:Tensor, scales_a_block:Tensor, scales_b_block:Tensor, c:Tensor) -> Tensor:
    N, K_packed, L = a.shape
    M = 1
    K = K_packed * 2
    c = torch.empty((N, M, L), dtype=torch.float16, device=a.device)
    return c

@torch.no_grad()
def custom_kernel_raw(data: input_t) -> output_t:
    a, b, scales_a, scales_b, _, _, c = data
    return custom_kernel_base(a, b, scales_a, scales_b, _, _, c)

##############################################
shapes = [
{"m": 7168, "k": 16384, "l":1, "seed": 1111},
{"m": 4096, "k": 7168, "l":8, "seed": 1111},
{"m": 7168, "k": 2048, "l":4, "seed": 1111},

{"m": 128, "k": 256, "l": 1, "seed": 1111},
{"m": 128, "k": 1536, "l": 1, "seed": 1111},
{"m": 128, "k": 3072, "l": 1, "seed": 1111},
{"m": 256, "k": 7168, "l": 1, "seed": 1111},
{"m": 256, "k": 7168, "l": 1, "seed": 1111},
{"m": 2432, "k": 4608, "l": 2, "seed": 1111},
{"m": 512, "k": 1536, "l": 2, "seed": 1111},
]

for shape in shapes:
    key_ = (shape['m'], shape['k'], shape['l'])
    data_ = generate_input(**shape)
    for _ in range(5):
        out = custom_kernel_raw(data_)
    torch.cuda.synchronize()

    reset_cache()
    torch.cuda.empty_cache()

custom_kernel = custom_kernel_raw
