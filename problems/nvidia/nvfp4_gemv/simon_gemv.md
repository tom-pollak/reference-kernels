NVFP4 GEMV
13 Nov, 2025

Introduction
GPU Mode and NVIDIA host a hackathon consisting of multiple challenges around the Blackwell GPU architecture and the new float 4 datatype NVFP4. In this blogpost I aim to give a brief introduction to the CuTeDSL which is able to target Blackwell specific features efficiently while having high productivity for the developer. The first challenge resolves around GEMV kernel which is simply a matrix vector multiplication.

NVP4
I will keep introduction to NVFP4 brief because it's sufficiently covered in this blogpost. NVFP4 is a 4 bit floating point format introduced on Blackwell GPUs. The main thing that we are interested in for our discussion is that NVFP4 is composed of two tensors: One in FP4 precision and another in FP8 precision. We call the FP8 term a scaling factor. The scaling factor is applied to every block of 16 values. This factor is introduced to minimize quantization error while still being able to reduce memory footprint. For more details please see the above mentioned blog.

Reference Kernel
The reference kernel for the competition which uses CuTeDSL can be found here. We will now give a quick onboarding to the kernel and explain how it works.

mma_tiler_mnk = (128, 1, 64)  # Tile sizes for M, N, K dimensions
ab_dtype = cutlass.Float4E2M1FN  # FP4 data type for A and B
sf_dtype = cutlass.Float8E4M3FN  # FP8 data type for scale factors
c_dtype = cutlass.Float16  # FP16 output type
sf_vec_size = 16  # Scale factor block size (16 elements share one scale)
threads_per_cta = 128  # Number of threads per CUDA thread block
This is the configuration for starters. Most of it is self explanatory, for example mma_tiler_mnk will define the tile size we are processing.

Let's now jump into the kernel:

@cute.kernel
def kernel(
    mA_mkl: cute.Tensor,
    mB_nkl: cute.Tensor,
    mSFA_mkl: cute.Tensor,
    mSFB_nkl: cute.Tensor,
    mC_mnl: cute.Tensor,
):
The kernel takes mA_mkl, i.e. a tensor of shape (m,k,l) and it's accompinying scale factor. The last dimension corresponds to the batch size, i.e. our tensor describes a matrix in each plane. The other matrix is mB_nkl where n = 1, i.e. our tensor describes a batch of vectors. It contains a scale factor as well. The output matrix is mC_mnl which is again a batch of vectors.

    # Get CUDA block and thread indices
    bidx, bidy, bidz = cute.arch.block_idx()
    tidx, _, _ = cute.arch.thread_idx()

    # Extract the local tile for input matrix A (shape: [block_M, block_K, rest_M, rest_K, rest_L])
    gA_mkl = cute.local_tile(
        mA_mkl, cute.slice_(mma_tiler_mnk, (None, 0, None)), (None, None, None)
    )
    # Extract the local tile for scale factor tensor for A (same shape as gA_mkl)
    # Here, block_M = (32, 4); block_K = (16, 4)
    gSFA_mkl = cute.local_tile(
        mSFA_mkl, cute.slice_(mma_tiler_mnk, (None, 0, None)), (None, None, None)
    )
    # Extract the local tile for input matrix B (shape: [block_N, block_K, rest_N, rest_K, rest_L])
    gB_nkl = cute.local_tile(
        mB_nkl, cute.slice_(mma_tiler_mnk, (0, None, None)), (None, None, None)
    )
    # Extract the local tile for scale factor tensor for B (same shape as gB_nkl)
    gSFB_nkl = cute.local_tile(
        mSFB_nkl, cute.slice_(mma_tiler_mnk, (0, None, None)), (None, None, None)
    )
    # Extract the local tile for output matrix C (shape: [block_M, block_N, rest_M, rest_N, rest_L])
    gC_mnl = cute.local_tile(
        mC_mnl, cute.slice_(mma_tiler_mnk, (None, None, 0)), (None, None, None)
    )
The first step is to calculate the relevant tile for our current block configuration. Let us look how these look like for a certain configuration (m: 128; k: 256; l: 1):

gA_mkl = raw_ptr(0x000075dae100aa00: f4E2M1FN, gmem, align<16>) o (128,64,1,4,1):(256,1,32768,64,32768)
gSFA_mkl = raw_ptr(0x000075dae1008200: f8E4M3FN, gmem, align<32>) o ((32,4),(16,4),1,4,(1,1)):((16,4),(0,1),2048,512,(0,2048))
gB_nkl = raw_ptr(0x000075dae100ea00: f4E2M1FN, gmem, align<16>) o (1,64,128,4,1):(0,1,256,64,32768)
gSFB_nkl = raw_ptr(0x000075dae1008a00: f8E4M3FN, gmem, align<32>) o (1,(16,4),(32,4),4,(1,1)):(0,(0,1),(16,4),512,(0,2048))
gC_mnl = raw_ptr(0x000075dae1009200: f16, gmem, align<16>) o (128,1,1,1,1):(1,0,128,0,128)
We see for example that gA_mkl has shape of (128,64,1,4,1) that is we have our tile of (128,64) and we need one tile in M dimension, four tiles in K dimension and one tile in L dimension to cover the whole tensor.

If we look at gSFA_mkl we see it has shape of ((32,4),(16,4),1,4,(1,1)). Note that 32 * 4 = 128, 16 * 4 = 64 and 1 * 1 = 1 so we essentially have the same size as above in each mode but the description is hierarchical. You can read about hierarchical layouts in CuTe in other blogposts I wrote. This analysis extends of course to the other tensors.

    # Select output element corresponding to this thread and block indices
    tCgC = gC_mnl[tidx, None, bidx, bidy, bidz]
    tCgC = cute.make_tensor(tCgC.iterator, 1)
    res = cute.zeros_like(tCgC, cutlass.Float32)
Here we index into our tensor. Note that this will give us the N tile because the second mode in the tensor gC_mnl corresponds to the N tile. Of the length of it is 1 because Ns dimension is 1 as it corresponds to the vector of shape 1 x K. make_tensor will essentially only simplify.

Note that:

gC_mnl = raw_ptr(0x000076a9fd009200: f16, gmem, align<16>) o (128,1,1,1,1):(1,0,128,0,128)
gC_mnl[tidx, None, bidx, bidy, bidz] = raw_ptr(0x000076a9fd009200: f16, gmem, align<2>) o (1):(0) =
  ( -0.017166 )
tCgC = raw_ptr(0x000076a9fd009200: f16, gmem, align<2>) o 1:0 =
  ( -0.017166 )
i.e. they all point to the same location in Memory. That is because a tensor in CuTe is essentially only a pointer associated with a Layout that defines how to offset the pointer for each entry. Again I refer to other blogposts I wrote if you are interested in this. res is tensor_value<vector<1xf32> o 1> and used to quickly accumulate the result and transferring it to tCgC in GMEM at the end.

    # Get the number of k tiles (depth dimension) for the reduction loop
    k_tile_cnt = gA_mkl.layout[3].shape
    for k_tile in range(k_tile_cnt):
This is our main loop. Note that above I explained that the fourth mode of gA_mkl Layout will give us the number of K tiles we need to cover the tensor. We loop over it because obviously for a GEMV we will need to cover the whole K dimension in order to recieve a meaningful result.

        tAgA = gA_mkl[tidx, None, bidx, k_tile, bidz]
        tBgB = gB_nkl[0, None, bidy, k_tile, bidz]
        tAgSFA = gSFA_mkl[tidx, None, bidx, k_tile, bidz]
        tBgSFB = gSFB_nkl[0, None, bidy, k_tile, bidz]
We extract the current values to process. We use tidx to index into the position of the M tile we are currently in. bidx tells us which of the M tiles we currently process. k_tile is the index into the K tile we are currently processing and bidz gives us index into the batch dimension. For B and its scale factor situation is different because we only have one N dimension of one so we index into it by providing a 0.

        tArA = cute.make_rmem_tensor_like(tAgA, cutlass.Float32)
        tBrB = cute.make_rmem_tensor_like(tBgB, cutlass.Float32)
        tArSFA = cute.make_rmem_tensor_like(tAgSFA, cutlass.Float32)
        tBrSFB = cute.make_rmem_tensor_like(tBgSFB, cutlass.Float32)

        # Load NVFP4 or FP8 values from global memory
        a_val_nvfp4 = tAgA.load()
        b_val_nvfp4 = tBgB.load()
        sfa_val_fp8 = tAgSFA.load()
        sfb_val_fp8 = tBgSFB.load()

        # Convert loaded values to float32 for computation (FFMA)
        a_val = a_val_nvfp4.to(cutlass.Float32)
        b_val = b_val_nvfp4.to(cutlass.Float32)
        sfa_val = sfa_val_fp8.to(cutlass.Float32)
        sfb_val = sfb_val_fp8.to(cutlass.Float32)
We load our values into registers and convert them to higher precision for accumulation. Note this is consistent with us defining res in Float32 above.

        # Store the converted values to RMEM CuTe tensors
        tArA.store(a_val)
        tBrB.store(b_val)
        tArSFA.store(sfa_val)
        tBrSFB.store(sfb_val)

        # Iterate over SF vector tiles and compute the scale&matmul accumulation
        for i in cutlass.range_constexpr(mma_tiler_mnk[2]):
            res += tArA[i] * tArSFA[i] * tBrB[i] * tBrSFB[i]
We store the converted values into registers. We'll than loop over all the values in our current K tile and perform multiplication. Apart from the usual multiplication we scale our FP4 values as demanded by NVFP4 format.

Let us look at the Layouts to get little bit better understanding:

tArA = (64):(1)
tBrB = (64):(1)
tArSFA = ((16,4)):((0,1))
tBrSFB = ((16,4)):((0,1))
What does this tell us? First it is clear that the size of both tensors is 64 because that is one K tile, i.e. it contains 64 values as we defined above. The Layout of the scale factors is ((16,4)):((0,1)). A stride of 0 corresponds to broadcast. So we really have only 4 different values in the scale factors. Each of them applied to 16 consecutive values.

    # Store the final float16 result back to global memory
    tCgC.store(res.to(cutlass.Float16))
    return
We finish of by converting the accumulated values to the desired precision and storing them in GMEM.
