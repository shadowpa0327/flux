#!/usr/bin/env python3
"""
Debug helper for Flux op_registry errors
"""
import torch
import flux

def check_gpu_compatibility():
    """Check if your GPU is compatible with Flux"""
    print("=" * 60)
    print("GPU Compatibility Check")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("❌ CUDA is not available")
        return False

    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)

    print(f"Device: {props.name}")
    print(f"Compute Capability: {props.major}.{props.minor}")

    arch = flux.util.get_arch()
    print(f"Flux Architecture Code: {arch}")

    # Supported architectures in Flux
    supported_archs = {
        80: "sm_80 (A100, A30, etc.)",
        89: "sm_89 (L20, L40, etc.)",
        90: "sm_90 (H100, H800, etc.)"
    }

    if arch in supported_archs:
        print(f"✅ GPU is supported: {supported_archs[arch]}")
        return True
    else:
        print(f"❌ GPU architecture {arch} may not be fully supported")
        print(f"   Supported architectures: {list(supported_archs.keys())}")
        return False


def check_operation_support(
    input_dtype=torch.bfloat16,
    output_dtype=torch.bfloat16,
    m=4096,
    n=8192,
    k=8192
):
    """
    Test if a basic GEMM operation works with given parameters
    """
    print("\n" + "=" * 60)
    print("Operation Support Check")
    print("=" * 60)
    print(f"Testing GEMM with:")
    print(f"  Input dtype: {input_dtype}")
    print(f"  Output dtype: {output_dtype}")
    print(f"  Dimensions: M={m}, N={n}, K={k}")

    try:
        # Create test tensors
        A = torch.randn(m, k, dtype=input_dtype, device='cuda')
        B = torch.randn(k, n, dtype=input_dtype, device='cuda')

        # Try to create a GemmOnly operation
        use_fp8 = flux.util.is_fp8_dtype(input_dtype)
        arch = flux.util.get_arch()

        gemm_op = flux.GemmOnly(
            input_dtype,
            output_dtype,
            use_fp8_gemm=(arch < 90 and use_fp8)
        )

        # Try to execute
        C = gemm_op(A, B.t())

        print(f"✅ GemmOnly operation succeeded")
        print(f"   Output shape: {C.shape}")
        return True

    except Exception as e:
        print(f"❌ GemmOnly operation failed")
        print(f"   Error: {type(e).__name__}: {e}")
        return False


def check_grouped_gemm_support(arch=None):
    """Check which grouped GEMM implementations are available"""
    print("\n" + "=" * 60)
    print("Grouped GEMM Support Check")
    print("=" * 60)

    if arch is None:
        arch = flux.util.get_arch()

    # Check GemmGroupedV2
    if hasattr(flux, 'GemmGroupedV2') and not isinstance(flux.GemmGroupedV2, flux.cpp_mod.NotCompiled):
        print("✅ GemmGroupedV2 is available")
    else:
        print("❌ GemmGroupedV2 is NOT available")

    # Check GemmGroupedV3 (for sm_90+)
    if hasattr(flux, 'GemmGroupedV3') and not isinstance(flux.GemmGroupedV3, flux.cpp_mod.NotCompiled):
        print("✅ GemmGroupedV3 is available")
        if arch >= 90:
            print("   (Recommended for your architecture)")
    else:
        print("❌ GemmGroupedV3 is NOT available")
        if arch >= 90:
            print("   (This is recommended for sm_90+ GPUs)")


def check_dtype_support():
    """Check supported data types"""
    print("\n" + "=" * 60)
    print("Data Type Support Check")
    print("=" * 60)

    test_dtypes = [
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.int8,
        torch.float8_e4m3fn,
        torch.float8_e5m2,
    ]

    for dtype in test_dtypes:
        try:
            # Quick sanity check
            tensor = torch.randn(4, 4, dtype=dtype, device='cuda')
            print(f"✅ {dtype} - tensor creation OK")
        except Exception as e:
            print(f"❌ {dtype} - Error: {e}")


def print_environment_vars():
    """Print relevant environment variables"""
    import os
    print("\n" + "=" * 60)
    print("Environment Variables")
    print("=" * 60)

    relevant_vars = [
        'FLUX_TUNE_CONFIG_FILE',
        'CUDA_VISIBLE_DEVICES',
        'NCCL_DEBUG',
        'NVSHMEM_HOME',
    ]

    for var in relevant_vars:
        value = os.environ.get(var, '<not set>')
        print(f"{var}: {value}")


def main():
    """Run all diagnostic checks"""
    print("\n")
    print("#" * 60)
    print("# Flux OpRegistry Diagnostic Tool")
    print("#" * 60)

    # Run checks
    gpu_ok = check_gpu_compatibility()

    if gpu_ok:
        check_dtype_support()
        check_grouped_gemm_support()

        # Test basic operation
        print("\n" + "=" * 60)
        print("Testing Basic Operations")
        print("=" * 60)

        # Test with bfloat16 (most common)
        check_operation_support(
            input_dtype=torch.bfloat16,
            output_dtype=torch.bfloat16,
            m=1024, n=1024, k=1024
        )

    print_environment_vars()

    print("\n" + "=" * 60)
    print("Diagnostic Complete")
    print("=" * 60)

    if not gpu_ok:
        print("\n⚠️  GPU compatibility issue detected!")
        print("   Your GPU may not be supported by Flux")

    print("\nFor more help, check:")
    print("  - Error messages above")
    print("  - Flux documentation")
    print("  - Set FLUX_TUNE_CONFIG_FILE for custom kernel configs")


if __name__ == "__main__":
    main()
