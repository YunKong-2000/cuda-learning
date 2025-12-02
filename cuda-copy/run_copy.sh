#!/bin/bash

exe_file="copy"
src_file="copy.cu"

# Check if source file exists
if [ ! -f $src_file ]; then
    echo "Error: Source file $src_file does not exist"
    exit 1
fi

# Compile if executable doesn't exist or source is newer
if [ ! -f $exe_file ] || [ $src_file -nt $exe_file ]; then
    echo "Compiling $src_file..."
    # Detect compute capability if available, otherwise use sm_80 as default
    COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')
    if [ -z "$COMPUTE_CAP" ]; then
        COMPUTE_CAP="80"
        echo "Warning: Could not detect compute capability, using default sm_80"
    else
        echo "Detected compute capability: $COMPUTE_CAP"
    fi
    # Compile with explicit architecture flag to avoid PTX toolchain mismatch
    nvcc -arch=sm_${COMPUTE_CAP} -o $exe_file $src_file
    if [ $? -ne 0 ]; then
        echo "Error: Compilation failed"
        exit 1
    fi
fi

# Check if ncu is available
NCU_AVAILABLE=false
# Check if sudo is available
SUDO_AVAILABLE=false
if command -v sudo &> /dev/null; then
    SUDO_AVAILABLE=true
fi

# Allow user to set USE_SUDO_NCU=1 environment variable to use sudo for ncu
if [ "${USE_SUDO_NCU:-0}" = "1" ]; then
    if [ "$SUDO_AVAILABLE" = true ]; then
        USE_SUDO_NCU=true
    else
        echo "Warning: USE_SUDO_NCU=1 but sudo not available, will run ncu without sudo"
        USE_SUDO_NCU=false
    fi
else
    USE_SUDO_NCU=false
fi

if command -v ncu &> /dev/null; then
    NCU_AVAILABLE=true
    echo "ncu is available, will collect DRAM metrics"
    if [ "$USE_SUDO_NCU" = true ]; then
        echo "Note: Will use sudo for ncu"
    fi
else
    echo "Warning: ncu not found, skipping DRAM metrics collection"
    echo "Install NVIDIA Nsight Compute to enable DRAM metrics collection"
fi

echo ""
echo "=========================================="
echo "Running copy kernel with different sizes"
echo "=========================================="
echo ""

# Kernel selection: "baseline", "loop_unroll", "vectorize", "vectorize_unroll", or "compare" (runs all kernels)
KERNEL_NAME="${KERNEL_NAME:-baseline}"

# Function to run a single kernel test
run_kernel_test() {
    local kernel_name=$1
    local block_dim=$2
    local num_elements=$3
    local n=$4
    local loop_unroll_times=${5:-0}
    
    echo "----------------------------------------"
    echo "Testing with 2^$n = $num_elements elements"
    echo "----------------------------------------"
    
    echo "----------------------------------------"
    echo "Testing with block_dim = $block_dim threads per block"
    if [ "$kernel_name" = "loop_unroll" ]; then
        echo "Testing with loop_unroll_times = $loop_unroll_times"
    fi
    echo "Kernel: $kernel_name"
    echo "----------------------------------------"
    
    # Set kernel-specific variables
    local NCU_KERNEL_NAME=""
    local CMD_ARGS=""
    
    if [ "$kernel_name" = "baseline" ]; then
        NCU_KERNEL_NAME="copy_baseline"
        CMD_ARGS="$num_elements $block_dim baseline"
    elif [ "$kernel_name" = "vectorize" ]; then
        NCU_KERNEL_NAME="copy_vectorize"
        CMD_ARGS="$num_elements $block_dim vectorize"
    elif [ "$kernel_name" = "vectorize_unroll" ]; then
        NCU_KERNEL_NAME="copy_vectorize_unroll"
        CMD_ARGS="$num_elements $block_dim vectorize_unroll"
    elif [ "$kernel_name" = "loop_unroll" ]; then
        NCU_KERNEL_NAME="copy_loop_unroll"
        CMD_ARGS="$num_elements $block_dim loop_unroll $loop_unroll_times"
    fi
    
    # Run the program normally
    ./$exe_file $CMD_ARGS
    
    # Run with ncu to collect DRAM metrics
    if [ "$NCU_AVAILABLE" = true ]; then
        echo ""
        echo "Collecting DRAM metrics with ncu..."
        
        # Determine ncu command (with or without sudo)
        if [ "$USE_SUDO_NCU" = true ] && [ "$SUDO_AVAILABLE" = true ]; then
            NCU_CMD="sudo ncu"
        else
            NCU_CMD="ncu"
        fi
        
        # Run ncu and capture output
        ncu_output=$($NCU_CMD \
            --kernel-name $NCU_KERNEL_NAME \
            --metrics dram__bytes_read.sum,dram__bytes_write.sum,gpu__time_duration.sum \
            --print-summary per-kernel \
            --target-processes all \
            ./$exe_file $CMD_ARGS 2>&1)
        
        ncu_exit_code=$?
        
        # Check for command not found error
        if [ $ncu_exit_code -eq 127 ]; then
            if echo "$ncu_output" | grep -q "sudo: command not found"; then
                echo ""
                echo "Warning: sudo command not found (likely in container environment)"
                echo "Trying ncu without sudo..."
                ncu_output=$(ncu \
                    --kernel-name $NCU_KERNEL_NAME \
                    --metrics dram__bytes_read.sum,dram__bytes_write.sum,gpu__time_duration.sum \
                    --print-summary per-kernel \
                    --target-processes all \
                    ./$exe_file $CMD_ARGS 2>&1)
                ncu_exit_code=$?
            fi
        fi
        
        # Check for permission error
        if echo "$ncu_output" | grep -q "ERR_NVGPUCTRPERM"; then
            echo ""
            echo "ERROR: Permission denied for GPU Performance Counters"
            echo "ncu requires special permissions to access GPU performance counters."
            echo ""
            if [ "$SUDO_AVAILABLE" = true ]; then
                echo "Solutions:"
                echo "  1. Run the script with sudo: sudo $0"
                echo "  2. Or set USE_SUDO_NCU=1 and run: USE_SUDO_NCU=1 $0"
                echo "  3. Or configure system permissions (see https://developer.nvidia.com/ERR_NVGPUCTRPERM)"
            else
                echo "Note: Running in container environment (sudo not available)"
                echo "If you have root privileges, try running as root user"
                echo "Or configure system permissions (see https://developer.nvidia.com/ERR_NVGPUCTRPERM)"
            fi
            echo ""
            return
        fi
        
        if [ $ncu_exit_code -eq 0 ]; then
            echo ""
            echo "--- ncu Raw Measurement Results ---"
            echo "$ncu_output"
            echo ""
        else
            echo "Warning: ncu profiling failed (exit code: $ncu_exit_code)"
            echo ""
            echo "ncu error output:"
            echo "$ncu_output" | tail -30
            echo ""
            if ! echo "$ncu_output" | grep -q "ERR_NVGPUCTRPERM"; then
                echo "Common issues:"
                echo "  1. Library version mismatch (GLIBC/GLIBCXX) - try running in the same environment where compiled"
                echo "  2. Kernel name mismatch - verify kernel name is '$NCU_KERNEL_NAME'"
                echo "  3. GPU not available"
            fi
        fi
    fi
    
    echo ""
}

# Check if comparing all kernels
if [ "$KERNEL_NAME" = "compare" ] || [ "$KERNEL_NAME" = "all" ]; then
    echo "Comparing all kernels: baseline, loop_unroll (times=4), vectorize, vectorize_unroll"
    echo "Fixed parameters: block_dim=256, loop_unroll_times=4"
    echo ""
    
    block_dim=256
    loop_unroll_times=4
    
    # Test all data sizes
    for ((n=10; n<=30; n+=2)); do
        num_elements=$((2**$n))
        
        echo "========================================"
        echo "Testing with 2^$n = $num_elements elements"
        echo "========================================"
        echo ""
        
        # Run baseline
        echo ">>> Running baseline kernel..."
        run_kernel_test "baseline" $block_dim $num_elements $n
        
        # Run loop_unroll
        echo ">>> Running loop_unroll kernel (times=$loop_unroll_times)..."
        run_kernel_test "loop_unroll" $block_dim $num_elements $n $loop_unroll_times
        
        # Run vectorize
        echo ">>> Running vectorize kernel..."
        run_kernel_test "vectorize" $block_dim $num_elements $n
        
        # Run vectorize_unroll
        echo ">>> Running vectorize_unroll kernel..."
        run_kernel_test "vectorize_unroll" $block_dim $num_elements $n
        
        echo "========================================"
        echo ""
    done
    
elif [ "$KERNEL_NAME" = "baseline" ] || [ "$KERNEL_NAME" = "vectorize" ] || [ "$KERNEL_NAME" = "vectorize_unroll" ]; then
    echo "Using kernel: $KERNEL_NAME"
    echo ""
    # Baseline, Vectorize, and Vectorize_unroll: test different block_dim and data sizes
    # All use the same parameter format (n, threadnum, kernel_name)
    
    block_dim_list=(256)
    # block_dim_list=(128 256 512 1024)
    
    for block_dim in "${block_dim_list[@]}"; do
    for ((n=10; n<=30; n+=2)); do
        num_elements=$((2**$n))
        run_kernel_test "$KERNEL_NAME" $block_dim $num_elements $n
    done
    done
    
elif [ "$KERNEL_NAME" = "loop_unroll" ]; then
    echo "Using kernel: $KERNEL_NAME"
    echo ""
    # Loop unroll: fixed block_dim=256, test different LOOP_UNROLL_TIMES and data sizes
    block_dim=256
    # Loop unroll times to test (can be customized)
    loop_unroll_times_list=(1 2 4 8)
    # loop_unroll_times_list=(1 2 4 8 16 32)
    
    for loop_unroll_times in "${loop_unroll_times_list[@]}"; do
    for ((n=10; n<=30; n+=2)); do
        num_elements=$((2**$n))
        run_kernel_test "loop_unroll" $block_dim $num_elements $n $loop_unroll_times
    done
    done
    
else
    echo "Error: Invalid KERNEL_NAME. Must be 'baseline', 'loop_unroll', 'vectorize', 'vectorize_unroll', or 'compare'"
    exit 1
fi

echo "=========================================="
echo "Testing completed"
echo "=========================================="
