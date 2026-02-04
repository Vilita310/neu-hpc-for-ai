import modal

app = modal.App("cuda-gemm-h100")

# CUDA + H100 
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.3.2-devel-ubuntu22.04",
        add_python="3.10"
    )
    .apt_install("git", "cmake", "ninja-build")
)

# GPU：H100
@app.function(
    image=image,
    gpu=modal.gpu.H100(),
    timeout=60 * 60,   # build + benchmark
)
def run_gemm():
    import subprocess
    import os

    def run(cmd):
        print(f"\n$ {cmd}")
        subprocess.run(cmd, shell=True, check=True)

    os.chdir("/root")

    # clone SGEMM_CUDA
    if not os.path.exists("SGEMM_CUDA"):
        run("git clone https://github.com/siboehm/SGEMM_CUDA.git")

    os.chdir("SGEMM_CUDA")

    # compute capability 为 H100 (sm_90)
    run(
        "sed -i 's/set(CUDA_COMPUTE_CAPABILITY 80)/set(CUDA_COMPUTE_CAPABILITY 90)/' "
        "CMakeLists.txt"
    )

    # build
    run("mkdir -p build")
    os.chdir("build")
    run("cmake .. -G Ninja")
    run("cmake --build .")

    kernels = [0, 1, 2, 3, 4, 5, 6, 9, 10]
    for k in kernels:
        run(f"DEVICE=0 ./sgemm {k}")


@app.local_entrypoint()
def main():
    run_gemm.remote()
