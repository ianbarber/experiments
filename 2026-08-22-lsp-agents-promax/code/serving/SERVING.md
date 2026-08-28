# Serving Qwen3.8-27B on the DGX Spark (GB10)

pip vLLM/SGLang do not work on this hardware (no aarch64 vLLM wheels; sgl-kernel's sm100
cubins lack kernels for sm_121; bundled triton ptxas predates sm_121a). The working stack
is the container setup from MiaAI-Lab/Qwen3.8-27B-SGLang-DGX-Spark (image
`lmsysorg/sglang:qwen38-27b`):

    git clone https://github.com/MiaAI-Lab/Qwen3.8-27B-SGLang-DGX-Spark spark-sglang
    cd spark-sglang && cp ../spark-sglang.env .env
    ./start-dspark.sh     # DSpark speculative decoding; OpenAI API on :8888

Measured: ~31 tok/s single-stream short-context code; ~14-15 tok/s per stream at 3-6
concurrent with 40k+ contexts. NOTE (see REPORT.md): hosted fp8 endpoints scored 20+
points higher on the same benchmark — treat this local stack as convenient, not as a
faithful reference for the model's ceiling.
