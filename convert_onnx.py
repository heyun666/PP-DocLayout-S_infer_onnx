from paddleocr import LayoutDetection
import subprocess, os

# 1. 初始化模型并触发下载
model = LayoutDetection(model_name="PP-DocLayout-S")

# 构造 paddle2onnx 命令
cmd = [
    "paddle2onnx",
    "--model_dir", "/root/.paddlex/official_models/PP-DocLayout-S",
    "--model_filename", "inference.json",
    "--params_filename", "inference.pdiparams",
    "--save_file", "./output/model.onnx",
    "--enable_onnx_checker", "True"
]

print("Running command:", " ".join(cmd))
result = subprocess.run(cmd, capture_output=True, text=True)

# 输出执行结果
print(result.stdout)
if result.returncode != 0:
    print(result.stderr)
    raise SystemExit("❌ paddle2onnx conversion failed")
else:
    print("✅ Model successfully converted to", output_file)
