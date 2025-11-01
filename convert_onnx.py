import os
import subprocess

model_dir = "models"
model_file = "model.pdmodel"
params_file = "model.pdiparams"
output_file = "model.onnx"

# 确认模型文件是否存在
if not os.path.exists(os.path.join(model_dir, model_file)):
    raise FileNotFoundError(f"{model_file} not found in {model_dir}")
if not os.path.exists(os.path.join(model_dir, params_file)):
    raise FileNotFoundError(f"{params_file} not found in {model_dir}")

# 构造 paddle2onnx 命令
cmd = [
    "paddle2onnx",
    "--model_dir", model_dir,
    "--model_filename", model_file,
    "--params_filename", params_file,
    "--save_file", output_file,
    "--opset_version", "11",
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
