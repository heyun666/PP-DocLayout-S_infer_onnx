from paddleocr import LayoutDetection
import subprocess, os

# 1. 初始化模型并触发下载（确保模型文件完整）
model = LayoutDetection(model_name="PP-DocLayout-S")
print(model._model_dir)
# 2. 确认模型文件路径（通常位于以下目录）


# 3. 构造修正后的paddle2onnx命令
cmd = [
    "paddle2onnx",
    "--model_dir", "/home/runner/.paddlex/official_models/PP-DocLayout-S",
    "--model_filename", "inference.pdmodel",  # 修正：使用正确的模型结构文件
    "--params_filename", "inference.pdiparams",  # 参数文件正确
    "--save_file", "model.onnx",
    "--input_shape_dict", "image:-1,3,-1,-1;scale_factor:-1,1,2",  # 正确参数和格式,  # 动态输入尺寸
    "--opset_version", "11",  # 指定opset版本，确保兼容性
    "--enable_onnx_checker", "True"  # 保留检查
]

print("Running command:", " ".join(cmd))
result = subprocess.run(cmd, capture_output=True, text=True)

# 输出执行结果
print(result.stdout)
if result.returncode != 0:
    print(result.stderr)
    raise SystemExit("❌ paddle2onnx conversion failed")
else:
    print("✅ Model successfully converted")
