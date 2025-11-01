from paddleocr import LayoutDetection
import subprocess

# 初始化模型，触发下载
model = LayoutDetection(model_name="PP-DocLayout-S")

# 获取模型真实路径
model_dir = model._model_dir
print("Model directory:", model_dir)

# 导出为 ONNX
onnx_path = "pp_doclayout_s.onnx"
cmd = [
    "paddle2onnx",
    "--model_dir", model_dir,
    "--model_filename", "inference.pdmodel",
    "--params_filename", "inference.pdiparams",
    "--save_file", onnx_path,
    "--opset_version", "14",
]
subprocess.run(cmd, check=True)

print("✅ Exported:", onnx_path)
