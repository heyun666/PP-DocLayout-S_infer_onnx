from paddleocr import LayoutDetection
import subprocess, os, glob

# 1. 触发模型下载
model = LayoutDetection(model_name="PP-DocLayout-S")

# 2. 自动定位模型目录
model_dir = os.path.expanduser("~/.paddlex/official_models/PP-DocLayout-S")
print(f"✅ Model directory confirmed: {model_dir}")

# 3. 自动检测文件名
pdmodel = glob.glob(os.path.join(model_dir, "*.pdmodel"))[0]
pdparams = glob.glob(os.path.join(model_dir, "*.pdparams"))[0]
print(f"Found model file: {os.path.basename(pdmodel)}")
print(f"Found params file: {os.path.basename(pdparams)}")

# 4. 导出为 ONNX
onnx_path = "pp_doclayout_s.onnx"
cmd = [
    "paddle2onnx",
    "--model_dir", model_dir,
    "--model_filename", os.path.basename(pdmodel),
    "--params_filename", os.path.basename(pdparams),
    "--save_file", onnx_path,
    "--opset_version", "14",
]
subprocess.run(cmd, check=True)
print(f"✅ Exported ONNX model to {onnx_path}")
