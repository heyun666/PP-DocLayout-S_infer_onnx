from paddleocr import LayoutDetection
import subprocess, os, glob

model_dir="./models"
# 1. 触发模型下载
model = LayoutDetection(model_dir=model_dir)


# 3. 自动检测文件名
pdmodel = glob.glob(os.path.join(model_dir, "*.pdmodel"))[0]
pdparams = glob.glob(os.path.join(model_dir, "*.pdparams"))[0]

# 4. 导出为 ONNX
onnx_path = "pp_doclayout_s.onnx"

cmd = [
    "paddle2onnx",
    "--model_dir", model_dir,
    "--model_filename", pdmodel,
    "--params_filename", pdparams,
    "--save_file", onnx_path,
    "--opset_version", "14",
]
subprocess.run(cmd, check=True)
print(f"✅ Exported ONNX model to {onnx_path}")
