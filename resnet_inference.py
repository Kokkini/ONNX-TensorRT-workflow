import numpy as np
from PIL import Image
import tensorrt as trt
import skimage.transform

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
trt_runtime = trt.Runtime(TRT_LOGGER)

import engine as eng
import inference as inf
import keras
import tensorrt as trt 

input_file_path = "/content/drive/My Drive/Colab Notebooks/deployment/Killerwhales_jumping.jpg"
onnx_file = "onnx/resnet50.onnx"
serialized_plan_fp32 = "resnet50.plan"
HEIGHT = 224
WIDTH = 224

def rescale_image(image, output_shape, order=1):
   image = skimage.transform.resize(image, output_shape,
               order=order, preserve_range=True, mode='reflect')
   return image

image = np.asarray(Image.open(input_file_path))
img = rescale_image(image, (224, 224),order=1)
im = np.array(img, dtype=np.float32, order='C')
im = im.transpose((2, 0, 1))

engine = eng.load_engine(trt_runtime, serialized_plan_fp32)
h_input, d_input, h_output, d_output, stream = inf.allocate_buffers(engine, 1, trt.float32)
out = inf.do_inference(engine, im, h_input, d_input, h_output, d_output, stream, 1, HEIGHT, WIDTH)

print(f"output: {out}")
print(f"argmax: {np.argmax(out, axis=1)}")
print(f"max prob: {np.max(out, axis=1)}")
print(f"shape: {out.shape}")