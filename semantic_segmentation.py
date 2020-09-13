import numpy as np
from PIL import Image
import tensorrt as trt
import labels  # from cityscapes evaluation script
import skimage.transform

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
trt_runtime = trt.Runtime(TRT_LOGGER)

MEAN = (71.60167789, 82.09696889, 72.30508881)
CLASSES = 20
HEIGHT = 512
WIDTH = 1024

def sub_mean_chw(data):
   data = data.transpose((1, 2, 0))  # CHW -> HWC
   data -= np.array(MEAN)  # Broadcast subtract
   data = data.transpose((2, 0, 1))  # HWC -> CHW
   return data

def rescale_image(image, output_shape, order=1):
   image = skimage.transform.resize(image, output_shape,
               order=order, preserve_range=True, mode='reflect')
   return image

def color_map(output):
   output = output.reshape(CLASSES, HEIGHT, WIDTH)
   out_col = np.zeros(shape=(HEIGHT, WIDTH), dtype=(np.uint8, 3))
   for x in range(WIDTH):
       for y in range(HEIGHT):

           if (np.argmax(output[:, y, x] )== 19):
               out_col[y,x] = (0, 0, 0)
           else:
               out_col[y, x] = labels.id2label[labels.trainId2label[np.argmax(output[:, y, x])].id].color
   return out_col  


import engine as eng
import inference as inf
import keras
import tensorrt as trt 

input_file_path = "/content/drive/My Drive/Colab Notebooks/deployment/cars.jpg"
onnx_file = "onnx/semantic.onnx"
serialized_plan_fp32 = "semantic.plan"
HEIGHT = 512
WIDTH = 1024

image = np.asarray(Image.open(input_file_path))
img = rescale_image(image, (512, 1024),order=1)
im = np.array(img, dtype=np.float32, order='C')
im = im.transpose((2, 0, 1))
im = sub_mean_chw(im)

engine = eng.load_engine(trt_runtime, serialized_plan_fp32)
h_input, d_input, h_output, d_output, stream = inf.allocate_buffers(engine, 1, trt.float32)
out = inf.do_inference(engine, im, h_input, d_input, h_output, d_output, stream, 1, HEIGHT, WIDTH, output_image=True)
out = color_map(out)

colorImage_trt = Image.fromarray(out.astype(np.uint8))
colorImage_trt.save("trt_output.png")

semantic_model = keras.models.load_model('/content/drive/My Drive/Colab Notebooks/deployment/semantic_segmentation.hdf5')
out_keras= semantic_model.predict(im.reshape(-1, 3, HEIGHT, WIDTH))

out_keras = color_map(out_keras)
colorImage_k = Image.fromarray(out_keras.astype(np.uint8))
colorImage_k.save("keras_output.png")