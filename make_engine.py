import engine as eng
import argparse
from onnx import ModelProto
import tensorrt as trt 

args = argparse.ArgumentParser()
args.add_argument('-engine-name')
args.add_argument('-onnx-path')

args = args.parse_args()

batch_size = 1 

model = ModelProto()
with open(args.onnx_path, "rb") as f:
    model.ParseFromString(f.read()) # load the model to get the input shape

dims = model.graph.input[0].type.tensor_type.shape.dim[1:4]
shape = [batch_size] + [d.dim_value for d in dims]
print(shape)

# build and save the engine
engine = eng.build_engine(args.onnx_path, shape)
eng.save_engine(engine, args.engine_name)