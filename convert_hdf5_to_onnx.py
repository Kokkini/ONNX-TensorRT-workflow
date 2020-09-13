import keras
import tensorflow as tf
from keras2onnx import convert_keras
import argparse

args = argparse.ArgumentParser()
args.add_argument("-infile")
args.add_argument("-outfile")
args = args.parse_args()

def keras_to_onnx(model, output_filename):
   onnx = convert_keras(model, output_filename)
   with open(output_filename, "wb") as f:
       f.write(onnx.SerializeToString())

semantic_model = keras.models.load_model(args.infile)
keras_to_onnx(semantic_model, args.outfile)