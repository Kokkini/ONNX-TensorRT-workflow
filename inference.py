import tensorrt as trt
import pycuda.driver as cuda
import numpy as np
import pycuda.autoinit 

def allocate_buffers(engine, batch_size, data_type):
    """
    allocate buffers for input and output in the device
    """
    h_input = cuda.pagelocked_empty(batch_size*trt.volume(engine.get_binding_shape(0)), dtype=trt.nptype(data_type))
    h_output = cuda.pagelocked_empty(batch_size * trt.volume(engine.get_binding_shape(1)), dtype=trt.nptype(data_type))
    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)

    stream = cuda.Stream()
    return h_input, d_input, h_output, d_output, stream

def load_images_to_buffer(pics, pagelocked_buffer):
    preprocessed = np.asarray(pics).ravel()
    np.copyto(pagelocked_buffer, preprocessed)

def do_inference(engine, pics, h_input, d_input, h_output, d_output, stream, batch_size, height, width, output_image=False):
    load_images_to_buffer(pics, h_input)

    with engine.create_execution_context() as context:
        # transfer input data to the GPU
        cuda.memcpy_htod_async(d_input, h_input, stream)

        # run inference
        context.profiler = trt.Profiler()
        context.execute(batch_size=1, bindings=[int(d_input), int(d_output)])

        # transfer predictions batch from the GPU
        cuda.memcpy_dtoh_async(h_output, d_output, stream)
        # synchronize the stream
        stream.synchronize()

        if output_image:
          out = h_output.reshape((batch_size, -1, height, width)) #TODO: why is the output a picture? Why is it channel first?
        else:
          out = h_output.reshape((batch_size, -1))
        return out