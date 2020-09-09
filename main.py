import tensorflow as tf
import keras
from tensorflow.keras.models import Model
import keras.backend as K
from tensorflow.python.tools import freeze_graph
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

K.set_learning_phase(0) #TODO: why set it here?

def keras_to_pb(model, output_dir, output_filename, output_node_names):
    # Get the name of the input and output nodes
    in_name = model.layers[0].get_output_at(0).name.split(':')[0]

    if output_node_names is None:
        output_node_names = [model.layers[-1].get_output_at(0).name.split(':')[0]]

    # Convert Keras model to ConcreteFunction
    full_model = tf.function(lambda x: model(x))
    full_model = full_model.get_concrete_function(
        tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))

    # Get frozen ConcreteFunction
    frozen_func = convert_variables_to_constants_v2(full_model)
    frozen_func.graph.as_graph_def()

    layers = [op.name for op in frozen_func.graph.get_operations()]
    print("-" * 50)
    print("Frozen model layers: ")
    for layer in layers:
        print(layer)

    print("-" * 50)
    print("Frozen model inputs: ")
    print(frozen_func.inputs)
    print("Frozen model outputs: ")
    print(frozen_func.outputs)

    # Save frozen graph from frozen ConcreteFunction to hard drive
    tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                      logdir=output_dir,
                      name=output_filename,
                      as_text=False)

    # sess = keras.backend.get_session() #TODO: why do we need to get session?

    # # tf freeze_graph needs a comma separated string of output node names
    # output_node_names_tf = ','.join(output_node_names) #TODO: why don't we use this?

    # # convert to constant so no changes to the model weight can happen
    # frozen_graph_def = tf.graph_util.convert_variable_to_constants(
    #     sess,
    #     sess.graph_def,
    #     output_node_names
    # )

    # sess.close()

    # wkdir = ''
    # tf.train.write_graph(frozen_graph_def, wkdir, output_filename, as_text=False)

    return in_name, output_node_names


model = keras.applications.ResNet50(include_top=True, input_tensor=None, input_shape=None, pooling=None, classes=1000, weights='imagenet')

in_tensor_name, out_tensor_names = keras_to_pb(model, "models", "resnet50.pb", None)

