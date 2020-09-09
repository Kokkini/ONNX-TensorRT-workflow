import tensorflow as tf
import keras
from tensorflow.keras.models import Model
import keras.backend as K

K.set_learning_phase(0) #TODO: why set it here?

def keras_to_pb(model, output_filename, output_node_names):
    # Get the name of the input and output nodes
    in_name = model.layers[0].get_output_at(0).name.split(':')[0]

    if output_node_names is None:
        output_node_names = [model.layers[-1].get_output_at(0).name.splot(':')[0]]

    sess = K.get_session() #TODO: why do we need to get session?

    # tf freeze_graph needs a comma separated string of output node names
    output_node_names_tf = ','.join(output_node_names) #TODO: why don't we use this?

    # convert to constant so no changes to the model weight can happen
    frozen_graph_def = tf.graph_util.convert_variable_to_constants(
        sess,
        sess.graph_def,
        output_node_names
    )

    sess.close()

    wkdir = ''
    tf.train.write_graph(frozen_graph_def, wkdir, output_filename, as_text=False)

    return in_name, output_node_names


model = keras.applications.ResNet50(include_top=True, input_tensor=None, input_shape=None, pooling=None, classes=1000, weights='imagenet')

in_tensor_name, out_tensor_names = keras_to_pb(model, "models/resnet50.pb", None)

