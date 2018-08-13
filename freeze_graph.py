import os, argparse

import tensorflow as tf

# The original freeze_graph function
# from tensorflow.python.tools.freeze_graph import freeze_graph

dir = os.path.dirname(os.path.realpath(__file__))


def create_graph(model_dir):

    with tf.Graph().as_default():

        # global_step = tf.Variable(0,name='global_step', trainable=False)
        init_op = tf.global_variables_initializer()

        session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        sess = tf.Session(config=session_conf)

        sess.run(init_op)

        saver = tf.train.Saver()

        saver.restore(sess, model_dir)
        # Save model graph
        tf_graph = sess.graph
        tf.train.write_graph(tf_graph.as_graph_def(), model_dir, 'graph.pbtxt', as_text=True)



def freeze_graph(model_dir):
    """Extract the sub graph defined by the output nodes and convert
    all its variables into constant
    Args:
        model_dir: the root folder containing the checkpoint state file
        output_node_names: a string, containing all the output node's names,
                            comma separated
    """
    if not tf.gfile.Exists(model_dir):
        raise AssertionError(
            "Export directory doesn't exists. Please specify an export "
            "directory: %s" % model_dir)

    # if not output_node_names:
    #     print("You need to supply the name of a node to --output_node_names.")
    #     return -1

    # We retrieve our checkpoint fullpath
    checkpoint = tf.train.get_checkpoint_state(model_dir)
    input_checkpoint = checkpoint.model_checkpoint_path

    # We precise the file fullname of our freezed graph
    absolute_model_dir = "/".join(input_checkpoint.split('/')[:-1])
    # output_graph = absolute_model_dir + "/graph.pbtxt"

    output_node_names = 'output/predictions'

    # We clear devices to allow TensorFlow to control on which device it will load operations
    clear_devices = True

    # We start a session using a temporary fresh Graph
    with tf.Session(graph=tf.Graph()) as sess:
        # We import the meta graph in the current default Graph
        saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)

        # We restore the weights
        saver.restore(sess, input_checkpoint)

        # We use a built-in TF helper to export variables to constants
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess,  # The session is used to retrieve the weights
            tf.get_default_graph().as_graph_def(),  # The graph_def is used to retrieve the nodes
            output_node_names.split(",")  # The output node names are used to select the useful nodes
        )
        with tf.gfile.FastGFile('./cnn-embeddings/trained_model_1534172737/checkpoints/frozen_model.pb', 'wb') as f:
            f.write(output_graph_def.SerializeToString())

        # Finally we serialize and dump the output graph to the filesystem
        # with tf.gfile.GFile(output_graph, "wb") as f:
        #     f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))



    # return output_graph_def

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--model_dir", type=str, default="", help="Model folder to export")
#     parser.add_argument("--output_node_names", type=str, default="",
#                         help="The name of the output nodes, comma separated.")
#     args = parser.parse_args()

    # freeze_graph(args.model_dir, args.output_node_names)

# create_graph('cnn-embeddings/64%_trained_model/checkpoints')
#
freeze_graph('cnn-embeddings/trained_model_1534172737/checkpoints')