
# Optimizing graph for inference
#
# python optimize_for_inference.py \
# --input=cnn-embeddings/trained_model_1534708793/checkpoints/frozen_model.pb \
# --output=cnn-embeddings/trained_model_1534708793/checkpoints/opt_model.pb \
# --frozen_graph=True \
# --input_names=input_x \
# --output_names=output/predictions

# From command line
# bazel build tensorflow/python/tools:optimize_for_inference && \
# bazel-bin/tensorflow/python/tools/optimize_for_inference \
# --input=cnn-embeddings/trained_model_1534255535/checkpoints/frozen_model.pb \
# --output=cnn-embeddings/trained_model_1534255535/checkpoints/opt_model.pb \
# --frozen_graph=True \
# --input_names=input_x \
# --output_names=output/predictions

# #
# toco \
# --graph_def_file=/Users/German/Desktop/Project/Algorithms/Emotion-Multiclass-CNN/cnn-embeddings/trained_model_1534708793/checkpoints/opt_model.pb \
# --input_format=TENSORFLOW_GRAPHDEF \
# --output_format=TFLITE \
# --inference_type=FLOAT \
# --input_type=INT \
# --input_arrays=input_x \
# --output_arrays=output/predictions \
# --input_shapes=1,40 \
# --output_file=/Users/German/Desktop/Project/Algorithms/Emotion-Multiclass-CNN/cnn-embeddings/trained_model_1534708793/checkpoints/model.tflite