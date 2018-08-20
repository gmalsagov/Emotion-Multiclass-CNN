import tfcoreml as tf_converter

input_tensor_shapes = {'input_x:0': [1, 78]}

tf_converter.convert(mlmodel_path = 'CNN.mlmodel',
                     tf_model_path = '/Users/German/Desktop/Project/Algorithms/Emotion-Multiclass-CNN/cnn-embeddings/trained_model_1534255535/checkpoints/opt_model.pb',
                     output_feature_names =['output/predictions:0'],
                     input_name_shape_dict=input_tensor_shapes,
                     class_labels= ["joy", "fear", "anger", "sadness", "disgust", "shame", "guilt"]
)