import tfcoreml as tf_converter

tf_converter.convert(tf_model_path = 'frozen_embeddings_no_dropout.pb',
                     mlmodel_path = 'CNN.mlmodel',
                     output_feature_names = ['output/predictions:0'])