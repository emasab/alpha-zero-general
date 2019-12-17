#/bin/bash
version=v10
tensorflowjs_converter  --input_format=tf_saved_model --output_format=tfjs_graph_model --signature_name=serving_default --quantization_bytes=2 out/$version connect4-web/$version.quant
