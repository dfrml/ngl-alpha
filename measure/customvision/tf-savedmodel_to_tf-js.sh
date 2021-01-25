tensorflowjs_converter \
    --input_format=tf_saved_model \
    --output_node_names='measureAPI' \
    --saved_model_tags=serve \
    tf-savedmodel \
    tf-webmodel
