def convert_lhs_model_to_quant_onnx(
    w2v2_model_path, onnx_model_name="last_hidden_state.onnx"
):
    model = Wav2Vec2Model.from_pretrained(w2v2_model_path, output_hidden_states=True)
    # # Convert to ONNX format
    # dummy_input = torch.randn(1, 16000*15, dtype=torch.float32)
    last_hidden_state_model = LastHiddenStateModel(model)
    # last_hidden_state_model(dummy_input)

    # torch.onnx.export(last_hidden_state_model, dummy_input, onnx_model_name)

    onnx_model_path = '/'.join([w2v2_model_path, onnx_model_name])

    audio_len = 250000
    x = torch.randn(1, audio_len, requires_grad=True)
    torch.onnx.export(
        last_hidden_state_model,  # model being run
        x,  # model input (or a tuple for multiple inputs)
        onnx_model_path,  # where to save the model (can be a file or file-like object)
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=11,  # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=['input'],  # the model's input names
        output_names=['output'],  # the model's output names
        dynamic_axes={
            'input': {1: 'audio_len'},  # variable length axes
            'output': {1: 'audio_len'},
        },
    )
    quantized_model_name = onnx_model_name.split("/")[-1].split('.')[0] + ".quant.onnx"
    quantized_model_path = '/'.join([w2v2_model_path, quantized_model_name])
    quantize_onnx_model(onnx_model_path, quantized_model_path)
