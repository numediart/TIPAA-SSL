import os

import onnx
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from torch import nn
from transformers import Wav2Vec2Model


class LastHiddenStateModel(nn.Module):
    """Expose only the final hidden state for ONNX export."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_values):
        return self.model(input_values, output_hidden_states=True).hidden_states[-1]


def quantize_onnx_model(model_path, quantized_model_path):
    """Dynamically quantize an ONNX model's weights to int8."""
    quantize_dynamic(
        model_input=model_path,
        model_output=quantized_model_path,
        weight_type=QuantType.QInt8,
    )


def convert_lhs_model_to_quant_onnx(
    w2v2_model_path, onnx_model_name="last_hidden_state.onnx"
):
    model = Wav2Vec2Model.from_pretrained(w2v2_model_path, output_hidden_states=True)
    model.eval()
    last_hidden_state_model = LastHiddenStateModel(model).eval()

    onnx_model_path = os.path.join(w2v2_model_path, onnx_model_name)

    audio_len = 250000
    x = torch.randn(1, audio_len, dtype=torch.float32)
    with torch.no_grad():
        torch.onnx.export(
            last_hidden_state_model,
            x,
            onnx_model_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {1: "audio_len"},
                "output": {1: "output_len"},
            },
        )

    # Validate the exported graph before attempting quantization. This turns a
    # later, less actionable quantizer error into an immediate export error.
    onnx.checker.check_model(onnx_model_path)

    quantized_model_name = os.path.splitext(os.path.basename(onnx_model_name))[0]
    quantized_model_path = os.path.join(
        w2v2_model_path, f"{quantized_model_name}.quant.onnx"
    )
    quantize_onnx_model(onnx_model_path, quantized_model_path)
