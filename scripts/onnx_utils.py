import gc
import logging
import os
import resource

import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from torch import nn
from transformers import Wav2Vec2Model


logger = logging.getLogger(__name__)


def _memory_usage_mb():
    """Return current and peak resident memory in MiB on Linux."""
    current_mb = None
    try:
        with open("/proc/self/status", encoding="utf-8") as status_file:
            for line in status_file:
                if line.startswith("VmRSS:"):
                    current_mb = int(line.split()[1]) / 1024
                    break
    except (FileNotFoundError, OSError, ValueError):
        pass

    # Linux reports ru_maxrss in KiB; this is a useful fallback and peak metric.
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    return current_mb, peak_mb


def log_memory(stage):
    """Log current and peak process RSS at an important conversion stage."""
    current_mb, peak_mb = _memory_usage_mb()
    current = f"{current_mb:.1f} MiB" if current_mb is not None else "unknown"
    logger.info("Memory %s: current RSS=%s, peak RSS=%.1f MiB", stage, current, peak_mb)


class LastHiddenStateModel(nn.Module):
    """Expose only the final hidden state for ONNX export."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_values):
        return self.model(
            input_values,
            output_hidden_states=False,
            return_dict=True,
        ).last_hidden_state


def quantize_onnx_model(model_path, quantized_model_path):
    """Dynamically quantize an ONNX model's weights to int8."""
    log_memory("before ONNX quantization")
    quantize_dynamic(
        model_input=model_path,
        model_output=quantized_model_path,
        weight_type=QuantType.QInt8,
    )
    log_memory("after ONNX quantization")


def convert_lhs_model_to_quant_onnx(
    w2v2_model_path, onnx_model_name="last_hidden_state.onnx"
):
    logger.info("Loading Wav2Vec2 model from %s", w2v2_model_path)
    log_memory("before model loading")
    model = Wav2Vec2Model.from_pretrained(
        w2v2_model_path,
        output_hidden_states=False,
    )
    model.eval()
    last_hidden_state_model = LastHiddenStateModel(model).eval()
    log_memory("after model loading")

    onnx_model_path = os.path.join(w2v2_model_path, onnx_model_name)

    # A short representative input is sufficient because audio_len is dynamic.
    audio_len = 16000
    logger.info("Preparing ONNX export input with audio_len=%d", audio_len)
    x = torch.zeros(1, audio_len, dtype=torch.float32)
    log_memory("after export input allocation")

    with torch.inference_mode():
        logger.info("Starting ONNX export to %s", onnx_model_path)
        torch.onnx.export(
            last_hidden_state_model,
            (x,),
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
    log_memory("after ONNX export")
    logger.info("Unquantized ONNX size: %.1f MiB", os.path.getsize(onnx_model_path) / 2**20)

    # The PyTorch model is no longer needed and can otherwise compete with
    # onnxruntime for memory during quantization.
    del x, last_hidden_state_model, model
    gc.collect()
    log_memory("after releasing PyTorch model")

    quantized_model_name = os.path.splitext(os.path.basename(onnx_model_name))[0]
    quantized_model_path = os.path.join(
        w2v2_model_path, f"{quantized_model_name}.quant.onnx"
    )
    quantize_onnx_model(onnx_model_path, quantized_model_path)
    logger.info(
        "Quantized ONNX size: %.1f MiB", os.path.getsize(quantized_model_path) / 2**20
    )
