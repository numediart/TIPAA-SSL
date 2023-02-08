# --- from https://github.com/ccoreilly/wav2vec2-service/blob/master/convert_torch_to_onnx.py ----
# MIT License
from onnxruntime.quantization.quantize import quantize
from transformers import Wav2Vec2ForCTC
import torch
import argparse


import torch
import torch.onnx

# Original transformer model
from transformers import Wav2Vec2Model, Wav2Vec2Processor, Wav2Vec2Config
w2v2_model_path="hf_models/facebook/wav2vec2-xlsr-53-espeak-cv-ft"

# Define a new model that outputs the last hidden state
class LastHiddenStateModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, input):
        # output = self.model(input)
        # return output[-1]
        # input_values = self.processor(torch.tensor(s), sampling_rate=fs, return_tensors="pt").input_values.to('cpu')
        with torch.no_grad(): 
            return self.model(input).hidden_states[-1]


def convert_to_onnx(model_id_or_path, onnx_model_name):
    print(f"Converting {model_id_or_path} to onnx")
    model = Wav2Vec2ForCTC.from_pretrained(model_id_or_path)
    audio_len = 250000

    x = torch.randn(1, audio_len, requires_grad=True)

    torch.onnx.export(model,                        # model being run
                    x,                              # model input (or a tuple for multiple inputs)
                    onnx_model_name,                # where to save the model (can be a file or file-like object)
                    export_params=True,             # store the trained parameter weights inside the model file
                    opset_version=11,               # the ONNX version to export the model to
                    do_constant_folding=True,       # whether to execute constant folding for optimization
                    input_names = ['input'],        # the model's input names
                    output_names = ['output'],      # the model's output names
                    dynamic_axes={'input' : {1 : 'audio_len'},    # variable length axes
                                'output' : {1 : 'audio_len'}})

def quantize_onnx_model(onnx_model_path, quantized_model_path):
    print("Starting quantization...")
    from onnxruntime.quantization import quantize_dynamic, QuantType
    quantize_dynamic(onnx_model_path,
                     quantized_model_path,
                     weight_type=QuantType.QUInt8)

    print(f"Quantized model saved to: {quantized_model_path}")

def convert_lhs_model_to_quant_onnx(w2v2_model_path, onnx_model_name="last_hidden_state.onnx"):
    model=Wav2Vec2Model.from_pretrained(w2v2_model_path, output_hidden_states=True) 
    # # Convert to ONNX format
    # dummy_input = torch.randn(1, 16000*15, dtype=torch.float32)
    last_hidden_state_model = LastHiddenStateModel(model)
    # last_hidden_state_model(dummy_input)
    
    # torch.onnx.export(last_hidden_state_model, dummy_input, onnx_model_name)

    onnx_model_path='/'.join([w2v2_model_path,onnx_model_name])

    audio_len = 250000
    x = torch.randn(1, audio_len, requires_grad=True)
    torch.onnx.export(last_hidden_state_model,                        # model being run
                    x,                              # model input (or a tuple for multiple inputs)
                    onnx_model_path,                # where to save the model (can be a file or file-like object)
                    export_params=True,             # store the trained parameter weights inside the model file
                    opset_version=11,               # the ONNX version to export the model to
                    do_constant_folding=True,       # whether to execute constant folding for optimization
                    input_names = ['input'],        # the model's input names
                    output_names = ['output'],      # the model's output names
                    dynamic_axes={'input' : {1 : 'audio_len'},    # variable length axes
                                'output' : {1 : 'audio_len'}})
    quantized_model_name = onnx_model_name.split("/")[-1].split('.')[0] + ".quant.onnx"
    quantized_model_path='/'.join([w2v2_model_path,quantized_model_name])
    quantize_onnx_model(onnx_model_name, quantized_model_path)


def inference_demo(quantized_model_name):
    audio_len = 250000
    x = torch.randn(1, audio_len, requires_grad=True)
    import onnxruntime as rt
    ONNX_PATH = quantized_model_name
    sess_options = rt.SessionOptions()
    sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = rt.InferenceSession(ONNX_PATH, sess_options)
    onnx_outputs = session.run(None, {session.get_inputs()[0].name: x.detach().numpy()})

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="ccoreilly/wav2vec2-large-100k-voxpopuli-catala",
        help="Model HuggingFace ID or path that will converted to ONNX",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Whether to use also quantize the model or not",
    )
    args = parser.parse_args()
    model_id_or_path = args.model
    onnx_model_name = model_id_or_path.split("/")[-1] + ".onnx"
    convert_to_onnx(model_id_or_path, onnx_model_name)
    if (args.quantize):
        quantized_model_name = model_id_or_path.split("/")[-1] + ".quant.onnx"
        quantize_onnx_model(onnx_model_name, quantized_model_name)
    





