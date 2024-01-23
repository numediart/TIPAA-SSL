ls /home/mambauser/hf_models/
sudo mkdir hf_models/
sudo cp /home/mambauser/hf_models/last_hidden_state.quant.onnx ./hf_models/last_hidden_state.quant.onnx
mfa server start
gunicorn -b 0.0.0.0:8000 wsgi:app --timeout 90 