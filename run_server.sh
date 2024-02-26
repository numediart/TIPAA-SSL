#ls /home/mambauser/hf_models/
#sudo mkdir -p hf_models/
#sudo cp /home/mambauser/hf_models/last_hidden_state.quant.onnx ./hf_models/last_hidden_state.quant.onnx
mfa server start
gunicorn -w ${NUM_SERVICES:-1} -b 0.0.0.0:8000 'app:init_app()' --timeout 90

