# https://flask.palletsprojects.com/en/1.1.x/deploying/wsgi-standalone/#gunicorn

# export FLASK_APP=flask_server.py
# flask run
# python flask_server.py
# gunicorn flask_server:app 
# uwsgi --http 127.0.0.1:5000 --module flask_server:app

# cron
# mfa model download g2p french_mfa && mfa model download g2p spanish_spain_mfa && mfa model download g2p spanish_latin_america_mfa && mfa model download g2p english_uk_mfa && mfa model download g2p english_us_mfa  
mfa model download acoustic english_us_arpa
ls /home/mambauser/hf_models/
sudo mkdir hf_models/
sudo cp /home/mambauser/hf_models/last_hidden_state.quant.onnx ./hf_models/last_hidden_state.quant.onnx
mfa server start
gunicorn -b 0.0.0.0:8000 flask_server:app --timeout 90 
# NEW_RELIC_CONFIG_FILE=newrelic.ini newrelic-admin run-program python flask_server.py
# NEW_RELIC_CONFIG_FILE=newrelic.ini newrelic-admin run-program gunicorn -b 0.0.0.0:8000 flask_server:app