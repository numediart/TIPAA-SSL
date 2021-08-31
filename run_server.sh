# https://flask.palletsprojects.com/en/1.1.x/deploying/wsgi-standalone/#gunicorn

# export FLASK_APP=flask_server.py
# flask run
# python flask_server.py
# gunicorn flask_server:app 
# uwsgi --http 127.0.0.1:5000 --module flask_server:app

export FLOWSPEECH_KEY="ThisIsTheFlowchaseSP-APIKey:MeaningOfLife=42"
cron
gunicorn -b 0.0.0.0:8000 flask_server:app
# python flask_server.py