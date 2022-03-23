from flask_apispec.extension import FlaskApiSpec

from app_definition import app

# ===================== API with DOC (above is less necessary:  ) =============

# docs: https://flask-apispec.readthedocs.io/en/latest/usage.html#decorators

# how to do a schema with a dict:
# https://marshmallow.readthedocs.io/en/stable/quickstart.html#declaring-schemas

# import routes of prefill
from server.prefill import *
from server.upload import *
from server.modules import *
# from server.prefill import prefill_from_phrase

if False:
    from server.demo import *

debug=True
def run_app():
    app.run(debug=debug, host='0.0.0.0', port=8000)

if False:
    docs = FlaskApiSpec(app)
    docs.register(send_audio)
    docs.register(prefill_from_phrase)
    docs.register(phoneme_contrast_api)
    docs.register(module_api)

if __name__ == '__main__':
    run_app()