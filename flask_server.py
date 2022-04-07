from flask_apispec.extension import FlaskApiSpec
from app_definition import app

# https://stackoverflow.com/questions/61444572/ignore-all-warnings-from-a-module
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


import pandas as pd
# disable pandas warning SettingWithCopyWarning
pd.options.mode.chained_assignment = None  # default='warn'

# ===================== API with DOC =============

# docs: https://flask-apispec.readthedocs.io/en/latest/usage.html#decorators

# how to do a schema with a dict:
# https://marshmallow.readthedocs.io/en/stable/quickstart.html#declaring-schemas

# import routes of different parts of the server
from server.prefill import *
from server.upload import *
from server.modules import *
from server.DL_modules import *

if False:
    from server.demo import *

debug=True
def run_app():
    app.run(debug=debug, host='0.0.0.0', port=8000)

if True:
    docs = FlaskApiSpec(app)
    docs.register(send_audio)
    docs.register(prefill_from_phrase)
    docs.register(phoneme_contrast_api)
    docs.register(module_api)

    docs.register(dl_module_api)
    docs.register(dl_phoneme_contrast_api)
    # docs.register(DL_syllable_contrast_api)

if __name__ == '__main__':
    run_app()