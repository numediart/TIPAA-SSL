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
# from server.prefill import *
# from server.prefill import bp as prefill_bp
# from server.upload import *
# from server.upload import bp as upload_bp

from server.DL_modules_v2 import *
from server.DL_modules_v2 import bp as DL_modules_bp2

# app.register_blueprint(upload_bp, url_prefix='/')
# app.register_blueprint(prefill_bp, url_prefix='/')

# app.register_blueprint(prefill_bp)
app.register_blueprint(DL_modules_bp2, url_prefix='/v2')

if False:
    from server.demo import *

debug=True
def run_app():
    app.run(debug=debug, host='0.0.0.0', port=8000)

if True:
    # https://github.com/jmcarp/flask-apispec/issues/16
    # use blueprints with flask apispec

    docs = FlaskApiSpec(app)
    # docs.register(prefill_from_phrase, blueprint="prefill")
    
    docs.register(dl_word_stress_api_v2, blueprint="DL_modules_v2")
    docs.register(dl_sentence_stress_api_v2, blueprint="DL_modules_v2")
    docs.register(dl_vowel_contrast_api_v2, blueprint="DL_modules_v2")
    docs.register(dl_consonant_contrast_api_v2, blueprint="DL_modules_v2")
    docs.register(dl_termination_contrast_api_v2, blueprint="DL_modules_v2")

    # To hide "OPTIONS" endpoints
    # https://github.com/jmcarp/flask-apispec/issues/111
    for key, value in docs.spec._paths.items():
        docs.spec._paths[key] = {
            inner_key: inner_value
            for inner_key, inner_value in value.items()
            if inner_key != 'options'
        }
if __name__ == '__main__':
    run_app()