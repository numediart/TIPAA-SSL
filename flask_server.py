# from flask_apispec.extension import FlaskApiSpec
from app_definition import app

# https://stackoverflow.com/questions/61444572/ignore-all-warnings-from-a-module
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


# As my unit tests sometimes send dicts withmore info than necessary, I have to eclude unknown field, not raise en error
# https://github.com/marshmallow-code/flask-smorest/issues/211
from marshmallow import EXCLUDE
from webargs.flaskparser import FlaskParser

FlaskParser.DEFAULT_UNKNOWN_BY_LOCATION["files"] = EXCLUDE
FlaskParser.DEFAULT_UNKNOWN_BY_LOCATION["form"] = EXCLUDE
FlaskParser.DEFAULT_UNKNOWN_BY_LOCATION["json"] = EXCLUDE

import pandas as pd

# disable pandas warning SettingWithCopyWarning
pd.options.mode.chained_assignment = None  # default='warn'

# ===================== API with DOC =============

# how to do a schema with a dict:
# https://marshmallow.readthedocs.io/en/stable/quickstart.html#declaring-schemas

# import routes of different parts of the server
# from server.prefill import *
# from server.prefill import bp as prefill_bp

from server.DL_modules_v2 import *
from server.DL_modules_v2 import bp as DL_modules_bp2


from server.DL_modules_v2_mp import *
from server.DL_modules_v2_mp import bp as DL_modules_bp2_mp

from server.content_tools import bp as content_tools_bp

from flask_smorest import Api

api = Api(app)

# api.register_blueprint(prefill_bp)
api.register_blueprint(DL_modules_bp2, url_prefix='/v2')
api.register_blueprint(DL_modules_bp2_mp, url_prefix='/v2_mp')
api.register_blueprint(content_tools_bp)

debug = True


def run_app():
    app.run(debug=debug, host='0.0.0.0', port=8001)


if False:
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
