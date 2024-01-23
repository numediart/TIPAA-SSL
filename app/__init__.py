import warnings

warnings.filterwarnings("ignore", category=UserWarning)

from flask import Flask
from flask_smorest import Api

# As my unit tests sometimes send dicts withmore info than necessary, I have to eclude unknown field, not raise en error
# https://github.com/marshmallow-code/flask-smorest/issues/211
from marshmallow import EXCLUDE
from webargs.flaskparser import FlaskParser

FlaskParser.DEFAULT_UNKNOWN_BY_LOCATION["files"] = EXCLUDE
FlaskParser.DEFAULT_UNKNOWN_BY_LOCATION["form"] = EXCLUDE
FlaskParser.DEFAULT_UNKNOWN_BY_LOCATION["json"] = EXCLUDE


def init_app() -> Flask:
    app = Flask(__name__)  # Flask app instance initiated

    app.config["API_TITLE"] = "Flowspeech Project"
    app.config["API_VERSION"] = "v2"
    app.config["API_PREFIX"] = "/api"
    app.config["OPENAPI_VERSION"] = "3.0.2"
    app.config["OPENAPI_JSON_PATH"] = "openapi.json"
    app.config["OPENAPI_URL_PREFIX"] = "/"
    app.config["OPENAPI_REDOC_PATH"] = "/redoc"
    app.config[
        "OPENAPI_REDOC_URL"
    ] = "https://cdn.jsdelivr.net/npm/redoc@next/bundles/redoc.standalone.js"
    app.config["OPENAPI_SWAGGER_UI_PATH"] = "/docs"
    app.config["OPENAPI_SWAGGER_UI_URL"] = "https://cdn.jsdelivr.net/npm/swagger-ui-dist/"

    app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

    # for not breaking the order of functions in the doc:
    # https://github.com/marshmallow-code/apispec/issues/193
    app.config["JSON_SORT_KEYS"] = False

    with app.app_context():
        api = Api(app)

        from .content_tools import bp as content_tools_bp
        from .DL_modules_v2 import bp as DL_modules_bp2  # noqa: N812
        from .DL_modules_v2_mp import bp as DL_modules_bp2_mp  # noqa: N812

        # api.register_blueprint(prefill_bp)
        api.register_blueprint(DL_modules_bp2, url_prefix="/v2")
        api.register_blueprint(DL_modules_bp2_mp, url_prefix="/v2_mp")
        api.register_blueprint(content_tools_bp)

        return app
