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


def init_app(config_class="app.config.ProductionConfig") -> Flask:
    app = Flask(__name__)  # Flask app instance initiated

    app.config.from_object(config_class)

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
