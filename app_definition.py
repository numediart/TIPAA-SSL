from flask import Flask

from apispec import APISpec
from apispec.ext.marshmallow import MarshmallowPlugin

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

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# for not breaking the order of functions in the doc:
# https://github.com/marshmallow-code/apispec/issues/193
app.config["JSON_SORT_KEYS"] = False
