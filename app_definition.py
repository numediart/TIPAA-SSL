from flask import Flask

from apispec import APISpec
from apispec.ext.marshmallow import MarshmallowPlugin

app = Flask(__name__)  # Flask app instance initiated

app.config.update({
    'APISPEC_SPEC': APISpec(
        title='Flowspeech Project',
        version='v1',
        plugins=[MarshmallowPlugin()],
        openapi_version='2.0.0'
    ),
    'APISPEC_SWAGGER_URL': '/swagger/',  # URI to access API Doc JSON
    'APISPEC_SWAGGER_UI_URL': '/swagger-ui/'  # URI to access UI of API Doc
})

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# for not breaking the order of functions in the doc:
# https://github.com/marshmallow-code/apispec/issues/193
app.config["JSON_SORT_KEYS"] = False

