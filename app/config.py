API_TITLE = "Flowspeech Project"
API_VERSION = "v2"
API_PREFIX = "/api"
OPENAPI_VERSION = "3.0.2"
OPENAPI_JSON_PATH = "openapi.json"
OPENAPI_URL_PREFIX = "/"
OPENAPI_REDOC_PATH = "/redoc"
OPENAPI_REDOC_URL = "https://cdn.jsdelivr.net/npm/redoc@next/bundles/redoc.standalone.js"
OPENAPI_SWAGGER_UI_PATH = "/docs"
OPENAPI_SWAGGER_UI_URL = "https://cdn.jsdelivr.net/npm/swagger-ui-dist/"

MAX_CONTENT_LENGTH = 16 * 1024 * 1024

# for not breaking the order of functions in the doc:
# https://github.com/marshmallow-code/apispec/issues/193
JSON_SORT_KEYS = False
