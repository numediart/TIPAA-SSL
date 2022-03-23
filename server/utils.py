from marshmallow import fields
from functools import wraps
from flask import current_app, abort

# make functions available only in debug mode:
# https://stackoverflow.com/questions/55719252/make-a-route-only-accessible-in-debug-mode-with-flask
def debug_only(f):
    @wraps(f)
    def wrapped(**kwargs):
        if not current_app.debug:
            abort(404)
        return f(**kwargs)
    return wrapped


def properties_to_args(properties, required=True):
    args={}
    for prop in properties:
        args[prop]=fields.String(required=required)
    return args

def access_property_error(content, property):
    try:
        content[property]
        return 0
    except:
        response='error: could not access "'+property+'" property of the request'
        return response
    
