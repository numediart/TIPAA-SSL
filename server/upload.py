# https://stackoverflow.com/questions/11994325/how-to-divide-flask-app-into-multiple-py-files
import os
import json
from app_definition import app

from flask import Response, request
from marshmallow import Schema, fields
from flask_apispec import marshal_with, doc, use_kwargs
from server.utils import access_property_error, properties_to_args
from speech_tech import prepare_audio_file

import uuid
import base64

upload_path="./upload_files/"


responseSchema=Schema.from_dict(
    {
    "status": fields.Str(), 
    "rID":fields.Str()
    }, name="audio_response"
)
properties=["audio", "API_KEY"]
@doc(description='send audio API.', tags=['audio'])
@use_kwargs(properties_to_args(properties), location=('form'))
@marshal_with(responseSchema, code=200)  # marshalling
@app.route('/send_base64_audio', methods=['POST'])
def send_audio():
    content = request.form
    properties=["audio", "API_KEY"]
    for prop in properties:
        err=access_property_error(content, prop)
        if err: return Response(err,status=400,mimetype="application/json")
    
    if os.environ['FLOWSPEECH_KEY']!=content['API_KEY']: 
        return Response(
            "error: wrong API key",
            status=400,
            mimetype="application/json"
        )
    # print(request.__dict__.keys())
    # print(content)
    audio=content['audio']
    temp_filename=str(uuid.uuid4())
    try:
        # based on:
        # https://stackoverflow.com/questions/50279380/how-to-decode-base64-string-directly-to-binary-audio-format
        wav_file = open(upload_path+temp_filename+".audio", "wb")
        decode_string = base64.b64decode(audio)
        wav_file.write(decode_string)
        wav_file.close()
    except:
        return Response(
            "error: could not save uploaded file",
            status=500,
            mimetype="application/json"
        )

    try:
        _, rID = prepare_audio_file(upload_path+temp_filename+".audio")
    except:
        return Response(
            "error: could not convert uploaded file",
            status=500,
            mimetype="application/json"
        )
    try:
        os.remove(upload_path+temp_filename+".audio")
    except:
        return Response(
            "error: could not remove uploaded file",
            status=500,
            mimetype="application/json"
        )
    res={"status": "success", "rID":rID}
    response=json.dumps(res)
    return response


