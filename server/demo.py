
# https://stackoverflow.com/questions/11994325/how-to-divide-flask-app-into-multiple-py-files
import os

from flask import send_from_directory
from server.utils import debug_only

from app_definition import app

from flask import Response, request
from server.utils import debug_only
from speech_tech import prepare_audio_file

from server.upload import upload_path

@app.route('/upload', methods=['POST'])
@debug_only
def upload_file():
    # import pdb;pdb.set_trace()
    try:
        uploaded_file = request.files['file']
    except:
        return Response(
            "error: could not access request.files['file']",
            status=400,
            mimetype="application/json"
        )
    if uploaded_file.filename != '':
        try:
            uploaded_file.save(upload_path+uploaded_file.filename)
        except:
            return Response(
                "error: could not save uploaded file",
                status=500,
                mimetype="application/json"
            )
        try:
            # import pdb;pdb.set_trace()
            status_conversion, rID = prepare_audio_file(upload_path+uploaded_file.filename)
            print(rID)
        except:
            return Response(
                "error: could not convert uploaded file",
                status=500,
                mimetype="application/json"
            )
        try:
            os.remove(upload_path+uploaded_file.filename)
        except:
            return Response(
                "error: could not remove uploaded file",
                status=500,
                mimetype="application/json"
            )
    else:
        return Response(
                "error: filename is empty",
                status=400,
                mimetype="application/json"
            )
    return rID


@app.route('/')
@debug_only
def index():
    return send_from_directory( './html/','index.html')

@app.route('/app.js')
@debug_only
def record_app():
    return send_from_directory( './html/','app.js')

@app.route('/phonemeContrast.html')
@debug_only
def phonemeContrast_html():
    return send_from_directory( './html/','phonemeContrast.html')

@app.route('/vowel_stresses.html')
@debug_only
def vowel_stresses_html():
    return send_from_directory( './html/','vowel_stresses.html')

