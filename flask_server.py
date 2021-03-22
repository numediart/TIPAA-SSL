from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__, template_folder='template')

from speech_tech import *
import json

@app.route('/')
def index():
    source='''
    <!doctype html>
<html>
  <head>
    <title>File Upload</title>
  </head>
  <body>
    <h1>File Upload</h1>
    <form method="POST" action="" enctype="multipart/form-data">
      <p><input type="file" name="file"></p>
      <p><input type="submit" value="Submit"></p>
    </form>
  </body>
</html>
'''
    return source


# app.config['UPLOAD_FOLDER'] = "./upload_files/"
# app.config['MAX_CONTENT_PATH']
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

upload_path="./upload_files/"
@app.route('/', methods=['POST'])
def upload_file():
    uploaded_file = request.files['file']
    if uploaded_file.filename != '':
        uploaded_file.save(upload_path+uploaded_file.filename)
    return redirect(url_for('index'))


@app.route("/ping")
def ping():
    return "pong"
  
@app.route('/flowspeech/<module>')
def call_module(module):
    # show the user profile for that user
    return 'User %s' % escape(module)
  
@app.route('/api/add_message/<module>', methods=['GET', 'POST'])
def add_message(module):
    content = request.json
    print(content['sentenceID'])
    print(content['filename'])
    # return jsonify({"uuid":uuid})
    p=set_params(sentenceID=content['sentenceID'], waveFileAddress=upload_path+content['filename'], module=module)
    if module=="wordStress":
      status,result=wordStress(p)
      result=result.tolist()
      result=json.dumps(result)
    if module=="sentenceStress":
      status,result=sentenceStress(p)
      result=result.tolist()
      result=json.dumps(result)
    else:
      print('no such module exists')
      result="-1"
    return result