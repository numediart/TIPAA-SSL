from flask import Flask, render_template, request, redirect, url_for
from flask import send_from_directory

app = Flask(__name__)

from speech_tech import *
import json
import speech_tech

@app.route('/')
def index():
    return send_from_directory( '.','index.html')

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

upload_path="./upload_files/"
@app.route('/', methods=['POST'])
def upload_file():
    uploaded_file = request.files['file']
    if uploaded_file.filename != '':
        uploaded_file.save(upload_path+uploaded_file.filename)
    return redirect(url_for('index'))

  
@app.route('/flowspeech/<module>', methods=['GET', 'POST'])
def add_message(module):
    content = request.json
    print(content['sentenceID'])
    print(content['filename'])
    # return jsonify({"uuid":uuid})
    p=set_params(sentenceID=content['sentenceID'], waveFileAddress=upload_path+content['filename'], module=module)
    try:
      method_to_call = getattr(speech_tech, module)
    except:
      print('no such module exists')
      response="-1"
      return response

    status,result=method_to_call(p)
    if not isinstance(result, list):
      print('result:',result)
      result=result.tolist()
    d={'status':status, 'result':result}
    response=json.dumps(d)
    return response
  
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')