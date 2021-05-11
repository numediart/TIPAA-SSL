from flask import Flask, render_template, request, redirect, url_for
from flask import send_from_directory

app = Flask(__name__)

from speech_tech import *
import json
import speech_tech

@app.route('/')
def index():
    return send_from_directory( '.','index.html')

@app.route('/phonemeContrast.html')
def phonemeContrast_html():
    return send_from_directory( '.','phonemeContrast.html')

@app.route('/vowel_stresses.html')
def vowel_stresses_html():
    return send_from_directory( '.','vowel_stresses.html')

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
    content = request.form
    # import pdb;pdb.set_trace()
    print(request.__dict__)
    print(content)
    print(content['sentenceID'])
    print(content['filename'])
    # return jsonify({"uuid":uuid})
    filename=content['filename']
    sentenceID=content['sentenceID']
    p=set_params(sentenceID=int(sentenceID), waveFileAddress=upload_path+filename, module=module)
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
  

  
@app.route('/vowel_stresses', methods=['GET', 'POST'])
def vowel_stresses_api():
    content = request.form
    # import pdb;pdb.set_trace()
    print(request.__dict__)
    print(content)
    print(content['text'])
    print(content['filename'])
    filename=content['filename']
    text=content['text']
    # p=set_params(waveFileAddress=upload_path+filename)
    # p=make_all_phones_annotation_files(p,text)
    # status,result=vowel_stresses(p)

    status,result=vowel_stresses_from_text_audio(text,upload_path+filename)
    print('result:',result)
    if not isinstance(result, list):
      print('result:',result)
      result=result.tolist()
    d={'status':status, 'result':result}
    response=json.dumps(d)
    return response

import ast
@app.route('/phonemeContrast', methods=['GET', 'POST'])
def phoneme_contrast_api():
    content = request.form
    # import pdb;pdb.set_trace()
    print(request.__dict__)
    # print(content)
    print(content['text'])
    print(content['filename'])
    print(content['word_id'])
    filename=content['filename']
    text=content['text']
    word_id=content['word_id']
    alternatives=ast.literal_eval(content['alternatives'])
    print(alternatives)
    
    # p=set_params(waveFileAddress=upload_path+filename)
    # p=make_pContrast_annotation_files(p,text=text,word_id=int(word_id), target_phones='D', 
    #             alternatives=alternatives)
    # status,result=phonemeContrast(p)
    status,result=phonemeContrast_from_text_audio(text, upload_path+filename, int(word_id), 'D', alternatives)
    print('status:',status)
    print('result:',result)
    if not isinstance(result, list):
      result=result.tolist()
      print('result:',result)

    if result!=[]:  
      phonetic_transcript=result[-1]
    else:
      phonetic_transcript=result
    d={'status':status, 'result':phonetic_transcript}
    response=json.dumps(d)
    return response

def run_app():
  app.run(debug=True, host='0.0.0.0')

if __name__ == '__main__':
    run_app()