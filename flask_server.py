from flask import Flask, request, redirect, url_for
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
@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        uploaded_file = request.files['file']
    except:
        return "error: could not access request.files['file'] "
    if uploaded_file.filename != '':
        try:
            uploaded_file.save(upload_path+uploaded_file.filename)
        except:
            return "error: could not save uploaded file"
        try:
            status_conversion, rID = prepare_audio_file(upload_path+uploaded_file.filename)
            os.remove(upload_path+uploaded_file.filename)
        except:
            return "error: could not convert uploaded file"
    else:
        return "error: filename is empty"
    return rID
  


@app.route('/vowel_stresses', methods=['GET', 'POST'])
def vowel_stresses_api():
    content = request.form
    # import pdb;pdb.set_trace()
    print(request.__dict__)
    print(content)
    print(content['phonetics'])
    print(content['rID'])
    rID=content['rID']
    phonetics=ast.literal_eval(content['phonetics'])
    p=set_params()
    p['rand_fileName']=rID
    status,result=vowel_stresses_from_phonetics_audio(phonetics,p=p)
    print('result:',result)
    if not isinstance(result, list):
        print('result:',result)
        result=result.tolist()
    
    result=[[int(x*100) for x  in sublist] for sublist in result]
    d={'status':status, 'result':result}
    response=json.dumps(d)
    return response


@app.route('/flowspeech/<module>', methods=['GET', 'POST'])
def module_api(module):
    content = request.form
    # import pdb;pdb.set_trace()
    print(request.__dict__)
    print(content)
    print(content['phonetics'])
    print(content['rID'])
    rID=content['rID']
    phonetics=ast.literal_eval(content['phonetics'])
    p=set_params()
    p['rand_fileName']=rID
    if module=='sentenceStress':
        res=sentenceStress_from_phonetics_audio(phonetics,p=p)
    elif module=='wordStress':
        res=wordStress_from_phonetics_audio(phonetics,p=p)
    else:
        res={"status": "error: no such module"}
    response=json.dumps(res)
    return response


import ast
@app.route('/phonemeContrast', methods=['GET', 'POST'])
def phoneme_contrast_api():
    content = request.form
    # import pdb;pdb.set_trace()
    print(request.__dict__)
    # print(content)
    print(content['phonetics'])
    # print(content['filename'])
    print(content['word_id'])
    print(content['rID'])
    # filename=content['filename']
    phonetics=ast.literal_eval(content['phonetics'])
    word_id=content['word_id']
    rID=content['rID']
    target=content['target']
    alternatives=ast.literal_eval(content['alternatives'])
    print(alternatives)

    p=set_params()
    p['rand_fileName']=rID
    status,result=phonemeContrast_from_phonetics_audio(phonetics, p, int(word_id), target, alternatives)
    print('status:',status)
    print('result:',result)
    if not isinstance(result, list):
        result=result.tolist()
        print('result:',result)
    if result!=[]:  
        # phonetic_transcript=result[-1]
        phonetic_detection=result[0][result[0].iloc[:,2].str.contains('_')].detected_transcription.tolist()
    else:
        phonetic_detection=result
    # phonetic_GT=phonetics_from_sentence(text)[int(word_id)]
    # d={'status':status, 'result':phonetic_transcript, 'ground_truth':phonetic_GT}
    
    d={'status':status, 'phonetic_detection':phonetic_detection}
    response=json.dumps(d)
    return response

def run_app():
    app.run(debug=True, host='0.0.0.0')


# deprecated functions
if False:
    @app.route('/flowspeech/<module>', methods=['GET', 'POST'])
    def module_api(module):
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
        res=method_to_call(p)
        print(res)
        response=json.dumps(res)
        return response

    
    upload_path="./upload_files/"
    @app.route('/', methods=['POST'])
    def upload_file():
        try:
            uploaded_file = request.files['file']
        except:
            return "error: could not access request.files['file'] "
        if uploaded_file.filename != '':
            try:
                uploaded_file.save(upload_path+uploaded_file.filename)
            except:
                return "error: could not save uploaded file"
        else:
            return "error: filename is empty"
        return "success"

if __name__ == '__main__':
    run_app()