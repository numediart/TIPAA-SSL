from flask import Flask, request
from flask_restful import Api

from apispec import APISpec
from marshmallow import Schema, fields
from apispec.ext.marshmallow import MarshmallowPlugin
from flask_apispec.extension import FlaskApiSpec
from flask_apispec import marshal_with, doc, use_kwargs

from flask import send_from_directory
from flask import Response

# from speech_tech import *
from speech_tech import stress_from_formatted_phonetics, prepare_audio_file, vowel_stresses_from_phonetics_audio, phonemeContrast_from_formatted_phonetics_audio, merge_list
import pandas as pd
import json
import speech_tech
from text_processing import generate_prefill_csv, prefill_for_sentence, check_phonemes

import uuid
import base64

import ast
from text_processing import cmu_to_gibberish
import os

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


syllables=pd.read_csv('data/syllables.csv')


app = Flask(__name__)  # Flask app instance initiated
api = Api(app)  # Flask restful wraps Flask app around it.
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

# for not breaking the order of functions in thr doc:
# https://github.com/marshmallow-code/apispec/issues/193
app.config["JSON_SORT_KEYS"] = False

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

@app.route('/prefill_from_phrases.html', methods=['GET'])
def prefill_from_phrases_html():
    return send_from_directory( './html/','prefill_from_phrases.html')

@app.route('/prefill_from_phrases', methods=['POST'])
def prefill_from_phrases():
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
        
        df=generate_prefill_csv(upload_path+uploaded_file.filename)#, out_path=upload_path+'prefill.csv')
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
    try:
        # from:
        # https://stackoverflow.com/questions/38634862/use-flask-to-convert-a-pandas-dataframe-to-csv-and-serve-a-download
        return Response(
                df.to_csv(),
                mimetype="text/csv",
                headers={"Content-disposition":
                "attachment; filename=filename.csv"})
    except Exception as e:
        return Response(
                "error: "+str(e),
                status=500,
                mimetype="application/json"
                )



upload_path="./upload_files/"
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



@app.route('/vowel_stresses', methods=['POST'])
@debug_only
def vowel_stresses_api():
    content = request.form

    properties=["phonetics","rID"]
    for prop in properties:
        err=access_property_error(content, prop)
        if err: return Response(err,status=400,mimetype="application/json")
    
    
    # print(request.__dict__)
    print(content)
    print(content['phonetics'])
    print(content['rID'])
    rID=content['rID']
    phonetics=ast.literal_eval(content['phonetics'])
    status,result=vowel_stresses_from_phonetics_audio(rID, phonetics)
    print('result:',result)
    if not isinstance(result, list):
        print('result:',result)
        result=result.tolist()
    
    result=[[int(x*100) for x  in sublist] for sublist in result]
    d={'status':status, 'result':result}
    response=json.dumps(d)
    return response


# ===================== API with DOC (above is less necessary:  ) =============

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
    
# docs: https://flask-apispec.readthedocs.io/en/latest/usage.html#decorators

# how to do a schema with a dict:
# https://marshmallow.readthedocs.io/en/stable/quickstart.html#declaring-schemas

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
    print(request.__dict__.keys())
    # print(content)
    # print(content['audio'])
    # print(content['extension'])
    audio=content['audio']
    # phonetics=ast.literal_eval(content['phonetics'])
    # extension=content['extension']

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





responseSchema=Schema.from_dict(
    {
        "status": fields.Str(), 
        "stress_intensities":fields.List(fields.Integer), 
        "stress_binaries":fields.List(fields.Integer)
    }, name="stress_response"
)
properties=["phonetics","rID","text"]
@doc(description='Detection sentence stress or word stress', tags=['stress'])
@use_kwargs(properties_to_args(properties), location=('form'))
@marshal_with(responseSchema, code=200)  # marshalling
@app.route('/flowspeech/<module>', methods=['POST'])
def module_api(module):
    content = request.form

    # import pdb;pdb.set_trace()
    
    properties=["phonetics","rID","text"]
    for prop in properties:
        err=access_property_error(content, prop)
        if err: return Response(err,status=400,mimetype="application/json")

    # print(request.__dict__)
    print(content)
    print(content['phonetics'])
    print(content['rID'])
    rID=content['rID']
    # phonetics=ast.literal_eval(content['phonetics'])
    phonetics=content['phonetics']

    
    split_phonetics=[[s.split('_') for s in w.split('|')] for w in phonetics.split(' ')]
    merged_phonetics=[merge_list(word) for word in split_phonetics]    
    not_p=check_phonemes(merge_list(merged_phonetics))
    if not_p is not None: 
        err="error: "+not_p+" is not a phoneme"
        return Response(err,status=400,mimetype="application/json")



    if module=='sentenceStress':
        text=content['text']
        res=stress_from_formatted_phonetics(rID, phonetics, text, level="sentence")
    elif module=='wordStress':
        res=stress_from_formatted_phonetics(rID, phonetics, level="word")
    else:
        res={"status": "error: no such module"}
    response=json.dumps(res)

    if res['status'].split(':')[0]=='error':
        return Response(response,status=500,mimetype="application/json")
    else:
        return Response(response,status=200,mimetype="application/json")

responseSchema=Schema.from_dict(
    {"status": fields.Str(), 
    "phonetic_detection": fields.Str(), 
    "gibberish_truth": fields.Str(),
    "gibberish_detected":fields.Str()},
    name="phonemeContrast_response"
)
properties=["phonetics","rID","word_idx","target","syl_idx","alternatives"]
@doc(description='Phoneme contrast', tags=['phonemeContrast'])
@use_kwargs(properties_to_args(properties), location=('form'))
@marshal_with(responseSchema, code=200)  # marshalling
@app.route('/phonemeContrast', methods=['POST'])
def phoneme_contrast_api():
    content = request.form
    properties=["phonetics","rID","word_idx","target","syl_idx","alternatives"]
    for prop in properties:
        err=access_property_error(content, prop)
        if err: return Response(err,status=400,mimetype="application/json")
    
    # print(request.__dict__)
    print(content['phonetics'])
    print(content['word_idx'])
    print(content['rID'])
    phonetics=content['phonetics']
    
    split_phonetics=[[s.split('_') for s in w.split('|')] for w in phonetics.split(' ')]
    merged_phonetics=[merge_list(word) for word in split_phonetics]    
    not_p=check_phonemes(merge_list(merged_phonetics))
    if not_p is not None: 
        err="error: "+not_p+" is not a phoneme"
        return Response(err,status=400,mimetype="application/json")


    word_idx=content['word_idx']
    syl_idx=content['syl_idx']
    rID=content['rID']
    target=content['target']
    alternatives=content['alternatives']
    print(alternatives)

    word_idx=int(word_idx)
    syl_idx=int(syl_idx)

    # split_phonetics=[[s.split('_') for s in w.split('|')] for w in phonetics.split(' ')]
    # extract the syllables which contain the target
    syls_with_target=[syl for syl in phonetics.split(' ')[word_idx].split('|') if target in syl]
    idx_syls_with_target=[i for i,syl in enumerate(phonetics.split(' ')[word_idx].split('|')) if target in syl]

    if syl_idx not in idx_syls_with_target:
        return Response("error: the syllable corresponding to syl_idx does not contain the target.",status=400,mimetype="application/json")
    
    target_idx=idx_syls_with_target.index(syl_idx)
    
    status,result=phonemeContrast_from_formatted_phonetics_audio(rID, phonetics, word_idx, target, alternatives)
    print('status:',status)
    print('result:',result)
    if not isinstance(result, list):
        result=result.tolist()
        print('result:',result)
    if result!=[]:  
        # phonetic_transcript=result[-1]
        phonetic_detection=result[0][result[0].iloc[:,2].str.contains('_')].detected_transcription.tolist()
    else:
        # phonetic_detection=result
        return Response(status,status=200,mimetype="application/json")

    if phonetic_detection==[]: Response("error: phonetic detection is empty",status=200,mimetype="application/json")
    # syls_detection=[syl.replace(target, phonetic_detection[i]) for i,syl in enumerate(syls_with_target)]
    syls_detection=[]
    for i,syl in enumerate(syls_with_target):
        try:
            phonetic_detection[i]
        except:
            return Response("error: index out of bounds in phonetic detection",status=500,mimetype="application/json")
        syls_detection.append(syl.replace(target, phonetic_detection[i]))

    gs_t=[]
    gs_d=[]
    for syl_t,syl_d in zip(syls_with_target,syls_detection):
        g_t=[cmu_to_gibberish[el] if not el[-1] in str([0,1,2]) else cmu_to_gibberish[el[:-1]] for el in syl_t.split('_')]
        g_d=[cmu_to_gibberish[el] if not el[-1] in str([0,1,2]) else cmu_to_gibberish[el[:-1]] for el in syl_d.split('_')]
        gs_t.append('_'.join(g_t))
        gs_d.append('_'.join(g_d))
    
    d={'status':status, 'phonetic_detection':phonetic_detection[target_idx], 'gibberish_truth':gs_t[target_idx], 'gibberish_detected':gs_d[target_idx]}
    response=json.dumps(d)
    return response



record={'text':fields.Str(),
        'cmu_phonetics':fields.Str(),
        'pronounciation_guide':fields.Str(),
        'pronounciation_guide_hr':fields.Str(),
        'syllable_parts':fields.Str(),
        'n_syl_mismatch':fields.Integer(),
        'n_syl_mismatches':fields.List(fields.Integer),
        'used_method_for_syl_text':fields.List(fields.Str()),
        'cmu_phonetics_alt':fields.List(fields.List(fields.Str())),
        'pronounciation_guide_alt':fields.List(fields.List(fields.Str())),
        'pronounciation_guide_hr_alt':fields.List(fields.List(fields.Str())),
        'n_alternatives':fields.List(fields.Integer)
        }
responseSchema=Schema.from_dict(record, name="prefill response")
@doc(description='Prefill from phrase', tags=['prefill'])
@use_kwargs({'phrase':fields.String(required=True, description="Text sentence to be processed. It can contain special characters etc.")}, location=('form'))
@marshal_with(responseSchema, code=200)  # marshalling
@app.route('/prefill_from_phrase', methods=['POST'])
@debug_only 
def prefill_from_phrase():
    content = request.form
    
    err=access_property_error(content, "phrase")
    if err: return Response(err,status=400,mimetype="application/json")

    # import pdb;pdb.set_trace()
    # print(request.__dict__)
    print(content)
    print(content['phrase'])

    d=prefill_for_sentence(content['phrase'], syllables)
    # print(d)
    response=json.dumps(d)
    # print(response)
    return response


def run_app():
    app.run(debug=True, host='0.0.0.0', port=8000)

docs = FlaskApiSpec(app)
docs.register(send_audio)
docs.register(prefill_from_phrase)
docs.register(phoneme_contrast_api)
docs.register(module_api)

# deprecated functions
if False:
    @app.route('/flowspeech/<module>', methods=['POST'])
    def module_api(module):
        content = request.form
        # import pdb;pdb.set_trace()
        # print(request.__dict__)
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