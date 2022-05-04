from flask import request
from flask_apispec import marshal_with, doc, use_kwargs
from flask import Response, Blueprint

import ast
import json
from DL_speech_tech import phonemeContrast_from_formatted_phonetics_audio, stress_from_formatted_phonetics, termination_contrast_from_formatted_phonetics_audio, syllable_contrast_from_formatted_phonetics_audio
from utils.text_processing import check_phonemes, cmu_vowels, cmu_consonants, chunk_text, split_phonetics

# from app_definition import app
from server.utils import default_example, access_property_error, properties_to_args, check_request, request_phoneme_contrast, request_stress, request_stress_v2, stress_responseSchema, contrast_responseSchema, syl_contrast_responseSchema

bp=Blueprint('DL_modules_v2', __name__, url_prefix='/')

d=default_example()

# documenting params: https://github.com/jmcarp/flask-apispec/issues/137
def params_def(properties, example=None):
    params={}
    for prop in properties:
        record={
                'in': 'query',
                'description': prop,
                'required': True,
                'type': 'string',
            }
        if example is not None:
            record['example']= example[prop]
        params[prop]=record
    return params

properties=["phonetics","audio64"]
example={"phonetics":d["phonetics_chunks"], "audio64": d["audio64"]}
@doc(description='Detection sentence stress or word stress', tags=['w2v_v2'], params=params_def(properties, example))
@marshal_with(stress_responseSchema, code=200)  # marshalling
@bp.route('/w2v/stress/<module>', methods=['GET'])
def dl_module_api_v2(module, **kwargs):
    d = request.args.to_dict()
    properties=["phonetics","audio64"]
    return request_stress_v2(d, properties, module, mode='base64')

properties=["phonetics","audio64","word_idx","target","syl_idx"]
example={"phonetics":d["phonetics"], "audio64": d["audio64"],"word_idx":d["vowel_w_idx"],"target":d["vowel_target"],"syl_idx":d["vowel_s_idx"]}
@doc(description='Vowel contrast', tags=['w2v_v2'], params=params_def(properties, example))
@marshal_with(contrast_responseSchema, code=200)  # marshalling
@bp.route('/w2v/contrast/vowel', methods=['GET'])
def dl_vowel_contrast_api_v2(**kwargs):
    d = request.args.to_dict()
    properties=["phonetics","audio64","word_idx","target","syl_idx"]
    return request_phoneme_contrast(d, properties, mode='base64')


properties=["phonetics","audio64","word_idx","target","syl_idx", "target_occurence_idx"]
example={"phonetics":d["phonetics"], "audio64": d["audio64"],"word_idx":d["consonant_w_idx"],"target":d["consonant_target"],"syl_idx":d["consonant_s_idx"], "target_occurence_idx":d["consonant_target_occurence_idx"]}
@doc(description='Consonant contrast', tags=['w2v_v2'], params=params_def(properties, example))
# @use_kwargs(properties_to_args(properties), location="query")
@marshal_with(contrast_responseSchema, code=200)  # marshalling
@bp.route('/w2v/contrast/consonant', methods=['GET'])
def dl_consonant_contrast_api_v2(**kwargs):
    d = request.args.to_dict()
    properties=["phonetics","audio64","word_idx","target","syl_idx", "target_occurence_idx"]
    target_occurence_idx=int(d['target_occurence_idx'])
    return request_phoneme_contrast(d, properties, target_occurence_idx, alternatives=cmu_consonants, mode='base64')


properties=["phonetics","audio64","word_idx","target"]
example={"phonetics":d["phonetics"], "audio64": d["audio64"],"target":d["termination_target"],"word_idx":d["termination_w_idx"]}
@doc(description='Termination contrast', tags=['w2v_v2'], params=params_def(properties, example))
# @use_kwargs(properties_to_args(properties), location="query")
@marshal_with(contrast_responseSchema, code=200)  # marshalling
@bp.route('/w2v/contrast/termination', methods=['GET'])
def dl_termination_contrast_api_v2(**kwargs):
    d = request.args.to_dict()
    properties=["phonetics","audio64","word_idx","target"]
    return request_phoneme_contrast(d, properties, tech_function=termination_contrast_from_formatted_phonetics_audio, mode='base64')

properties=["phonetics","audio64","word_idx","syl_idx"]
example={"phonetics":d["phonetics"], "audio64": d["audio64"],"word_idx":d["termination_w_idx"],"syl_idx":"1"}
@doc(description='Syllable contrast', tags=['w2v_v2'], params=params_def(properties, example))
# @use_kwargs(properties_to_args(properties), location="query")
@marshal_with(syl_contrast_responseSchema, code=200)  # marshalling
@bp.route('/w2v/contrast/syllable', methods=['GET'])
def dl_syllable_contrast_api_v2(**kwargs):
    # TODO: refactor with "request_phoneme_contrast" by parametrizing the function call
    d = request.args.to_dict()
    properties=["phonetics","audio64","word_idx","syl_idx"]
    err=check_request(d, properties)
    if err is not None: return Response(err,status=400,mimetype="application/json")

    word_idx=int(d['word_idx'])
    syl_idx=int(d['syl_idx'])

    if word_idx>=len(d['phonetics'].split(' ')):
        return Response("error: word_idx >= number of words",status=400,mimetype="application/json")    
    res=syllable_contrast_from_formatted_phonetics_audio(d['audio64'],phonetics=d['phonetics'], target_word_idx=word_idx, target_syllable_idx=syl_idx, mode='base64')
    response=json.dumps(res)
    if res['status'].split(':')[0]=='error':
        return Response(response,status=500,mimetype="application/json")
    else:
        return Response(response,status=200,mimetype="application/json")

