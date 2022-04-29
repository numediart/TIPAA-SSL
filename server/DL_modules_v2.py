from flask import request
from flask_apispec import marshal_with, doc, use_kwargs
from flask import Response, Blueprint

import ast
import json
from DL_speech_tech import phonemeContrast_from_formatted_phonetics_audio, stress_from_formatted_phonetics, termination_contrast_from_formatted_phonetics_audio, syllable_contrast_from_formatted_phonetics_audio

# from app_definition import app
from server.utils import access_property_error, properties_to_args, check_request, request_phoneme_contrast, request_stress, request_stress_v2, stress_responseSchema, contrast_responseSchema, syl_contrast_responseSchema

bp=Blueprint('DL_modules_v2', __name__, url_prefix='/')

properties=["phonetics","audio64"]
@doc(description='Detection sentence stress or word stress', tags=['stress'])
@use_kwargs(properties_to_args(properties), location=('form'))
@marshal_with(stress_responseSchema, code=200)  # marshalling
@bp.route('/w2v/stress/<module>', methods=['POST'])
def dl_module_api(module):
    d = request.form
    properties=["phonetics","audio64"]
    return request_stress_v2(d, properties, module, mode='base64')

properties=["phonetics","audio64","word_idx","target","syl_idx"]
@doc(description='Vowel contrast', tags=['vowel_contrast'])
@use_kwargs(properties_to_args(properties), location=('form'))
@marshal_with(contrast_responseSchema, code=200)  # marshalling
@bp.route('/w2v/contrast/vowel', methods=['POST'])
def dl_vowel_contrast_api():
    d = request.form
    properties=["phonetics","audio64","word_idx","target","syl_idx"]
    return request_phoneme_contrast(d, properties, mode='base64')


properties=["phonetics","audio64","word_idx","target","syl_idx", "target_occurence_idx"]
@doc(description='Consonant contrast', tags=['consonant_contrast'])
@use_kwargs(properties_to_args(properties), location=('form'))
@marshal_with(contrast_responseSchema, code=200)  # marshalling
@bp.route('/w2v/contrast/consonant', methods=['POST'])
def dl_consonant_contrast_api():
    d = request.form
    properties=["phonetics","audio64","word_idx","target","syl_idx", "target_occurence_idx"]
    target_occurence_idx=int(d['target_occurence_idx'])
    return request_phoneme_contrast(d, properties, target_occurence_idx, alternatives=cmu_consonants, mode='base64')


properties=["phonetics","audio64","word_idx","target"]
@doc(description='Termination contrast', tags=['termination_contrast'])
@use_kwargs(properties_to_args(properties), location=('form'))
@marshal_with(contrast_responseSchema, code=200)  # marshalling
@bp.route('/w2v/contrast/termination', methods=['POST'])
def dl_termination_contrast_api():
    d = request.form
    properties=["phonetics","audio64","word_idx","target"]
    return request_phoneme_contrast(d, properties, tech_function=termination_contrast_from_formatted_phonetics_audio, mode='base64')

properties=["phonetics","audio64","word_idx","syl_idx"]
@doc(description='Syllable contrast', tags=['syllable_contrast'])
@use_kwargs(properties_to_args(properties), location=('form'))
@marshal_with(syl_contrast_responseSchema, code=200)  # marshalling
@bp.route('/w2v/contrast/syllable', methods=['POST'])
def dl_syllable_contrast_api():
    # TODO: refactor with "request_phoneme_contrast" by parametrizing the function call
    d = request.form
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

