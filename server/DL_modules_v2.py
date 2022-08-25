from flask import request
from flask_apispec import marshal_with, doc, use_kwargs
from flask import Response, Blueprint
from marshmallow import fields, Schema, EXCLUDE

import json
from DL_speech_tech import phonemeContrast_from_formatted_phonetics_audio, stress_from_formatted_phonetics, start_end_contrast_from_formatted_phonetics_audio, syllable_contrast_from_formatted_phonetics_audio
from utils.text_processing import check_phonemes, chunk_text, split_phonetics
from utils.pronunciation_dictionaries import cmu_vowels, cmu_consonants

# from app_definition import app
from server.utils import default_example, request_phoneme_contrast, request_syl_contrast, request_stress, request_stress_v2, contrast_responseSchema, syl_contrast_responseSchema, sentence_stress_responseSchema_v2, word_stress_responseSchema_v2

bp=Blueprint('DL_modules_v2', __name__, url_prefix='/')

d=default_example()


def kwargs_def(example):
    params={}
    for prop in example:
        type_dict={str : fields.String(required=True, example=example[prop]), int: fields.Integer(required=True, example=example[prop]), list: fields.List(fields.Str(),required=True, example=example[prop])}
        param_type = type_dict[type(example[prop])]
        params[prop]=param_type
    return params

def check_schema(d, params):
    # https://marshmallow.readthedocs.io/en/stable/api_reference.html  -> loof for "from_dict"
    input_schema=Schema.from_dict(params)
    my_input_schema = input_schema(unknown=EXCLUDE)
    errors = my_input_schema.validate(d)
    if errors:
        return str(errors)

# @bp.app_errorhandler(500)
# def internal_error(error):
#     etype, value, tb = sys.exc_info()
#     return Response(json.dumps({"status":"The server encountered an internal error and was unable to complete your request. Either the server is overloaded or there is an error in the application.", "error":True }),status=500,mimetype="application/json")

example_sentence_stress={"phonetics":d["phonetics_chunks"], "audio64": d["audio64"]}
sentence_stress_params=kwargs_def(example_sentence_stress)
@doc(description='Detection of sentence stress', tags=['w2v_v2'])
@use_kwargs(sentence_stress_params, location="json")
@marshal_with(sentence_stress_responseSchema_v2, code=200)  # marshalling
@bp.route('/w2v/stress/sentence', methods=['POST'])
def dl_sentence_stress_api_v2(**kwargs):
    
    # d = request.values.to_dict()
    try:
        d=json.loads(request.get_json())
    except TypeError:
        d=request.get_json()
    global sentence_stress_params
    err=check_schema(d, sentence_stress_params)
    if err is not None: return Response(json.dumps({"status":"wrong payload:"+err, "error":True }),status=400,mimetype="application/json")
    properties=["phonetics","audio64"]
    return request_stress_v2(d, properties, "sentence", mode='base64')

example_word_stress={"phonetics":d["phonetics_chunks"], "audio64": d["audio64"]}
word_stress_params=kwargs_def(example_word_stress)
@doc(description='Detection of word stress', tags=['w2v_v2'])
@use_kwargs(word_stress_params, location="json")
@marshal_with(word_stress_responseSchema_v2, code=200)  # marshalling
@bp.route('/w2v/stress/word', methods=['POST'])
def dl_word_stress_api_v2(**kwargs):
    # d = request.values.to_dict()
    try:
        d=json.loads(request.get_json())
    except TypeError:
        d=request.get_json()
    global word_stress_params
    err=check_schema(d, word_stress_params)
    if err is not None: return Response(json.dumps({"status":"wrong payload:"+err, "error":True }),status=400,mimetype="application/json")
    properties=["phonetics","audio64"]
    return request_stress_v2(d, properties, "word", mode='base64')

example_vowel_contrast={"phonetics":d["phonetics"], "audio64": d["audio64"],"word_idx":d["vowel_w_idx"],"target":d["vowel_target"],"syl_idx":d["vowel_s_idx"]}
vowel_contrast_params=kwargs_def(example_vowel_contrast)
@doc(description='Vowel contrast', tags=['w2v_v2'])
@use_kwargs(vowel_contrast_params, location="json")
@marshal_with(contrast_responseSchema, code=200)  # marshalling
@bp.route('/w2v/contrast/vowel', methods=['POST'])
def dl_vowel_contrast_api_v2(**kwargs):
    # d = request.values.to_dict()
    try:
        d=json.loads(request.get_json())
    except TypeError:
        d=request.get_json()
    global vowel_contrast_params
    err=check_schema(d, vowel_contrast_params)
    if err is not None: return Response(json.dumps({"status":"wrong payload:"+err, "error":True }),status=400,mimetype="application/json")
    properties=["phonetics","audio64","word_idx","target","syl_idx"]
    return request_phoneme_contrast(d, properties, mode='base64')


example_consonant_contrast={"phonetics":d["phonetics"], "audio64": d["audio64"],"word_idx":d["consonant_w_idx"],"target":d["consonant_target"],"syl_idx":d["consonant_s_idx"], "target_occurence_idx":d["consonant_target_occurence_idx"]}
consonant_contrast_params=kwargs_def(example_consonant_contrast)
@doc(description='Consonant contrast', tags=['w2v_v2'])
@use_kwargs(consonant_contrast_params, location="json")
@marshal_with(contrast_responseSchema, code=200)  # marshalling
@bp.route('/w2v/contrast/consonant', methods=['POST'])
def dl_consonant_contrast_api_v2(**kwargs):
    # d = request.values.to_dict()
    try:
        d=json.loads(request.get_json())
    except TypeError:
        d=request.get_json()
    global consonant_contrast_params
    err=check_schema(d, consonant_contrast_params)
    if err is not None: return Response(json.dumps({"status":"wrong payload:"+err, "error":True }),status=400,mimetype="application/json")
    properties=["phonetics","audio64","word_idx","target","syl_idx", "target_occurence_idx"]
    target_occurence_idx=int(d['target_occurence_idx'])
    return request_phoneme_contrast(d, properties, target_occurence_idx, alternatives=cmu_consonants, mode='base64')


example_termination_contrast={"phonetics":d["phonetics"], "audio64": d["audio64"],"target":d["termination_target"],"word_idx":d["termination_w_idx"]}
termination_contrast_params=kwargs_def(example_termination_contrast)
@doc(description='Termination contrast', tags=['w2v_v2'])
@use_kwargs(termination_contrast_params, location="json")
@marshal_with(contrast_responseSchema, code=200)  # marshalling
@bp.route('/w2v/contrast/termination', methods=['POST'])
def dl_termination_contrast_api_v2(**kwargs):
    # d = request.values.to_dict()
    try:
        d=json.loads(request.get_json())
    except TypeError:
        d=request.get_json()
    global termination_contrast_params
    err=check_schema(d, termination_contrast_params)
    if err is not None: return Response(json.dumps({"status":"wrong payload:"+err, "error":True }),status=400,mimetype="application/json")
    properties=["phonetics","audio64","word_idx","target"]
    return request_phoneme_contrast(d, properties, tech_function=start_end_contrast_from_formatted_phonetics_audio, mode='base64')

example_syllable_contrast={"phonetics":d["phonetics"], "audio64": d["audio64"],"word_idx":d["termination_w_idx"],"syl_idx":1}
syllable_contrast_params=kwargs_def(example_syllable_contrast)
@doc(description='Syllable contrast', tags=['w2v_v2'])
@use_kwargs(syllable_contrast_params, location="json")
@marshal_with(syl_contrast_responseSchema, code=200)  # marshalling
@bp.route('/w2v/contrast/syllable', methods=['POST'])
def dl_syllable_contrast_api_v2(**kwargs):
    # d = request.values.to_dict()
    try:
        d=json.loads(request.get_json())
    except TypeError:
        d=request.get_json()
    global syllable_contrast_params
    err=check_schema(d, syllable_contrast_params)
    if err is not None: return Response(json.dumps({"status":"wrong payload:"+err, "error":True }),status=400,mimetype="application/json")
    properties=["phonetics","audio64","word_idx","syl_idx"]
    return request_syl_contrast(d, properties, tech_function=syllable_contrast_from_formatted_phonetics_audio, mode='base64')
