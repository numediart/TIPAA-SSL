from flask import request
from flask_apispec import marshal_with, doc, use_kwargs
from flask import Response, Blueprint

import ast
import json
from DL_speech_tech import phonemeContrast_from_formatted_phonetics_audio, stress_from_formatted_phonetics, start_end_contrast_from_formatted_phonetics_audio, syllable_contrast_from_formatted_phonetics_audio
from utils.text_processing import check_phonemes, cmu_vowels, cmu_consonants, chunk_text, split_phonetics

# from app_definition import app
from server.utils import default_example, access_property_error, properties_to_args, check_request, request_phoneme_contrast, request_syl_contrast, request_stress, request_stress_v2, stress_responseSchema, contrast_responseSchema, syl_contrast_responseSchema, sentence_stress_responseSchema_v2, word_stress_responseSchema_v2


bp=Blueprint('DL_modules_v2', __name__, url_prefix='/')

d=default_example()



# documenting params: https://github.com/jmcarp/flask-apispec/issues/137
def params_def(example):
    params={}
    for prop in example:
        type_dict={str : 'string', int: 'integer'}
        param_type = type_dict[type(example[prop])]
        record={
                'in': 'query',
                'description': prop,
                'required': True,
                'type': param_type,
            }
        # if example is not None:
        record['example']= example[prop]
        params[prop]=record
    return params

properties=["phonetics","audio64"]
example={"phonetics":d["phonetics_chunks"], "audio64": d["audio64"]}
@doc(description='Detection of sentence stress', tags=['w2v_v2'], params=params_def(example))
@marshal_with(sentence_stress_responseSchema_v2, code=200)  # marshalling
@bp.route('/w2v/stress/sentence', methods=['POST'])
def dl_sentence_stress_api_v2(**kwargs):
    # d = request.values.to_dict()
    try:
        d=json.loads(request.get_json())
    except TypeError:
        d=request.get_json()
    properties=["phonetics","audio64"]
    return request_stress_v2(d, properties, "sentence", mode='base64')

properties=["phonetics","audio64"]
example={"phonetics":d["phonetics"], "audio64": d["audio64"]}
@doc(description='Detection of word stress', tags=['w2v_v2'], params=params_def(example))
@marshal_with(word_stress_responseSchema_v2, code=200)  # marshalling
@bp.route('/w2v/stress/word', methods=['POST'])
def dl_word_stress_api_v2(**kwargs):
    # d = request.values.to_dict()
    try:
        d=json.loads(request.get_json())
    except TypeError:
        d=request.get_json()
    properties=["phonetics","audio64"]
    return request_stress_v2(d, properties, "word", mode='base64')

properties=["phonetics","audio64","word_idx","target","syl_idx"]
example={"phonetics":d["phonetics"], "audio64": d["audio64"],"word_idx":d["vowel_w_idx"],"target":d["vowel_target"],"syl_idx":d["vowel_s_idx"]}
@doc(description='Vowel contrast', tags=['w2v_v2'], params=params_def(example))
@marshal_with(contrast_responseSchema, code=200)  # marshalling
@bp.route('/w2v/contrast/vowel', methods=['POST'])
def dl_vowel_contrast_api_v2(**kwargs):
    # d = request.values.to_dict()
    try:
        d=json.loads(request.get_json())
    except TypeError:
        d=request.get_json()
    properties=["phonetics","audio64","word_idx","target","syl_idx"]
    return request_phoneme_contrast(d, properties, mode='base64')


properties=["phonetics","audio64","word_idx","target","syl_idx", "target_occurence_idx"]
example={"phonetics":d["phonetics"], "audio64": d["audio64"],"word_idx":d["consonant_w_idx"],"target":d["consonant_target"],"syl_idx":d["consonant_s_idx"], "target_occurence_idx":d["consonant_target_occurence_idx"]}
@doc(description='Consonant contrast', tags=['w2v_v2'], params=params_def(example))
# @use_kwargs(properties_to_args(properties), location="query")
@marshal_with(contrast_responseSchema, code=200)  # marshalling
@bp.route('/w2v/contrast/consonant', methods=['POST'])
def dl_consonant_contrast_api_v2(**kwargs):
    # d = request.values.to_dict()
    try:
        d=json.loads(request.get_json())
    except TypeError:
        d=request.get_json()
    properties=["phonetics","audio64","word_idx","target","syl_idx", "target_occurence_idx"]
    target_occurence_idx=int(d['target_occurence_idx'])
    return request_phoneme_contrast(d, properties, target_occurence_idx, alternatives=cmu_consonants, mode='base64')


properties=["phonetics","audio64","word_idx","target"]
example={"phonetics":d["phonetics"], "audio64": d["audio64"],"target":d["termination_target"],"word_idx":d["termination_w_idx"]}
@doc(description='Termination contrast', tags=['w2v_v2'], params=params_def(example))
# @use_kwargs(properties_to_args(properties), location="query")
@marshal_with(contrast_responseSchema, code=200)  # marshalling
@bp.route('/w2v/contrast/termination', methods=['POST'])
def dl_termination_contrast_api_v2(**kwargs):
    # d = request.values.to_dict()
    try:
        d=json.loads(request.get_json())
    except TypeError:
        d=request.get_json()
    properties=["phonetics","audio64","word_idx","target"]
    return request_phoneme_contrast(d, properties, tech_function=start_end_contrast_from_formatted_phonetics_audio, mode='base64')

properties=["phonetics","audio64","word_idx","syl_idx"]
example={"phonetics":d["phonetics"], "audio64": d["audio64"],"word_idx":d["termination_w_idx"],"syl_idx":"1"}
@doc(description='Syllable contrast', tags=['w2v_v2'], params=params_def(example))
# @use_kwargs(properties_to_args(properties), location="query")
@marshal_with(syl_contrast_responseSchema, code=200)  # marshalling
@bp.route('/w2v/contrast/syllable', methods=['POST'])
def dl_syllable_contrast_api_v2(**kwargs):
    # d = request.values.to_dict()
    try:
        d=json.loads(request.get_json())
    except TypeError:
        d=request.get_json()
    properties=["phonetics","audio64","word_idx","syl_idx"]
    return request_syl_contrast(d, properties, tech_function=syllable_contrast_from_formatted_phonetics_audio, mode='base64')
