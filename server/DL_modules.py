from flask import request

from marshmallow import Schema, fields
from flask_apispec import marshal_with, doc, use_kwargs
from flask import Response

from DL_speech_tech import phonemeContrast_from_formatted_phonetics_audio, stress_from_formatted_phonetics, termination_contrast_from_formatted_phonetics_audio, syllable_contrast_from_formatted_phonetics_audio
import json
from utils.text_processing import check_phonemes, cmu_vowels, cmu_consonants

from app_definition import app
from server.utils import access_property_error, properties_to_args


def check_request(d, properties):
    for prop in properties:
        err=access_property_error(d, prop)
        if err: return err
    split_phonetics=[[s.split('_') for s in w.split('|')] for w in d['phonetics'].split(' ')]
    merged_phonetics=[sum(word,[]) for word in split_phonetics]    
    not_p=check_phonemes(sum(merged_phonetics,[]))
    if not_p is not None: 
        err="error: "+not_p+" is not a phoneme"
        return err


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
@app.route('/w2v/stress/<module>', methods=['POST'])
def dl_module_api(module):
    d = request.form
    properties=["phonetics","rID","text"]

    err=check_request(d, properties)
    if err is not None: return Response(err,status=400,mimetype="application/json")

    if module=='sentence':
        res=stress_from_formatted_phonetics(d['rID'], d['phonetics'], d['text'], level="sentence")
    elif module=='word':
        res=stress_from_formatted_phonetics(d['rID'], d['phonetics'], level="word")
    else:
        res={"status": "error: no such module"}
        return Response(json.dumps(res),status=400,mimetype="application/json")
    response=json.dumps(res)

    if res['status'].split(':')[0]=='error':
        return Response(response,status=500,mimetype="application/json")
    else:
        return Response(response,status=200,mimetype="application/json")


def request_phoneme_contrast(d, properties, target_occurence_idx=0, tech_function=phonemeContrast_from_formatted_phonetics_audio, alternatives=cmu_vowels):
    err=check_request(d, properties)
    if err is not None: return Response(err,status=400,mimetype="application/json")
    word_idx=int(d['word_idx'])
    syl_idx=int(d['syl_idx'])
    if word_idx>=len(d['phonetics'].split(' ')):
        return Response("error: word_idx >= number of words",status=400,mimetype="application/json")
    # extract the syllables which contain the target
    syls_with_target=[syl for syl in d['phonetics'].split(' ')[word_idx].split('|') if d['target'] in syl]
    idx_syls_with_target=[i for i,syl in enumerate(d['phonetics'].split(' ')[word_idx].split('|')) if d['target'] in syl]

    if syl_idx not in idx_syls_with_target:
        return Response("error: the syllable corresponding to syl_idx does not contain the target.",status=400,mimetype="application/json")
    
    res=tech_function(d['rID'], d['phonetics'], target_word_idx=word_idx, target_syllable_idx=syl_idx, target_occurence_idx=target_occurence_idx, target_phones=d['target'], alternatives=alternatives)
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
    name="vowel_contrast_response"
)
properties=["phonetics","rID","word_idx","target","syl_idx"]
@doc(description='Vowel contrast', tags=['vowel_contrast'])
@use_kwargs(properties_to_args(properties), location=('form'))
@marshal_with(responseSchema, code=200)  # marshalling
@app.route('/w2v/contrast/vowel', methods=['POST'])
def dl_vowel_contrast_api():
    d = request.form
    properties=["phonetics","rID","word_idx","target","syl_idx"]
    return request_phoneme_contrast(d, properties)


responseSchema=Schema.from_dict(
    {"status": fields.Str(), 
    "phonetic_detection": fields.Str(),
    "gibberish_truth": fields.Str(),
    "gibberish_detected":fields.Str()},
    name="consonant_contrast_response"
)
properties=["phonetics","rID","word_idx","target","syl_idx", "target_occurence_idx"]
@doc(description='Consonant contrast', tags=['consonant_contrast'])
@use_kwargs(properties_to_args(properties), location=('form'))
@marshal_with(responseSchema, code=200)  # marshalling
@app.route('/w2v/contrast/consonant', methods=['POST'])
def dl_consonant_contrast_api():
    d = request.form
    properties=["phonetics","rID","word_idx","target","syl_idx", "target_occurence_idx"]
    target_occurence_idx=int(d['target_occurence_idx'])
    return request_phoneme_contrast(d, properties, target_occurence_idx, alternatives=cmu_consonants)


responseSchema=Schema.from_dict(
    {"status": fields.Str(), 
    "phonetic_detection": fields.Str(),
    "gibberish_truth": fields.Str(),
    "gibberish_detected":fields.Str()},
    name="termination_contrast_response"
)
properties=["phonetics","rID","word_idx","target","syl_idx", "target_occurence_idx"]
@doc(description='Termination contrast', tags=['termination_contrast'])
@use_kwargs(properties_to_args(properties), location=('form'))
@marshal_with(responseSchema, code=200)  # marshalling
@app.route('/w2v/contrast/termination', methods=['POST'])
def dl_termination_contrast_api():
    d = request.form
    properties=["phonetics","rID","word_idx","target"]
    return request_phoneme_contrast(d, properties, tech_function=termination_contrast_from_formatted_phonetics_audio)



responseSchema=Schema.from_dict(
    {"status": fields.Str(), 
    "gibberish_truth": fields.Str(),
    "gibberish_detected":fields.Str()},
    name="syllable_contrast_response"
)
properties=["phonetics","rID","word_idx","syl_idx"]
@doc(description='Syllable contrast', tags=['syllable_contrast'])
@use_kwargs(properties_to_args(properties), location=('form'))
@marshal_with(responseSchema, code=200)  # marshalling
@app.route('/w2v/contrast/syllable', methods=['POST'])
def dl_syllable_contrast_api():
    # TODO: refactor with "request_phoneme_contrast" by parametrizing the function call
    d = request.form
    properties=["phonetics","rID","word_idx","syl_idx"]
    err=check_request(d, properties)
    if err is not None: return Response(err,status=400,mimetype="application/json")

    word_idx=int(d['word_idx'])
    syl_idx=int(d['syl_idx'])

    if word_idx>=len(d['phonetics'].split(' ')):
        return Response("error: word_idx >= number of words",status=400,mimetype="application/json")    
    res=syllable_contrast_from_formatted_phonetics_audio(d['rID'],phonetics=d['phonetics'], target_word_idx=word_idx, target_syllable_idx=syl_idx)
    response=json.dumps(res)
    if res['status'].split(':')[0]=='error':
        return Response(response,status=500,mimetype="application/json")
    else:
        return Response(response,status=200,mimetype="application/json")

