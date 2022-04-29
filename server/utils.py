from marshmallow import fields
from functools import wraps
from flask import current_app, abort
from marshmallow import Schema, fields
import json
from DL_speech_tech import phonemeContrast_from_formatted_phonetics_audio, stress_from_formatted_phonetics, termination_contrast_from_formatted_phonetics_audio, syllable_contrast_from_formatted_phonetics_audio
from utils.text_processing import check_phonemes, cmu_vowels, cmu_consonants
from flask import Response

# make functions available only in debug mode:
# https://stackoverflow.com/questions/55719252/make-a-route-only-accessible-in-debug-mode-with-flask
def debug_only(f):
    @wraps(f)
    def wrapped(**kwargs):
        if not current_app.debug:
            abort(404)
        return f(**kwargs)
    return wrapped


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
    



audio_property_dict={'file':'rID', 'base64':'audio64'}

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

def request_phoneme_contrast(d, properties, target_occurence_idx=0, tech_function=phonemeContrast_from_formatted_phonetics_audio, alternatives=cmu_vowels, mode='file'):
    # import pdb;pdb.set_trace()
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
    
    res=tech_function(d[audio_property_dict[mode]], d['phonetics'], target_word_idx=word_idx, target_syllable_idx=syl_idx, target_occurence_idx=target_occurence_idx, target_phones=d['target'], alternatives=alternatives, mode=mode)
    response=json.dumps(res)

    if res['status'].split(':')[0]=='error':
        return Response(response,status=500,mimetype="application/json")
    else:
        return Response(response,status=200,mimetype="application/json")

def request_stress(d, properties, module, mode='file'):
    err=check_request(d, properties)
    if err is not None: return Response(err,status=400,mimetype="application/json")

    if module=='sentence':
        res=stress_from_formatted_phonetics(d[audio_property_dict[mode]], d['phonetics'], d['text'], level="sentence", mode=mode)
    elif module=='word':
        res=stress_from_formatted_phonetics(d[audio_property_dict[mode]], d['phonetics'], level="word", mode=mode)
    else:
        res={"status": "error: no such module"}
        return Response(json.dumps(res),status=400,mimetype="application/json")
    response=json.dumps(res)

    if res['status'].split(':')[0]=='error':
        return Response(response,status=500,mimetype="application/json")
    else:
        return Response(response,status=200,mimetype="application/json")



stress_responseSchema=Schema.from_dict(
    {
        "status": fields.Str(), 
        "stress_intensities":fields.List(fields.Integer), 
        "stress_binaries":fields.List(fields.Integer)
    }, name="stress_response"
)

contrast_responseSchema=Schema.from_dict(
    {"status": fields.Str(), 
    "phonetic_detection": fields.Str(),
    "gibberish_truth": fields.Str(),
    "gibberish_detected":fields.Str()},
    name="contrast_response"
)

syl_contrast_responseSchema=Schema.from_dict(
    {"status": fields.Str(), 
    "gibberish_truth": fields.Str(),
    "gibberish_detected":fields.Str()},
    name="syllable_contrast_response"
)
