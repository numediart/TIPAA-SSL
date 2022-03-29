from flask import request

from marshmallow import Schema, fields
from flask_apispec import marshal_with, doc, use_kwargs
from flask import Response

from DL_speech_tech import phonemeContrast_from_formatted_phonetics_audio, stress_from_formatted_phonetics

import json
from utils.text_processing import check_phonemes, cmu_to_gibberish
import ast
from server.utils import debug_only

from app_definition import app
from server.utils import debug_only, access_property_error, properties_to_args



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
def DL_module_api(module):
    content = request.form
    properties=["phonetics","rID","text"]
    for prop in properties:
        err=access_property_error(content, prop)
        if err: return Response(err,status=400,mimetype="application/json")

    # print(request.__dict__)
    print(content)
    # print(content['phonetics'])
    # print(content['rID'])
    rID=content['rID']
    # phonetics=ast.literal_eval(content['phonetics'])
    phonetics=content['phonetics']
    split_phonetics=[[s.split('_') for s in w.split('|')] for w in phonetics.split(' ')]
    merged_phonetics=[sum(word,[]) for word in split_phonetics]    
    not_p=check_phonemes(sum(merged_phonetics,[]))
    if not_p is not None: 
        err="error: "+not_p+" is not a phoneme"
        return Response(err,status=400,mimetype="application/json")
    if module=='sentence':
        text=content['text']
        res=stress_from_formatted_phonetics(rID, phonetics, text, level="sentence")
    elif module=='word':
        res=stress_from_formatted_phonetics(rID, phonetics, level="word")
    else:
        res={"status": "error: no such module"}
        return Response(json.dumps(res),status=400,mimetype="application/json")
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
@app.route('/w2v/contrast/vowel', methods=['POST'])
def DL_phoneme_contrast_api():
    content = request.form
    # properties=["phonetics","rID","word_idx","target","syl_idx","alternatives"]
    properties=["phonetics","rID","word_idx","target","syl_idx"]
    for prop in properties:
        err=access_property_error(content, prop)
        if err: return Response(err,status=400,mimetype="application/json")
    
    # print(request.__dict__)
    print(content['phonetics'])
    print(content['word_idx'])
    print(content['rID'])
    phonetics=content['phonetics']
    
    split_phonetics=[[s.split('_') for s in w.split('|')] for w in phonetics.split(' ')]
    merged_phonetics=[sum(word,[]) for word in split_phonetics]    
    not_p=check_phonemes(sum(merged_phonetics,[]))
    if not_p is not None: 
        err="error: "+not_p+" is not a phoneme"
        return Response(err,status=400,mimetype="application/json")

    word_idx=content['word_idx']
    syl_idx=content['syl_idx']
    rID=content['rID']
    target=content['target']
    
    word_idx=int(word_idx)
    syl_idx=int(syl_idx)

    if word_idx>=len(phonetics.split(' ')):
        return Response("error: word_idx >= number of words",status=400,mimetype="application/json")
    
    # extract the syllables which contain the target
    syls_with_target=[syl for syl in phonetics.split(' ')[word_idx].split('|') if target in syl]
    idx_syls_with_target=[i for i,syl in enumerate(phonetics.split(' ')[word_idx].split('|')) if target in syl]

    if syl_idx not in idx_syls_with_target:
        return Response("error: the syllable corresponding to syl_idx does not contain the target.",status=400,mimetype="application/json")
    
    target_idx=idx_syls_with_target.index(syl_idx)
    res=phonemeContrast_from_formatted_phonetics_audio(rID, phonetics, target_word_idx=word_idx, target_syllable_idx=syl_idx, target_phones=target)


    if False:
        # print('status:',status)
        # print('result:',result)

        # if not isinstance(result, list):
        #     result=result.tolist()
        #     print('result:',result)
        # if result!=[]:  
        #     # phonetic_transcript=result[-1]
        #     phonetic_detection=result[0][result[0].iloc[:,2].str.contains('_')].detected_transcription.tolist()
        # else:
        #     # phonetic_detection=result
        #     return Response(status,status=200,mimetype="application/json")

        # if phonetic_detection==[]: return Response("success: phonetic detection is empty",status=200,mimetype="application/json")
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
        
        # A posteriori correction for phonemes that are too close to be considered different
        if 'AA' in target or 'AO' in target:
            if 'AA' in phonetic_detection[target_idx] or 'AO' in phonetic_detection[target_idx]:
                phonetic_detection[target_idx]=target
                gs_d[target_idx]=gs_t[target_idx]
        if target=='D' or target=='T':
            if phonetic_detection[target_idx]=='D' or phonetic_detection[target_idx]=='T':
                phonetic_detection[target_idx]=target
                gs_d[target_idx]=gs_t[target_idx]
    
    response=json.dumps(res)

    if res['status'].split(':')[0]=='error':
        return Response(response,status=500,mimetype="application/json")
    else:
        return Response(response,status=200,mimetype="application/json")

