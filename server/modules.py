from flask import request

from marshmallow import Schema, fields
from flask_apispec import marshal_with, doc, use_kwargs
from flask import Response, Blueprint

from speech_tech import stress_from_formatted_phonetics, vowel_stresses_from_phonetics_audio, phonemeContrast_from_formatted_phonetics_audio, merge_list
import json
from utils.text_processing import check_phonemes, cmu_to_gibberish
import ast
from server.utils import debug_only

# from app_definition import app
from server.utils import debug_only, access_property_error, properties_to_args

bp=Blueprint('modules', __name__, url_prefix='/')


# set alternatives from targets
def set_alternatives_from_target(target):
    ED_targets=['_T','_D','_IH0_D']

    if False:
        target_to_alternatives={
            'IH':['IH','IY','AA','AO','AW','AY','ER','OY'], # from   https://docs.google.com/spreadsheets/d/1tzb7ZKQOifCHXh-Aoz4PdAKPlIquThzk80EvXW1UxLw/edit#gid=0
            'IY':['IY','IH','AA','AE','AH','AO','AW','AY','EH','ER','OW','OY','UH'],
            'AO':['AO','OW','AW','EH','ER','EY','IH','IY','OY','UH','UW'],
            'AA':['AA','OW','AW','EH','ER','EY','IH','IY','OY','UH','UW'],
            'OW':['OW','AA','AO','AE','AY','ER','EY','IH','IY','OY','UH'],
            "IH0_D":['T', 'D', 'IH0_D'],
            "D":['T', 'D', 'IH0_D'],
            "T":['T', 'D', 'IH0_D']
        }

    # New version based on a_test_GE_linguistic_data_content results in test_api
    target_to_alternatives={
        'IH':['IH','IY','AO','AW','AY','ER','OY'],
        'IY':['IY','IH','AA','AE','AH','AO','AW','AY','EH','ER','OW','OY','UH'],
        'AO':['AO','AA','OW','AW','ER','IY','OY'],
        'AA':['AA','AO','OW','AW','ER'],#,'IY','OY'
        'OW':['OW','AA','AO'],#,'IY','OY'
        "IH0_D":['T', 'D', 'IH0_D'],
        "D":['T', 'D', 'IH0_D'],
        "T":['T', 'D', 'IH0_D']
    }

    alternatives=float('nan')

    if target[-1] in str([0,1,2]):
        alternatives=' '.join([el+target[-1] for el in target_to_alternatives[target[:-1]]])
    elif '_'+target in ED_targets:
        alternatives=' '.join(target_to_alternatives[target])
    return alternatives


@bp.route('/vowel_stresses', methods=['POST'])
@debug_only
def vowel_stresses_api():
    content = request.form

    properties=["phonetics","rID"]
    for prop in properties:
        err=access_property_error(content, prop)
        if err: return Response(err,status=400,mimetype="application/json")
    
    
    # print(request.__dict__)
    print(content)
    # print(content['phonetics'])
    # print(content['rID'])
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


responseSchema=Schema.from_dict(
    {
        "status": fields.Str(), 
        "stress_intensities":fields.List(fields.Integer), 
        "stress_binaries":fields.List(fields.Integer)
    }, name="stress_response"
)
properties=["phonetics","rID","text"]
@doc(description='Detection sentence stress or word stress', tags=['HMM_v1'])
@use_kwargs(properties_to_args(properties), location=('form'))
@marshal_with(responseSchema, code=200)  # marshalling
@bp.route('/flowspeech/<module>', methods=['POST'])
def module_api(module):
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
@doc(description='Phoneme contrast', tags=['HMM_v1'])
@use_kwargs(properties_to_args(properties), location=('form'))
@marshal_with(responseSchema, code=200)  # marshalling
@bp.route('/phonemeContrast', methods=['POST'])
def phoneme_contrast_api():
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
    merged_phonetics=[merge_list(word) for word in split_phonetics]    
    not_p=check_phonemes(merge_list(merged_phonetics))
    if not_p is not None: 
        err="error: "+not_p+" is not a phoneme"
        return Response(err,status=400,mimetype="application/json")

    word_idx=content['word_idx']
    syl_idx=content['syl_idx']
    rID=content['rID']
    target=content['target']
    # alternatives=content['alternatives']
    

    alternatives=set_alternatives_from_target(target)
    print(alternatives)
    word_idx=int(word_idx)
    syl_idx=int(syl_idx)

    # split_phonetics=[[s.split('_') for s in w.split('|')] for w in phonetics.split(' ')]

    if word_idx>=len(phonetics.split(' ')):
        return Response("error: word_idx >= number of words",status=400,mimetype="application/json")
    
    # extract the syllables which contain the target
    syls_with_target=[syl for syl in phonetics.split(' ')[word_idx].split('|') if target in syl]
    idx_syls_with_target=[i for i,syl in enumerate(phonetics.split(' ')[word_idx].split('|')) if target in syl]

    if syl_idx not in idx_syls_with_target:
        return Response("error: the syllable corresponding to syl_idx does not contain the target.",status=400,mimetype="application/json")
    
    target_idx=idx_syls_with_target.index(syl_idx)
    
    status,result=phonemeContrast_from_formatted_phonetics_audio(rID, phonetics, word_idx, target, alternatives)
    # print('status:',status)
    # print('result:',result)
    if not isinstance(result, list):
        result=result.tolist()
        print('result:',result)
    if result!=[]:  
        # phonetic_transcript=result[-1]
        phonetic_detection=result[0][result[0].iloc[:,2].str.contains('_')].detected_transcription.tolist()
    else:
        # phonetic_detection=result
        return Response(status,status=200,mimetype="application/json")

    if phonetic_detection==[]: return Response("success: phonetic detection is empty",status=200,mimetype="application/json")
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
    
    d={'status':status, 'phonetic_detection':phonetic_detection[target_idx], 'gibberish_truth':gs_t[target_idx], 'gibberish_detected':gs_d[target_idx]}
    response=json.dumps(d)
    return response



responseSchema=Schema.from_dict(
    {"status": fields.Str(), 
    "phonetic_detection": fields.Str(), 
    "gibberish_truth": fields.Str(),
    "gibberish_detected":fields.Str()},
    name="phonemeContrast_response"
)
properties=["phonetics","rID","word_idx","target","syl_idx","alternatives"]
@doc(description='Phoneme contrast', tags=['HMM_v1'])
@use_kwargs(properties_to_args(properties), location=('form'))
@marshal_with(responseSchema, code=200)  # marshalling
@bp.route('/terminationContrast', methods=['POST'])
def termination_contrast_api():
    final_phoneme=True

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
    merged_phonetics=[merge_list(word) for word in split_phonetics]    
    not_p=check_phonemes(merge_list(merged_phonetics))
    if not_p is not None: 
        err="error: "+not_p+" is not a phoneme"
        return Response(err,status=400,mimetype="application/json")

    word_idx=content['word_idx']
    syl_idx=content['syl_idx']
    rID=content['rID']
    target=content['target']
    # alternatives=content['alternatives']
    
    alternatives=set_alternatives_from_target(target)
    print(alternatives)
    word_idx=int(word_idx)
    syl_idx=int(syl_idx)

    # split_phonetics=[[s.split('_') for s in w.split('|')] for w in phonetics.split(' ')]

    if word_idx>=len(phonetics.split(' ')):
        return Response("error: word_idx >= number of words",status=400,mimetype="application/json")
    
    # extract the syllables which contain the target
    syls_with_target=[syl for syl in phonetics.split(' ')[word_idx].split('|') if target in syl]
    idx_syls_with_target=[i for i,syl in enumerate(phonetics.split(' ')[word_idx].split('|')) if target in syl]

    if syl_idx not in idx_syls_with_target:
        return Response("error: the syllable corresponding to syl_idx does not contain the target.",status=400,mimetype="application/json")
    target_idx=idx_syls_with_target.index(syl_idx)
    if final_phoneme:
        # to discriminate between with/without termination, 
        # we put the pretermination in the target and put it at the start of each alternative + add it alone
        target_syl=[syl for syl in phonetics.split(' ')[word_idx].split('|')][syl_idx]
        
        # TODO: I take only the first occurence of target with this. So it assumes there is only one, which is not general
        # the problem is that I will have a different preterminantion for each target then, and therefore different resulting targets, which is not usable with "phonemeContrast_from_formatted_phonetics_audio"

        # maybe I should use the assumptions that there is only one, but it is the last one (as we are in final_phoneme)

        # idx_target_in_syl=target_syl.replace(target,'TAR').split('_').index('TAR')
        idxs=[i for i,el in enumerate(target_syl.replace(target,'TAR').split('_')) if el=='TAR']
        idx_target_in_syl=idxs[-1]
        if idx_target_in_syl==0:
            print("idx=0")
            # return Response("error: the target cannot be the first phoneme",status=400,mimetype="application/json")

            # if the target is the first phoneme, then there is no pretermination, let it be ''
            pretermination=''
        else:
            pretermination=target_syl.split('_')[idx_target_in_syl-1]

            alternatives=' '.join([pretermination+'_'+el for el in alternatives.split(' ')]+[pretermination])
            target=pretermination+'_'+target

    status,result=phonemeContrast_from_formatted_phonetics_audio(rID, phonetics, word_idx, target, alternatives)
    # print('status:',status)
    # print('result:',result)
    if not isinstance(result, list):
        result=result.tolist()
        print('result:',result)
    if result!=[]:  
        # phonetic_transcript=result[-1]
        phonetic_detection=result[0][result[0].iloc[:,2].str.contains('_')].detected_transcription.tolist()
    else:
        # phonetic_detection=result
        return Response(status,status=200,mimetype="application/json")

    if phonetic_detection==[]: return Response("success: phonetic detection is empty",status=200,mimetype="application/json")
    
    if final_phoneme:
        phonetic_detection=[el.replace(pretermination+'_','') if el!=pretermination else 'not pronounced' for el in phonetic_detection]
        # remove the pretermination that was added at the start of target
        if pretermination!='':
            target='_'.join(target.split('_')[1:])

    syls_detection=[]
    for i,syl in enumerate(syls_with_target):
        try:
            phonetic_detection[i]
        except:
            return Response("error: index out of bounds in phonetic detection",status=500,mimetype="application/json")
        if phonetic_detection[i]!='not pronounced':
            syls_detection.append(syl.replace(target, phonetic_detection[i]))          
        else:
            # if it is "not pronounced", then remove the target, and make sure it is consistent in terms of "_" vy splitting, filtering empty strings (None) and rejoining
            syls_detection.append('_'.join(list(filter(None, syl.replace(target, '').split('_')))))        

    gs_t=[]
    gs_d=[]
    for syl_t,syl_d in zip(syls_with_target,syls_detection):
        g_t=[cmu_to_gibberish[el] if not el[-1] in str([0,1,2]) else cmu_to_gibberish[el[:-1]] for el in syl_t.split('_')]
        g_d=[]
        for el in syl_d.split('_'):
            # if el=='not pronounced': g_d.append(el)
            if not el[-1] in str([0,1,2]): g_d.append(cmu_to_gibberish[el])
            else: g_d.append(cmu_to_gibberish[el[:-1]])

        # g_d=[cmu_to_gibberish[el] if not el[-1] in str([0,1,2]) else cmu_to_gibberish[el[:-1]] for el in syl_d.split('_')]
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
        if 'Y' in pretermination and "IH" in phonetic_detection[target_idx]:
            phonetic_detection[target_idx]=target
            gs_d[target_idx]=gs_t[target_idx]
    
    d={'status':status, 'phonetic_detection':phonetic_detection[target_idx], 'gibberish_truth':gs_t[target_idx], 'gibberish_detected':gs_d[target_idx]}
    response=json.dumps(d)
    return response

