from marshmallow import fields
from functools import wraps
from flask import current_app, abort
from marshmallow import Schema, fields
import json
import ast
from DL_speech_tech import phonemeContrast_from_formatted_phonetics_audio, stress_from_formatted_phonetics
from src.text_processing import check_phonemes, chunk_text, split_phonetics
from src.pronunciation_dictionaries import cmu_vowels, cmu_consonants
from flask import Response
from src.audio_processing import audio64_from_file

from marshmallow import fields, Schema, EXCLUDE


success_messages={
    "success", #--> "speech"
    "success: audio is too short compared to the expected number of syllables", # --> "nospeech"
    "success: no voiced sound detected (no pitch detected)", #--> "nospeech"
    "success: no voiced sound detected inside supposed vowels (no pitch detected)" #--> "nospeech"
    "success: the phrase was not recognized in expected phonemes"  #--> "nonsense"
}


# server_errors={
#     1: "error: audio file not found",
#     2: "error: mode for audio_load_and_check() must be file or base64",
#     3: "error: not a valid level in stress_from_formatted_phonetics. It has to be either 'word' or 'sentence'."
# }

# request_errors={
#     1:"error: could not access a property of the request",
#     2:"error: invalid phoneme in phonetics",
#     3:"error: the syllable corresponding to syl_idx does not contain the target.",
#     4:"error: word_idx >= n of words"
# }
# err="error: target phone "+not_p+" is not a phoneme"
# err="error: target_occurence_idx >= n of ocurrences in syllable"
# err="error: target not in syllable"
# err="error: syl_idx >= n of syllables"
# err="error: word_idx >= n of words"


base_response_dict={   "error":fields.Boolean(),
        "detected": fields.Str(),
        "status": fields.Str() }

stress_response_dict=base_response_dict
stress_response_dict["stress_intensities"]=fields.List(fields.List(fields.Integer))
stress_response_dict["stress_binaries"]=fields.List(fields.List(fields.Integer))

contrast_response_dict=base_response_dict
contrast_response_dict["gibberish_truth"]=fields.Str()
contrast_response_dict["gibberish_detected"]=fields.Str()

phone_contrast_response_dict=contrast_response_dict
contrast_response_dict["phonetic_detection"]=fields.Str()

word_stress_responseSchema_v2=Schema.from_dict(
    stress_response_dict, name="word_stress_response_v2"
)
sentence_stress_responseSchema_v2=Schema.from_dict(
    stress_response_dict, name="sentence_stress_response_v2"
)

contrast_responseSchema=Schema.from_dict(
    phone_contrast_response_dict,
    name="contrast_response"
)

syl_contrast_responseSchema=Schema.from_dict(
    contrast_response_dict,
    name="syllable_contrast_response"
)


def define_detected_flag(status):
    if status=="success": flag = "speech"
    elif "not recognized" in status: flag="nonsense"
    elif "empty" in status: flag="empty_audio"
    else: flag = "nospeech"
    return flag



from io import _io
def kwargs_def(example):
    params={}
    for prop in example:
        type_dict={
                str : fields.Str( metadata= {'example':example[prop]}), 
                int: fields.Int( metadata= {'example':example[prop]}), 
                list: fields.List(fields.Str(), example=example[prop]),
                _io.BytesIO: fields.Raw(metadata={'type':'file', 'required':True}) ,  # https://stackoverflow.com/questions/59642902/how-to-handle-file-upload-validations-using-flask-marshmallow
                _io.BufferedReader: fields.Raw(metadata={'type':'file', 'required':True}) 
            }
        param_type = type_dict[type(example[prop])]
        params[prop]=param_type
    return params

def check_schema(d, params):
    # https://marshmallow.readthedocs.io/en/stable/api_reference.html  -> look for "from_dict"
    input_schema=Schema.from_dict(params)
    my_input_schema = input_schema(unknown=EXCLUDE)
    errors = my_input_schema.validate(d)
    if errors:
        return str(errors)



def default_example():
    path="data/audio_recordings/M1_two-hundred-dollars-way-too-expensive.mp3"
    audio64=audio64_from_file(path)
    text="*Two* hundred *dollars*? That's *way* too expensive!"
    p='T_UW1 HH_AH1_N|D_R_IH0_D D_AA1|L_ER0_Z DH_AE1_T_S W_EY1 T_UW1 IH0_K_S|P_EH1_N|S_IH0_V'
    n_words_by_chunk=chunk_text(text)
    cumsum=0
    chunks_p=[]
    for n in n_words_by_chunk:
        chunks_p.append(' '.join(p.split(' ')[cumsum:cumsum+n]))
        cumsum+=n
    
    d={"phonetics":p,
        # "phonetics_chunks":json.dumps(chunks_p),
        "phonetics_chunks":chunks_p,
        "audio64":audio64.decode('utf-8'),
        "audio":open(path,'rb'),
        "vowel_target":"AA1",
        "vowel_w_idx":2,
        "vowel_s_idx":0,
        "consonant_target":"DH",
        "consonant_w_idx":3,
        "consonant_s_idx":0,
        "cluster_target":"HH",
        "cluster_basis":"HH",
        "cluster_w_idx":1,
        "cluster_s_idx":0,
        "cluster_position":"start",
        "consonant_target_occurence_idx":0,
        "termination_target":"IH0_D",
        "termination_w_idx":1,
        "API_KEY":"[FOWSPEECH_KEY]"
    }
    return d

# make functions available only in debug mode:
# https://stackoverflow.com/questions/55719252/make-a-route-only-accessible-in-debug-mode-with-flask
def debug_only(f):
    @wraps(f)
    def wrapped(**kwargs):
        if not current_app.debug:
            abort(404)
        return f(**kwargs)
    return wrapped

# def properties_to_args(properties, default=None, required=True):
#     args={}
#     for i,prop in enumerate(properties):
#         if default is not None and prop in default: 
#             d=default[prop]
#             args[prop]=fields.Str(required=required,example=d,default=d)
#         else:
#             args[prop]=fields.Str(required=required)
#     return args

def access_property_error(content, property):
    try:
        content[property]
        return 0
    except:
        response='error: could not access "'+property+'" property of the request'
        return response

# TODO: I think I should add "numpy" here, and in the default_example() function to put the waveform signal as numpy array. maybe I should get rid of the rID
audio_property_dict={'file':'rID', 'base64':'audio64', 'bytes':'audio'}

def check_request(d, properties):
    for prop in properties:
        err=access_property_error(d, prop)
        if err: return err

def check_phonetics(phonetics):
    split_phonetics_lists=split_phonetics(phonetics)
    merged_words_phonetics=[sum(word,[]) for word in split_phonetics_lists]    
    not_p=check_phonemes(sum(merged_words_phonetics,[]))
    if not_p is not None: 
        err="error: "+not_p+" is not a phoneme"
        return err


def check_target_access(d, target_occurence_idx=0, target_type='phone'):  # target_type='phone',  'termination' or 'cluster'
    split_phonetics_lists=split_phonetics(d['phonetics'])

    target_phone_list=d['target'].split('_')
    not_p=check_phonemes(target_phone_list)
    if not_p is not None: 
        err="error: target phone "+not_p+" is not a phoneme"
        return err

    if int(d['word_idx'])<len(split_phonetics_lists):
        word_lists=split_phonetics_lists[int(d['word_idx'])]
        if target_type=='termination': d['syl_idx']=-1
        if int(d['syl_idx'])<len(word_lists):
            syl_list=word_lists[int(d['syl_idx'])]
            syl='_'.join(syl_list)

            if d['target'] in syl:
                if target_type=='phone':
                    idxs_of_target_occurences=[i for i,p in enumerate(syl_list) if d['target'] == p]
                    if target_occurence_idx<len(idxs_of_target_occurences): pass
                    else:
                        err="error: target_occurence_idx >= n of ocurrences in syllable"
                        return err
            else:
                err="error: target not in syllable"
                return err
        else:
            err="error: syl_idx >= n of syllables"
            return err
    else:
        err="error: word_idx >= n of words"
        return err

def request_contrast(d, properties, target_occurence_idx=0, tech_function=phonemeContrast_from_formatted_phonetics_audio, alternatives=cmu_vowels, mode='file', target_type="phone"):
    """
    calls corresponding DL_speech_tech function with parameters

    Args:
        d (_type_): _description_
        properties (_type_): _description_
        target_occurence_idx (int, optional): _description_. Defaults to 0.
        tech_function (_type_, optional): _description_. Defaults to phonemeContrast_from_formatted_phonetics_audio.
        alternatives (_type_, optional): _description_. Defaults to cmu_vowels.
        mode (str, optional): from a "file", from "base64" encoding, from "bytes", or directly a "numpy" array . Defaults to 'file'.
        target_type (str, optional): "phone", "termination" or "cluster". Defaults to "phone".

    Returns:
        _type_: _description_
    """
    err=check_request(d, properties)
    if err is not None: return Response(json.dumps({"status":err, "error":True }),status=400,mimetype="application/json")
    err=check_phonetics(d['phonetics'])
    if err is not None: return Response(json.dumps({"status":err, "error":True }),status=400,mimetype="application/json")

    word_idx=int(d['word_idx'])
    # if tech_function==phonemeContrast_from_formatted_phonetics_audio:
    if target_type=="phone":
        syl_idx=int(d['syl_idx'])
        basis=None
        position=None
    elif target_type=="termination":
        syl_idx=None
        basis=None
        position="end"
    elif target_type=="cluster":
        syl_idx=int(d['syl_idx'])
        # basis=d['basis']
        basis=None
        position=d['position']


    err=check_target_access(d, target_occurence_idx=target_occurence_idx, target_type=target_type)
    if err is not None: return Response(json.dumps({"status":err, "error":True }),status=400,mimetype="application/json")

    res=tech_function(d[audio_property_dict[mode]], d['phonetics'], target_word_idx=word_idx, target_syllable_idx=syl_idx, target_occurence_idx=target_occurence_idx, target_phones=d['target'], basis=basis, position=position, alternatives=alternatives, mode=mode)
    
    if res['status'].split(':')[0]=='error':
        res['error']=True
        # res['detected']=define_detected_flag(res['status'])
        response=json.dumps(res)
        return Response(response,status=500,mimetype="application/json")
    else:
        res['error']=False
        res['detected']=define_detected_flag(res['status'])
        response=json.dumps(res)
        return Response(response,status=200,mimetype="application/json")



def group_by_chunk(scores, n_words_by_chunk):
    scores_grouped_by_chunk=[]
    cumsum=0
    for n in n_words_by_chunk:
        scores_grouped_by_chunk.append(scores[cumsum:cumsum+n])
        cumsum+=n
    return scores_grouped_by_chunk

def call_stress_fn(audio, p, module, n_words_by_chunk=[], mode='file', version='v1'):
    if module=='sentence':
        res=stress_from_formatted_phonetics(audio, p, n_words_by_chunk, level=module, mode=mode)
        if version=='v2':
            if res['status']=="success":
                res['stress_intensities']=group_by_chunk(res['stress_intensities'], n_words_by_chunk)
                res['stress_binaries']=group_by_chunk(res['stress_binaries'], n_words_by_chunk)

    elif module=='word':
        res=stress_from_formatted_phonetics(audio, p, level=module, mode=mode)
    else:
        res={"status": "error: no such module"}
        return Response(json.dumps(res),status=500,mimetype="application/json")
    
    if res['status'].split(':')[0]=='error':
        res['error']=True
        # res['detected']=define_detected_flag(res['status'])
        response=json.dumps(res)
        return Response(response,status=500,mimetype="application/json")
    else:
        res['error']=False
        res['detected']=define_detected_flag(res['status'])
        response=json.dumps(res)
        return Response(response,status=200,mimetype="application/json")

# def request_stress(d, properties, module, mode='file'):
#     err=check_request(d, properties)
#     if err is not None: return Response(json.dumps({"status":err, "error":True }),status=400,mimetype="application/json")
#     err=check_phonetics(d['phonetics'])
#     if err is not None: return Response(json.dumps({"status":err, "error":True }),status=400,mimetype="application/json")

#     p=d['phonetics']
#     n_words_by_chunk=chunk_text(d['text'])

#     audio=d[audio_property_dict[mode]]

#     return call_stress_fn(audio, p, module, n_words_by_chunk=n_words_by_chunk, mode=mode)

def request_stress_v2(d, properties, module, mode='file'):
    err=check_request(d, properties)
    if err is not None: return Response(json.dumps({"status":err, "error":True }),status=400,mimetype="application/json")

    p=d['phonetics']
    n_words_by_chunk=[len(c.split(' ')) for c in p]
    p=' '.join(p)
    err=check_phonetics(p)
    if err is not None: return Response(json.dumps({"status":err, "error":True }),status=400,mimetype="application/json")

    audio=d[audio_property_dict[mode]]

    if module=='sentence':
        return call_stress_fn(audio, p, module, n_words_by_chunk=n_words_by_chunk, mode=mode, version='v2')
    else:
        return call_stress_fn(audio, p, module, mode=mode, version='v2')


# to be removed
if False:
    from DL_speech_tech import phonemeContrast_from_formatted_phonetics_audio, stress_from_formatted_phonetics, syllable_contrast_from_formatted_phonetics_audio

    def request_syl_contrast(d, properties, tech_function=syllable_contrast_from_formatted_phonetics_audio, mode='file'):
        err=check_request(d, properties)
        if err is not None: return Response(json.dumps({"status":err, "error":True }),status=400,mimetype="application/json")
        err=check_phonetics(d['phonetics'])
        if err is not None: return Response(json.dumps({"status":err, "error":True }),status=400,mimetype="application/json")

        word_idx=int(d['word_idx'])
        syl_idx=int(d['syl_idx'])

        if word_idx>=len(d['phonetics'].split(' ')):
            res={"status": "error: word_idx >= n of words"}
            res['error']=True
            # res['detected']=define_detected_flag(res['status'])
            response=json.dumps(res)
            return Response(response,status=400,mimetype="application/json")    
        res=tech_function(d['audio64'],phonetics=d['phonetics'], target_word_idx=word_idx, target_syllable_idx=syl_idx, mode=mode)
        
        if res['status'].split(':')[0]=='error':
            res['error']=True
            # res['detected']=define_detected_flag(res['status'])
            response=json.dumps(res)
            return Response(response,status=500,mimetype="application/json")
        else:
            res['error']=False
            res['detected']=define_detected_flag(res['status'])
            response=json.dumps(res)
            return Response(response,status=200,mimetype="application/json")

