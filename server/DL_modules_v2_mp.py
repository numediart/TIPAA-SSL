from flask import request
# from flask_apispec import marshal_with, doc, use_kwargs
from flask import Response#, Blueprint
from flask_smorest import Blueprint
from flask.views import MethodView
from marshmallow import fields, Schema, EXCLUDE

import json
from DL_speech_tech import phonemeContrast_from_formatted_phonetics_audio, start_end_contrast_from_formatted_phonetics_audio
from src.pronunciation_dictionaries import cmu_vowels, cmu_consonants

# from app_definition import app
from server.utils import kwargs_def, check_schema, default_example, request_contrast, request_stress_v2, contrast_responseSchema, syl_contrast_responseSchema, sentence_stress_responseSchema_v2, word_stress_responseSchema_v2

# bp=Blueprint('DL_modules_v2', __name__, url_prefix='/')

bp = Blueprint("DL_modules_v2_mp", "DL_modules_v2_mp", url_prefix="/", description="DL_modules_v2 in multipart/form-data")


d=default_example()

import sys, traceback

@bp.app_errorhandler(500)
def internal_error(error):
    etype, value, tb = sys.exc_info()
    
    content={
        'type':str(etype),
        'value':str(value),
        'traceback':str(traceback.format_tb(tb))
    }
    return Response(json.dumps({"status":content, "error":True }),status=500,mimetype="application/json")


example_stress_file={"audio":d["audio"], "phonetics":d["phonetics_chunks"]}
stress_params_file=kwargs_def(example_stress_file)

# https://flask-smorest.readthedocs.io/en/latest/arguments.html#file-upload

@bp.route('/w2v/stress/sentence', methods=['POST'])
class dl_sentence_stress_api_v2_mp(MethodView):
    # @bp.arguments(Schema.from_dict(sentence_stress_params_form), location="form")
    @bp.arguments(Schema.from_dict(stress_params_file), location='files', description='Some description')
    @bp.response(200, sentence_stress_responseSchema_v2)
    def post(self, data):
        d = request.values.to_dict()
        d['phonetics']=d['phonetics'].split(',')
        d["audio"]=request.files['audio'].read()
        global stress_params_file
        err=check_schema(d, stress_params_file)
        if err is not None: return Response(json.dumps({"status":"wrong payload:"+err, "error":True }),status=400,mimetype="application/json")
        properties=["phonetics","audio"]
        return request_stress_v2(d, properties, "sentence", mode='bytes')



@bp.route('/w2v/stress/word', methods=['POST'])
class dl_word_stress_api_v2_mp(MethodView):
    @bp.arguments(Schema.from_dict(stress_params_file), location="files")
    @bp.response(200, word_stress_responseSchema_v2)
    def post(self, data):
        d = request.values.to_dict()
        d['phonetics']=d['phonetics'].split(',')
        d["audio"]=request.files['audio'].read()

        global stress_params_file
        err=check_schema(d, stress_params_file)
        if err is not None: return Response(json.dumps({"status":"wrong payload:"+err, "error":True }),status=400,mimetype="application/json")
        properties=["phonetics","audio"]
        return request_stress_v2(d, properties, "word", mode='bytes')

example_vowel_contrast={"phonetics":d["phonetics"], "audio":d["audio"] ,"word_idx":d["vowel_w_idx"],"target":d["vowel_target"],"syl_idx":d["vowel_s_idx"]}
vowel_contrast_params=kwargs_def(example_vowel_contrast)
@bp.route('/w2v/contrast/vowel', methods=['POST'])
class dl_vowel_contrast_api_v2_mp(MethodView):
    @bp.arguments(Schema.from_dict(vowel_contrast_params), location="files")
    @bp.response(200, contrast_responseSchema)
    def post(self, data):
        d = request.values.to_dict()
        d["audio"]=request.files['audio'].read()
            
        global vowel_contrast_params
        err=check_schema(d, vowel_contrast_params)
        if err is not None: return Response(json.dumps({"status":"wrong payload:"+err, "error":True }),status=400,mimetype="application/json")
        properties=["phonetics","audio","word_idx","target","syl_idx"]
        return request_contrast(d, properties, mode='bytes')


example_consonant_contrast={"phonetics":d["phonetics"], "audio": d["audio"],"word_idx":d["consonant_w_idx"],"target":d["consonant_target"],"syl_idx":d["consonant_s_idx"], "target_occurence_idx":d["consonant_target_occurence_idx"]}
consonant_contrast_params=kwargs_def(example_consonant_contrast)
@bp.route('/w2v/contrast/consonant', methods=['POST'])
class dl_consonant_contrast_api_v2_mp(MethodView):
    @bp.arguments(Schema.from_dict(consonant_contrast_params), location="files")
    @bp.response(200, contrast_responseSchema)
    def post(self, data):
        d = request.values.to_dict()
        d["audio"]=request.files['audio'].read()
        global consonant_contrast_params
        err=check_schema(d, consonant_contrast_params)
        if err is not None: return Response(json.dumps({"status":"wrong payload:"+err, "error":True }),status=400,mimetype="application/json")
        properties=["phonetics","audio","word_idx","target","syl_idx", "target_occurence_idx"]
        target_occurence_idx=int(d['target_occurence_idx'])
        return request_contrast(d, properties, target_occurence_idx, alternatives=cmu_consonants, mode='bytes')


example_termination_contrast={"phonetics":d["phonetics"], "audio": d["audio"],"target":d["termination_target"],"word_idx":d["termination_w_idx"]}
termination_contrast_params=kwargs_def(example_termination_contrast)
@bp.route('/w2v/contrast/termination', methods=['POST'])
class dl_termination_contrast_api_v2_mp(MethodView):
    @bp.arguments(Schema.from_dict(termination_contrast_params), location="files")
    @bp.response(200, contrast_responseSchema)
    def post(self, data):
        d = request.values.to_dict()
        d["audio"]=request.files['audio'].read()
        global termination_contrast_params
        err=check_schema(d, termination_contrast_params)
        if err is not None: return Response(json.dumps({"status":"wrong payload:"+err, "error":True }),status=400,mimetype="application/json")
        properties=["phonetics","audio","word_idx","target"]
        return request_contrast(d, properties, tech_function=start_end_contrast_from_formatted_phonetics_audio, mode='bytes', target_type="termination")



# example_cluster_contrast={"phonetics":d["phonetics"], "audio": d["audio"],"target":d["cluster_target"],"basis":d["cluster_basis"],"word_idx":d["cluster_w_idx"],"syl_idx":d["cluster_s_idx"],"position":d["cluster_position"]}
example_cluster_contrast={"phonetics":d["phonetics"], "audio": d["audio"],"target":d["cluster_target"],"word_idx":d["cluster_w_idx"],"syl_idx":d["cluster_s_idx"],"position":d["cluster_position"]}
cluster_contrast_params=kwargs_def(example_cluster_contrast)
@bp.route('/w2v/contrast/cluster', methods=['POST'])
class dl_cluster_contrast_api_v2_mp(MethodView):
    @bp.arguments(Schema.from_dict(cluster_contrast_params), location="files")
    @bp.response(200, contrast_responseSchema)
    def post(self, data):
        d = request.values.to_dict()
        d["audio"]=request.files['audio'].read()
        global cluster_contrast_params
        err=check_schema(d, cluster_contrast_params)
        if err is not None: return Response(json.dumps({"status":"wrong payload:"+err, "error":True }),status=400,mimetype="application/json")
        # properties=["phonetics","audio","target","basis","word_idx","syl_idx","position"]
        properties=["phonetics","audio","target","word_idx","syl_idx","position"]
        return request_contrast(d, properties, tech_function=start_end_contrast_from_formatted_phonetics_audio, mode='bytes', target_type="cluster")
