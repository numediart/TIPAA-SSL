from flask import request

# from flask_apispec import marshal_with, doc, use_kwargs
from flask import Response  # , Blueprint
from flask_smorest import Blueprint
from flask.views import MethodView
from marshmallow import fields, Schema, EXCLUDE

import json
from DL_speech_tech import (
    phonemeContrast_from_formatted_phonetics_audio,
    start_end_contrast_from_formatted_phonetics_audio,
)
from src.pronunciation_dictionaries import cmu_vowels, cmu_consonants

# from app_definition import app
from server.utils import (
    kwargs_def,
    check_schema,
    default_example,
    request_contrast,
    request_stress_v2,
    contrast_responseSchema,
    syl_contrast_responseSchema,
    sentence_stress_responseSchema_v2,
    word_stress_responseSchema_v2,
)

bp = Blueprint(
    "DL_modules_v2", "DL_modules_v2", url_prefix="/", description="DL_modules_v2"
)


d = default_example()


import sys, traceback


@bp.app_errorhandler(500)
def internal_error(error):
    etype, value, tb = sys.exc_info()

    content = {
        'type': str(etype),
        'value': str(value),
        'traceback': str(traceback.format_tb(tb)),
    }
    return Response(
        json.dumps({"status": content, "error": True}),
        status=500,
        mimetype="application/json",
    )


example_sentence_stress = {"phonetics": d["phonetics_chunks"], "audio64": d["audio64"]}
sentence_stress_params = kwargs_def(example_sentence_stress)


# @doc(description='Detection of sentence stress', tags=['w2v_v2'])
# @use_kwargs(sentence_stress_params, location="json")
# @marshal_with(200, sentence_stress_responseSchema_v2)  # marshalling
@bp.route('/w2v/stress/sentence', methods=['POST'])
class dl_sentence_stress_api_v2(MethodView):
    @bp.arguments(Schema.from_dict(sentence_stress_params), location="json")
    @bp.response(200, sentence_stress_responseSchema_v2)
    def post(self, data):
        try:
            d = json.loads(request.get_json())
        except TypeError:
            d = request.get_json()
        global sentence_stress_params
        err = check_schema(d, sentence_stress_params)
        if err is not None:
            return Response(
                json.dumps({"status": "wrong payload:" + err, "error": True}),
                status=400,
                mimetype="application/json",
            )
        properties = ["phonetics", "audio64"]
        return request_stress_v2(d, properties, "sentence", mode='base64')


example_word_stress = {"phonetics": d["phonetics_chunks"], "audio64": d["audio64"]}
word_stress_params = kwargs_def(example_word_stress)


# @doc(description='Detection of word stress', tags=['w2v_v2'])
# @use_kwargs(word_stress_params, location="json")
# @marshal_with(200, word_stress_responseSchema_v2)  # marshalling
@bp.route('/w2v/stress/word', methods=['POST'])
class dl_word_stress_api_v2(MethodView):
    @bp.arguments(Schema.from_dict(word_stress_params), location="json")
    @bp.response(200, word_stress_responseSchema_v2)
    def post(self, data):
        try:
            d = json.loads(request.get_json())
        except TypeError:
            d = request.get_json()
        global word_stress_params
        err = check_schema(d, word_stress_params)
        if err is not None:
            return Response(
                json.dumps({"status": "wrong payload:" + err, "error": True}),
                status=400,
                mimetype="application/json",
            )
        properties = ["phonetics", "audio64"]
        return request_stress_v2(d, properties, "word", mode='base64')


example_vowel_contrast = {
    "phonetics": d["phonetics"],
    "audio64": d["audio64"],
    "word_idx": d["vowel_w_idx"],
    "target": d["vowel_target"],
    "syl_idx": d["vowel_s_idx"],
}
vowel_contrast_params = kwargs_def(example_vowel_contrast)


# @doc(description='Vowel contrast', tags=['w2v_v2'])
# @use_kwargs(vowel_contrast_params, location="json")
# @marshal_with(200, contrast_responseSchema)  # marshalling
@bp.route('/w2v/contrast/vowel', methods=['POST'])
class dl_vowel_contrast_api_v2(MethodView):
    @bp.arguments(Schema.from_dict(vowel_contrast_params), location="json")
    @bp.response(200, contrast_responseSchema)
    def post(self, data):
        try:
            d = json.loads(request.get_json())
        except TypeError:
            d = request.get_json()
        global vowel_contrast_params
        err = check_schema(d, vowel_contrast_params)
        if err is not None:
            return Response(
                json.dumps({"status": "wrong payload:" + err, "error": True}),
                status=400,
                mimetype="application/json",
            )
        properties = ["phonetics", "audio64", "word_idx", "target", "syl_idx"]
        return request_contrast(d, properties, mode='base64')


example_consonant_contrast = {
    "phonetics": d["phonetics"],
    "audio64": d["audio64"],
    "word_idx": d["consonant_w_idx"],
    "target": d["consonant_target"],
    "syl_idx": d["consonant_s_idx"],
    "target_occurence_idx": d["consonant_target_occurence_idx"],
}
consonant_contrast_params = kwargs_def(example_consonant_contrast)


# @doc(description='Consonant contrast', tags=['w2v_v2'])
# @use_kwargs(consonant_contrast_params, location="json")
# @marshal_with(200, contrast_responseSchema)  # marshalling
@bp.route('/w2v/contrast/consonant', methods=['POST'])
class dl_consonant_contrast_api_v2(MethodView):
    @bp.arguments(Schema.from_dict(consonant_contrast_params), location="json")
    @bp.response(200, contrast_responseSchema)
    def post(self, data):
        try:
            d = json.loads(request.get_json())
        except TypeError:
            d = request.get_json()
        global consonant_contrast_params
        err = check_schema(d, consonant_contrast_params)
        if err is not None:
            return Response(
                json.dumps({"status": "wrong payload:" + err, "error": True}),
                status=400,
                mimetype="application/json",
            )
        properties = [
            "phonetics",
            "audio64",
            "word_idx",
            "target",
            "syl_idx",
            "target_occurence_idx",
        ]
        target_occurence_idx = int(d['target_occurence_idx'])
        return request_contrast(
            d,
            properties,
            target_occurence_idx,
            alternatives=cmu_consonants,
            mode='base64',
        )


example_termination_contrast = {
    "phonetics": d["phonetics"],
    "audio64": d["audio64"],
    "target": d["termination_target"],
    "word_idx": d["termination_w_idx"],
}
termination_contrast_params = kwargs_def(example_termination_contrast)


# @doc(description='Termination contrast', tags=['w2v_v2'])
# @use_kwargs(termination_contrast_params, location="json")
# @marshal_with(200, contrast_responseSchema)  # marshalling
@bp.route('/w2v/contrast/termination', methods=['POST'])
class dl_termination_contrast_api_v2(MethodView):
    @bp.arguments(Schema.from_dict(termination_contrast_params), location="json")
    @bp.response(200, contrast_responseSchema)
    def post(self, data):
        try:
            d = json.loads(request.get_json())
        except TypeError:
            d = request.get_json()
        global termination_contrast_params
        err = check_schema(d, termination_contrast_params)
        if err is not None:
            return Response(
                json.dumps({"status": "wrong payload:" + err, "error": True}),
                status=400,
                mimetype="application/json",
            )
        properties = ["phonetics", "audio64", "word_idx", "target"]
        return request_contrast(
            d,
            properties,
            tech_function=start_end_contrast_from_formatted_phonetics_audio,
            mode='base64',
            target_type="termination",
        )


# example_cluster_contrast={"phonetics":d["phonetics"], "audio64": d["audio64"],"target":d["cluster_target"],"basis":d["cluster_basis"],"word_idx":d["cluster_w_idx"],"syl_idx":d["cluster_s_idx"],"position":d["cluster_position"]}
example_cluster_contrast = {
    "phonetics": d["phonetics"],
    "audio64": d["audio64"],
    "target": d["cluster_target"],
    "word_idx": d["cluster_w_idx"],
    "syl_idx": d["cluster_s_idx"],
    "position": d["cluster_position"],
}
cluster_contrast_params = kwargs_def(example_cluster_contrast)


@bp.route('/w2v/contrast/cluster', methods=['POST'])
class dl_cluster_contrast_api_v2(MethodView):
    @bp.arguments(Schema.from_dict(cluster_contrast_params), location="json")
    @bp.response(200, contrast_responseSchema)
    def post(self, data):
        try:
            d = json.loads(request.get_json())
        except TypeError:
            d = request.get_json()
        global cluster_contrast_params
        err = check_schema(d, cluster_contrast_params)
        if err is not None:
            return Response(
                json.dumps({"status": "wrong payload:" + err, "error": True}),
                status=400,
                mimetype="application/json",
            )
        # properties=["phonetics","audio64","target","basis","word_idx","syl_idx","position"]
        properties = ["phonetics", "audio64", "target", "word_idx", "syl_idx", "position"]
        return request_contrast(
            d,
            properties,
            tech_function=start_end_contrast_from_formatted_phonetics_audio,
            mode='base64',
            target_type="cluster",
        )


if False:
    from DL_speech_tech import syllable_contrast_from_formatted_phonetics_audio
    from server.utils import request_syl_contrast

    example_syllable_contrast = {
        "phonetics": d["phonetics"],
        "audio64": d["audio64"],
        "word_idx": d["termination_w_idx"],
        "syl_idx": 1,
    }
    syllable_contrast_params = kwargs_def(example_syllable_contrast)

    @doc(description='Syllable contrast', tags=['w2v_v2'])
    @use_kwargs(syllable_contrast_params, location="json")
    @marshal_with(syl_contrast_responseSchema, code=200)  # marshalling
    @bp.route('/w2v/contrast/syllable', methods=['POST'])
    def dl_syllable_contrast_api_v2(**kwargs):
        # d = request.values.to_dict()
        try:
            d = json.loads(request.get_json())
        except TypeError:
            d = request.get_json()
        global syllable_contrast_params
        err = check_schema(d, syllable_contrast_params)
        if err is not None:
            return Response(
                json.dumps({"status": "wrong payload:" + err, "error": True}),
                status=400,
                mimetype="application/json",
            )
        properties = ["phonetics", "audio64", "word_idx", "syl_idx"]
        return request_syl_contrast(
            d,
            properties,
            tech_function=syllable_contrast_from_formatted_phonetics_audio,
            mode='base64',
        )
