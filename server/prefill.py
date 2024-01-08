# https://stackoverflow.com/questions/11994325/how-to-divide-flask-app-into-multiple-py-files
import os
import json

# from app_definition import app

from flask import send_from_directory, Response, request  # , Blueprint
from flask_smorest import Blueprint
from flask.views import MethodView


from marshmallow import Schema, fields

# from flask_apispec import marshal_with, doc, use_kwargs

from src.text_processing import generate_prefill_csv, prefill_for_sentence, syllables_dfs
from server.utils import kwargs_def, check_schema, debug_only, access_property_error

# from server.upload import upload_path
upload_path = "./upload_files/"

# bp=Blueprint('prefill', __name__, url_prefix='/')

bp = Blueprint("prefill", "prefill", url_prefix="/", description="prefill")


@bp.route('/prefill_from_phrases.html', methods=['GET'])
def prefill_from_phrases_html():
    return send_from_directory('./html/', 'prefill_from_phrases.html')


@bp.route('/prefill_from_phrases', methods=['POST'])
def prefill_from_phrases():
    try:
        uploaded_file = request.files['file']
    except:
        return Response(
            "error: could not access request.files['file']",
            status=400,
            mimetype="application/json",
        )
    if uploaded_file.filename != '':
        try:
            uploaded_file.save(upload_path + uploaded_file.filename)
        except:
            return Response(
                "error: could not save uploaded file",
                status=500,
                mimetype="application/json",
            )

        df = generate_prefill_csv(
            upload_path + uploaded_file.filename
        )  # , out_path=upload_path+'prefill.csv')
        try:
            os.remove(upload_path + uploaded_file.filename)
        except:
            return Response(
                "error: could not remove uploaded file",
                status=500,
                mimetype="application/json",
            )
    else:
        return Response(
            "error: filename is empty", status=400, mimetype="application/json"
        )
    try:
        # from:
        # https://stackoverflow.com/questions/38634862/use-flask-to-convert-a-pandas-dataframe-to-csv-and-serve-a-download
        return Response(
            df.to_csv(),
            mimetype="text/csv",
            headers={"Content-disposition": "attachment; filename=filename.csv"},
        )
    except Exception as e:
        return Response("error: " + str(e), status=500, mimetype="application/json")


example_prefill = {
    "phrase": "I paid a $3000 bill when visiting UCLA, it's an expensive hotel, for the 21st century!",
    "mode": "CMU",
    "lang": "en_US",
}
example_prefill_params = kwargs_def(example_prefill)
record = {
    'text': fields.Str(),
    'phonetics': fields.Str(),
    'pronounciation_guide': fields.Str(),
    'pronounciation_guide_hr': fields.Str(),
    'syllable_parts': fields.Str(),
    'n_syl_mismatch': fields.Integer(),
    'n_syl_mismatches': fields.List(fields.Integer),
    'used_method_for_syl_text': fields.List(fields.Str()),
    'phonetics_alt': fields.List(fields.List(fields.Str())),
    'pronounciation_guide_alt': fields.List(fields.List(fields.Str())),
    'pronounciation_guide_hr_alt': fields.List(fields.List(fields.Str())),
    'n_alternatives': fields.List(fields.Integer),
}
responseSchema = Schema.from_dict(record, name="prefill response")


# @doc(description='Prefill from phrase', tags=['prefill'])
# @use_kwargs(example_prefill_params, location=('json'))
# @marshal_with(200, responseSchema)  # marshalling
@bp.route('/prefill_from_phrase', methods=['POST'])
# @debug_only
class prefill_from_phrase(MethodView):
    @bp.arguments(Schema.from_dict(example_prefill_params), location="json")
    @bp.response(200, responseSchema)
    def post(self, data):
        # content = request.form
        try:
            content = json.loads(request.get_json())
        except TypeError:
            content = request.get_json()
        global example_prefill_params
        err = check_schema(content, example_prefill_params)
        if err:
            return Response(err, status=400, mimetype="application/json")

        # d=prefill_for_sentence(content['phrase'], syllables_df, lang=content['lang'], mode=content['mode'])
        lang = content['lang']
        mode = content['mode']
        d = prefill_for_sentence(
            sentence=content['phrase'],
            syllables_df=syllables_dfs[lang],
            lang=lang,
            mode=mode,
        )  # "CMU" or "MFA_IPA"
        # print(d)
        response = json.dumps(d)
        # print(response)
        return Response(response, status=200, mimetype="application/json")
