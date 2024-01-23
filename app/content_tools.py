from flask.views import MethodView
from flask_smorest import Blueprint
from marshmallow import Schema, fields, validate

from flowspeech.text_processing import prefill_content

bp = Blueprint(
    "content-tools", "content-tools", url_prefix="/content", description="content tools"
)


class PhonetizerQueryArgsSchema(Schema):
    lang = fields.Str(
        required=False,
        description="Language code for additional pronunciation and syllabification dictionaries",
        load_default="en_US",
    )
    mode = fields.Str(
        required=False,
        description="Pronunciation dictionary to use",
        load_default="CMU",
        validate=validate.OneOf(["CMU", "MFA_IPA"]),
    )


class PhonetizerBodySchema(Schema):
    sentences = fields.List(
        fields.Str(required=True, description="Text to phonetize"),
        required=True,
        description="Sentences to process",
    )


class PhonetizerResponseSchema(Schema):
    text = fields.Str(required=True, description="Original text")
    phonetics = fields.Str(required=True, description="Phonetic transcription")
    pronunciation_guide_hr = fields.Str(required=True, description="Pronunciation guide")
    segmented_text = fields.Str(required=True, description="Text segmented by syllables")
    n_syl_mismatches = fields.List(
        fields.Int(), required=False, description="Number of syllable mismatches"
    )
    n_stress_inconsistencies = fields.List(
        fields.Int(), required=True, description="Number stress inconsistencies"
    )
    used_method_for_syl_text = fields.List(
        fields.Str(),
        required=True,
        description="Method used for syllabificiation of each word",
    )
    phonetics_alt = fields.List(
        fields.Str(),
        required=True,
        description="Alternative phonetics for each word (separated by ',')",
    )
    n_alernatives = fields.List(
        fields.Int(),
        required=True,
        description="Number of alternative phonetics for each word",
    )


@bp.route("/phonetizer", methods=["POST"])
class Phonetizer(MethodView):
    @bp.arguments(
        PhonetizerBodySchema,
        location="json",
        example={"sentences": ["I would love to go to Ireland.", "How lovely!"]},
    )
    @bp.arguments(
        PhonetizerQueryArgsSchema,
        location="query",
    )
    @bp.response(200, PhonetizerResponseSchema(many=True))
    def post(self, body, query_args):
        """Obtain phonetization and syllabification of given sentences"""
        df, _ = prefill_content(body["sentences"], **query_args)
        response = df.to_dict(orient="records")
        return response
