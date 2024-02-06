import ast
import re

import exercise_generation.auxiliary as aux
import exercise_generation.exercises as ex
import streamlit as st
from utils import check_password


def make_exercise():
    """
    pre: make_exercise is the main function for this interface and will be launched everytime the interface is refreshed or one of the parameters is modified.
    post: an exercise that follows the requirements that were selected is created and accessible in a JSON file called data.json. The content of this JSON file is also
        directly printed at the bottom of the interface and can be copy and pasted easily.
    """

    st.title('Exercise Generation Interface')

    if check_password():
        st.markdown(
            """
        This tool allows you to generate exercises based on OpenAI API and obtain an output JSON compatible with the Content Entry Interface from this [list of content tools](https://docs.google.com/spreadsheets/d/1McYu6sPxRJsOcYFdWp-3k9u6GkzgbHedcX2MbIb7F_o/edit?usp=sharing)

        The output can only be a list of exercises of a specific exercise type, but you can choose until what level you want to wrap it ('Exercise','Activity','Activity Set','Module').
        You can then paste the result in the Content Entry Interface at the corresponding level by clicking on the JSON button in the interface.
        """
        )

        format = st.selectbox(
            '',
            [
                'Please Select Level inside Learning Program',
                'Exercise',
                'Activity',
                'Activity Set',
                'Module',
            ],
        )
        # The next item to be selected will only appear once the format has been selected beforehand
        if format != 'Please Select Level inside Learning Program':
            exercise_type = st.selectbox(
                'What type of exercise would you like to generate?',
                (
                    'Options...',
                    'Count Syllables',
                    'Input Stress Pattern',
                    'Match Audio to Stress Pattern',
                    'Match Word to Audio',
                    'Match Word to Stress Pattern',
                    'Pick Stress Pattern',
                    'Pick Stressed Syllable',
                    'Pick Phonetics',
                    'Pick Audio Unlike Others',
                    'SpokenCard',
                    'Pick Stressed Word',
                    'Pick Meaning of Audio',
                    'Match Audio to Meaning',
                    'SpokenSentence',
                ),
            )
            if exercise_type != 'Options...':
                # Some exercise types focus on a certain pronunciation aspect, others don't
                if exercise_type in [
                    'Match Word to Audio',
                    'Pick Phonetics',
                    'Pick Audio Unlike Others',
                    'SpokenCard',
                    'SpokenSentence',
                ]:
                    if exercise_type == 'Match Word to Audio':
                        pronunciation_aspect = st.selectbox(
                            'What pronunciation aspect would you like your exercises to be based on?',
                            ('Options...', 'VC1', 'VC2', 'VC3', 'VC4', '-ED'),
                        )
                    elif exercise_type == 'Pick Phonetics':
                        pronunciation_aspect = st.selectbox(
                            'What pronunciation aspect would you like your exercises to be based on?',
                            ('Options...', 'VC1', 'VC2', 'VC3', 'VC4', '-ED', 'TH', '-S'),
                        )
                    elif exercise_type == 'Pick Audio Unlike Others':
                        pronunciation_aspect = st.selectbox(
                            'What pronunciation aspect would you like your exercises to be based on?',
                            ('Options...', 'VC1', 'VC2', '-ED'),
                        )
                    elif exercise_type == 'SpokenCard':
                        pronunciation_aspect = st.selectbox(
                            'What pronunciation aspect would you like your exercises to be based on?',
                            ('Options...', 'VC1', 'VC2', '-ED', 'WS'),
                        )
                    elif exercise_type == 'SpokenSentence':
                        pronunciation_aspect = st.selectbox(
                            'What pronunciation aspect would you like your exercises to be based on?',
                            ('Options...', 'VC1', 'VC2', '-ED', 'FW', 'WS'),
                        )
                # some exercise types have 'FW' as default pronunciation aspect (cf. Gianie's structure)
                elif exercise_type in ['Pick Stressed Word', 'Pick Meaning of Audio']:
                    pronunciation_aspect = 'FW'
                # some exercise types have 'WS' as default pronunciation aspect (cf. Gianie's structure)
                else:
                    pronunciation_aspect = 'WS'
                if pronunciation_aspect != 'Options...':
                    n_exercises = st.number_input(
                        'How many exercises would you like to generate?',
                        min_value=1,
                        max_value=30,
                        value=10,
                    )
                    if n_exercises != 0:
                        # Match Word to Audio is the only exercise that is not topic-specific, since it focuses on minimal pairs
                        if exercise_type != 'Match Word to Audio':
                            topic = st.text_input(
                                'What topic would you like your exercises to be based on?'
                            )
                        else:
                            topic = 'no topic'
                        if topic != '':
                            final_button = st.button('Generate ' + format)
                            if final_button:
                                if exercise_type in [
                                    'Match Audio to Stress Pattern',
                                    'Match Word to Stress Pattern',
                                ]:
                                    st.write(
                                        'Your exercise might take a while to generate. Please be patient.'
                                    )
                                # With all the collected information, the main function in exercice.py will take over
                                ex.generate_exercise(
                                    exercise_type,
                                    pronunciation_aspect,
                                    n_exercises,
                                    topic,
                                )
                                # the generated content is accessible in the file data.json, but we also need to print the output in the interface directly so that it can be copy and pasted into Gianie's interface.
                                with open('data.json') as file:
                                    x = file.read()
                                liste = ast.literal_eval(x)
                                d_exercise = (
                                    {}
                                )  # creating a dictionary to follow the same structure as JSON Editor (cf. Gianie's work)
                                d_exercise['Pronunciation aspect'] = pronunciation_aspect
                                if (
                                    exercise_type == 'SpokenCard'
                                ):  # SpokenCard and SpokenSentence need to be adapted differently to be compatible with Gianie's interface
                                    d_exercise['Exercise type'] = 'spoken_card'
                                elif exercise_type == 'SpokenSentence':
                                    d_exercise['Exercise type'] = 'spoken_sentence'
                                else:
                                    d_exercise['Exercise type'] = re.sub(
                                        '\s', '_', exercise_type.lower()
                                    )
                                if (
                                    exercise_type == 'SpokenCard'
                                ):  # SpokenCard and SpokenSentence need to be adapted differently to be compatible with Gianie's interface
                                    d_exercise['content_spoken_card'] = liste
                                elif exercise_type == 'SpokenSentence':
                                    d_exercise['content_spoken_sentence'] = liste
                                else:
                                    d_exercise[
                                        'content_'
                                        + (re.sub('\s', '_', exercise_type.lower()))
                                    ] = liste
                                # depending on the format, the embedding into different dictionaries and lists to correspond to Gianie's interface will have a different amount of levels
                                if format == 'Exercise':  # level 1 (exercise)
                                    st.write(d_exercise)
                                    aux.jsonfile([d_exercise])
                                else:
                                    d_activity_title = {
                                        'Activity title': "Unfortunately, no title was generated..."
                                    }  # creating a dictionary to follow the same structure as JSON Editor (cf. Gianie's work)
                                    d_activity_title['Activity title'] = aux.create_title(
                                        d_activity_title, topic=topic
                                    )
                                    if exercise_type in ['SpokenCard', 'SpokenSentence']:
                                        d_activity_title['Activity type'] = "Speaking"
                                        d_activity_title['Speaking activity'] = [
                                            d_exercise
                                        ]
                                    else:
                                        d_activity_title['Activity type'] = "Listening"
                                        d_activity_title['Listening activity'] = [
                                            d_exercise
                                        ]
                                    if format == 'Activity':  # level 2 (activity)
                                        st.write(d_activity_title)
                                        aux.jsonfile([d_activity_title])
                                    else:
                                        d_activity_set = {
                                            'Activity set': "Unfortunately, no title was generated..."
                                        }  # creating a dictionary to follow the same structure as JSON Editor (cf. Gianie's work)
                                        d_activity_set['Activity set'] = aux.create_title(
                                            d_activity_set, topic=topic
                                        )
                                        d_activity_set['Activity'] = [d_activity_title]
                                        if (
                                            format == 'Activity Set'
                                        ):  # level 3 (activity set)
                                            st.write(d_activity_set)
                                            aux.jsonfile([d_activity_set])
                                        else:  # level 4 (module)
                                            d_module = {
                                                'Module title': "Unfortunately, no title was generated..."
                                            }  # creating a dictionary to follow the same structure as JSON Editor (cf. Gianie's work)
                                            d_module['Module title'] = aux.create_title(
                                                d_module, topic=topic
                                            )
                                            d_module['Activity sets'] = [d_activity_set]
                                            st.write(d_module)
                                            aux.jsonfile([d_module])
                                if len(liste) == 0:
                                    st.write(
                                        'Sorry, no exercises were generated... Please try again.'
                                    )
                                # else:
                                #     st.write('Your exercises are also available in the file data.json')


make_exercise()
