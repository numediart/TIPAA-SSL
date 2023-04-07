import json
import pandas as pd
from copy import deepcopy


import sys
sys.path.append('./')
from src.text_processing import group_consecutive_duplicates, prefill_content, remove_special_characters


param_to_name={
    'module_title': 'Module title',
    'activity_set': 'Activity sets',
    'activity_set_title': 'Activity set',
    'activity': 'Activity',
    'activity_title': 'Activity title',
    'activity_type': 'Activity type',
    'pronunciation_aspect': 'Pronunciation aspect',
    'exercise_type': 'Exercise type'
}

def nested_dict_to_df(content):
    d={}
    records=[]
    for module in content:
        d['module_title']=module[param_to_name['module_title']]
        for activity_set in module[param_to_name['activity_set']]:
            d['activity_set_title']=activity_set[param_to_name['activity_set_title']]
            for activity in activity_set[param_to_name['activity']]:
                d['activity_title']=activity[param_to_name['activity_title']]
                d['activity_type']=activity[param_to_name['activity_type']]
                activity.keys()
                k = 'Speaking activity' if 'Speaking activity' in activity.keys() else 'Listening activity'
                for exercise in activity[k]:
                    d['exercise_type']=exercise[param_to_name['exercise_type']]

                    if 'Listening' in k:
                        d['pronunciation_aspect']=exercise[param_to_name['pronunciation_aspect']]

                    content_key=[el for el in exercise.keys() if "content_" in el]
                    assert len(content_key)==1

                    k2=content_key[0]
                    for ex in exercise[k2]:
                        record=deepcopy(d)

                        if 'Speaking' in k:
                            record['pronunciation_aspect']=ex['pronunciation_aspect']
                        record['phrase_data']=ex
                        records.append(record)
    df=pd.DataFrame.from_records(records)
    return df


def df_to_nested_dict(df):
    module_groups=list(df.groupby(['module_title'], sort=False))
    modules=[]
    for m_g, m in module_groups:
        m_d={}
        m_title=m_g
        activity_set_groups=list(m.groupby(['module_title','activity_set_title'], sort=False))
        activity_sets=[]
        for act_set_g, act_set in activity_set_groups:
            act_set_d={}
            act_set_title=act_set_g[-1]
            activity_groups=list(act_set.groupby(['module_title','activity_set_title','activity_title'], sort=False))
            activities=[]
            for act_g, act in activity_groups:
                act_d={}
                act_title=act_g[-1]
                exercise_groups=list(act.groupby(['module_title','activity_set_title','activity_title','exercise_type'], sort=False))
                exercises=[]
                for ex_g, exs  in exercise_groups:
                    d={}

                    if ex_g[-1] in ['spoken_card','spoken_sentence']:
                        # put pronunciation aspect in phrase_data so that it'll be in the exercise unit data
                        exs['phrase_data']=exs.apply(lambda r: {'pronunciation_aspect':r.pronunciation_aspect} | r.phrase_data, axis=1)

                    d['content_'+ex_g[-1]]=exs['phrase_data'].tolist()
                    assert len(exs['exercise_type'].unique())==1
                    d[param_to_name['exercise_type']]=exs['exercise_type'].iloc[0]
                    # assert len(exs['pronunciation_aspect'].unique())==1
                    if ex_g[-1] not in ['spoken_card','spoken_sentence']:
                        d[param_to_name['pronunciation_aspect']]=exs['pronunciation_aspect'].iloc[0]

                    assert len(exs['activity_type'].unique())==1
                    exercises.append(d)
                act_d[param_to_name['activity_type']]=exs['activity_type'].iloc[0]
                act_d[exs['activity_type'].iloc[0].split(' ')[0]+' activity']=exercises
                act_d[param_to_name['activity_title']]=act_title
                activities.append(act_d)
            act_set_d[param_to_name['activity_set_title']]=act_set_title
            act_set_d[param_to_name['activity']]=activities
            activity_sets.append(act_set_d)
        m_d[param_to_name['module_title']]=m_title
        m_d[param_to_name['activity_set']]=activity_sets
        modules.append(m_d)
    return modules


def process_content(df):
    not_content_keys={'stress_category', 'text_to_display', 'pronunciation_aspect'}

    all_sentences=df.phrase_data.apply(lambda r: [r[k] for k in r if k not in not_content_keys]).sum()

    # make a string containing infos of all columns except phrase data for making ids
    L=df.apply(lambda r: ['__'.join([remove_special_characters(el).replace(' ','_') for el in r[:-1]]) for k in r.phrase_data if k not in not_content_keys], axis=1).sum()
    # group them, this extract the consecutive duplicates and get a the number of occurences
    L_c=group_consecutive_duplicates(L)
    # make a dataframe with the two columns
    L_c_df=pd.DataFrame(L_c)
    # make the ids by concatenating an index to all occurences for making unique ids
    ids=L_c_df.apply(lambda r: [el+'_'+str(i) for i,el in enumerate([r[0]]*r[1])] , axis=1).sum()

    all_sentences_df=pd.DataFrame([ids, all_sentences]).T

    all_sentences_df.columns=['id','sentence']
    linguistic_data,df_errors=prefill_content(all_sentences)


    return all_sentences_df, linguistic_data, df_errors

def remove_duplicates(db_content, all_phrases_df):
    all_phrases_df[all_phrases_df.apply(lambda r: r['sentence'] in db_content.sentence.tolist(), axis=1)]

    len(all_phrases_df)
    all_phrases_df_drop=all_phrases_df.iloc[all_phrases_df.sentence.drop_duplicates().index]
    len(all_phrases_df_drop)
    all_phrases_df_drop_db=all_phrases_df_drop[~all_phrases_df_drop.apply(lambda r: r['sentence'] in db_content.sentence.tolist(), axis=1)]
    len(all_phrases_df_drop_db)
    return all_phrases_df_drop_db

def use_tests():

    db_content=pd.read_csv('/mnt/c/Users/noe_t/Downloads/All content minus audio 2023-02-14 - Sheet1.csv')
    db_content.columns=['sentence','id']

    PE_all_phrases=pd.read_csv('/mnt/c/Users/noe_t/Downloads/all_phrases_df (7).csv')

    
    standard_payload={
                        'text_to_display': 'text_to_display',
                        'incorrect1': 'incorrect1',
                        'incorrect2': 'incorrect2',
                        'incorrect': 'incorrect',
                        'question': 'question',
                        'answer': 'answer',
                        'category': 'stress_category',

                        'phrase_audiorecorded': 'phrase_audio_recorded',
                        'phrase_prompt': 'question',
                        'phrase_annotated': 'target_content',
                        'target_content': 'target_content',
                        'text_correct': 'target_content',
                        'text_incorrect1': 'incorrect1',
                        'text_incorrect2': 'incorrect2',
                        'correct': 'target_content',
                        'displayed_text': 'text_to_display',
                        'phrase_correct': 'target_content',
                        'phrase_incorrect': 'incorrect',
                        'word': 'target_content',
                        # 'phrase_annotated': 'target_content',
                        'phrase_answertorecord': 'answer'
                        }

    def replace_keys(dictionary):
        for k in standard_payload:
            dict_keys_orig=dictionary.keys()
            dict_keys_lower=[el.lower() for el in dictionary.keys()]
            new_dict={}
            for k_low, k in zip(dict_keys_lower, dict_keys_orig):
                new_dict[standard_payload[k_low]] = dictionary[k]
        return new_dict

    

    import ast
    df=pd.read_csv("content_tools/GE_S_syllabus_tree_03_09_2023_15_59_59.csv")
    # df=pd.read_csv("scripts/PE_syllabus_tree_and_stress.csv")
    df.phrase_data=df.phrase_data.str.replace('…','...')
    df.phrase_data=df.phrase_data.apply(lambda r: ast.literal_eval(r))

    df.phrase_data=df.phrase_data.apply(lambda r: {k.lower().strip():r[k] for k in r})
    keys=set(df.phrase_data.apply(lambda r: list(r.keys())).sum())

    properties_phrase_data_to_correct={k:k for k in keys}
    properties_phrase_data_to_correct["phrase_annotated"]="target_content"
    df.phrase_data=df.phrase_data.apply(lambda r: {properties_phrase_data_to_correct[k.lower().strip()]:r[k] for k in r})
    # df.phrase_data=df.phrase_data.apply(lambda r: {k.lower().strip():r[k] for k in r if k not in properties_phrase_data_to_correct else properties_phrase_data_to_correct[k.lower().strip()]:r[k] })
    df.phrase_data=df.phrase_data.apply(lambda r:{k.lower().strip():r[k].strip() for k in r if k not in ['pos','phrase_raw']})
    # df['phrase_data']=df['phrase_data'].apply(lambda r: replace_keys(r))

    import re
    # for single words, there shouldn't be asterisks
    df_star=df[df.phrase_data.apply(lambda r: re.match("^\*[a-z]*\*$", r["target_content"]) if "target_content" in r else False).astype(bool)]
    df.loc[df.phrase_data.apply(lambda r: re.match("^\*[a-z]*\*$", r["target_content"]) if "target_content" in r else False).astype(bool), 'phrase_data']=df_star.phrase_data.apply(lambda r: {k:r[k].replace('*','') for k in r})


    df.columns=['id', 'phrase_data', 'module_title', 'activity_set_title', 'activity_title', 'activity_type', 'exercise_type', 'pronunciation_aspect']#, 'stress_category']
    df.activity_type=df.activity_type.apply(lambda r: r.strip())
    df.pronunciation_aspect=df.pronunciation_aspect.apply(lambda r: r.strip())
    df.exercise_type=df.exercise_type.apply(lambda r: r.lower().strip().replace(' ','_'))

    modules=df_to_nested_dict(df)

    with open("scripts/GE_S_data_entry.json", 'w') as f: json.dump(modules, f, indent=2)

    with open("scripts/PE_data_entry_manually_corrected.json", 'r') as f: content=json.load(f)

    df_back=nested_dict_to_df(content)

    for c in df_back.columns:
        if len(df[df[c]!=df_back[c]])>0:
            print(c)
            print(df[df[c]!=df_back[c]])

    c='pronunciation_aspect'
    df[df[c]!=df_back[c]]
    df_back[df[c]!=df_back[c]]



    df[df[c]!=df_back[c]].phrase_data.apply(lambda r: r.keys())
    
    with open("scripts/marie_program_modified.json", 'r') as f: content=json.load(f)

    with open("scripts/marie_program_modified.json", 'r') as f: content=json.load(f)
    df=nested_dict_to_df(content)
    df.to_csv("data/marie_program_syllabus_tree.csv")

    not_content_keys=['stress_category', 'text_to_display']
    set(df.phrase_data.apply(lambda r:list(r.keys())).sum())


    with open("scripts/data_entry_output_ex.json", 'r') as f: content=json.load(f)

    df=nested_dict_to_df(content)
    modules=df_to_nested_dict(df)
    with open("scripts/data_entry_conversion_ex.json", 'w') as f: json.dump(modules, f, indent=2)


    BE=pd.read_csv('data/BE_syllabus_tree_10_26_2022_15_37_04.csv')
    BE['phrase_data']=BE['phrase_data'].apply(lambda r: ast.literal_eval(r))

    set(BE['phrase_data'].apply(lambda r: list(r.keys())).sum())
    # modules==content


    BE['phrase_data']=BE['phrase_data'].apply(lambda r: replace_keys(r))
    BE['module_title']=BE['module']
    BE['activity_set_title']=BE['activity_set']

    BE.loc[BE.exercise_type=='SpokenSentence','exercise_type']="Spoken sentence"
    BE.loc[BE.exercise_type=='SpokenCard','exercise_type']="Spoken card"

    BE.exercise_type=BE.exercise_type.apply(lambda r: '_'.join(r.lower().split(' ')))
    BE.loc[BE.exercise_type=='count_syllable','exercise_type']="count_syllables"
    BE.loc[BE.exercise_type=='pick_stressed_pattern','exercise_type']="pick_stress_pattern"

    select_wrong_ex_type=BE.loc[BE.exercise_type=='spoken_sentence','phrase_data'].apply(lambda r: list(r.keys())!=['question','answer'])

    BE.loc[BE.exercise_type=='spoken_sentence'].loc[select_wrong_ex_type]

    modules=df_to_nested_dict(BE)

    with open("scripts/PE_data_entry.json", 'w') as f: json.dump(modules, f, indent=2)



    BE['phrase_data'].iloc[0]
    import ast

    listening_types=[
                "Count syllable",
                "Input stress pattern",
                "Match audio to meaning",
                "Match word to audio",
                "Match word to stress pattern",
                "Pick audio unlike others",
                "Pick meaning of audio",
                "Pick phonetics",
                "Pick stressed word",
                "Pick stressed pattern",
                "Pick stressed syllable"
                ]

    speaking_types=[
                "Spoken card",
                "Spoken sentence"
                ]

    exercise_type_to_payload={
        "count_syllables": ["target_content"],
        "input_stress_pattern": ["target_content"],
        "pick_phonetics": ["target_content"],
        "pick_stress_pattern": ["target_content"],
        "pick_stressed_syllabe": ["target_content"],
        "pick_stressed_word": ["target_content"],

        "match_audio_to_meaning": ["text_to_display", "target_content", "incorrect"],
        "match_word_to_audio": ["target_content", "incorrect1"],  #, (incorrect2)
        "match_word_to_stress_pattern": ["target_content", "incorrect1", "incorrect2"],
        "pick_audio_unlike_others": ["target_content", "incorrect1", "incorrect2"],
        "pick_meaning_of_audio": ["phrase_audio_recorded", "target_content", "incorrect1", "incorrect2"],

        "spoken_card":[ "target_content"],  #, (stress_category), (pronunciation_aspect)
        "spoken_sentence":[ "question", "answer"]  #, (stress_category), (pronunciation_aspect)
    }