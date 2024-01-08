import os
import openai
import re
import random
import ast
import json  # everything needs to be accessible in a JSON fileimport json # everything needs to be accessible in a JSON file
import cmudict

cmudict_dict = cmudict.dict()  # initializes an access to the CMU dictionary

# cmu_phones_info=cmudict.phones()
# cmu_phones=set([el[0] for el in cmu_phones_info])
# cmu_vowels=set([p[0] for p in cmu_phones_info if p[1][0]=='vowel'])
# cmu_consonants=set([p[0] for p in cmu_phones_info if p[1][0]!='vowel'])

openai.api_key = "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"
import csv
import pandas as pd

# Generate a response based on a prompt using OpenAI
# https://wandb.ai/ivangoncharov/GPT-3%20in%20Python/reports/Use-GPT-3-in-Python-with-the-OpenAI-API-and-W-B-Tables--VmlldzoxOTg4NTMz


"""
Auxiliary functions
"""


def generate_response(
    prompt,
    temperature=0.7,
    max_tokens=1000,
    top_p=1.0,
    frequency_penalty=0.4,
    presence_penalty=0.4,
    model="text-davinci-002",
):
    response = openai.Completion.create(
        model=model,
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
    )
    return response


def remove_numbering(liste):
    newlist = []
    for el in liste:
        newlist.append(
            re.sub("[0-9]+[.)]\s*", "", el.strip())
        )  # removes the numbering in front of the list element
    return newlist


def filter(text):  # filtering to required format
    list = text.split('\n')
    del list[0:2]  # the first 2 elements are "\n"
    list = remove_numbering(list)
    list = [el.lower() for el in list]
    list = [el for el in list if el in cmudict_dict]  # removing elements not in cmudict
    return list


def jsonfile(list):  # putting the returned list into a JSON file
    jsonString = json.dumps(list)
    jsonFile = open("data.json", "w")
    jsonFile.write(jsonString)
    jsonFile.close()


has_a_target_phone = lambda phones, target_phoneme_regexes: sum(
    [
        sum([re.search(reg, phon) != None for phon in phones])
        for reg in target_phoneme_regexes
    ]
)

has_target_text_pattern = lambda text, target_word_regexes: sum(
    [
        re.search(target_word_regex, text) != None
        for target_word_regex in target_word_regexes
    ]
)

has_at_least_2_syls = lambda text, target_word_regexes: syllable_count(text) >= 2


def word_has_a_target_phone(word, target_phoneme_regexes):
    if word in cmudict_dict:
        if len(cmudict_dict[word]) > 0:
            return has_a_target_phone(cmudict_dict[word][0], target_phoneme_regexes)
        else:
            return False
    else:
        return False


def syllable_count(word):
    """
    pre: word is a string that can be found in the keys of the CMU dictionary
    post: returns the number of syllables contained in 'word'
    """
    phword = cmudict_dict[word]
    if len(phword) == 0:
        return 0
    i = 0
    for el in phword[0]:
        for symbol in el:
            if symbol in ['0', '1', '2']:
                i += 1
    return i


def syllable_sort(list_of_words):
    """
    pre: list_of_words is a list of words that can be found in the keys of the CMU dictionary
    post: returns a dictonary where the keys are numbers of syllables and the values the words of the list
          that have that amount of syllables
    Example: {3: ['Insulting'], 4: ['Disrespectful', 'Condescending']}
    """
    d = {}
    for word in list_of_words:
        syll_count = syllable_count(word)
        if syll_count not in d:
            d[syll_count] = [word]
        else:
            d[syll_count].append(word)
    return d


def stressed_syllable(word):
    """
    pre: word is a string that can be found in the keys of the CMU dictionary
    post: returns which syllable is stressed in 'word'.
    """
    word = word.lower()  # needs to be lowercase
    phword = cmudict_dict[word]
    i = 0
    for el in phword[0]:
        if len(el) == 3:  # vowels are always written with 3 symbols
            i += 1
            for symbol in el:
                if symbol == '1':
                    return i


def stress_pattern(word):
    """
    pre: word is a string that can be found in the keys of the CMU dictionary
    post: returns a dictionary with two keys: n_syll which gives the number of syllables of the word, and stress_index which gives the index of the stressed
        syllable.
    Example: {"n_syll":5,"stress_index":3}
    """
    d = {"n_syll": syllable_count(word), "stress_index": stressed_syllable(word)}
    return d


def stress_pattern_sort(list_of_words):
    """
    pre: list_of_words is a list of words that can be found in the keys of the CMU dictionary
    post: returns a dictonary where the keys are stress patterns and the values are the words of the list
          that have that specific stress pattern
    Example: {{"n_syll":1,"stress_index":1}: ['Big', 'Gross'], {"n_syll":2,"stress_index":1}: ['Purple', 'Slimy', 'Bumpy', 'Stinky', 'Healthy'], {"n_syll":3,"stress_index":2}: ['Delicious', 'Nutritious']}
    """
    d = {}
    for word in list_of_words:
        pattern = stress_pattern(word)
        if str(pattern) not in d:
            d[str(pattern)] = [word]
        else:
            d[str(pattern)].append(word)
    return d


def sort_ed(list_of_words):
    """
    pre: list_of_words is a list of words, most of which normally having a final -ed.
    post: returns a dictionary where the keys are the phonetic pattern of the final -ed (either 'T','D' or 'AH0D' for /t/,/d/
    and /əd/ or /ɪd/ respectively) and the values are the elements from list_of_words that fit into one of these categories
    Example: {'T': ['developed', 'introduced', 'produced', 'replaced', 'increased', 'decreased', 'searched', 'touched', 'introduced'], 'D': ['installed', 'repaired', 'maintained', 'recognized', 'determined', 'achieved', 'verified', 'justified', 'classified', 'described', 'combined', 'remained', 'identified', 'transformed', 'happened', 'considered', 'required'], 'AH0D': ['expanded', 'contracted', 'invested', 'imported', 'exported', 'conducted', 'objected', 'constructed', 'instructed', 'directed', 'collected', 'operated', 'separated', 'united', 'illustrated', 'investigated', 'presented', 'formulated', 'tested', 'committed', 'affected', 'conducted', 'respected', 'affected', 'created', 'depended', 'visited', 'existed', 'negotiated', 'illustrated', 'invested']}
    """
    d = {'T': [], 'D': [], 'AH0D': []}
    for word in list_of_words:
        if re.search(
            'ed$', word
        ):  # word has to end in -ed (entries like 'heard' are not valid)
            reversed_word = list(reversed(cmudict_dict[word][0]))
            if reversed_word[0] == 'T':  # the final -ed is pronounced /t/
                d['T'].append(word)
            elif reversed_word[0] == 'D':
                if reversed_word[1] == 'AH0' or reversed_word[1] == 'IH0':
                    d['AH0D'].append(word)  # the final -ed is pronounced /əd/ or /ɪd/
                else:
                    d['D'].append(word)  # the final -ed is pronounced /d/
    return d


def filter_index_nans(string):
    """
    pre: string is a string that comes from an Panda Dataframe extraction and that needs to be cleaned from all superfluous info
    post returns a string that is clean data that can be used as such in prompts
    """
    string = re.sub('\d+\s+|NaN\n*', '', string)
    return string


def create_syllabus_tree(topic="biology"):
    # create titles for 10 modules, each 4 activity sets, each 4 activities

    df = pd.read_csv("content_tools/content_examples_csvs/titles_theme_based.csv").iloc[
        :, :-3
    ]  # drop last columns

    # drop tutorials
    df = df[~df["Type of activities"].isin(["Listening", "Speaking"])]

    print("Genetating module names...")
    data_module = '\n'.join(df["Module (week)"].dropna().unique().tolist())
    prompt = (
        """Here are example of titles for exercise modules from a learning program for English relating to the topic of "Legal profession"
    """
        + data_module
        + "\nGenerate a numbered list of 10 exercise module titles of the same kind, for a learning program for English relating to the topic of "
        + topic
    )  # fillers[pronunciation_aspect]
    response = generate_response(prompt)
    text = response.choices[0]['text']
    liste = text.split("\n")
    liste = remove_numbering(liste)
    module_list = [x for x in liste if x != '']

    print("Genetating activity set names for modules...")
    module_to_activity_set = {
        module: [] for module in df["Module (week)"].dropna().unique().tolist()
    }
    activity_sets = df["Activity set"].dropna().unique().tolist()
    for activity_set in activity_sets:
        module = df[df["Activity set"] == activity_set]["Module (week)"].values[0]
        module_to_activity_set[module].append(activity_set)

    module_to_activity_set_str = ""
    for key, values in module_to_activity_set.items():
        module_to_activity_set_str += key + ": " + ", ".join(values) + "\n"

    activity_set_to_activities = {}
    for module in module_to_activity_set:
        for act_set in module_to_activity_set[module]:
            activity_set_to_activities[module + ", " + act_set] = (
                df[df["Activity set"] == act_set]["Activity"].dropna().tolist()
            )

    activity_set_to_activities_str = ""
    for key, values in activity_set_to_activities.items():
        activity_set_to_activities_str += key + ": " + ", ".join(values) + "\n"

    activity_set_to_activities_list = [
        key + ": " + ", ".join(values)
        for key, values in activity_set_to_activities.items()
    ]
    random.Random(0).shuffle(activity_set_to_activities_list)
    activity_set_to_activities_list = activity_set_to_activities_list[:5]
    activity_set_to_activities_str = "\n".join(activity_set_to_activities_list)

    gen_module_to_act_sets_dict = {}
    for module in module_list:
        prompt = (
            """Here are example of titles for exercise modules and associated submodules, from a learning program for English relating to the topic of "Legal profession":
        """
            + module_to_activity_set_str
            + "\nGenerate a numbered list of 4 short submodule titles of the same kind, for a learning program for English relating to the topic of "
            + topic
            + ", within the module \""
            + module
            + '".'
        )
        response = generate_response(prompt)
        text = response.choices[0]['text']
        liste = text.split("\n")
        liste = remove_numbering(liste)
        act_set_list = [x for x in liste if x != '']
        gen_module_to_act_sets_dict[module] = act_set_list

    tree = {}
    for k in gen_module_to_act_sets_dict:
        print(k)
        act_set_to_acts_dict = {}
        for act_set in gen_module_to_act_sets_dict[k]:
            theme = k + ", " + act_set
            print(theme)
            prompt = (
                """Here are example of titles for exercise modules (before ":") and associated submodules (after ":"), from a learning program for English relating to the topic of "Legal profession":
            """
                + activity_set_to_activities_str
                + "\nGenerate a numbered list of 4 short submodule titles (after \":\") of the same kind, for a learning program for English relating to the topic of "
                + topic
                + ", within the module \""
                + theme
                + '".'
            )
            response = generate_response(prompt)
            text = response.choices[0]['text']
            liste = text.split("\n")
            liste = remove_numbering(liste)
            act_list = [x for x in liste if x != '']
            act_set_to_acts_dict[act_set] = act_list
            tree[k] = act_set_to_acts_dict

    with open("scripts/tree_" + topic + ".json", 'w') as f:
        json.dump(tree, f, indent=2)
    return tree


def create_title(d, topic="biology"):
    """
    pre:
    post:
    """
    # Creating a Panda Dataframe to retrieve data in CSV file
    # df = pd.read_csv("content_tools/content_examples_csvs/titles.csv")
    df = pd.read_csv("content_tools/content_examples_csvs/titles_theme_based.csv")

    # '\n'.join(df["Activity set"].dropna().unique()[:5].tolist())

    # creating a dictionary with the related prompt complement depending on the pronunciation aspect
    # fillers = {'FW':'focus words.','WS':'word stress.','-ED':'words ending in -ed.','VC1':'the vowel contrast between /ɪ/ and /i:/','VC2':'the vowel contrast between /ɔː/ vs. /əʊ/'}
    if list(d.keys())[0] == 'Module title':
        # data_module = filter_index_nans(df['Module'].to_string())
        data_module = '\n'.join(df["Module (week)"].dropna().unique()[:5].tolist())
        prompt = (
            """Here are example of titles for exercise modules from a learning program for English relating to the topic of "Legal profession":\n"""
            + data_module
            + "\nGenerate 5 exercise module titles of the same kind, for a learning program for English relating to the topic of "
            + topic
        )  # fillers[pronunciation_aspect]
    elif list(d.keys())[0] == 'Activity set':
        # data_activity_set = filter_index_nans(df['Activity Set'].to_string())
        data_activity_set = '\n'.join(df["Activity set"].dropna().unique()[:5].tolist())
        prompt = (
            """Here are example of titles for exercise modules from a learning program for English relating to the topic of "Legal profession":\n"""
            + data_activity_set
            + "\nGenerate 5 exercise module titles of the same kind, for a learning program for English relating to the topic of "
            + topic
        )  # fillers[pronunciation_aspect]
    elif list(d.keys())[0] == 'Activity title':
        # data_activity = filter_index_nans(df['Activity'].to_string())
        data_activity = '\n'.join(df["Activity"].dropna().unique()[:5].tolist())
        prompt = (
            """Here are example of titles for exercise modules from a learning program for English relating to the topic of "Legal profession":\n"""
            + data_activity
            + "\nGenerate 5 exercise module titles of the same kind, for a learning program for English relating to the topic of "
            + topic
        )  # fillers[pronunciation_aspect]
    response = generate_response(prompt)
    text = response.choices[0]['text']
    liste = text.split("\n")
    liste = remove_numbering(liste)
    filtered_list = [x for x in liste if x != '']
    final_title = random.choice(filtered_list)
    return final_title
