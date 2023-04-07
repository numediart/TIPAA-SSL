import pandas as pd
import re
from glob import glob

def remove_special_characters(sentence="Where's the best place to have coffee?", lowercase=True, chars_to_ignore_regex = '[\,\?\.\!\;\:\"\*]'):
    """Normalize text by lowercasing (if option is True), and remove a set of punctuation characters

    Args:
        sentence (str, optional): [description]. Defaults to "Where's the best place to have coffee ?".
        lowercase (bool, optional): [description]. Defaults to True.
        chars_to_ignore_regex (str, optional): [description]. Defaults to '[\,\?\.\!\;\:\"\*]'.

    Returns:
        str: normalized sentence
    """
    # from https://huggingface.co/blog/fine-tune-wav2vec2-english
    sentence = re.sub(chars_to_ignore_regex, '', sentence)
    # I saw this weird quote show up and mess the rest up
    sentence=sentence.replace("’","'")
    if lowercase:
        sentence=sentence.lower()
    # This is to make sure there will not be empty strings after a splitting. So here I split, remove Nones, and rejoin
    sentence=' '.join(list(filter(None, sentence.split(' '))))
    return sentence

def sum_list(l):
    sum_l=[]
    for el in l:
        sum_l+=el
    return sum_l

def lemmatize_from_df(df):
    from nltk.stem import WordNetLemmatizer
    import nltk
    nltk.download('wordnet')
    # To have a list of vocabulary: remove special characters, split with spaces, concatenate results
    w_list=sum_list(df.apply(lambda r: remove_special_characters(r.text, lowercase=False), axis=1).apply(lambda r: r.split(' ')).tolist())
    w_set=set(w_list)

    lemmatizer = WordNetLemmatizer()
    lemmatized_words=[]
    for w in w_set:
        lemmatized_words.append(lemmatizer.lemmatize(w))
    lemmatized_words=sorted(list(set(lemmatized_words)))

    # get rid of isn't, we're, I'll, etc.
    lemmatized_words=[w for w in lemmatized_words if not "'" in w]

    return lemmatized_words


def use_tests():


    # remove duplicates to existing phrases from a new learning program
    df_phrases=pd.read_csv('data/query_results-2022-08-31_101957.csv')
    path='data/SE_content_all_phrases_08_31_2022_11_56_57.csv'
    df_content=pd.read_csv(path)

    df_phrases['n_text']=df_phrases.apply(lambda r: remove_special_characters(r.text, chars_to_ignore_regex = '[\,\?\.\!\;\:\"\']', lowercase=True), axis=1)
    df_content['n_text']=df_content.apply(lambda r: remove_special_characters(r.sentence, chars_to_ignore_regex = '[\,\?\.\!\;\:\"\']', lowercase=True), axis=1)

    df_content[df_content.sentence.str.contains(' ')].sentence.tolist()
    df_content[~df_content.sentence.str.contains(' ')]

    duplicates=df_content[df_content.n_text.isin(set(df_phrases.n_text.unique()))]
    duplicates[duplicates.sentence.str.contains('\*')]
    duplicates[duplicates.sentence.str.contains(' ')]
    duplicates[~duplicates.sentence.str.contains(' ')]

    no_duplicates_path=path.split('.')[0]+'_no_duplicates.csv'

    df_content[~df_content.n_text.isin(set(df_phrases.n_text.unique()))][['id', 'sentence']].to_csv(no_duplicates_path)



    df=pd.read_csv('data/query_results-2022-08-31_101957.csv')
    ws=lemmatize_from_df(df)
    lexique=pd.DataFrame(ws)
    lexique.to_csv("data/lexique.csv")

    df['n_text']=df.apply(lambda r: remove_special_characters(r.text, chars_to_ignore_regex = '[\*]', lowercase=False), axis=1)
    df_audio_names=pd.read_csv('data/tasks_audio_names.csv')

    filipes_ids=[]
    for i,r in df_audio_names.iterrows():
        n_text=remove_special_characters(r.text, chars_to_ignore_regex = '[\*]', lowercase=False)
        print(len(df['n_text'][df['n_text']==n_text]))

        if len(df['n_text'][df['n_text']==n_text])>1:
            import pdb;pdb.set_trace()
        filipes_ids.append(df[df['n_text']==n_text].id.values[0])