import os
from tqdm import tqdm
from glob import glob
import shutil
# from src.libri_phonetization_data import get_all_phone_with_timings
import textgrid
import numpy as np
import librosa
from src.text_processing import unstress
from src.text_processing import remove_stress_annots
import pandas as pd
import csv
import ast
import soundfile as sf
from Bio import pairwise2


def get_all_phone_with_timings(f='data/librispeech_alignments/dev-clean/8842/304647/8842-304647-0013.TextGrid'):
    """get all phonemes of a sentence located in tg[1], and filter silence and empty parts, then convert to DataFrame
    """
    tg = textgrid.TextGrid.fromFile(f)
    # get phones and drop "sp", "sil" and empty strings
    phones=[[el.minTime, el.maxTime, el.mark] for el in tg[1] if el.mark not in ['sil','sp','','spn']]
    phones=pd.DataFrame(phones)
    phones.columns=["start", "end", "phone"]
    return phones





# os.system('mfa model download acoustic english')
# os.system('mfa model download dictionary english')
# os.system('mfa models download acoustic spanish_mfa')
# os.system('mfa models download dictionary spanish_mfa')
# os.system('mfa models download acoustic english_mfa')
# os.system('mfa models download dictionary english_mfa')
# os.system('mfa models download dictionary english_uk_mfa')
# os.system('mfa models download dictionary english_us_mfa')
# os.system('mfa models download acoustic french_mfa')
# os.system('mfa models download dictionary french_mfa')

# mfa models download dictionary english_us_arpa
# mfa models download acoustic english_us_arpa



# modles are stored there:
# ~/Documents/MFA/pretrained_models/

# os.system('mfa model inspect acoustic english')
# os.system('mfa model inspect dictionary english')

def prepare_files(root_path, wav_paths, texts, mfa_path, fs):
    if not os.path.exists(mfa_path): os.makedirs(mfa_path)
    print('Preparing files for alignment')
    for text, wav_path in tqdm(zip(texts,wav_paths)):
        fp='/'.join([root_path,wav_path])
        fn='_'.join(wav_path.split('/'))
        wav_fp=os.path.join(mfa_path, fn).split(".")[0]+'.wav'
        text_fp=os.path.join(mfa_path, fn).split(".")[0]+'.txt'
        s,fs=librosa.load(fp, sr=fs)
        sf.write(wav_fp, s, fs)
        with open(text_fp, 'w') as f: f.write(text)

def launch_mfa( mfa_path='mfa_data', mfa_result_path='mfa_result', dictionary='english_us_arpa', acoustic_model='english_us_arpa'):
    print('Performing alignment')
    cmd='mfa align --clean '+mfa_path+' '+dictionary+' '+acoustic_model+' '+mfa_result_path+ " --speaker_characters 16"
    os.system(cmd)

    
    print('Put results in dataframe')
    textgrids=glob(mfa_result_path+'/*.TextGrid')
    phone_dfs=[]
    failures=[]
    for t in tqdm(textgrids):
        try:
            phone_dfs.append(get_all_phone_with_timings(t))
        except:
            phone_dfs.append(float('nan'))
            failures.append(t)
    
    textgrids_df=pd.DataFrame(pd.DataFrame(textgrids).apply(lambda r: os.path.split(r[0])[-1].split('.')[0], axis=1).tolist())
    textgrids_df.columns=['filename']
    textgrids_df['phone_df']=phone_dfs
    return textgrids_df, failures


def align_common_voice(root='data/cv-corpus-10.0-delta-2022-07-04/', lang='en', split='train', mfa_model='english_mfa', mfa_path='mfa_data', mfa_result_path='mfa_result', fs=16000, out_json='data/align_common_voice_en_train_mfa.json'):
    
    if os.path.exists(mfa_path):
        shutil.rmtree(mfa_path)
    if os.path.exists(mfa_result_path):
        shutil.rmtree(mfa_result_path)
    
    tsv_file_path='/'.join([root,lang,split])+'.tsv'
    df=pd.read_csv(tsv_file_path, sep='\t')

    df.accents.unique()
    df.locale.unique()

    path='/'.join([root,lang,'clips'])
    texts=df.sentence.tolist()
    
    prepare_files(path, wav_paths=df.path.tolist(), texts=texts, mfa_path=mfa_path, fs=fs)
    textgrids_df, failures=launch_mfa(mfa_path, mfa_result_path, dictionary=mfa_model, acoustic_model=mfa_model)

    # df['phone_df']=phone_dfs
    textgrids_df.to_json(out_json)
    
    shutil.rmtree(mfa_path)
    shutil.rmtree(mfa_result_path)

    return df

def align_synth_speech(path='data/synth_audio/speechocean762', mfa_path='mfa_data', mfa_result_path='mfa_result', fs=16000, out_json='data/align_synth_speech.json'):
    df=pd.read_csv(path+'data.csv')
    wav_paths=(df.filename+'.mp3').tolist()
    texts=df.text.tolist()

    prepare_files(path, wav_paths=wav_paths, texts=texts, mfa_path=mfa_path, fs=fs)
    textgrids_df, failures=launch_mfa(mfa_path, mfa_result_path, dictionary='english_us_arpa', acoustic_model='english_us_arpa')

    # df['phone_df']=phone_dfs
    textgrids_df.to_json(out_json)
    
    shutil.rmtree(mfa_path)
    shutil.rmtree(mfa_result_path)

    return df


# def align_synth_words(path='data/synth_audio/speechocean762', mfa_path='mfa_data', mfa_result_path='mfa_result', fs=16000, out_json='data/align_synth_speech.json'):
#     from src.label_data_processing import synth_words_data

#     df=synth_words_data()

#     wav_paths=(df.filename+'.mp3').tolist()
#     texts=df.text.tolist()

#     prepare_files(path, wav_paths=wav_paths, texts=texts, mfa_path=mfa_path, fs=fs)
#     textgrids_df, failures=launch_mfa(mfa_path, mfa_result_path, dictionary='english_us_arpa', acoustic_model='english_us_arpa')

#     # df['phone_df']=phone_dfs
#     textgrids_df.to_json(out_json)
    
#     shutil.rmtree(mfa_path)
#     shutil.rmtree(mfa_result_path)

#     return df

def speech_ocean_data(path='data/speechocean762'):
    """Aggregate score information at sentence level and word level as well as wav path and data set (train or test) in a single dataframe
    for which each row is an example and columns are:
        ['accuracy', 'completeness', 'fluency', 'prosodic', 'total', 'text',
       'words_accuracy', 'words_stress', 'words_phones', 'words_total',
       'words_phones_accuracy', 'wav_path', 'data_set']
       For details on scores meaning: https://github.com/jimbozhang/speechocean762

    Args:
        path (str, optional): [description]. Defaults to 'data/speechocean762'.

    Returns:
        [type]: [description]
    """
    if not os.path.exists(path): 
        print(path+" does not exist. Download at least a subset of the dataset")
        return None
    train_metadata={}
    for p in glob(path+'/train/*'):
        fn=p.split('/')[-1]
        train_metadata[fn]=pd.read_csv(p, header=None, sep="\t")
    
    test_metadata={}
    for p in glob(path+'/test/*'):
        fn=p.split('/')[-1]
        test_metadata[fn]=pd.read_csv(p, header=None, sep="\t")

    glob(path+'/resource/*')
    scores=pd.read_json('data/speechocean762/resource/scores.json').T
    # scores_details=pd.read_json('data/speechocean762/resource/scores-detail.json').T  # this ones has the scores of each expert judging the pronounciation

    # filter out 
    # scores[scores.completeness!=10]
    # scores=scores[scores.completeness==10]

    # There are 5 experts scoring samples
    # scores.iloc[0]
    # scores.iloc[0].words
    # pd.DataFrame.from_records(scores.iloc[0].words)

    scores['words_accuracy']=scores.apply(lambda r: pd.DataFrame.from_records(r.words).accuracy.tolist(), axis=1)
    scores['words_stress']=scores.apply(lambda r: pd.DataFrame.from_records(r.words).stress.tolist(), axis=1)
    scores['words_phones']=scores.apply(lambda r: pd.DataFrame.from_records(r.words).phones.tolist(), axis=1)
    scores['words_total']=scores.apply(lambda r: pd.DataFrame.from_records(r.words).total.tolist(), axis=1)
    # scores['words_text']=scores.apply(lambda r: pd.DataFrame.from_records(r.words).text.tolist(), axis=1)
    scores['words_phones_accuracy']=scores.apply(lambda r: pd.DataFrame.from_records(r.words)['phones-accuracy'].tolist(), axis=1)

    columns=['accuracy', 'completeness', 'fluency', 'prosodic', 'total',
       'text', 'words_accuracy', 'words_stress', 'words_phones', 'words_total',
       'words_phones_accuracy']

    scores=scores[columns]

    train_metadata['wav.scp'].columns=['sentence_id', 'wav_path']
    test_metadata['wav.scp'].columns=['sentence_id', 'wav_path']

    train_metadata['wav.scp']['data_set']=['train']*len(train_metadata['wav.scp'])
    test_metadata['wav.scp']['data_set']=['test']*len(test_metadata['wav.scp'])

    metadata=pd.concat([train_metadata['wav.scp'],test_metadata['wav.scp']], axis=0)
    metadata.index=metadata['sentence_id']

    scores['wav_path']=metadata['wav_path']
    scores['data_set']=metadata['data_set']

    return scores



def align_speech_ocean(path='data/speechocean762', mfa_path='mfa_data', mfa_result_path='mfa_result', fs=16000):
    print('Loading speech ocean metadata')
    df=speech_ocean_data(path=path)
    df['id']=df.apply(lambda r: "_".join(r.wav_path.split("/")).split(".")[0], axis=1)
        
    prepare_files(path, wav_paths=df.wav_path.tolist(), texts=df.text.tolist(), mfa_path=mfa_path, fs=fs)
    _, _=launch_mfa( mfa_path, mfa_result_path, dictionary='english_us_arpa', acoustic_model='english_us_arpa')

    print('Put results in dataframe')
    phone_dfs=[]
    failures=[]
    for i,r in tqdm(df.iterrows()):
        f='mfa_result/'+r.id+'.TextGrid'
        try:
            phone_dfs.append(get_all_phone_with_timings(f))
        except:
            phone_dfs.append(float('nan'))
            failures.append(r.id)
    
    df['phone_df']=phone_dfs
    df.to_json('data/align_speechocean.json')

    shutil.rmtree(mfa_path)
    shutil.rmtree(mfa_result_path)
    return df.dropna()

# https://www.biostars.org/p/246408/
# 
def words_to_unicode_chars(LISTA, LISTB):
    charcode=ord(u"一")-1
    LATtoHAN={}
    HANtoLAT={}
    LISTA_=[]
    LISTB_=[]
    for x in LISTA:
        if x in LATtoHAN.keys():
            LISTA_.append(LATtoHAN[x])
        else:
            charcode+=1
            LATtoHAN[x]=chr(charcode)
            HANtoLAT[chr(charcode)]=x
            LISTA_.append(LATtoHAN[x])
    for x in LISTB:
        if x in LATtoHAN.keys():
            LISTB_.append(LATtoHAN[x])
        else:
            charcode+=1
            LATtoHAN[x]=chr(charcode)
            HANtoLAT[chr(charcode)]=x
            LISTB_.append(LATtoHAN[x])

    LISTA__="".join(LISTA_)
    LISTB__="".join(LISTB_)

    return LISTA__, LISTB__, LATtoHAN, HANtoLAT

def unicode_chars_to_words(seqA, seqB, LATtoHAN, HANtoLAT):

    RESA=[]
    RESB=[]
    for w in seqA:
        if (w == "-"):
            RESA.append("-")
        else:
            RESA.append(HANtoLAT[w])
    for w in seqB:
        if (w == "-"):
            RESB.append("-")
        else:
            RESB.append(HANtoLAT[w])
    return RESA, RESB


def align_user_recordings(path='data/user_recordings_annotated/', fs=16000):
    df=pd.read_csv(path+'/turkle_annotations.csv')

    # df[['Input.phonetics', 'Input.target_phoneme',
    #    'Input.text','Answer.comment', 'Answer.gop',
    #    'Answer.validity', 'Answer.vowel', 'Turkle.Username']]

    df=df[['Input.a1','Input.phonetics', 'Input.target_phoneme', 'Input.text', 'Answer.gop', 'Answer.vowel']].dropna()
    df['gop']=df['Answer.gop'].str[:1].astype(int)
    
    mfa_path='mfa_data'
    if not os.path.exists(mfa_path): os.makedirs(mfa_path)

    prepare_files(path, wav_paths=df['Input.a1'].tolist(), texts=df['Input.text'].tolist(), mfa_path=mfa_path, fs=fs)
    _, _=launch_mfa( mfa_path, mfa_result_path='mfa_result')

    # print('Preparing files for alignment')
    # for i,r in tqdm(df.iterrows()):
    #     fp='/'.join([path,r['Input.a1']])
    #     fn=r['Input.a1'].split('/')[-1]
        
    #     wav_fp=os.path.join(mfa_path, fn).split(".")[0]+'.wav'
    #     text_fp=os.path.join(mfa_path, fn).split(".")[0]+'.txt'

    #     s, fs = sf.read(fp)
    #     sf.write(wav_fp, s, fs)
    #     with open(text_fp, 'w') as f: f.write(r['Input.text'])
    
    # print('Performing alignment')
    # mfa_result_path='mfa_result'
    # cmd='mfa align --clean '+mfa_path+' english_us_arpa english_us_arpa '+mfa_result_path# + " --speaker_characters 16"
    # os.system(cmd)

    print('Put results in dataframe')
    phone_dfs=[]
    failures=[]
    for i,r in tqdm(df.iterrows()):
        id=r['Input.a1'].split('/')[-1].split('.')[0]
        f='mfa_result/'+id+'.TextGrid'
        try:
            phone_dfs.append(get_all_phone_with_timings(f))
        except:
            phone_dfs.append(float('nan'))
            failures.append(id)

    df['phone_df']=phone_dfs

    cols=['audio_path', 'phonetics', 'target_phoneme', 'text', 'Answer.gop', 'vowel', 'gop', 'phone_df']
    df.columns=cols

    # we assume there is only 1 stressed word (I maybe should check that, it should be the case in this content)
    target_word_indexes=df.text.apply(lambda r: [i for i,el in enumerate(r.split(' ')) if '*' in el])
    target_word_indexes=target_word_indexes.apply(lambda r: r[0] if len(r)>0 else 0)

    df['target_word_indexes']=target_word_indexes
    
    target_phone_indexes=[]
    status=[]
    for i,r in df.iterrows():
        if r['audio_path'].split('/')[-1].split('.')[0] not in failures:
            target_word=r.phonetics.split(' ')[r['target_word_indexes']]
            p=r['phonetics'].replace('|',' ').replace('-',' ').replace('_',' ').split(' ')            
            word_p=target_word.replace('|',' ').replace('-',' ').replace('_',' ').split(' ')
            
            # We assume it's the first occurence of the phoneme in the word that is targetted. Again, it should be the case in this content
            # but not sure 100% that it is always the case

            if r.target_phoneme in word_p:
                phoneme_index=word_p.index(r.target_phoneme)
                # count the phonemes before the target word
                if r.target_word_indexes>0:
                    n_phones_before_target=len(' '.join(r['phonetics'].split(' ')[:r['target_word_indexes']]).replace('|',' ').replace('-',' ').replace('_',' ').split(' '))
                else: n_phones_before_target=0
                global_phoneme_index=n_phones_before_target+phoneme_index

                assert p[global_phoneme_index]==r['target_phoneme'], "error: phoneme at global_phoneme_index is not the target_phone"

                phones_mfa=list(r['phone_df'].phone)
                
                # the phonetics come from 2 different sources, they are mostly the same but not exactly
                # because of that, I have to do pairwise alignment to retrieve the index in the mfa phonetics
                LISTA__, LISTB__, LATtoHAN, HANtoLAT = words_to_unicode_chars(p, phones_mfa)
                alignments=pairwise2.align.globalxx(LISTA__,LISTB__)
                RESA, RESB=unicode_chars_to_words(alignments[0].seqA, alignments[0].seqB, LATtoHAN, HANtoLAT)

                print(RESA)
                print(RESB)

                # convert phoneme index from p to phones_mfa by counting the dashes in p before global_phoneme_index, 
                # then substracting the number of dashes in phones_mfa before that
                n=0
                for i,el in enumerate(RESA):
                    if el!='-':
                        if n==global_phoneme_index: global_phoneme_index_dash=i
                        n+=1

                assert RESA[global_phoneme_index_dash]==r['target_phoneme'], "error: phoneme at global_phoneme_index is not the target_phone"



                global_phoneme_index_B=global_phoneme_index_dash - RESB[:global_phoneme_index_dash].count('-')

                if global_phoneme_index_B<len(phones_mfa):  #phonetization or alignment prblem
                    if phones_mfa[global_phoneme_index_B]==r['target_phoneme']:#, "error: phoneme at global_phoneme_index is not the target_phone"
                        target_phone_indexes.append(global_phoneme_index_B)
                        status.append('success')
                    
                    # here I want to handle cases where mha phonemizer does not do the same as our ground truth phonemtics and switch to the mfa version if it's just a stress variation or a AA vs AO
                    elif unstress(phones_mfa[global_phoneme_index_B])==unstress(r['target_phoneme']):
                        # target_phoneme=p[global_phoneme_index]
                        target_phone_indexes.append(global_phoneme_index_B)
                        status.append('success')
                        df.loc[i,'target_phoneme']=phones_mfa[global_phoneme_index_B]
                    elif (unstress(phones_mfa[global_phoneme_index_B]) in ['AA', 'AO']) and (unstress(r['target_phoneme']) in ['AA', 'AO']):
                        target_phone_indexes.append(global_phoneme_index_B)
                        status.append('success')
                        df.loc[i,'target_phoneme']=phones_mfa[global_phoneme_index_B]
                    else:
                        # if both phonemizers don't agree on the target phoneme (e.g. AA1 instead of AO1)
                        
                        # target_phoneme=r['target_phoneme']
                        # if unstress(phones_mfa[global_phoneme_index_B])==unstress(r['target_phoneme']):
                        #     target_phoneme=p[global_phoneme_index]
                        # elif (unstress(p[global_phoneme_index]) in ['AA', 'AO']) and (r['target_phoneme'] in ['AA', 'AO']):
                        #     target_phoneme=p[global_phoneme_index]

                        # import pdb;pdb.set_trace()

                        target_phone_indexes.append(np.nan)
                        status.append('error: phoneme at global_phoneme_index is not the target_phone, it is instead: '+phones_mfa[global_phoneme_index_B])
                else:
                    target_phone_indexes.append(np.nan)
                    status.append('error: phonetization or alignment problem')
            else:
                target_phone_indexes.append(np.nan)
                status.append('error: target phoneme not in word')
        else:
            target_phone_indexes.append(np.nan)
            status.append('error: in previous mfa alignment failure')

    df['target_phone_global_indexes']=target_phone_indexes
    df['status']=status

    errors=list(df[df.status!='success'].status.unique())

    for e in errors:
        print(df[df.status==e])
    
    df[df.status=='error: target phoneme not in word'].text

    df[df.status.str.contains('rror: phoneme at global_ph')].status.str[-3:]
    df[df.status.str.contains('rror: phoneme at global_ph')].target_phoneme
    df[df.status.str.contains('rror: phoneme at global_ph')].text




    target_phone_indexes_failures=df[df.target_phone_global_indexes.isna()]

    ids_failures=df[df.target_phone_global_indexes.isna()].audio_path.str[6:-4]

    for id in ids_failures:
        path_textgrid='mfa_result/'+id+'.TextGrid'
        cmd='cat '+path_textgrid
        os.system(cmd)

    df=df.dropna()

    shutil.rmtree(mfa_path)

    df.to_json('data/align_user_recordings.json')

    return df



MAILABS_lang_dict={'en_UK':'en-gb',
'en_US':'en-us',
'fr_FR':'fr-fr',
'es_ES':'es'
}
MAILABS_lang_to_MFA_model={'en_UK':'english_mfa',
'en_US':'english_mfa',
'fr_FR':'french_mfa',
'es_ES':'spanish_mfa'
}
MAILABS_lang_to_MFA_dict={'en_UK':'english_uk_mfa',
'en_US':'english_us_mfa',
'fr_FR':'french_mfa',
'es_ES':'spanish_mfa'
}
MAILABS_lang_to_CMU_model={'en_UK':'english_us_arpa',
'en_US':'english_us_arpa'
}

def MAILABS_data(path="data/MAILABS", lang_code="en_US"):
    if not os.path.exists(path): 
        print(path+" does not exist. Download at least a subset of the dataset")
        return None
    # working_dir= os.getcwd()
    # os.chdir(path)
    metadatas = glob(path+'/'+lang_code + "/*/*/*/*/*csv")
    wavs=glob(path+'/'+lang_code+"/*/*/*/*/*/*wav")
    dfs=[]
    for metadata in metadatas:
        try:
            # QUOTE_NONE to avoid error "EOF inside string starting at row 341": https://stackoverflow.com/questions/18016037/pandas-parsererror-eof-character-when-reading-multiple-csv-files-to-hdf5
            df=pd.read_csv(metadata, sep='|', header=None, quoting=csv.QUOTE_NONE)
        except:
            import pdb;pdb.set_trace()
            # pass
        dfs.append(df)

    metadatas_df=pd.concat(dfs, axis=0)

    len(metadatas_df)
    len(wavs)
    len(set(wavs))

    metadatas_df.columns=['filename','text_raw','text']

    wavs_names_to_paths={}
    for el in wavs:
        wavs_names_to_paths[el.split('/')[-1].split('.')[0]]=el

    paths=[]
    for i,r in metadatas_df.iterrows():
        try:
            paths.append(wavs_names_to_paths[r.filename])
        except KeyError:
            paths.append(float('nan'))

    metadatas_df['path']=paths
    metadatas_df=metadatas_df.dropna()
    metadatas_df['language_code']=lang_code

    lang=MAILABS_lang_dict[lang_code]
    
    # os.chdir(working_dir)
    return metadatas_df

def align_MAILABS(path="data/MAILABS", lang_code="en_US", phone_set='MFA_IPA', n=10000):
    print('Loading metadata')
    df=MAILABS_data(path=path, lang_code=lang_code)
    df=df.sample(frac=1, random_state=0)
    df=df[:n]
    mfa_path='/'.join(['./mfa_data', path.split("/")[-1], lang_code])
    if os.path.exists(mfa_path): shutil.rmtree(mfa_path)
    if not os.path.exists(mfa_path): os.makedirs(mfa_path)

    print('Preparing files for alignment')
    print("len of df:", len(df))
    for i,r in tqdm(df.iterrows()):
        # fp='/'.join([path,r.path])
        # fn='_'.join(r.wav_path.split('/'))
        fp=r.path
        fn=r.filename
        wav_fp=os.path.join(mfa_path, fn)+'.wav'
        text_fp=os.path.join(mfa_path, fn)+'.txt'
        os.symlink(fp, wav_fp)
        with open(text_fp, 'w') as f: f.write(r.text)

    print('Performing alignment')
    mfa_result_path='/'.join(['mfa_result', path.split("/")[-1], lang_code])
    if not os.path.exists(mfa_result_path): os.makedirs(mfa_result_path)
    if phone_set=='MFA_IPA':
        mfa_model=MAILABS_lang_to_MFA_model[lang_code]
        mfa_dict=MAILABS_lang_to_MFA_dict[lang_code]
    else: 
        mfa_model=MAILABS_lang_to_CMU_model[lang_code]
        mfa_dict=MAILABS_lang_to_CMU_model[lang_code]
    cmd='mfa align -j 16 --clean '+mfa_path+' '+mfa_dict+' '+mfa_model+' '+mfa_result_path
    os.system(cmd)

    print('Put results in dataframe')
    phone_dfs=[]
    failures=[]
    for i,r in tqdm(df.iterrows()):
        f=mfa_result_path+'/'+r.filename+'.TextGrid'
        try:
            phone_dfs.append(get_all_phone_with_timings(f).to_dict())
        except:
            phone_dfs.append(float('nan'))
            failures.append(r.filename)
    df['phone_df']=phone_dfs

    # df.drop('path', inplace=True, axis=1)
    df.to_csv(path+'/'+'MAILABS_shuffled_aligned-'+lang_code+'_'+phone_set+'.csv')
    return df, failures

def load_aligned_MAILABS(path="data/MAILABS", lang_code="en_US"):
    mfa_result_path='/'.join(['mfa_result', path.split("/")[-1], lang_code])
    df=pd.read_csv(path+'/'+'MAILABS_aligned-'+lang_code+'.csv')
    df=df.dropna()
    phone_dfs=df.phone_df.apply(lambda r: pd.DataFrame.from_dict(ast.literal_eval(r)))
    df.phone_df=phone_dfs

    return df


def use_tests():

    align_common_voice(root='data/cv-corpus-10.0-delta-2022-07-04/', lang='en', split='train', out_json='data/align_common_voice_en_train_mfa.json')
    align_common_voice(root='data/cv-corpus-10.0-delta-2022-07-04/', lang='en', split='dev', out_json='data/align_common_voice_en_dev_mfa.json')
    align_common_voice(root='data/cv-corpus-10.0-delta-2022-07-04/', lang='en', split='test', out_json='data/align_common_voice_en_test_mfa.json')
    
    align_common_voice(root='data/cv-corpus-10.0-delta-2022-07-04/', lang='en', split='train', mfa_model='english_us_arpa', out_json='data/align_common_voice_en_train_cmu.json')
    align_common_voice(root='data/cv-corpus-10.0-delta-2022-07-04/', lang='en', split='dev', mfa_model='english_us_arpa', out_json='data/align_common_voice_en_dev_cmu.json')
    align_common_voice(root='data/cv-corpus-10.0-delta-2022-07-04/', lang='en', split='test', mfa_model='english_us_arpa', out_json='data/align_common_voice_en_test_cmu.json')
    
    df, failures=align_MAILABS(path='/mnt/c/Users/noe_t/OneDrive - UMONS/flowchase/datasets/MAILABS', lang_code="en_US", phone_set='CMU', n=None)
    df, failures=align_MAILABS(path='/mnt/c/Users/noe_t/OneDrive - UMONS/flowchase/datasets/MAILABS', lang_code="en_UK", phone_set='CMU', n=None)

    
    df, failures=align_MAILABS(path='/mnt/c/Users/noe_t/OneDrive - UMONS/flowchase/datasets/MAILABS', lang_code="en_US", n=1000)
    df, failures=align_MAILABS(path='/mnt/c/Users/noe_t/OneDrive - UMONS/flowchase/datasets/MAILABS', lang_code="en_UK", n=1000)
    df, failures=align_MAILABS(path='/mnt/c/Users/noe_t/OneDrive - UMONS/flowchase/datasets/MAILABS', lang_code="fr_FR", n=1000)
    df, failures=align_MAILABS(path='/mnt/c/Users/noe_t/OneDrive - UMONS/flowchase/datasets/MAILABS', lang_code="es_ES", n=1000)

    df=load_aligned_MAILABS(path='/mnt/c/Users/noe_t/OneDrive - UMONS/flowchase/datasets/MAILABS', lang_code="es_ES")


    df=align_speech_ocean(path='data/speechocean762')
    # the phonemes from annotations and alignments (that has a step of text to phonmeme inside it) are not the same:
    # I already remove the stress annotations from vowels to have less differences
    phones_1=df.words_phones.apply(lambda r: remove_stress_annots(sum(r, [])))
    phones_2=df.apply(lambda r: remove_stress_annots(r.phone_df.cmu_phone.tolist()), axis=1)

    phones_1[phones_1!=phones_2]
    phones_2[phones_1!=phones_2]

    phones_1[phones_1.apply(lambda r:len(r))!=phones_2.apply(lambda r:len(r))]
    phones_2[phones_1.apply(lambda r:len(r))!=phones_2.apply(lambda r:len(r))]

    phones_1[phones_1.apply(lambda r:len(r))==phones_2.apply(lambda r:len(r))]
    phones_2[phones_1.apply(lambda r:len(r))==phones_2.apply(lambda r:len(r))]

    # select df such that the number of phonemes are the same
    df[phones_1.apply(lambda r:len(r))==phones_2.apply(lambda r:len(r))]
