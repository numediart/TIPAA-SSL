import pandas as pd
from glob import glob
import os
import numpy as np
exercise_data=pd.read_csv('data/flwc-recordings/QueryResultsForNoe-2021-12-23_120638.csv')

# there are files at this level (when there is no date), I ignore them for now, it's like 2% of the files
# all_files=glob('data/flwc-recordings/*/*')

all_files=glob('data/flwc-recordings/*/*/*')
len(all_files)

# different possible extensions in dataset
extensions=set([f.split('.')[-1] for f in all_files if '.' in f])

# get filenames
fpaths=[f for f in all_files if f.split('.')[-1] in extensions]
user_ids=[f.split('/')[-2] for f in fpaths]
# user_ids=set([f.split('/')[-2] for f in fpaths])

fnames=[os.path.split(f.split('.')[0])[-1] for f in all_files if f.split('.')[-1] in extensions]
len(fpaths)
len(fnames)

# big number corresponding to the specific audio file
audio_file_idx=[f.split('.')[0].split('__')[-1] for f in fnames]

# remove user info from filename because it can contain '__' which is the separator and thus messes with parsing if not removed
# and I keep the information inside user_ids list
# metadata=[f.replace(user_ids[i]+'__', '') for i,f in enumerate(fnames)]
# len(metadata)

# split with '__' separator of metadata, discard last columns because user_id may contain '__' which messes the rest of the columns
df=pd.DataFrame([f.split('__') for f in fnames]).iloc[:,:3]
df.columns=['exercise_id', 'processing_status', 'answer']
df['user_id']=user_ids
df['audio_file_idx']=audio_file_idx
df['fpath']=fpaths
df['fname']=fnames

# 0=error,    1=success
df.loc[df.processing_status=='P0','processing_status']=0
df.loc[df.processing_status=='P1','processing_status']=1

# 0=error,    1=  , 2=  
df.loc[df.answer=='A0','answer']=0
df.loc[df.answer=='A1','answer']=1
df.loc[df.answer=='A2','answer']=2

# df[df.processing_status==0]
# df[df.processing_status==1]

# df[df.answer==0]
# df[df.answer==1]
# df[df.answer==2]

# df[df.processing_status==0][df.answer==0]
# df[df.processing_status==0][df.answer!=0]

df_errors=df[df.processing_status==0]
len(df_errors)/len(df)

ex_ids=df_errors.exercise_id.unique()

n_errors={}
for ex_id in ex_ids:
    n_errors[ex_id]=len(df_errors[df_errors.exercise_id==ex_id])

df_errors['module_type']=df_errors.exercise_id.apply(lambda r: exercise_data[exercise_data.exercise_id==r].module_type.values[0])


# sort dict by value
# https://stackoverflow.com/questions/613183/how-do-i-sort-a-dictionary-by-value
exs_sort_by_n_errors=list({k: v for k, v in sorted(n_errors.items(), key=lambda item: item[1])})[::-1]

# the most errors:
# n_errors[exs_sort_by_n_errors[0]]
# exs_sort_by_n_errors[0]

# data[data.exercise_id==exs_sort_by_n_errors[0]]