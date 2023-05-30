##### Data processing

# from src.load_data import load_dataset_MAILABS, build_df_all_frames, build_df_all_phoneme_instances
# df_t_train = load_dataset_MAILABS(['en_US', 'en_UK'], path='/mnt/c/Users/noe_t/OneDrive - UMONS/flowchase/datasets/MAILABS')
# df_t_train=df_t_train.dropna()

import pandas as pd

# df_all_phoneme_instances = build_df_all_phoneme_instances(df_t_train, 'phone', number_of_examples=None)
# df_all_instances.to_pickle('df_all_phonemes_instances_MAILABS_train.pkl')
# df_all_instances=pd.read_pickle('df_all_phonemes_instances_MAILABS_train.pkl')

# df_all_phoneme_instances.to_pickle('df_all_phonemes_instances_sequences_MAILABS_train.pkl')
df_all_phoneme_instances=pd.read_pickle('df_all_phonemes_instances_sequences_MAILABS_train.pkl')


dfs=[]
for i,r in df_all_phoneme_instances.iterrows():
    p_df=pd.DataFrame()
    p_df['vector']=list(r.vector_sequence)
    p_df['phoneme']=r.phoneme
    dfs.append(p_df)
df_all_instances=pd.concat(dfs)


# df_all_instances = build_df_all_frames(df_t_train, 'phone')
df_all_instances.to_pickle('df_all_frames_MAILABS_train.pkl')

# df_all_instances=pd.read_pickle('df_all_frames_MAILABS_train_w2v_xlsr_no_ft.pkl')
# df_all_instances=pd.read_pickle('df_all_frames_MAILABS_train_w2v_base.pkl')

# df_all_instances=pd.read_pickle('df_all_frames_commonvoice_en_dev.pkl')
# df_all_instances=pd.read_pickle('df_all_frames_MAILABS_train_ipa.pkl')
# df_all_instances=pd.read_pickle('df_all_frames_MAILABS_UK_US_FR_ES_train_ipa.pkl')


# df_all_instances=pd.read_pickle('df_all_frames_MAILABS_train.pkl')

df_all_instances_frames_with_silences=pd.read_pickle('df_all_frames_MAILABS_train.pkl')



df_all_instances_test=pd.read_pickle('df_all_frames_commonvoice_en_test.pkl')
from itertools import groupby
group_consecutive_duplicates= lambda L:[(k, sum(1 for i in g)) for k,g in groupby(L)]
# group_consecutive_duplicates(df_all_instances_test.phoneme)



# df_all_instances['sum']=df_all_instances.vector.apply(lambda r: r.sum())
# df_all_instances=df_all_instances.dropna()


# # shuffle, then select max 1000 instances of each phoneme
# df_all_instances_select = df_all_instances.sample(frac=1, random_state=0).groupby('phoneme').apply(lambda x: x.sample(n=min(len(x), 500))).reset_index(drop=True)
# df_all_instances_select.to_pickle('df_all_instances_select_MAILABS_train.pkl')
# df_all_instances_select=pd.read_pickle('df_all_instances_select_MAILABS_train.pkl')

# silences=df_all_instances_frames_with_silences[df_all_instances_frames_with_silences.phoneme.str.contains('SIL')].sample(n=500)
# df_all_instances_select_with_silences=pd.concat([silences, df_all_instances_select[['phoneme','vector']] ])

# df_all_instances_select_with_silences.to_pickle('df_all_instances_select_with_silences.pkl')



#####################  Frame classifier

df_all_instances_select_with_silences=pd.read_pickle('df_all_instances_select_with_silences.pkl')

df_all_instances_select_with_silences=df_all_instances_select_with_silences[(df_all_instances_select_with_silences.phoneme!="CH")&(df_all_instances_select_with_silences.phoneme!="JH")]

df_all_instances_select_with_silences.to_pickle('df_all_instances_select_with_silences_no_CH_JH.pkl')

df_all_instances_test=df_all_instances_test[(df_all_instances_test.phoneme!="CH")&(df_all_instances_test.phoneme!="JH")]


phoneme_set=sorted(list(set(df_all_instances_select_with_silences.phoneme)))
id_to_p={i:p for i,p in enumerate(phoneme_set)}
p_to_id={p:i for i,p in enumerate(phoneme_set)}
my_dict={p:len(df_all_instances_select_with_silences[df_all_instances_select_with_silences.phoneme==p]) for p in phoneme_set}
sorted_dict = dict(sorted(my_dict.items(), key=lambda item: item[1]))

# from src.load_data import load_libri_dataset, df_all_frames_to_X_y, load_libri_dataset_audio_timings

# df_t_train, df_t_test = load_libri_dataset()
# data = load_libri_dataset_audio_timings(df_t_test)
# data = load_libri_dataset_audio_timings(df_t_train)

# data[data.apply(lambda r: "ZH" in r.cmu_phones, axis=1)]



import numpy as np
# X=np.array(list(df_all_instances.iloc[:,1].values))
X=np.array(list(df_all_instances_select_with_silences.vector.values))
y=df_all_instances_select_with_silences.phoneme.apply(lambda r: p_to_id[r]).values

from sklearn.decomposition import PCA

reducer = PCA(n_components=0.95, random_state=42)

# from umap.umap_ import UMAP
# reducer2 = UMAP(n_components=100, n_neighbors=30, min_dist=0.0, random_state=42)

X_reduced=reducer.fit_transform(X)
# X_reduced=reducer2.fit_transform(X_reduced)

X_reduced.shape

from sklearn.model_selection import train_test_split

# X_train_reduced, X_test_reduced, y_train, y_test = train_test_split(X_reduced, y, random_state=1, test_size=0.2)

X_train_reduced=X_reduced
y_train=y

X_test=np.array(list(df_all_instances_test.iloc[:,1].values))
X_test_reduced=reducer.transform(X_test)
y_test=df_all_instances_test.phoneme.apply(lambda r: p_to_id[r]).values


############# try a few sklearn classifiers
# https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html#sphx-glr-auto-examples-classification-plot-classifier-comparison-py

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression

classifiers = [
    KNeighborsClassifier(10),
    KNeighborsClassifier(10, weights='distance'),
    KNeighborsClassifier(10, metric='cosine'),
    # KNeighborsClassifier(1, weights='distance', metric='cosine'),
    # KNeighborsClassifier(2, weights='distance', metric='cosine'),
    KNeighborsClassifier(5, weights='distance', metric='cosine'),
    # KNeighborsClassifier(7, weights='distance', metric='cosine'),
    KNeighborsClassifier(10, weights='distance', metric='cosine'),
    # KNeighborsClassifier(20, weights='distance', metric='cosine'),
    # SVC(kernel="linear", C=0.025, probability=True),
    # SVC(gamma=2, C=1, probability=True),
    # GaussianProcessClassifier(1.0 * RBF(1.0)),
    # DecisionTreeClassifier(max_depth=5),
    # RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    # MLPClassifier(alpha=1, max_iter=1000),
    # AdaBoostClassifier(),
    # GaussianNB(),
    # QuadraticDiscriminantAnalysis(),
    LinearDiscriminantAnalysis(),
    LogisticRegression(max_iter=1000)
]


# https://scikit-learn.org/stable/modules/ensemble.html#voting-classifier

# clf=LogisticRegression(max_iter=1000)


from time import time
score_dict={}
fit_time_dict={}
predict_time_dict={}
# iterate over classifiers
scores=[]
for clf in classifiers:
    start=time()
    clf.fit(X_train_reduced, y_train)
    fit_time_dict[clf.__str__()]=time()-start

    start=time()
    score = clf.score(X_test_reduced, y_test)
    predict_time_dict[clf.__str__()]=time()-start
    print(clf,': ', score)
    score_dict[clf.__str__()]=score
    scores.append(score)
print(fit_time_dict)
print(predict_time_dict)


# try a classifier with cmu_reducer if data is originally in IPA
clf=classifiers[0]
y_predicted=clf.predict(X_test_reduced)
predicted_phonemes=[id_to_p[el] for el in y_predicted]
test_phonemes=[id_to_p[el] for el in y_test]

df=pd.DataFrame([predicted_phonemes, test_phonemes]).T
sum(df[0]==df[1])/len(df)


from src.pronunciation_dictionaries import cmu_reducer

df_cmu_reduced=df.applymap(lambda r: cmu_reducer[r] if r!='[SIL]' else r)
sum(df_cmu_reduced[0]==df_cmu_reduced[1])/len(df_cmu_reduced)


probas_test=classifiers[1].predict_proba(X_test_reduced)

df_all_instances_test['probas']=list(probas_test)

# df_all_instances_test

np.array(df_all_instances_test[df_all_instances_test.phoneme=="[SIL]"].probas.values)

np.array(list(df_all_instances_test[df_all_instances_test.phoneme=="[SIL]"].probas)).mean(axis=0)

# scores=[score_dict[el] for el in list(score_dict)]


# https://scikit-learn.org/stable/modules/ensemble.html#weighted-average-probabilities-soft-voting
from sklearn.ensemble import VotingClassifier

# estimators=[(n,c) for n,c in zip(names, classifiers) if score_dict[n]>0.8]
# scores_estimators=[el for el in scores if el>0.8]
# eclf = VotingClassifier(estimators=estimators,
#                        voting='soft', weights=scores_estimators)

# estimators=[('10 Nearest Neighbors weighted', KNeighborsClassifier(10, weights='distance')),
#  ('QDA', QuadraticDiscriminantAnalysis())]

# estimators=[('5-NN cosine metric weighted', KNeighborsClassifier(5, weights='distance', metric='cosine')), ('LogisticRegression', LogisticRegression(max_iter=1000))]
estimators=[('LDA', LinearDiscriminantAnalysis()), ('LogisticRegression', LogisticRegression(max_iter=1000))]
# eclf = VotingClassifier(estimators=estimators, voting='soft', weights=[1 for _ in estimators])
eclf = VotingClassifier(estimators=estimators, voting='soft', weights=[1,5])
eclf.fit(X_train_reduced, y_train)
score = eclf.score(X_test_reduced, y_test)
print(score)

eclf.predict_proba(X_test_reduced[0].reshape(1,-1))

classifiers[1].predict_proba(X_test_reduced[0].reshape(1,-1))

classifiers[0].predict_proba(X_test_reduced[0].reshape(1,-1))
classifiers[1].predict_proba(X_test_reduced[0].reshape(1,-1))
classifiers[3].predict_proba(X_test_reduced[0].reshape(1,-1))
classifiers[-2].predict_proba(X_test_reduced[0].reshape(1,-1))
classifiers[-1].predict_proba(X_test_reduced[0].reshape(1,-1))


clf=eclf
y_predicted=clf.predict(X_test_reduced)
predicted_phonemes=[id_to_p[el] for el in y_predicted]
test_phonemes=[id_to_p[el] for el in y_test]

df=pd.DataFrame([predicted_phonemes, test_phonemes]).T
sum(df[0]==df[1])/len(df)

df_cmu_reduced=df.applymap(lambda r: cmu_reducer[r] if r!='[SIL]' else r)
sum(df_cmu_reduced[0]==df_cmu_reduced[1])/len(df_cmu_reduced)


#################### Investigate KNN-DTW


from sequentia.models import KNNClassifier
from sklearn.decomposition import PCA

df_all_phoneme_instances_sample=df_all_phoneme_instances.sample(frac=1, random_state=0).groupby('phoneme').apply(lambda x: x.sample(n=min(len(x), 500))).reset_index(drop=True)

# remove empty phonemes, that have 0 frame
df_all_phoneme_instances_sample=df_all_phoneme_instances_sample[df_all_phoneme_instances_sample.apply(lambda r: len(r.vector_sequence)!=0, axis=1)]

dfs=[]
for i,r in df_all_phoneme_instances_sample.iterrows():
    p_df=pd.DataFrame()
    p_df['vector']=list(r.vector_sequence)
    p_df['phoneme']=r.phoneme
    dfs.append(p_df)
df_all_instances_frames=pd.concat(dfs)


phoneme_set=sorted(list(set(df_all_phoneme_instances_sample.phoneme)))
id_to_p={i:p for i,p in enumerate(phoneme_set)}
p_to_id={p:i for i,p in enumerate(phoneme_set)}
my_dict={p:len(df_all_phoneme_instances_sample[df_all_phoneme_instances_sample.phoneme==p]) for p in phoneme_set}
sorted_dict = dict(sorted(my_dict.items(), key=lambda item: item[1]))


X=np.array(df_all_instances_frames.vector.tolist())

# y=df_all_instances_frames.phoneme.apply(lambda r: p_to_id[r]).values


reducer = PCA(n_components=0.95, random_state=42)
X_reduced=reducer.fit_transform(X)

df_all_phoneme_instances_sample['reduced_vector_sequence']=df_all_phoneme_instances_sample.apply(lambda r: reducer.transform(r.vector_sequence), axis=1)
lengths=df_all_phoneme_instances_sample.reduced_vector_sequence.apply(len).tolist()

y=df_all_phoneme_instances_sample.phoneme.apply(lambda r: p_to_id[r]).values

clf = KNNClassifier(k=1, use_c=True, n_jobs=-1).fit(X_reduced, y, lengths)
examples_lens=lengths[:3]
y_pred = clf.predict(X_reduced[:sum(examples_lens)], examples_lens)


group_consecutive_duplicates(df_all_instances_frames_with_silences.phoneme)

df_all_instances_test=pd.read_pickle('df_all_frames_commonvoice_en_test.pkl')

# try without silences for now
# TODO: also get silences. I think recompute the build_all_frames and then do the group_consecutive_duplicates

from itertools import groupby
group_consecutive_duplicates= lambda L:[(k, sum(1 for i in g)) for k,g in groupby(L)]


df_all_instances_test=df_all_instances_test[df_all_instances_test.phoneme!='[SIL]']
group_consecutive_duplicates(df_all_instances_test.phoneme)

X_test=np.array(list(df_all_instances_test.vector.values))
X_test_reduced=reducer.transform(X_test)

y_test=df_all_instances_test.phoneme.apply(lambda r: p_to_id[r]).values

y_test=[p_to_id[el[0]] for el in group_consecutive_duplicates(df_all_instances_test.phoneme)]

lengths=[el[-1] for el in group_consecutive_duplicates(df_all_instances_test.phoneme)]
examples_lens=lengths[:3]
y_pred_test = clf.predict(X_test_reduced[:sum(examples_lens)], examples_lens)

y_pred_test = clf.predict_proba(X_test_reduced[:sum(examples_lens)], examples_lens)


############## tslearn, not working
from tslearn.neighbors import KNeighborsTimeSeriesClassifier
clf = KNeighborsTimeSeriesClassifier(n_neighbors=10, metric="dtw", weights="distance")


X_seq=list(df_all_phoneme_instances['reduced_vector_sequence'].values)
from tslearn.utils import to_time_series_dataset, fillna
X_dataset = to_time_series_dataset(X_seq)

y=df_all_phoneme_instances.phoneme.apply(lambda r: p_to_id[r]).values


# clf.fit(X_filled,list(y))

X_filled = np.where(np.isnan(X_dataset), 0, X_dataset)
clf.predict(X_filled)

######### trying autoML, not working or not convinced
# import os
# os.environ["OPENBLAS_NUM_THREADS"] = "8"
# import autosklearn.classification
# # cls = autosklearn.classification.AutoSklearnClassifier()


# cls = autosklearn.classification.AutoSklearnClassifier(
#     memory_limit=1000,
#     include = {
#         'classifier': ["random_forest", "k_nearest_neighbors", "liblinear_svc"],
#         'feature_preprocessor': ["no_preprocessing"]
#     },
#     exclude=None
# )

# cls.fit(X_train_reduced, y_train)
# y_pred = cls.predict(X_test_reduced)
# score = cls.score(y_pred, y_test)
# print(score)


# # nn_model = NeuralNetwork(target_dims=2, loss_fn=nn.CrossEntropyLoss()).to(device)
# # nn_model.fit(X_train, torch.LongTensor(y_train))


# # initialise Auto-PyTorch api
# automl = TabularClassificationTask(seed=1)

# # Search for an ensemble of machine learning algorithms
# automl.search(
#     X_train=X_train_reduced,
#     y_train=y_train,
#     X_test=X_test_reduced,
#     y_test=y_test,
#     optimize_metric='accuracy',
#     total_walltime_limit=10000,
#     func_eval_time_limit_secs=1000
# )

# # Calculate test accuracy
# y_pred = automl.predict(X_test_reduced)
# score = automl.score(y_pred, y_test)
# print("Accuracy score", score)

# print(automl.show_models())