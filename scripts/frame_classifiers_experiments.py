##### Get data
# from src.load_data import load_dataset_MAILABS, build_df_all_frames, build_df_all_phoneme_instances
# df_t_train = load_dataset_MAILABS(['en_US', 'en_UK'], path='/mnt/c/Users/noe_t/OneDrive - UMONS/flowchase/datasets/MAILABS')
# df_t_train=df_t_train.dropna()

import pandas as pd

# df_all_instances = build_df_all_phoneme_instances(df_t_train, 'phone')
# df_all_instances.to_pickle('df_all_phonemes_instances_MAILABS_train.pkl')
# df_all_instances=pd.read_pickle('df_all_phonemes_instances_MAILABS_train.pkl')

# df_all_instances = build_df_all_frames(df_t_train, 'phone')
# df_all_instances.to_pickle('df_all_frames_MAILABS_train.pkl')


# df_all_instances = build_df_all_frames(df_t_train, 'phone')
# df_all_instances.to_pickle('df_all_frames_MAILABS_train.pkl')


# df_all_instances=pd.read_pickle('df_all_frames_MAILABS_train_w2v_xlsr_no_ft.pkl')
# df_all_instances=pd.read_pickle('df_all_frames_MAILABS_train_w2v_base.pkl')

# df_all_instances=pd.read_pickle('df_all_frames_MAILABS_train.pkl')
# df_all_instances=pd.read_pickle('df_all_frames_MAILABS_train_ipa.pkl')
df_all_instances=pd.read_pickle('df_all_frames_MAILABS_UK_US_FR_ES_train_ipa.pkl')

# X_train, _ = df_all_frames_to_X_y(df_all_instances)



df_all_instances['sum']=df_all_instances.iloc[:,1].apply(lambda r: r.sum())
df_all_instances=df_all_instances.dropna()

import numpy as np
X=np.array(list(df_all_instances.iloc[:,1].values))
phoneme_set=list(set(df_all_instances.phoneme))
id_to_p={i:p for i,p in enumerate(phoneme_set)}
p_to_id={p:i for i,p in enumerate(phoneme_set)}

y=df_all_instances.phoneme.apply(lambda r: p_to_id[r]).values

from sklearn.decomposition import PCA

reducer = PCA(n_components=0.99, random_state=42)

# from umap.umap_ import UMAP
# reducer2 = UMAP(n_components=100, n_neighbors=30, min_dist=0.0, random_state=42)

X_reduced=reducer.fit_transform(X)
# X_reduced=reducer2.fit_transform(X_reduced)

X_reduced.shape

from sklearn.model_selection import train_test_split

X_train_reduced, X_test_reduced, y_train, y_test = train_test_split(X_reduced, y, random_state=1, test_size=0.2)


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

names = [
    "10 Nearest Neighbors",
    "10 Nearest Neighbors weighted distance",
    "10 Nearest Neighbors cosine dist",
    # "1 Nearest Neighbors cosine weighted",
    # "2 Nearest Neighbors cosine weighted",
    "5 Nearest Neighbors cosine weighted",
    # "7 Nearest Neighbors cosine weighted",
    # "10 Nearest Neighbors cosine weighted",
    # "20 Nearest Neighbors cosine weighted",
    # "50 Nearest Neighbors cosine weighted",
    # "100 Nearest Neighbors cosine weighted",
    # "Linear SVM",
    # "RBF SVM",
    # "Gaussian Process",
    # "Decision Tree",
    # "Random Forest",
    # "Neural Net",
    # "AdaBoost",
    # "Naive Bayes",
    # "QDA",
    "LDA",
    "logistic regression"
]

classifiers = [
    KNeighborsClassifier(10),
    KNeighborsClassifier(10, weights='distance'),
    KNeighborsClassifier(10, metric='cosine'),
    # KNeighborsClassifier(1, weights='distance', metric='cosine'),
    # KNeighborsClassifier(2, weights='distance', metric='cosine'),
    KNeighborsClassifier(5, weights='distance', metric='cosine'),
    # KNeighborsClassifier(7, weights='distance', metric='cosine'),
    # KNeighborsClassifier(10, weights='distance', metric='cosine'),
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
for name, clf in zip(names, classifiers):
    start=time()
    clf.fit(X_train_reduced, y_train)
    fit_time_dict[name]=time()-start

    start=time()
    score = clf.score(X_test_reduced, y_test)
    predict_time_dict[name]=time()-start
    print(clf,': ', score)
    score_dict[name]=score
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

# scores=[score_dict[el] for el in list(score_dict)]


# https://scikit-learn.org/stable/modules/ensemble.html#weighted-average-probabilities-soft-voting
from sklearn.ensemble import VotingClassifier

# estimators=[(n,c) for n,c in zip(names, classifiers) if score_dict[n]>0.8]
# scores_estimators=[el for el in scores if el>0.8]
# eclf = VotingClassifier(estimators=estimators,
#                        voting='soft', weights=scores_estimators)

# estimators=[('10 Nearest Neighbors weighted', KNeighborsClassifier(10, weights='distance')),
#  ('QDA', QuadraticDiscriminantAnalysis())]

estimators=[('5-NN cosine metric weighted', KNeighborsClassifier(5, weights='distance', metric='cosine')), ('LogisticRegression', LogisticRegression(max_iter=1000))]
# eclf = VotingClassifier(estimators=estimators, voting='soft', weights=[1 for _ in estimators])
eclf = VotingClassifier(estimators=estimators, voting='soft', weights=[1,5])
eclf.fit(X_train_reduced, y_train)
score = eclf.score(X_test_reduced, y_test)
print(score)

eclf.predict_proba(X_test_reduced[0].reshape(1,-1))

classifiers[0].predict_proba(X_test_reduced[0].reshape(1,-1))
classifiers[3].predict_proba(X_test_reduced[0].reshape(1,-1))
clf.predict_proba(X_test_reduced[0].reshape(1,-1))


clf=eclf
y_predicted=clf.predict(X_test_reduced)
predicted_phonemes=[id_to_p[el] for el in y_predicted]
test_phonemes=[id_to_p[el] for el in y_test]

df=pd.DataFrame([predicted_phonemes, test_phonemes]).T
sum(df[0]==df[1])/len(df)

df_cmu_reduced=df.applymap(lambda r: cmu_reducer[r] if r!='[SIL]' else r)
sum(df_cmu_reduced[0]==df_cmu_reduced[1])/len(df_cmu_reduced)


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