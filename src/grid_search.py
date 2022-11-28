import numpy as np
from src.wav2vec2_GMM_ipa import Wav2Vec2ForFrameGMMAssignment
# from src.wav2vec2_GMM_no_reducer import Wav2Vec2ForFrameGMMAssignmentNoReducer
from src.load_data import *
from math import log
from sklearn.metrics import mean_squared_error
from src.metrics import compute_PER
from src.text_processing import remove_stress_annots
from time import time
import pickle

# calculate aic for regression
def calculate_aic(n, mse, num_params):
    aic = n * log(mse) + 2 * num_params
    return aic

# calculate bic for regression
def calculate_bic(n, mse, num_params):
    bic = n * log(mse) + num_params * log(n)
    return bic

if __name__ == '__main__':
    n_clusters = [50, 100, 150, 200]
    # n_clusters = [250, 300]
    n_dimensions = [4, 8]

    results = {}

    df_t_train = load_dataset_MAILABS(['en_US', 'en_UK'])
    # df_all_frames = build_df_all_frames(df_t_train, 'ipa_phone')
    df_all_frames = pd.read_pickle('./data/df_all_frames_MAILABS.pkl')
    X, y = df_all_frames_to_X_y(df_all_frames)

    data = load_test_dataset(df_t_test)
    target_phonemes_list = data.ipa_phones.tolist()
    x = [[el.s,el.fs,el.ipa_phones] for _,el in data.iterrows()]
    x = x[:50]
    target_phonemes_list = target_phonemes_list[:50]
    model = Wav2Vec2ForFrameGMMAssignment(30,2,'cmu','t-sne')

    for c in n_clusters:
        for d in n_dimensions:
            
            start=time()
            print("Test for {} clusters and {} dimensions".format(c,d))
            model.nbr_clusters = c
            model.target_dim = d
            model.fit(np.array(X), y)
            duration_fit=time()-start

            # pickle.dump(model.gmm, open("./data/test_model_gmm_{}_{}.pkl".format(c,d),"wb"))
            # pickle.dump(model.reducer, open("./data/test_model_pca_{}_{}.pkl".format(c,d),"wb"))
            
            start=time()
            print("Predictions for {} clusters and {} dimensions".format(c,d))
            preds = model.predict(x)
            duration_pred=time()-start

            results["({},{})".format(c,d)] = {
                "preds" : preds,
                # "AIC" : calculate_aic( len(preds), mean_squared_error(preds, target_phonemes_list), c),
                # "BIC" : calculate_bic( len(preds), mean_squared_error(preds, target_phonemes_list), c),
                "mean_PER" : np.mean(compute_PER(preds, target_phonemes_list)),
                "duration_fit" : duration_fit,
                "duration_pred" : duration_pred,
                "timestamps": model.timestamps
            }

    for i in results.values():
        print(i['mean_PER'])

    from src.metrics import plot_cf_matrix

    # ax = plot_cf_matrix([], flat_list_predictions, labels = cmu_alphabet) 
    # plt.show()