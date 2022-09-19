import numpy as np
from utils.wav2vec2_GMM_ipa import Wav2Vec2ForFrameGMMAssignment
from utils.load_data import *
from math import log
from sklearn.metrics import mean_squared_error
from utils.metrics import compute_PER

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
    n_dimensions = [2, 4, 8, 16]

    results = {}

    df_t_train, df_t_test = load_cmu_dataset_MAILABS('en_UK', ['en_US', 'en_UK'])
    # df_all_frames = build_df_all_frames(df_t_train, 'ipa_phone')
    df_all_frames = pd.read_pickle('./data/df_all_frames_MAILABS.pkl')
    X, y = df_all_frames_to_X_y(df_all_frames)

    data = load_test_dataset(df_t_test)
    target_phonemes_list = data.ipa_phones.tolist()
    x = [[el.s,el.fs,el.ipa_phones] for _,el in data.iterrows()]
    x = x[:50]

    for c in n_clusters:
        for d in n_dimensions:
            print("Test for {} clusters and {} dimensions".format(c,d))
            model = Wav2Vec2ForFrameGMMAssignment(c,d,'cmu')
            model.fit(X, y)

            preds = model.predict(x)

            results["({},{})".format(c,d)] = {
                "preds" : preds,
                # "AIC" : calculate_aic( len(preds), mean_squared_error(preds, target_phonemes_list), c),
                # "BIC" : calculate_bic( len(preds), mean_squared_error(preds, target_phonemes_list), c),
                "mean_PER" : np.mean(compute_PER(preds, target_phonemes_list))
            }