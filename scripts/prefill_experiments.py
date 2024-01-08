from src.text_processing import *


def prefill_content_variations(sentences):
    db = pd.read_csv('data/query_results-2023-02-21_102331.csv')
    sentences = db.words.tolist()

    PE = pd.read_csv('data/PE_all_phrases_no_duplicates.csv')
    sentences = PE.sentence.tolist()

    df_MFA_IPA_US, df_errors = prefill_content(sentences, lang='en_US', mode='MFA_IPA')
    df_MFA_IPA_GB, df_errors = prefill_content(sentences, lang='en_GB', mode='MFA_IPA')
    df_CMU, df_errors = prefill_content(sentences, lang='en_US', mode='CMU')

    # df_MFA_IPA_US.to_csv('prefill_export_2023-02-21_MFA_IPA_en_US.csv')
    # df_MFA_IPA_GB.to_csv('prefill_export_2023-02-21_MFA_IPA_en_GB.csv')
    # df_CMU.to_csv('prefill_export_2023-02-21_CMU_en_US.csv')

    # df_MFA_IPA_US=pd.read_csv('prefill_export_2023-02-21_MFA_IPA_en_US.csv')
    # df_MFA_IPA_GB=pd.read_csv('prefill_export_2023-02-21_MFA_IPA_en_GB.csv')
    # df_CMU=pd.read_csv('prefill_export_2023-02-21_CMU_en_US.csv')

    df_MFA_IPA_US.to_csv('prefill_PE_no_duplicates_MFA_IPA_en_US.csv')
    df_MFA_IPA_GB.to_csv('prefill_PE_no_duplicates_MFA_IPA_en_GB.csv')
    df_CMU.to_csv('prefill_PE_no_duplicates_CMU_en_US.csv')

    df_MFA_IPA_US = pd.read_csv('prefill_PE_no_duplicates_MFA_IPA_en_US.csv')
    df_MFA_IPA_GB = pd.read_csv('prefill_PE_no_duplicates_MFA_IPA_en_GB.csv')
    df_CMU = pd.read_csv('prefill_PE_no_duplicates_CMU_en_US.csv')

    # assert (df_MFA_IPA_GB.text==df_CMU.text).sum()==len(df_CMU)
    assert (df_MFA_IPA_GB.segmented_text != df_CMU.segmented_text).sum() == 0
    assert (df_MFA_IPA_US.segmented_text != df_CMU.segmented_text).sum() == 0

    n_syls_new = df_MFA_IPA_GB.segmented_text.str.lower().apply(
        lambda r: [w.count('|') + 1 for w in r.split(' ')]
    )
    # n_syls_db=db.syllables.str.lower().apply(lambda r: [w.count('|')+1 for w in r.split(' ')])

    # db[n_syls_new!=n_syls_db].syllables.str.lower()

    # df_MFA_IPA_GB[n_syls_new!=n_syls_db].segmented_text.str.lower()
    # syllable_comparison=pd.DataFrame()
    # syllable_comparison['DB']=db[n_syls_new!=n_syls_db].syllables
    # syllable_comparison['now']=df_MFA_IPA_GB[n_syls_new!=n_syls_db].segmented_text
    # syllable_comparison['id']=db.id[syllable_comparison.index]

    # syllable_comparison.to_csv('syllables_text_comparison.csv')

    # df_MFA_IPA_GB[df_MFA_IPA_GB.segmented_text!=df_CMU.segmented_text].segmented_text
    # df_CMU[df_MFA_IPA_GB.segmented_text!=df_CMU.segmented_text].segmented_text

    import ast

    # mismatch syllables between text and phonetics, for each phonetics variation
    df_CMU_mis = df_CMU[df_CMU.n_syl_mismatches.str.len() != 0][
        ["segmented_text", "phonetics"]
    ]
    df_MFA_IPA_GB_syl_mis = df_MFA_IPA_GB[df_MFA_IPA_GB.n_syl_mismatches.str.len() != 0][
        ["segmented_text", "phonetics"]
    ]
    df_MFA_IPA_US_syl_mis = df_MFA_IPA_US[df_MFA_IPA_US.n_syl_mismatches.str.len() != 0][
        ["segmented_text", "phonetics"]
    ]

    # df_CMU_mis=df_CMU[df_CMU.n_syl_mismatches!="[]"][["segmented_text","phonetics"]]
    # df_MFA_IPA_GB_syl_mis=df_MFA_IPA_GB[df_MFA_IPA_GB.n_syl_mismatches!="[]"][["segmented_text","phonetics"]]
    # df_MFA_IPA_US_syl_mis=df_MFA_IPA_US[df_MFA_IPA_US.n_syl_mismatches!="[]"][["segmented_text","phonetics"]]

    df_MFA_IPA_GB_syl_mis[
        df_MFA_IPA_GB_syl_mis.segmented_text.isin(df_MFA_IPA_US_syl_mis.segmented_text)
    ]
    df_MFA_IPA_GB_syl_mis[
        ~df_MFA_IPA_GB_syl_mis.segmented_text.isin(df_MFA_IPA_US_syl_mis.segmented_text)
    ]

    # # check that there is no "ɪ|ə" in US
    # mask=df_MFA_IPA_US["phonetics"].apply(lambda r: "ɪ|ə" in r[-1])
    # df_MFA_IPA_US[mask]

    # # check that this "ɪ|ə" exist only in a mismatch situation. Here when there is no mismatch, it does not exists
    # mask=df_MFA_IPA_GB[df_MFA_IPA_GB.n_syl_mismatches=="[]"]["phonetics"].apply(lambda r: "ɪ|ə" in r[-1])
    # df_MFA_IPA_GB[df_MFA_IPA_GB.n_syl_mismatches=="[]"][mask]

    # syl_inconsistencies_CMU=df_CMU[df_CMU.n_syl_mismatches!="[]"].apply(lambda r: (r.segmented_text.split(' ')[ast.literal_eval(r.n_syl_mismatches)[0]],r.phonetics.split(' ')[ast.literal_eval(r.n_syl_mismatches)[0]]), axis=1)
    # syl_inconsistencies_MFA_GB=df_MFA_IPA_GB[df_MFA_IPA_GB.n_syl_mismatches!="[]"].apply(lambda r: (r.segmented_text.split(' ')[ast.literal_eval(r.n_syl_mismatches)[0]],r.phonetics.split(' ')[ast.literal_eval(r.n_syl_mismatches)[0]]), axis=1)
    # syl_inconsistencies_MFA_US=df_MFA_IPA_US[df_MFA_IPA_US.n_syl_mismatches!="[]"].apply(lambda r: (r.segmented_text.split(' ')[ast.literal_eval(r.n_syl_mismatches)[0]],r.phonetics.split(' ')[ast.literal_eval(r.n_syl_mismatches)[0]]), axis=1)

    syl_inconsistencies_CMU = df_CMU[df_CMU.n_syl_mismatches.str.len() != 0].apply(
        lambda r: (
            r.segmented_text.split(' ')[r.n_syl_mismatches[0]],
            r.phonetics.split(' ')[r.n_syl_mismatches[0]],
        ),
        axis=1,
    )
    syl_inconsistencies_MFA_GB = df_MFA_IPA_GB[
        df_MFA_IPA_GB.n_syl_mismatches.str.len() != 0
    ].apply(
        lambda r: (
            r.segmented_text.split(' ')[r.n_syl_mismatches[0]],
            r.phonetics.split(' ')[r.n_syl_mismatches[0]],
        ),
        axis=1,
    )
    syl_inconsistencies_MFA_US = df_MFA_IPA_US[
        df_MFA_IPA_US.n_syl_mismatches.str.len() != 0
    ].apply(
        lambda r: (
            r.segmented_text.split(' ')[r.n_syl_mismatches[0]],
            r.phonetics.split(' ')[r.n_syl_mismatches[0]],
        ),
        axis=1,
    )

    df_CMU_mis["word_syl_mismatch"] = syl_inconsistencies_CMU
    # df_CMU_mis['id']=db.id[syl_inconsistencies_CMU.index]
    df_MFA_IPA_GB_syl_mis["word_syl_mismatch"] = syl_inconsistencies_MFA_GB
    # df_MFA_IPA_GB_syl_mis['id']=db.id[syl_inconsistencies_MFA_GB.index]
    df_MFA_IPA_US_syl_mis["word_syl_mismatch"] = syl_inconsistencies_MFA_US
    # df_MFA_IPA_US_syl_mis['id']=db.id[syl_inconsistencies_MFA_US.index]

    df_CMU_mis.to_csv('syllables_CMU_comparison.csv')
    df_MFA_IPA_GB_syl_mis.to_csv('syllables_IPA_GB_comparison.csv')
    df_MFA_IPA_US_syl_mis.to_csv('syllables_IPA_US_comparison.csv')

    for i in range(int(len(syl_inconsistencies_CMU) / 10)):
        print(syl_inconsistencies_CMU[i * 10 : (i + 1) * 10])
    for i in range(int(len(syl_inconsistencies_MFA_GB) / 10)):
        print(syl_inconsistencies_MFA_GB[i * 10 : (i + 1) * 10])
    for i in range(int(len(syl_inconsistencies_MFA_US) / 10)):
        print(syl_inconsistencies_MFA_US[i * 10 : (i + 1) * 10])

    map_phonemes = lambda mapper, formatted_phonetics: ' '.join(
        [
            '|'.join(
                ['_'.join([mapper[unstress(p).lower()] for p in syl]) for syl in word]
            )
            for word in split_phonetics(formatted_phonetics)
        ]
    )

    from src.pronunciation_dictionaries import arpabet_to_2_char_ipa, mfa_to_display_ipa

    df_MFA_IPA_US_disp = df_MFA_IPA_US.phonetics.apply(
        lambda r: map_phonemes(
            mfa_to_display_ipa, r.replace('{', '').replace('}', '').replace('-', ' ')
        )
    )
    df_MFA_IPA_GB_disp = df_MFA_IPA_GB.phonetics.apply(
        lambda r: map_phonemes(
            mfa_to_display_ipa, r.replace('{', '').replace('}', '').replace('-', ' ')
        )
    )

    result_df = pd.DataFrame()
    # result_df['id']=db.id
    # result_df['syllables']=db.syllables
    result_df['text'] = df_CMU.text
    result_df['segmented_text'] = df_CMU.segmented_text
    result_df['CMU'] = df_CMU.phonetics
    result_df['IPA en_US'] = df_MFA_IPA_US.phonetics
    result_df['IPA en_GB'] = df_MFA_IPA_GB.phonetics
    result_df['simple IPA en_US'] = df_MFA_IPA_US_disp.str.replace('_', '').str.replace(
        '|', '‧'
    )
    result_df['simple IPA en_GB'] = df_MFA_IPA_GB_disp.str.replace('_', '').str.replace(
        '|', '‧'
    )

    result_df.to_csv('phonetization_PE_export.csv')

    ## the rest is for comparing with an export from DB

    syl_inconsistencies_MFA_GB[syl_inconsistencies_MFA_GB.apply(lambda r: "ɪ|ə" in r[-1])]
    syl_inconsistencies_MFA_GB[
        syl_inconsistencies_MFA_GB.apply(lambda r: "ɪ|ə" not in r[-1])
    ]

    # df_CMU[df_CMU.n_stress_inconsistencies.apply(lambda r:len(r))==0]
    # df_MFA_IPA_GB[df_MFA_IPA_GB.n_stress_inconsistencies.apply(lambda r:len(r))==0]
    # df_MFA_IPA_US[df_MFA_IPA_US.n_stress_inconsistencies.apply(lambda r:len(r))==0]

    df_compare = pd.DataFrame()
    df_compare['DB'] = db[df_CMU.phonetics != db.phonetics].phonetics
    df_compare['now'] = df_CMU[df_CMU.phonetics != db.phonetics].phonetics
    df_compare["word_syl_mismatch"] = df_compare.apply(
        lambda r: [el for el in zip(r.DB.split(' '), r.now.split(' ')) if el[0] != el[1]],
        axis=1,
    )
    df_compare['id'] = db.id[df_compare.index]

    df_compare.to_csv('CMU_comparison.csv')

    # import difflib
    # diff_string = lambda case_a, case_b : [li for li in difflib.ndiff(case_a, case_b) if li[0] != ' ']
    # df_compare.apply(lambda r: diff_string(r.db, r.prefill), axis=1)

    # df_compare.apply(lambda r: [el for el in zip(r.db.split(' '),r.prefill.split(' ')) if el[0]!=el[1]], axis=1)

    diff_iz = df_compare.apply(
        lambda r: [
            el
            for el in zip(r.DB.split(' '), r.now.split(' '))
            if (el[0] != el[1] and el[1].endswith("_IH0_Z"))
        ],
        axis=1,
    )
    diff_non_iz = df_compare.apply(
        lambda r: [
            el
            for el in zip(r.DB.split(' '), r.now.split(' '))
            if el[0] != el[1] and (not el[1].endswith("_IH0_Z"))
        ],
        axis=1,
    )

    diff_non_iz = diff_non_iz[diff_non_iz.apply(lambda r: len(r)) > 0]
    diff_iz = diff_iz[diff_iz.apply(lambda r: len(r)) > 0]

    db.loc[diff_non_iz.index, :]
    iz_modifs = db.loc[diff_iz.index, :]

    iz_modifs['modification'] = diff_iz

    df_compare_non_iz_CMU = pd.DataFrame()
    df_compare_non_iz_CMU['DB'] = df_compare.DB[diff_non_iz.index]
    df_compare_non_iz_CMU['now'] = df_compare.now[diff_non_iz.index]
    df_compare_non_iz_CMU['mismatch'] = diff_non_iz
    df_compare_non_iz_CMU['id'] = db.id[diff_non_iz.index]

    df_compare_non_iz_CMU.to_csv('CMU_comparison_non_iz.csv')


def prefill_words():
    from src.pronunciation_dictionaries import (
        get_augmented_mfa_dict,
        lang_to_MFA_g2p_models,
        get_mfa_dict,
    )

    langs = ['es_ES', 'fr_FR', 'en_GB', 'en_US']
    # langs=lang_to_MFA_g2p_models.keys()
    mfa_dicts = {lang: get_augmented_mfa_dict(lang) for lang in langs}
    # mfa_dicts={lang:get_augmented_mfa_dict(lang) for lang in lang_to_MFA_g2p_models}

    d = get_mfa_dict(path='data/' + lang_to_MFA_g2p_models['fr_FR'] + '.dict')
    mfa_dicts['fr_FR'] = d

    for lang in langs:
        mfa_d = mfa_dicts[lang]
        print(lang, len(mfa_d))
        sentences = list(mfa_d.keys())
        df_MFA_IPA, df_errors = prefill_content(sentences, lang=lang, mode='MFA_IPA')
        # df_MFA_IPA.to_csv('prefill_MFA_IPA_dict_'+lang+'.csv')

    for lang in langs:
        df_MFA_IPA = pd.read_csv('prefill_MFA_IPA_dict_' + lang + '.csv')

        df_MFA_IPA.dropna().apply(lambda r: r.phonetics.count('|') + 1, axis=1).tolist()

        df_MFA_IPA = df_MFA_IPA.sample(n=1000, random_state=0)
        import ast

        df_MFA_IPA.n_syl_mismatches = df_MFA_IPA.n_syl_mismatches.apply(ast.literal_eval)
        df_MFA_IPA_syl_mis = df_MFA_IPA[df_MFA_IPA.n_syl_mismatches.str.len() != 0][
            ["segmented_text", "phonetics", "n_syl_mismatches"]
        ]
        print(df_MFA_IPA_syl_mis)

        print("nans", df_MFA_IPA_syl_mis[df_MFA_IPA_syl_mis.isna().any(axis=1)])
        df_MFA_IPA_syl_mis = df_MFA_IPA_syl_mis.dropna()
        syl_inconsistencies_MFA = df_MFA_IPA_syl_mis.apply(
            lambda r: (
                r.segmented_text.split(' ')[r.n_syl_mismatches[0]],
                r.phonetics.split(' ')[r.n_syl_mismatches[0]],
            ),
            axis=1,
        )
        syl_mismatch_tot = df_MFA_IPA.n_syl_mismatches.apply(lambda r: len(r)).sum()
        n_words_tot = len(df_MFA_IPA)
        mismatch_rate = syl_mismatch_tot / n_words_tot
        print(lang, (1 - mismatch_rate) * 100)

        # lang
        tqdm.pandas()
        df_MFA_IPA['segmented_text_SSP_text'] = df_MFA_IPA.progress_apply(
            lambda r: syllabified_text(
                r.text,
                n_vowels(r.text, lang=lang),
                pd.DataFrame(
                    columns=['n_syls', 'n_syls_SonoriPy', 'normalized_text', 'syllables']
                ),
                lang,
                combined_sonoripy_dtw=False,
            )[0],
            axis=1,
        )
        df_MFA_IPA['segmented_text_SSP_DTW'] = df_MFA_IPA.progress_apply(
            lambda r: syllabified_text(
                r.text,
                n_vowels(r.text, lang=lang),
                pd.DataFrame(
                    columns=['n_syls', 'n_syls_SonoriPy', 'normalized_text', 'syllables']
                ),
                lang,
            )[0],
            axis=1,
        )
        df_MFA_IPA['segmented_text_lookup_SSP_DTW'] = df_MFA_IPA.progress_apply(
            lambda r: syllabified_text(
                r.text, n_vowels(r.text, lang=lang), syllables_dfs[lang], lang
            )[0],
            axis=1,
        )
        # st=df_MFA_IPA_syl_mis.progress_apply(lambda r: syllabified_text(r.segmented_text.replace('|',''), n_vowels(r.segmented_text.replace('|',''), lang=lang), syllables_dfs[lang], lang)[0], axis=1)

        text_cols = [el for el in df_MFA_IPA.columns if "segmented_" in el]

        for c in text_cols:
            df_MFA_IPA_syl_mis = df_MFA_IPA[
                df_MFA_IPA.phonetics.str.count('\|') != df_MFA_IPA[c].str.count('\|')
            ]
            st = syl_inconsistencies_MFA.apply(
                lambda r: syllabified_text(
                    r[0], n_vowels(r[0], lang=lang), syllables_dfs[lang], lang
                )
            )
            mismatch_rate = len(df_MFA_IPA_syl_mis) / n_words_tot
            print(c, lang, (1 - mismatch_rate) * 100)

    from src.pronunciation_dictionaries import get_augmented_cmudict

    d = get_augmented_cmudict()
    sentences = list(d.keys())
    df_CMU, df_errors = prefill_content(sentences, mode='CMU')
    # if len(df_errors)==0:
    #     df_CMU['text']=df['sentence']
    # df_CMU.to_csv('prefill_CMU_dict_'+lang+'.csv')

    df_CMU = pd.read_csv('prefill_CMU_dict.csv')


def prefill_cmuarctic():
    from src.pronunciation_dictionaries import (
        get_augmented_mfa_dict,
        lang_to_MFA_g2p_models,
        get_mfa_dict,
    )
    import pandas as pd
    from collections import Counter

    df = pd.read_csv('data/cmuarctic.data', sep='(', header=None)
    df['id'] = df.iloc[:, 1].str[1:13]
    df['sentence'] = df.iloc[:, 1].str[15:-3].str.replace('-- ', '')
    df.index = df.id
    langs = ['en_US', 'en_GB']

    sentences = df['sentence'].tolist()
    for lang in langs:
        df_MFA_IPA, df_errors = prefill_content(sentences, lang=lang, mode='MFA_IPA')
        if len(df_errors) == 0:
            df_MFA_IPA['text'] = sentences
        df_MFA_IPA.to_csv('prefill_MFA_IPA_cmuarctic_' + lang + '.csv')

    for lang in langs:
        df_MFA_IPA = pd.read_csv('prefill_MFA_IPA_cmuarctic_' + lang + '.csv')

        import ast

        df_MFA_IPA.n_syl_mismatches = df_MFA_IPA.n_syl_mismatches.apply(ast.literal_eval)
        # mismatch syllables between text and phonetics, for each phonetics variation
        # df_CMU_mis=df_CMU[df_CMU.n_syl_mismatches.str.len()!=0][["segmented_text","phonetics"]]
        df_MFA_IPA_syl_mis = df_MFA_IPA[df_MFA_IPA.n_syl_mismatches.str.len() != 0][
            ["segmented_text", "phonetics", "n_syl_mismatches"]
        ]
        syl_inconsistencies_MFA = df_MFA_IPA_syl_mis.apply(
            lambda r: (
                r.segmented_text.split(' ')[r.n_syl_mismatches[0]],
                r.phonetics.split(' ')[r.n_syl_mismatches[0]],
            ),
            axis=1,
        )
        # print(syl_inconsistencies_MFA)
        # for i in range(int(len(syl_inconsistencies_MFA)/10)): print(syl_inconsistencies_MFA[i*10:(i+1)*10])

        syl_mismatch_tot = df_MFA_IPA.n_syl_mismatches.apply(len).sum()
        n_words_tot = df_MFA_IPA.n_alternatives.apply(len).sum()
        mismatch_rate = syl_mismatch_tot / n_words_tot
        print(lang, (1 - mismatch_rate) * 100)

        df_MFA_IPA['segmented_text_lookup_SSP_DTW'] = df_MFA_IPA.progress_apply(
            lambda r: ' '.join(
                [
                    syllabified_text(
                        word, n_vowels(word, lang=lang), syllables_dfs[lang], lang
                    )[0]
                    for word in r.segmented_text.replace('|', '').split(' ')
                ]
            ),
            axis=1,
        )
        c = "segmented_text_lookup_SSP_DTW"

        df_MFA_IPA['n_syl_mismatches_best'] = df_MFA_IPA.apply(
            lambda r: [
                i
                for i, (el1, el2) in enumerate(
                    zip(r.phonetics.split(' '), r.segmented_text.split(' '))
                )
                if el1.count('|') != el2.count('|')
            ],
            axis=1,
        )

        df_MFA_IPA_syl_mis = df_MFA_IPA[df_MFA_IPA.n_syl_mismatches_best.str.len() != 0][
            ["segmented_text", "phonetics", "n_syl_mismatches_best"]
        ]
        syl_inconsistencies_MFA = df_MFA_IPA_syl_mis.apply(
            lambda r: (
                r.segmented_text.split(' ')[r.n_syl_mismatches_best[0]],
                r.phonetics.split(' ')[r.n_syl_mismatches_best[0]],
            ),
            axis=1,
        )

        df_MFA_IPA['n_syls'] = df_MFA_IPA.apply(
            lambda r: [el.count('|') + 1 for el in r.phonetics.split(' ')], axis=1
        )

        # df_MFA_IPA['n_syls'].sum()

        inventory_dict = dict(Counter(df_MFA_IPA['n_syls'].sum()))
        sorted_inventory = dict(sorted(inventory_dict.items()))

        x_values = sorted_inventory.keys()
        y_values = sorted_inventory.values()

        syl_mismatch_tot = df_MFA_IPA.n_syl_mismatches_best.apply(len).sum()
        n_words_tot = df_MFA_IPA.n_alternatives.apply(len).sum()
        mismatch_rate = syl_mismatch_tot / n_words_tot
        print(lang, (1 - mismatch_rate) * 100)

        # df_MFA_IPA_syl_mis=df_MFA_IPA[df_MFA_IPA.phonetics.str.count('\|')!=df_MFA_IPA[c].str.count('\|')]
        # st=syl_inconsistencies_MFA.apply(lambda r: syllabified_text(r[0], n_vowels(r[0], lang=lang), syllables_dfs[lang], lang))
        # mismatch_rate=len(df_MFA_IPA_syl_mis)/n_words_tot
        # print(c, (1-mismatch_rate)*100)

    import matplotlib.pyplot as plt

    plt.clf()
    fig = plt.figure()
    ax = plt.subplot(121)
    plt.bar(x_values, np.array(list(y_values)) / sum(y_values) * 100)

    ax.set_title("CMU ARCTIC sentences")

    ax2 = plt.subplot(122)
    df_MFA_IPA_dict = pd.read_csv('prefill_MFA_IPA_dict_' + lang + '.csv')
    inventory_dict2 = dict(
        Counter(
            df_MFA_IPA_dict.dropna()
            .apply(lambda r: r.phonetics.count('|') + 1, axis=1)
            .tolist()
        )
    )
    sorted_inventory2 = dict(sorted(inventory_dict2.items()))

    x_values2 = sorted_inventory2.keys()
    y_values2 = sorted_inventory2.values()
    plt.bar(x_values2, np.array(list(y_values2)) / sum(y_values2) * 100)
    ax2.set_title("MFA dict words (en_US)")

    fig.suptitle('Number of word occurences per number of syllables')
    plt.savefig('n_syl_hist.png')

    df_CMU, df_errors = prefill_content(sentences, mode='CMU')
    if len(df_errors) == 0:
        df_CMU['text'] = sentences
    df_CMU.to_csv('prefill_CMU_cmuarctic.csv')
    df_CMU = pd.read_csv('prefill_CMU_cmuarctic.csv')

    df_MFA_IPA_dict = pd.read_csv('prefill_MFA_IPA_dict_' + lang + '.csv')
    df_MFA_IPA_dict.dropna().apply(lambda r: r.phonetics.count('|') + 1, axis=1).tolist()


def prefill_mls():
    root_path = "data/MLS_transcripts/"

    sets = ['test', 'dev']
    dirs = [
        "mls_english",
        # "mls_dutch",
        "mls_french",
        # "mls_german",
        # "mls_italian",
        # "mls_polish",
        # "mls_portuguese",
        "mls_spanish",
    ]

    from tqdm import tqdm

    dfs = {}

    for l in tqdm(dirs):
        for s in tqdm(sets):
            f = '/'.join([root_path, l, s, "transcripts.txt"])
            print(f)
            df = pd.read_csv(f, sep='\t', header=None)
            dfs['/'.join([l, s])] = df

    # df2=pd.read_csv('data/MLS_transcripts/unrated_transcripts/french/unrated_dev_transcripts.txt', sep='\t', header=None)

    to_lang = {"mls_french": "fr_FR", "mls_english": "en_US", "mls_spanish": "es_ES"}
    for k in dfs:
        if k.split('/')[0] in to_lang:
            lang = to_lang[k.split('/')[0]]
            print(k)
            df = dfs[k]
            sentences = df.iloc[:, 1].tolist()
            print(k, len(df))
            df_MFA_IPA, df_errors = prefill_content(sentences, lang=lang, mode='MFA_IPA')
            df_MFA_IPA.to_csv(
                'prefill_MFA_IPA_' + k.replace('/', '_') + '_' + lang + '.csv'
            )

    for k in dfs:
        if k.split('/')[0] in to_lang:
            lang = to_lang[k.split('/')[0]]
            df_MFA_IPA = pd.read_csv(
                'prefill_MFA_IPA_' + k.replace('/', '_') + '_' + lang + '.csv'
            )
            import ast

            df_MFA_IPA.n_syl_mismatches = df_MFA_IPA.n_syl_mismatches.apply(
                ast.literal_eval
            )
            # mismatch syllables between text and phonetics, for each phonetics variation
            # df_CMU_mis=df_CMU[df_CMU.n_syl_mismatches.str.len()!=0][["segmented_text","phonetics"]]
            df_MFA_IPA_syl_mis = df_MFA_IPA[df_MFA_IPA.n_syl_mismatches.str.len() != 0][
                ["segmented_text", "phonetics", "n_syl_mismatches"]
            ]
            # print(df_MFA_IPA_syl_mis)

            syl_inconsistencies_MFA = df_MFA_IPA_syl_mis.apply(
                lambda r: (
                    r.segmented_text.split(' ')[r.n_syl_mismatches[0]],
                    r.phonetics.split(' ')[r.n_syl_mismatches[0]],
                ),
                axis=1,
            )
            # print(syl_inconsistencies_MFA)
            # for i in range(int(len(syl_inconsistencies_MFA)/10)): print(syl_inconsistencies_MFA[i*10:(i+1)*10])

            syl_mismatch_tot = df_MFA_IPA.n_syl_mismatches.apply(len).sum()
            n_words_tot = df_MFA_IPA.n_alternatives.apply(len).sum()
            mismatch_rate = syl_mismatch_tot / n_words_tot
            print(k, mismatch_rate)
