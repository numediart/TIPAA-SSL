# API docs for Flowspeech v2

The source of truth for API endpoints (requests and responses) is swagger.
https://spt.flowchase.app/docs

Different endpoint on new version (now on dev)
https://spdev.flowchase.app/docs

Requests and responses should be JSON.
The audio is base64 encoded files from these format: ogg, caf. It can be others as long as it is handled by libsndfile:
http://www.mega-nerd.com/libsndfile/

Mpeg related format are handled thanks to pydub:
https://github.com/jiaaro/pydub


There could still be responses in HTML for unexpected errors (e.g. from nginx, a 502 bad gateway)

For 400 and 500 errors, it will be JSON with only `error` and `status` payloads

Otherwise, there is a `detected` flag and other payloads as described in swagger.
```typescript
type Response = {
    error: bool
    status: string| {"type": string, "value": string,   "traceback":string}
    detected: 'speech'|'nospeech'|'nonsense'|'empty_audio'
 ...
}
```

Hereunder, I will try to describe the mechanics of the different types of endpoint, without detailing types of every parameters, as for that you can refer to the source of truth.

## Definitions
### formatted phonetics
representation of phonetics with different separators. There is no punctuation mark in them except for the hyphen because it is considered inside a word. The separators are:
    - " " to separate words
    - "|" to separate syllables
    - "_" to separate phonemes

However as mentioned, the hyphen stays present for hyphenated words. There is an additional set of symbols (curly brackets "{}") that we use to process compound words.

Examples:
- 22 -> twen|ty-two -> T_W_EH1_N|T_IY0-T_UW1
- 122 -> {one hun|dred twen|ty-two} -> {W_AH1_N HH_AH1_N|D_R_IH0_D T_W_EH1_N|T_IY0-T_UW1}
- UCL -> {UCL} -> {Y_UW1 S_IY1 EH1_L}

### Breath group
A breath group or tone group is a group of words that is expected to be pronounced "in one go", without taking a breath.
Typically, it also means that in terms of intonation, there is no big discontinuity, but a continuous variation.

We assume that in sentences, the pauses between the breath groups are given by dedicated punctuation mrks (".", ",", ":", etc.)

## Stress related endpoints

Study of the stress patterns in phrases. This study can be done at the sentence level or at the word level.

Both endpoints need audio and phonetics as a list of formatted phonetics of each breath group, that we consider limited by punctuation mark.

The sentence stress endpoint will return stress intensities of each word, and grouped by breath group.
The word stress will return stress intensities of each syllable, and grouped by word.

## Phonetic contrast related endpoints

Phonetic contrasts endpoints aim at analyzing a specific part of a word. The needed information is thus the audio, the formatted phonetics, and the information allowing to locate the target.
Depending of the nature of the target, the way of locating this target is different.

- Vowels is the easiest case: We need a word index, a syllable index and the target itslef (the phoneme). In our assumptions, there is always one and only one vowel per syllable. Therefore there is a redundancy here that is used to make sure of a correct identification of the target. (We can check that the vowel in a specified syllable of a specified word is indeed the target)
- Consonants: We also need a word index, a syllable index and the target itslef. But as there can be several occurences of a consonant in a syllable (crisps, clock), we add an additional information, the target occurence index.
- Termination: This type of target is not necessarily a single phoneme. It can be used to analyze ending of words, for example final -ed words (IH0_D, D, T), or final -s (IH0_Z, Z, S) words. We also need a word index, and the target itslef. However we do not need the syllable index, as we assume it is always in the last syllable since it is a termination.
- Cluster: As for termination, this type of target is not necessarily a single phoneme. It is a generalization of "termination", so that it can be either at the sart or the end, and it is at the syllable level instead of the word level. Therefore it means that we can target any group of phoneme at the start or end of any syllable.
The information needed for identifying the target are therefore word index, syllable index, position (start or end), target

This endpoint will be used for /h/ sound pronunciation aspect, and probably consonant clusters afterwards. 
As this feature is quite general it might be used for other things.

### How cluster actually works

First, as intro, in the phonological structure of words, I consider that:

a word is constituted of a sequence of syllables (minimum 1)
every syllable contain exactly 1 vowel (the nucleus) (a diphthong is labeled as 1 vowel)
there can be consonants before and after that vowel (respectively onset and coda), but not necessarily
Vowel and consonant contrasts target only 1 phoneme and assumes its presence and only does a classification, i.e. output the most likely phoneme pronounced by the speaker.

I wanted a way to target several phonemes inside a syllable, because it is what happens for final -ed, final -s, h initial, initial/final consonant clusters, …

Cluster contrasts can target more than 1 phoneme, but also do not assume its presence (i.e., it can output that the phoneme was absent). It is basically a part of a syllable, it can contain onset, onset+nucleus, …

But to be able to locate it, this cluster is either at “start” or at “end” of the syllable. It cannot be a couple phonemes inside the syllable.

#### Examples

crisps → K_R_IH1_S_P_S

possible start clusters: K, K_R, K_R_IH1, …

possible end clusters: S, P_S, S_P_S, …

However, these are not possible: R_IH1, R_IH1_S, IH1_S, IH1_S_P, …

#### How to find it from content info

There is a way to look based on the pronunciation aspect. For vowel contrasts, we defined rules based on a set of possible targets. We have to do something similar.

For h_initial, if I understood and remember correctly what was done:

target in CMU is HH
always in the first syllable
always “start” or this syllable