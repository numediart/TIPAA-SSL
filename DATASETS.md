# Datasets

To test the tech's performance, develop new features, ..., you will need speech datasets.

## Data of actor recordings

Go in this [drive folder](https://drive.google.com/drive/folders/1-c34uCaNL8PvokYFPaqWYin6FGPXq-3t?usp=share_link)

- Download the folder "audio-with-analysis-ids"
  Paste it inside "data/" folder of this repo.
  You can now use `actor_recordings()` function in "flowspeech/label_data_processing.py".
- Create a folder called "synth_audio" in "data/" and drop the folder "cmu_words" in it after extracting the zip
- Drop "flwc-phrase-audios" after extracting the zip

For extracting a big zip, windows zip extractor might crash (although it was able to create it...). If that's the case "unzip" command on a WSL Ubuntu (install with `sudo apt install unzip`) should work.

## Librispeech data

https://www.openslr.org/12/

I use dev-clean and test-clean sets.

```
cd data
wget https://www.openslr.org/resources/12/dev-clean.tar.gz
tar xvfz dev-clean.tar.gz
rm dev-clean.tar.gz
wget https://www.openslr.org/resources/12/test-clean.tar.gz
tar xvfz test-clean.tar.gz
rm test-clean.tar.gz
cd ..
```

Get phonetic alignments data:

```
sudo apt install unzip
cd data
mkdir librispeech_alignments
cd librispeech_alignments
curl https://zenodo.org/record/2619474/files/librispeech_alignments.zip?download=1 --output librispeech_alignments.zip
unzip librispeech_alignments.zip
rm librispeech_alignments.zip
cd ..
cd ..
```

## MAILABS dataset

Download data archives from [MAILABS website](https://www.caito.de/2019/01/03/the-m-ailabs-speech-dataset/#:~:text=Statistics%20%26%20Download%20Links)
Download the ones with the tags "en_UK", "en_US" in priority. (You can also download "es_ES" and "fr_FR" for experimenting later, but not necessary right now)

Commands for doing it for "en_UK" if you are at the root of the repo:

```
cd data
mkdir MAILABS
cd MAILABS
wget https://data.solak.de/data/Training/stt_tts/en_UK.tgz
tar xvfz en_UK.tgz
rm en_UK.tgz
cd ../..
```

## Other sets of data

There are other useful sets of data, we don't have a place for them online yet. Ask Noé