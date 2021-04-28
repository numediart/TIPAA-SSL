import os

text='<speak>Hello world</speak>'
cmd= "aws polly synthesize-speech \
--text-type ssml \
--text "+text+" \
--output-format mp3 \
--voice-id Joanna \
speech.mp3"