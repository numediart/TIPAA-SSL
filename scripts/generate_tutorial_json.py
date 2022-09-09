import json
import os
import requests
from text_processing import remove_special_characters
from bs4 import BeautifulSoup
special_chars_dict={'&#8230;':'…','&#160;&#187;':'»','&#171;&#160;':'«'}

# No… 🤔
# Yup! 😉
# a=" No… 🤔"

def rgb_to_hex(r,g,b):
    # check if in range 0~255
    assert 0 <= r <= 255
    assert 0 <= g <= 255
    assert 0 <= b <= 255
 
    r = hex(r).lstrip('0x')
    g = hex(g).lstrip('0x')
    b = hex(b).lstrip('0x')
    # re-write '7' to '07'
    r = (2 - len(r)) * '0' + r
    g = (2 - len(g)) * '0' + g
    b = (2 - len(b)) * '0' + b
 
    hex_color = '#' + r + g + b
    return hex_color

def parse_html(path='data/Focus_words_introduction_1___48.html'):
    tuto_id=path.split('/')[-1].split('.')[0]

    with open(path, "r") as f: code=f.read()

    soup = BeautifulSoup(code)
    body = soup.find('body')

    elements=str(body).split('\n')
    phrases=[el.strip('\t').strip('</span>').replace('</span>','') for i,el in enumerate(elements) if '<span>' in el]

    for phrase in phrases:
        pieces=phrase.split('<')
        # print(len(phrase.split('<')))
        words=[]
        colors=[]
        areBold=[]
        areUnderline=[]
        for piece in pieces:
            text=piece.replace('span style=','').replace('span>','')
            idx=text.find('>')
            tag=''
            color='white' #default to be overwritten
            if idx!=-1:
                tag=text[:idx]
                text=text[idx+1:]
            if 'color:rgba' in tag:
                rgb_start=tag.find('(')
                rgb_end=tag.find(')')
                rgb=tag[rgb_start+1:rgb_end]
                rgba_int=[int(el) for el in rgb.split(',')]
                r,g,b=rgba_int[0],rgba_int[1],rgba_int[2]
                color=rgb_to_hex(r,g,b)
            if 'uppercase' in tag: text=text.upper()
            isBold='false'
            if 'bold' in tag: isBold='true'
            isUnderline='false'
            if 'underline' in tag: isUnderline='true'
            # print('tag:',tag)
            n_words=len(text.strip(' ').split(' '))
            words+=text.strip(' ').split(' ')
            colors+=[color]*n_words
            areBold+=[isBold]*n_words
            areUnderline+=[isUnderline]*n_words

        print(words)
        print(colors)
        print(areBold)

        concatenation=concat_from_words(words, colors, areBold, areUnderline)

        
        filename='_'.join(words)
        filename=filename.replace('/','-').replace(':','')
        filename=remove_special_characters(filename).encode('ascii','ignore').decode('ascii')+'.json'
        outpath='/'.join(["data",tuto_id])
        if not os.path.exists(outpath):os.makedirs(outpath)
        print(filename)
        text_file = open(outpath+'/'+filename, "w")
        n = text_file.write(concatenation)
        text_file.close()

def concat_from_words(words, colors, areBold, areUnderline):
    concatenation=""
    for word, color, isBold, isUnderline in zip(words,colors,areBold,areUnderline):
        template='''
                            {
                                "id": uuidv4(),
                                "word": "'''+ word +'''",
                                "color": "'''+color+'''",
                                "isBold": '''+isBold+''',
                                "isUnderline": '''+isUnderline+''',
                                "creationDate": new Date(),
                                "modificationDate": new Date()
                            },'''
        concatenation+=template
    concatenation=concatenation[:-1]
    text_intro='''                    "text": {
                            "id": uuidv4(),
                            "words": ['''
    text_outro='''],
                            "creationDate": new Date(),
                            "modificationDate": new Date()
                        }'''
    concatenation=text_intro+concatenation+text_outro

    return concatenation

def validate_json(path="data/Tutorials 1.0/JSON Final -ED/page100/when_talking_about_the_past_learners_of_english_often_forget_to_pronounce_the_-ed_ending_of_the_verbs_they_use.json"):
    with open(path, 'r') as file:
        text=file.read()
    text=text.replace('uuidv4()', '"uuidv4()"').replace('new Date()','"new Date()"').replace('"text": ','')
    d=json.loads(text)

if __name__=='__main__':
    sentence="Hello world, how are you?"
    len_words=len(sentence.split(' '))
    colors=["white"]*len_words
    areBold=["false"]*len_words
    areUnderline=["false"]*len_words

    concatenation=concat_from_words(sentence.split(' '), colors, areBold, areUnderline)

    text_file = open("concatenation.ts", "w")
    n = text_file.write(concatenation)
    text_file.close()

    parse_html(path='data/Focus_words_introduction_2___1.html')
    parse_html(path='data/Focus_words_introduction_1___31.html')
    parse_html(path='data/Focus_words_introduction_1___48.html')
    parse_html(path='data/Focus_words_introduction_1___49.html')
    parse_html(path='data/Focus_words_introduction_1___50.html')
    parse_html(path='data/Focus_words_introduction_1___51.html')

    import glob
    path_htmls=glob.glob('data/Tutorials 1.0/HTML Final -ED/*')
    path_htmls=glob.glob('data/Tutorials 1.0/HTML VC02-lowlaw/*')

    for p in path_htmls:
        parse_html(p)
