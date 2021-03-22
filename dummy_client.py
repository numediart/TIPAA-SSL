# import requests
# def send_audio():
#     #print('attempting to send audio')
#     url = 'http://127.0.0.1:5000/'
#     with open('audio_recordings/WS_111_toothpaste.wav', 'rb') as file:
#         data = {'uuid':'-jx-1', 'alarmType':1, 'timeDuration':10}
#         files = {'messageFile': file}

#         req = requests.post(url, files=files, json=data)
#         print(req.status_code)
#         print(req.text)

import requests
res = requests.post('http://localhost:5000/api/add_message/sentenceStress', json={"filename":"SS_1_i_would_love_to_go_to_ireland.wav", "sentenceID":1})
print(res.__dict__)