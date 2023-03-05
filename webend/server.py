from flask import Flask
from text import vid_to_aud, get_text
from ai import get_rating
from Naked.toolshed.shell import execute_js
import os
import firebase_admin
from firebase_admin import firestore, credentials

cred = credentials.Certificate("wellness-shshacks-firebase-adminsdk-fltv5-97f33d2b36.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

app = Flask(__name__)

@app.route('/')
def index():
    return '<p>Hello World</p>'

# POST after uploading video to dl and do data shit & upload
@app.route('/video/<uid>', methods=['POST'])
def video(uid):
    js_command = 'node storage.js ' + str(uid)
    r = False
    r = os.system(js_command)
    while not r:
        r = os.system(js_command)
    vid_to_aud(f'./videos/{uid}.mp4')
    text = get_text(f'./audio/{uid}.wav')
    rating = get_rating(text)
    arr = db.collection('users').document(uid).get().to_dict()[uid]
    arr.append(rating)
    db.collection('users').document(uid).set({ uid: arr })    
