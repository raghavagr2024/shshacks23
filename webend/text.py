from pydub import AudioSegment
import speech_recognition as sr
from argparse import ArgumentParser
from pathlib import Path
from uuid import uuid1
from os import remove, devnull, path
import json
import sys
from moviepy.editor import VideoFileClip

parser = ArgumentParser()
parser.add_argument('--filename', '-f', type=str)

def vid_to_aud(f):
    filename, _ = path.splitext(f)
    vid = VideoFileClip(f)
    vid.audio.write_audiofile(f'./audio/{filename}.wav')
    remove(f)

def convert(filepath: str) -> str:
    sound = AudioSegment.from_mp3(filepath)
    output = str(uuid1())
    output = f'{output}.wav'
    sound.export(output, format='wav')
    #remove(filepath)
    return output

def get_text(filepath: str) -> str:
    listener = sr.Recognizer()

    with sr.AudioFile(filepath) as audio:
        atext = listener.listen(audio)
        try:
            with open(devnull, "w") as f:
                out = sys.stdout
                sys.stdout = f
                text = listener.recognize_google(atext)
                sys.stdout = out
        except:
            print('error')
    remove(filepath)
    return text

if __name__ == '__main__':
    args = parser.parse_args()
    txt = get_text( convert( args.filename ) )
    print(txt)
