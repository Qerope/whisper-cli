import argparse
import os
import subprocess
import tempfile
import whisper


def to_wav(path, out):
    subprocess.run(['ffmpeg', '-y', '-i', path, '-ar', '16000', '-ac', '1', out],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('input')
    args = ap.parse_args()

    model = whisper.load_model('base')
    base = os.path.splitext(args.input)[0]
    tmp = tempfile.mkdtemp()
    wav = os.path.join(tmp, 'in.wav')
    to_wav(args.input, wav)
    result = model.transcribe(wav)
    segments = result['segments']
    print(result['text'])


if __name__ == '__main__':
    main()
