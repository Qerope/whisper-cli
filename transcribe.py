import argparse
import os
import subprocess
import tempfile
import whisper

def fmt_ts(t, sep=','):
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return '%02d:%02d:%06.3f' % (h, m, s) if sep == '.' else \
        ('%02d:%02d:%02d,%03d' % (h, m, int(s), round((s - int(s)) * 1000)))

def to_wav(path, out):
    subprocess.run(['ffmpeg', '-y', '-i', path, '-ar', '16000', '-ac', '1', out],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('input')
    ap.add_argument('--model', default='base')
    args = ap.parse_args()

    model = whisper.load_model('base')
    model = whisper.load_model(args.model)
    base = os.path.splitext(args.input)[0]
    tmp = tempfile.mkdtemp()
    wav = os.path.join(tmp, 'in.wav')
    to_wav(args.input, wav)
    result = model.transcribe(wav)
    segments = result['segments']
    print(result['text'])
    with open(base + '.srt', 'w') as f:
        for i, (a, b, text) in enumerate(segments, 1):
            f.write('%d\n%s --> %s\n%s\n\n' % (i, fmt_ts(a), fmt_ts(b), text.strip()))


if __name__ == '__main__':
    main()
