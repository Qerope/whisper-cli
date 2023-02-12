import argparse
import os
import subprocess
import tempfile
import math
from faster_whisper import WhisperModel

def fmt_ts(t, sep=','):
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return '%02d:%02d:%06.3f' % (h, m, s) if sep == '.' else \
        ('%02d:%02d:%02d,%03d' % (h, m, int(s), round((s - int(s)) * 1000)))

def to_wav(path, out):
    subprocess.run(['ffmpeg', '-y', '-i', path, '-ar', '16000', '-ac', '1', out],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def duration(path):
    out = subprocess.check_output(['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                                   '-of', 'default=nw=1:nk=1', path])
    return float(out)


def split(wav, chunk, outdir):
    total = duration(wav)
    parts = []
    for i in range(math.ceil(total / chunk)):
        part = os.path.join(outdir, 'part%03d.wav' % i)
        subprocess.run(['ffmpeg', '-y', '-ss', str(i * chunk), '-t', str(chunk), '-i', wav, part],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        parts.append((i * chunk, part))
    return parts

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('input')
    ap.add_argument('--model', default='base')
    ap.add_argument('--language')
    args = ap.parse_args()

    model = WhisperModel(args.model, device='cpu', compute_type='int8')
    base = os.path.splitext(args.input)[0]
    tmp = tempfile.mkdtemp()
    wav = os.path.join(tmp, 'in.wav')
    to_wav(args.input, wav)
    segments = []
    for offset, part in split(wav, 600, tmp):
        it, info = model.transcribe(part, language=args.language
            , vad_filter=True
            )
        for seg in it:
            segments.append((seg.start + offset, seg.end + offset, seg.text.strip()))
        print('chunk at %ds done' % offset, flush=True)
    with open(base + '.srt', 'w') as f:
        for i, (a, b, text) in enumerate(segments, 1):
            f.write('%d\n%s --> %s\n%s\n\n' % (i, fmt_ts(a), fmt_ts(b), text.strip()))


if __name__ == '__main__':
    main()
