# whisper-cli

    pip install -r requirements.txt
    python transcribe.py talk.m4a --model small --language en

Writes `talk.srt`, `talk.vtt` and `talk.txt` next to the input. Uses
faster-whisper (CTranslate2) so it runs on CPU in int8 without needing torch.
Recordings longer than 10 minutes are split into chunks so memory stays flat;
a word cut at a chunk boundary can be repeated or dropped.

`ffmpeg` needs to be on PATH.
