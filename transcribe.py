import argparse
import os
import whisper


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('input')
    args = ap.parse_args()

    model = whisper.load_model('base')
    result = model.transcribe(args.input)
    print(result['text'])


if __name__ == '__main__':
    main()
