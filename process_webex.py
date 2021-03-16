#!/usr/bin/env python

import argparse
import fnmatch
import json
import logging
import os
import subprocess
import tarfile
import urllib.request
from pathlib import Path

from sox import Transformer
from tqdm import tqdm

logging.basicConfig(level=logging.DEBUG)


def get_args():
    parser = argparse.ArgumentParser(description='Process webex data')
    parser.add_argument("--data_root", required=True, default=None, type=str)
    parser.add_argument("--data_set", default="webex.tst", type=str)
    args = parser.parse_args()
    return(args)


def process_data(data_folder: str, dst_folder: str, manifest_file: str):
    """
    Converts flac to wav and build manifests's json
    Args:
        data_folder: source with flac files
        dst_folder: where wav files will be stored
        manifest_file: where to store manifest
    Returns:
    """

    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)

    files = []
    entries = []
    print(data_folder)
    # get list of .wrd transcripts
    for root, dirnames, filenames in os.walk(data_folder):
        for filename in fnmatch.filter(filenames, '*.wrd'):
            files.append((os.path.join(root, filename), root))

    for transcripts_file, root in tqdm(files):
        with open(transcripts_file, encoding="utf-8") as fin:
            for line in fin:
                transcript_text = line.lower().strip()
                id = Path(transcripts_file).stem
                # Convert FLAC file to WAV
                flac_file = os.path.join(root, id + ".flac")
                wav_file = os.path.join(dst_folder, id + ".wav")
                if not os.path.exists(wav_file):
                    Transformer().build(flac_file, wav_file)
                # check duration
                duration = subprocess.check_output("soxi -D {0}".format(wav_file), shell=True)

                entry = {}
                entry['audio_filepath'] = os.path.abspath(wav_file)
                entry['duration'] = float(duration)
                entry['text'] = transcript_text
                entries.append(entry)

    with open(manifest_file, 'w') as fout:
        for m in entries:
            fout.write(json.dumps(m) + '\n')



def main():
    args = get_args()
    data_root = args.data_root
    data_set = args.data_set

    logging.info("\n\nWorking on: {0}".format(data_set))

    logging.info("Processing {0}".format(data_set))
    process_data(
        os.path.join(data_root, data_set),
        os.path.join(os.path.join(data_root, data_set), "wav"),
        os.path.join(data_root, data_set + ".json"))
    logging.info('Done!')

if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter