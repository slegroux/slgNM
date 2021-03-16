#!/usr/bin/env python
# (c) 2021 Sylvain Le Groux <slegroux@ccrma.stanford.edu>

import argparse
import fnmatch
import json
import logging
import os
import subprocess
import tarfile
import urllib.request
from pathlib import Path
import pandas as pd
import multiprocessing as mp
from sox import Transformer
from tqdm import tqdm
import swifter

logging.basicConfig(level=logging.DEBUG)

def get_args():
    parser = argparse.ArgumentParser(description='Process webex400 data')
    parser.add_argument("--data_root", required=True, default=None, type=str)
    parser.add_argument("--data_set", default="webex.tst", type=str)
    args = parser.parse_args()
    return(args)

def flac2wav(flac_file:str, wav_file:str):
    if not os.path.exists(wav_file):
        Transformer().build(flac_file, wav_file)

def process_data(data_folder: str, data_set:str, dst_folder: str, manifest_file: str):
    """
    process
    """

    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)    

    df = pd.read_csv(os.path.join(data_folder,data_set), sep='|', names=['path', 'transcript'])
    df['path'] = df['path'].swifter.apply(lambda x: os.path.join(data_folder,x))
    df['wav_path'] = df['path'].replace('flac','wav',regex=True)
    # TODO: parallelize duration
    df['duration'] = df['path'].apply(lambda x: float(subprocess.check_output("soxi -D {0}".format(x), shell=True)))

    path_pair = zip(df['path'], df['wav_path'])
    
    num_processes = max(1,mp.cpu_count()-2)
    with mp.Pool(num_processes) as pool:
        list(tqdm(pool.starmap(flac2wav, path_pair), total=len(df)))

    entries = []
    for _, row in df.iterrows():
        entry = {}
        entry['audio_filepath'] = row['wav_path']
        entry['duration'] = row['duration']
        entry['text'] = row['transcript']
        entries.append(entry)

    with open(manifest_file, 'w') as fout:
        for m in entries:
            fout.write(json.dumps(m) + '\n')


def main():
    args = get_args()
    data_root = args.data_root
    data_set = args.data_set
    test_id = Path(data_set).stem
    
    logging.info("\n\nWorking on: {0}".format(data_set))
    logging.info("Processing {0}".format(data_set))

    process_data(
        data_root,
        data_set,
        os.path.join(data_root, "wav"),
        os.path.join(data_root, test_id) + ".json"
    )
    logging.info('Done!')

if __name__ == '__main__':
    main()
