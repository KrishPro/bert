"""
Written by KrishPro @ KP

filename: `prepare_data.py`
"""

import multiprocessing
import os
from tqdm import tqdm

import string
import emoji
import nltk
import re

from utils import embed_num_sentences_into_file, ensure_parent_dir, extract_num_sentences


class ExtractEnglish:
    extract_word = lambda txt: re.sub(r'[\s0-9]', '', txt).translate(str.maketrans('', '', string.punctuation)).lower().strip()
    ENGLISH_WORDS = set(nltk.corpus.words.words())
    
    @staticmethod
    def extract_english(text: str):
        return ' '.join([word for word in text.split(' ') if ExtractEnglish.extract_word(word) in ExtractEnglish.ENGLISH_WORDS])


def prepare_text(text: str) -> str:
    text = text.strip()
    text = emoji.demojize(text)
    text = ExtractEnglish.extract_english(text)
    if len(text.split(' ')) >= 5:
        return text
    else:
        return None

def prepare_file(input_path: str, output_path: str, chunk_size = 512):
    ensure_parent_dir(output_path)

    num_sentences = extract_num_sentences(input_path)
    pool = multiprocessing.Pool(processes=os.cpu_count())
    chunk = []

    with open(input_path) as input_file:
        with open(output_path, 'w') as output_file:
        
            for sentence in tqdm(input_file, total=num_sentences, desc=f"Processing {input_path}"):
                chunk.append(sentence)

                if len(chunk) == chunk_size:
                    chunk = list(filter(lambda x: bool(x), pool.map(prepare_text, chunk)))
                    output_file.write('\n'.join(chunk) + '\n')
                    chunk = []

            chunk = list(filter(lambda x: bool(x), pool.map(prepare_text, chunk)))
            output_file.write('\n'.join(chunk) + '\n')

    embed_num_sentences_into_file(output_path)
