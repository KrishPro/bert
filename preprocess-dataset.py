"""
Written by KrishPro @ KP

filename: `preprocess_dataset.py` 
"""

import random
from typing import Set
from tqdm import tqdm
import string
import emoji
import nltk
import os
import re


class ExtractEnglish:
    extract_word = lambda txt: re.sub(r'[\s0-9]', '', txt).translate(str.maketrans('', '', string.punctuation)).lower().strip()
    ENGLISH_WORDS: Set[str] = set(nltk.corpus.words.words())
    
    @staticmethod
    def extract_english(text: str):
        return ' '.join([word for word in text.split(' ') if ExtractEnglish.extract_word(word) in ExtractEnglish.ENGLISH_WORDS])


def merge_into_one(data_dir: str, output_path: str):
    """
    This function can be used, in-case your is corpus already divided into multiple files but you want to merge them and divide as-per your requirements
    """
    with open(output_path, 'a') as dataset:

        for file_name in enumerate(tqdm(os.listdir(data_dir))):
            file_path = os.path.join(data_dir, file_name)

            with open(file_path, 'r') as file:
                text = file.read()
                
            dataset.write(text + '\n')

def process_text(text: str):
    """
    This function will process the text and return it
    """

    # Filtering sentences shorter than 5 words
    if len(text.split()) >= 5:

        # Removing new-line if it comes in the staring of the sentences
        text.lstrip('\n')

        # Removes emojis
        text = emoji.demojize(text)

        # NOTE: Modify this file to filter everything except text in the language you want
        # For Example- THe below line filters out everything non-english
        # text = ExtractEnglish.extract_english(text)

        return text
    
    else:
        return None


def main(data_path: str, output_path: str, chunk_size: int = 20_000):

    with open(output_path, 'w') as output_file:

        input_file = open(data_path)

        chunk = []
        num_sentences = 0
        for sentence in tqdm(input_file):
            sentence = process_text(sentence)
            if sentence:
                chunk.append(sentence)
                if len(chunk) == chunk_size:
                    random.shuffle(chunk)
                    output_file.write(''.join(chunk))
                    num_sentences += len(chunk)
                    chunk = []

    # Adding the information about the number of sentences to the file name
    output_dir, output_name, output_ext = os.path.dirname(output_path), *os.path.splitext(os.path.basename(output_path))
    os.rename(os.path.join(output_dir, output_name+output_ext), os.path.join(output_dir, output_name+f'-{num_sentences}'+output_ext))


if __name__ == "__main__":
    # the input file can contain any type of text, but the output file will contain pure text no un-necessary newlines, no emojis, no short sentences
    # main("input.txt", "output.txt")
    pass
