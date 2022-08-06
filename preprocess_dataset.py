"""
Written by KrishPro @ KP

filename: `preprocess_dataset.py` 
"""

from multiprocessing import Pool
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

    # Creating multiprocessing pool
    pool = Pool(processes=os.cpu_count())

    # Opening the output file, it will be opened through out the process. We'll be writting sentences as we want
    with open(output_path, 'w') as output_file:

        # Opening the input file, we'll be looping over it's sentences
        input_file = open(data_path)

        # Creating a blank chunk
        chunk = []

        # Creating a sentence counter
        num_sentences = 0

        # Looping over each sentence of the input file
        for sentence in tqdm(input_file):

            # Appening the sentence to the chunk
            chunk.append(sentence)

            # If the chunk_size has reached a pre-setted size, we'll shuffle it and write it on the output_file
            if len(chunk) == chunk_size:

                # Shuffling the chunk
                random.shuffle(chunk)

                # Processing sentences parrelly
                chunk = pool.map(process_text, chunk)
                chunk = [s for s in chunk if s] # Filtering None items from the list

                # Joining all the sentences of the chunk
                # NOTE: All the sentences end with a new_line, so we can just join them using a blank str
                output_file.write(''.join(chunk))

                # Increasing the sentence count by the chunk size
                num_sentences += len(chunk)

                # Resetting the chunk
                chunk = []

    # Adding the information about the number of sentences to the file name
    output_dir, output_name, output_ext = os.path.dirname(output_path), *os.path.splitext(os.path.basename(output_path))
    os.rename(os.path.join(output_dir, output_name+output_ext), os.path.join(output_dir, output_name+f'-{num_sentences}'+output_ext))


if __name__ == "__main__":
    # the input file can contain any type of text, but the output file will contain pure text no un-necessary newlines, no emojis, no short sentences
    # main("input.txt", "output.txt")
    pass
