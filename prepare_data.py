"""
Written by KrishPro @ KP

Raw data can be downloaded from:
    - https://t.co/J3EaSEgwW0 ('This is a shorterned path, it redirect to the below url')
    - https://battle.shawwn.com/sdb/books1/books1.tar.gz

filename: `prepare_data.py`
"""

import multiprocessing
import random
import shutil
from tokenizers import Tokenizer
import time
from typing import List, Tuple
from tqdm import tqdm
import string
import emoji
import nltk
import os
import re

from vocab import create_vocab

class ExtractEnglish:
    extract_word = lambda txt: re.sub(r'[\s0-9]', '', txt).translate(str.maketrans('', '', string.punctuation)).lower().strip()
    ENGLISH_WORDS = set(nltk.corpus.words.words())
    
    @staticmethod
    def extract_english(text: str):
        return ' '.join([word for word in text.split(' ') if ExtractEnglish.extract_word(word) in ExtractEnglish.ENGLISH_WORDS])

class PrepareData:
    @staticmethod
    def merge_into_one(data_dir: str, output_path: str):
        assert os.path.exists(data_dir), "Data dir doesn't exists"

        assert len(os.listdir(data_dir)) >= 1, "Data dir doesn't contain any file"

        # Creating the parent directories of output_path, if they don't exists
        if not os.path.exists(os.path.dirname(output_path)) and os.path.dirname(output_path): os.makedirs(os.path.dirname(output_path))

        num_sentences = 0
        output_file = open(output_path, 'w')

        # Looping over all the input_files
        for fname in tqdm(os.listdir(data_dir)):
            fpath: str = os.path.join(data_dir, fname)

            # Reading the content of the input_file
            with open(fpath) as file:
                # Appending the content of the input_file to the output_file
                text = file.read()
                output_file.write(text + '\n')
                num_sentences += len(text.split('\n'))

        # Closing the output_file
        output_file.close()

        # Adding the information about the number of sentences to the file name
        os.rename(output_path, PrepareData._embed_num_sentences(output_path, num_sentences))
        return PrepareData._embed_num_sentences(output_path, num_sentences)


    @staticmethod
    def _prepare_sentence(sentence: str):
        sentence = sentence.strip('\n')

        if len(sentence.split()) >= 5:
            sentence = emoji.demojize(sentence)

            # NOTE: Filter everything from this sentence except the language you want
            # For example- The below line filters out everything except english
            sentence = ExtractEnglish.extract_english(sentence)

            return sentence

        else:
            return None

    @staticmethod
    def _extract_num_sentences(file_path: str):
        return int(os.path.splitext(os.path.basename(file_path))[0].split('-')[1])

    @staticmethod
    def _embed_num_sentences(file_path: str, num_sentences: int):
        dir, file_name, file_ext = os.path.dirname(file_path), *os.path.splitext(os.path.basename(file_path))
        return os.path.join(dir, f"{file_name}-{num_sentences}" + file_ext)


    @staticmethod
    def prepare_file(input_path: str, output_path: str, chunk_size: int = 200_000):
        # Creating pool for multiprocessing
        pool = multiprocessing.Pool(processes=os.cpu_count())

        orignal_num_sentences = PrepareData._extract_num_sentences(input_path)
        num_sentences = 0
        input_file, output_file = open(input_path, 'r'), open(output_path, 'w')

        chunk = []
        for sentence in tqdm(input_file, total=orignal_num_sentences):

            # Apending sentences into chunk till the chunk_size is reached
            chunk.append(sentence) 
            if len(chunk) == chunk_size:
                    # Processing the whole chunk parallelly, and filtering None items
                    processed_sentences = filter(lambda x: x, pool.map(PrepareData._prepare_sentence, chunk))
                    
                    output_file.write('\n'.join(processed_sentences) + '\n')
                    num_sentences += len(chunk)

                    chunk = []
        
        pool.close()

        # Adding the information about the number of sentences to the file name
        os.rename(output_path, PrepareData._embed_num_sentences(output_path, num_sentences))
        return PrepareData._embed_num_sentences(output_path, num_sentences)


    @staticmethod
    def prepare(data_dir: str, output_path: str = None, chunk_size = 100_000):
        tmp: List[str] = [str(time.time()) for _ in range(2)]
        if output_path: tmp[1] = output_path

        tmp[0] = PrepareData.merge_into_one(data_dir, tmp[0])
        tmp[1] = PrepareData.prepare_file(tmp[0], tmp[1], chunk_size=chunk_size)
        os.remove(tmp[0])
        return tmp[1]



class ProcessData:
    def __init__(self, raw_sentences: str, output_path: str, vocab_path: str, chunk_size = 512):
        self.raw_sentences = raw_sentences
        self.output_path = output_path
        self.chunk_size = chunk_size

        self.load_tokenizer(vocab_path)

    def load_tokenizer(self, vocab_path: str):
         self.tokenizer: Tokenizer = Tokenizer.from_file(vocab_path)

    def _process_sentence(self, sentence: str) -> Tuple[List[int], List[int]]:
        ids: List[int] = self.tokenizer.encode(sentence).ids
        src, tgt = [], []

        for id in ids:
            # Sampling a random number b/w 0 and 1
            prob = random.random()

            # Masking 15% of tokens
            if prob < 0.15 and (id != self.tokenizer.token_to_id("[SOS]") and id != self.tokenizer.token_to_id("[EOS]")):
                prob /= 0.15

                # Replacing 80% tokens with [MASK] token
                if prob < 0.8: src.append(self.tokenizer.token_to_id("[MASK]"))

                # Replacing 10% tokens with random token
                elif prob < 0.9: src.append(random.randrange(self.tokenizer.get_vocab_size()))

                # Leaving 10% tokens un-changed
                else: src.append(id)

                # Appending actual token to tgt
                tgt.append(id)
            
            else:
                # If the token is decided to be un-masked, the actual token and pad_id are appended to src and tgt respectively.
                src.append(id)
                tgt.append(self.tokenizer.padding['pad_id'])

        return src, tgt

    def process(self):
        pool = multiprocessing.Pool(processes=os.cpu_count())
        num_sentences = 0
        chunk = []

        with open(self.raw_sentences) as input_file, open(self.output_path, 'w') as output_file:
            for sentence in tqdm(input_file, total=PrepareData._extract_num_sentences(self.raw_sentences)):
                for _ in range(int(1/0.15)):

                    chunk.append(sentence.strip('\n'))
                    if len(chunk) == self.chunk_size:

                        processed_pairs: List[Tuple[List[int], List[int]]] = pool.map(self._process_sentence, chunk)

                        output_file.write('\n'.join(['\t'.join([' '.join(map(str, s)) for s in p]) for p in processed_pairs]) + '\n')
                        num_sentences += len(processed_pairs)

                        chunk = []

        pool.close()

        os.rename(self.output_path, PrepareData._embed_num_sentences(self.output_path, num_sentences))
        return PrepareData._embed_num_sentences(self.output_path, num_sentences)



def main(data_dir: str, output_path: str, vocab_path: str, chunk_size=100_000, save_disk=False):
    raw_sentences = PrepareData.prepare(data_dir, chunk_size=chunk_size)
    if save_disk: shutil.rmtree(data_dir)

    create_vocab(raw_sentences, vocab_path)

    data_processor = ProcessData(raw_sentences, output_path, vocab_path, chunk_size=chunk_size)

    output_path = data_processor.process()
    os.remove(raw_sentences)

    print()
    print(f"Output Path ==> {output_path}")
    print(f"Vocab Path  ==> {vocab_path}")









if __name__ == '__main__':
    main('.ignore/books1/epubtxt', 'dataset.txt', 'vocab.json')