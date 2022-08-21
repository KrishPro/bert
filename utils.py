"""
Written by KrishPro @ KP

filename: `utils.py`
"""

from typing import Tuple, Union
from tqdm import tqdm
import os


def count_sentences(file_path: str, predicted_num_sentences: int = None) -> int:
    r"""Counts the number of sentences in a file, Usually does within 2 minutes"""

    num_sentences = 0

    with open(file_path) as file:
        for _ in tqdm(file, total=predicted_num_sentences, desc=f"Counting sentences ({file_path})"): num_sentences += 1

    return num_sentences


def split_path(file_path: str) -> Tuple[str, str, str]:
    f"""Splits the path into- dirname, filename and extension"""
    return (os.path.dirname(file_path), *os.path.splitext(os.path.basename(file_path)))


def extract_num_sentences(file_path: str, return_file_path=False) -> Union[Tuple[str, int], int]:

    dirname, filename, extension = split_path(file_path)

    try:
        num_sentences = int(filename.split('-')[-1])
        filename = '-'.join(filename.split('-')[:-1])

    except:
        num_sentences = None

    if return_file_path:
        return os.path.join(dirname, filename+extension), num_sentences
    else:
        return num_sentences


def embed_num_sentences(file_path: str, num_sentences: int) -> str:
        
    dirname, filename, extension = split_path(file_path)

    filename = f"{filename}-{num_sentences}"

    return os.path.join(dirname, filename+extension)


def ensure_dir(dir_path: str):
    if not os.path.exists(dir_path) and dir_path:
        os.makedirs(dir_path)


def ensure_parent_dir(file_path: str):
    ensure_dir(os.path.dirname(file_path))


def merge_into_one(input_dir: str, output_path: str):
    r"""Assuming that input_dir contains multiple text files, This function merge the content of all files and write them into output_path"""

    ensure_parent_dir(output_path)
    assert os.path.exists(input_dir), "The input dir doesn't exists"

    input_files = [os.path.join(input_dir, file) for file in os.listdir(input_dir) if os.path.splitext(file)[1] == '.txt'] 

    with open(output_path, 'w') as output_file:
        for input_file_path in tqdm(input_files, desc=f"Merging ({input_dir})"):
            with open(input_file_path) as input_file:
                output_file.write(input_file.read() + '\n')

    output_path = embed_num_sentences_into_file(output_path)

    return output_path


def embed_num_sentences_into_file(file_path: str):
    r"""Makes sure that the file_path contains correct count of the sentences and returns the new_path"""
    orignal_file_path = file_path
    num_sentences = extract_num_sentences(file_path)
    num_sentences = count_sentences(file_path, num_sentences)
    file_path = embed_num_sentences(file_path, num_sentences)
    os.rename(orignal_file_path, file_path)
    return file_path
