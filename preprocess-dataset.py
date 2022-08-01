"""
Written by KrishPro @ KP

filename: `preprocess_dataset.py` 
"""

from tqdm import tqdm
import emoji
import os


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


        return text
    
    else:
        return None


def main(data_path: str, output_path: str,):
    with open(output_path, 'w') as output_file:

        input_file = open(data_path)

        for sentence in tqdm(input_file):
            sentence = process_text(sentence)
            if sentence:
                output_file.write(sentence)


if __name__ == "__main__":
    # the input file can contain any type of text, but the output file will contain pure text no un-necessary newlines, no emojis, no short sentences
    # main("input.txt", "output.txt")
    pass
