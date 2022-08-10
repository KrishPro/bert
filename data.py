"""
Written by KrishPro @ KP

filename: `data.py`
"""

from prepare_data import PrepareData
import torch.utils.data as data

class Dataset(data.Dataset):
    def __init__(self, data_path: str, chunk_size: int = 2 ** 23):
        self.chunk_size = chunk_size
        self.file = open(data_path)

        try:
            self.length = PrepareData._extract_num_sentences(data_path)
        except:
            self.length = 0
            self.file.seek(0)
            for _ in self.file: self.length += 1
            self.file.seek(0)

        self._reset()

    def _load_chunk(self):
        if not hasattr(self, 'previous_left'): self.previous_left = ''
       
        chunk = (self.previous_left + self.file.read(self.chunk_size))
    
        last_new_line_char = chunk.rfind('\n')

        if last_new_line_char >= 0:
            self.previous_left = chunk[last_new_line_char+1:]
            chunk = chunk[:last_new_line_char]
            if not chunk:
                chunk = self.previous_left
                self.previous_left = ''
        else:
            self.previous_left = ''

        chunk = chunk.splitlines()

        try:
            chunk = list( map( lambda line: tuple(map(lambda s: list(map(int, s.strip().split(' '))), line.strip().split('\t'))), filter(lambda c: c and len(c.split('\t')) == 2, chunk) )) 
        except:
            try:
                final_chunk = []
                for c in chunk:
                    ss = c.strip().split('\t')
                    final_c = []
                    if len(ss) == 2:
                        for s in ss:
                            final_s = []
                            for i in s.strip().split(' '):
                                try:
                                    final_s.append(int(i))
                                except:
                                    print(i)
                            final_c.append(final_s)
                 
                        final_chunk.append(tuple(final_c))
                chunk = final_chunk
                del final_chunk, final_c, final_s, ss, c
            except:
                print("[WARN]: A chunk failed to load and is skipped. So, the dataset will have unexpectedly shorter length")
                chunk = self._load_chunk()

        return chunk

    def _reset(self):
        self.current_chunk_start = 0
        self.previous_left = ''
        self.file.seek(0)

        self.current_chunk = self._load_chunk()

        self.current_chunk_end = len(self.current_chunk) - 1

    def load_chunk(self):
        chunk = self._load_chunk()

        # If we've a chunk, Simply load it
        if chunk:
            self.current_chunk_start += len(self.current_chunk)
            self.current_chunk = chunk
            self.current_chunk_end += len(self.current_chunk)
        
        # If we don't, Reset the dataset and stop iterations
        else:
            self._reset()
            raise StopIteration

    
    def __getitem__(self, idx):
        if idx == (self.current_chunk_end + 1):
            self.load_chunk()

        assert self.current_chunk_start <= idx <= self.current_chunk_end, f"Index Error, Sampled out of chunk"

        item = self.current_chunk[idx - self.current_chunk_start]

        return item
    
    def __len__(self):
        return self.length
