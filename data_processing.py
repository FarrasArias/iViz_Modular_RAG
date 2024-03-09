import numpy as np


class DataProcessor:
    def __init__(self):
        pass
    
    def split_text(self, path, separator=None, chunk_length=100, force_length=False):
        """
        Splits a text file into a numpy array of substrings based on a separator or chunk length.
        
        :param path: Path to the .txt file.
        :param separator: The separator to split the text. If None, uses chunk_length.
        :param chunk_length: The length of each chunk if separator is not used.
        :param force_length: If True, forces splitting by chunk_length even when separator is found.
        :return: Numpy array of substrings.
        """
        with open(path, 'r', encoding='utf-8') as file:
            text = file.read()
        
        if separator and not force_length:
            chunks = text.split(separator)
        ####TODO: NOT WORKING PROPERLY YET
        elif separator and force_length: 
            chunks = []
            temp_chunk = ''
            for part in text.split(separator):
                temp_chunk += part
                while len(temp_chunk) >= chunk_length:
                    chunks.append(temp_chunk[:chunk_length])
                    temp_chunk = temp_chunk[chunk_length:]
                temp_chunk += separator  
            if temp_chunk:  
                chunks.append(temp_chunk)
        ####
        else:
            chunks = [text[i:i+chunk_length] for i in range(0, len(text), chunk_length)]
        
        return np.array(chunks)