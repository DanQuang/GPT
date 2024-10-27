import re

# Define a simple tokenizer, if not using BPE (Byte Pair Encoding) for GPT models
class SimpleTokenizer:
    # Constructor takes a vocabulary (a dictionary mapping strings to integers)
    def __init__(self, vocab: dict):
        # Store the string-to-integer mapping for encoding text
        self.str_to_int = vocab
        # Create a reverse mapping (integer-to-string) for decoding text
        self.int_to_str = {i: s for s, i in vocab.items()}
    
    # Method to encode a string into a list of integers (token IDs)
    def encode(self, text: str) -> list[int]:
        # Split the text into tokens using regular expressions. It captures
        # punctuation marks like , . : ; ? _ ! " ' () and spaces as separate tokens.
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)

        # Strip whitespaces from each token and filter out empty strings
        preprocessed = [item.strip() for item in preprocessed if item.strip()]

        # Replace any token not in the vocabulary with a special unknown token "<|unk|>"
        preprocessed = [
            item if item in self.str_to_int 
            else "<|unk|>" for item in preprocessed
        ]

        # Convert the tokens into their corresponding integer IDs
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids  # Return the list of token IDs
        
    # Method to decode a list of token IDs back into a string
    def decode(self, ids: list[int]) -> str:
        # Convert the list of token IDs back into strings using the reverse mapping
        text = " ".join([self.int_to_str[i] for i in ids])

        # Remove extra spaces before punctuation marks like , . : ; ? ! " ( )
        # This helps to clean up the tokenized text into a readable format
        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
        return text  # Return the decoded string
