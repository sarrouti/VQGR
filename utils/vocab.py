"""
Created on Tue Jun 23 20:15:11 2020

@author: sarroutim2
"""

"""Creates a vocabulary using vqg_dataset for the vqa dataset.
"""

from collections import Counter
from utils import Vocabulary
#train_
import argparse
import json
import logging
import nltk
import numpy as np
import re
import base64

def process_text(text, vocab, max_length=20):
    """Converts text into a list of tokens surrounded by <start> and <end>.

    Args:
        text: String text.
        vocab: The vocabulary instance.
        max_length: The max allowed length.

    Returns:
        output: An numpy array with tokenized text.
        length: The length of the text.
    """

    tokens = tokenize(text.lower().strip())
    output = []
    output.append(vocab(vocab.SYM_SOQ))  # <start>
    output.extend([vocab(token) for token in tokens])
    output.append(vocab(vocab.SYM_EOS))  # <end>
    length = min(max_length, len(output))

    return np.array(output[:length]), length


def load_vocab(vocab_path):
    """Load Vocabulary object from a pickle file.

    Args:
        vocab_path: The location of the vocab pickle file.

    Returns:
        A Vocabulary object.
    """
    vocab = Vocabulary()
    vocab.load(vocab_path)
    return vocab


def tokenize(sentence):
    """Tokenizes a sentence into words.

    Args:
        sentence: A string of words.

    Returns:
        A list of words.
    """
    if len(sentence) == 0:
        return []
    if isinstance(sentence,str):
        sentence=sentence
    else:
        sentence=sentence.decode('utf-8')
    sentence = re.sub('\.+', r'.', sentence)
    sentence = re.sub('([a-z])([.,!?()])', r'\1 \2 ', sentence)
    sentence = re.sub('\s+', ' ', sentence)

    tokens = nltk.tokenize.word_tokenize(
            sentence.strip().lower())
    return tokens


def build_vocab(questions, threshold):
    """Build a vocabulary from the annotations.

    Args:
        annotations: A json file containing the questions and images.
        threshold: The minimum number of times a work must occur. Otherwise it
            is treated as an `Vocabulary.SYM_UNK`.

    Returns:
        A Vocabulary object.
    """
    with open(questions) as f:
        questions = json.load(f)
    
    words = []
    counter = Counter()
    for i, entry in enumerate(questions):
        question = entry["question"].encode('utf8')
        q_tokens = tokenize(question)
        counter.update(q_tokens)
        if i % 1000 == 0:
            logging.info("Tokenized %d questions." % (i))


    # If a word frequency is less than 'threshold', then the word is discarded.
    words.extend([word for word, cnt in counter.items() if cnt >= threshold])
    words = list(set(words))
    vocab = create_vocab(words)
    return vocab


def create_vocab(words):
    # Adds the words to the vocabulary.
    vocab = Vocabulary()
    for i, word in enumerate(words):
        vocab.add_word(word)
    return vocab


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Inputs.
    parser.add_argument('--questions', type=str,
                        default='data/vqg/'
                        'train_questions_RAD_captions__titles_v1.json',
                        help='Path for train questions file.')
    # Hyperparameters.
    parser.add_argument('--threshold', type=int, default=1,
                        help='Minimum word count threshold.')
    # Outputs.
    parser.add_argument('--vocab-path', type=str,
                        default='data/processed/vocab_vqg.json',
                        help='Path for saving vocabulary wrapper.')
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(level=logging.INFO)
    vocab = build_vocab(args.questions,args.threshold)
    logging.info("Total vocabulary size: %d" % len(vocab))
    vocab.save(args.vocab_path)
    logging.info("Saved the vocabulary wrapper to '%s'" % args.vocab_path)
