import pickle
import torch

class TextPreprocessor:
    """Converts text to numerical format for neural network"""

    def __init__(self, max_words=10000, max_len=100):
        self.max_words = max_words
        self.max_len = max_len
        self.word_to_idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx_to_word = {0: "<PAD>", 1: "<UNK>"}

    def build_vocab(self, texts):
        """Build vocabulary from texts"""
        word_freq = {}
        for text in texts:
            for word in text.lower().split():
                word_freq[word] = word_freq.get(word, 0) + 1

        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        for idx, (word, _) in enumerate(sorted_words[:self.max_words-2], start=2):
            self.word_to_idx[word] = idx
            self.idx_to_word[idx] = word

    def text_to_sequence(self, text):
        """Convert text to sequence of indices"""
        words = text.lower().split()
        sequence = [self.word_to_idx.get(word, 1) for word in words[:self.max_len]]

        if len(sequence) < self.max_len:
            sequence += [0] * (self.max_len - len(sequence))

        return torch.tensor(sequence, dtype=torch.long)

    def save(self, path):
        """Save preprocessor state"""
        with open(path, 'wb') as f:
            pickle.dump({
                'word_to_idx': self.word_to_idx,
                'idx_to_word': self.idx_to_word,
                'max_words': self.max_words,
                'max_len': self.max_len
            }, f)
        print(f"Vocabulary saved to {path}")

    def load(self, path):
        """Load preprocessor state"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.word_to_idx = data['word_to_idx']
            self.idx_to_word = data['idx_to_word']
            self.max_words = data['max_words']
            self.max_len = data['max_len']
        print(f"Vocabulary loaded from {path}")
