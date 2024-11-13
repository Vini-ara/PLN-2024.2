import os
import json

class Tokenizer:
    def __init__(self):
        return
    
    def get_stats(self, textIds):
        counter = {}
        for pair in zip(textIds, textIds[1:]):
            counter[pair] = counter.get(pair, 0) + 1

        return counter

    def merge(self, textIds, pair, newPairIdx):
        newIds = []
        i = 0
        while i < len(textIds):
            if i < len(textIds) - 1 and (textIds[i], textIds[i + 1]) == pair:
                newIds.append(newPairIdx)
                i += 2
            else:
                newIds.append(textIds[i])
                i += 1

        return newIds

    def create_merges(self, textIds, maxVocabSize):
        merges = {}
        num_merges = maxVocabSize - 256
        for i in range(num_merges):
            stats = self.get_stats(textIds)
            max_pair = max(stats, key=stats.get)
            newPairIdx = 256 + i
            textIds = self.merge(textIds, max_pair, newPairIdx)
            merges[max_pair] = newPairIdx

        return merges

    def create_vocab(self, merges):
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]

        return vocab
        
    def decode(self, textIds, vocab):
        tokens = b"".join(vocab[idx] for idx in textIds)
        text = tokens.decode("utf-8", errors="replace")
        return text

    def encode(self, text, merges):
        tokens = list(text.encode("utf-8"))
        while len(tokens) >= 2:
            stats = self.get_stats(tokens)
            pair = min(stats, key=lambda pair: merges.get(pair, float("inf")))
            if pair not in merges:
                break
            idx = merges[pair]
            tokens = self.merge(tokens, pair, idx)
        return tokens

    def getTextByCorpus(self):
        text = ""

        directory = os.fsencode("corpus")

        for file in os.listdir(directory):
            filename = os.fsdecode(file)
            with open("corpus/" + filename, "r") as f:
                fileObj = json.load(f)
                text += fileObj["text"] + " "
        
        return text
