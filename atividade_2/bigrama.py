import torch

class Bigram:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.bigram = torch.zeros((vocab_size, vocab_size), dtype=torch.int32)
    
    def train(self, tokenizedText):
        for token1, token2 in zip(tokenizedText, tokenizedText[1:]):
            self.bigram[token1, token2] += 1
        
        return
    
    def test(self, tokenizer): 
        P = (self.bigram+1).float()
        P /= P.sum(1, keepdims=True)

        g = torch.Generator().manual_seed(2147483647)
        
        for i in range(5):
            out = []
            ix = 0
            while True:
                p = P[ix]
                ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
                out.append(tokenizer.decode([ix]))
                if ix == 0:
                    break

            print(''.join(out))