import json
from tqdm import tqdm
import os
import pickle
from collections import Counter
from nltk.tokenize import word_tokenize


data_folder = '/data/users/vkhanh/toy/' 
dataset = 'wit_toy'
threshold = 20


class Vocabulary(object):
	"""Simple vocabulary wrapper."""

	def __init__(self):
		self.idx = 0
		self.word2idx = {}
		self.idx2word = {}

	def add_word(self, word):
		if word not in self.word2idx:
			self.word2idx[word] = self.idx
			self.idx2word[self.idx] = word
			self.idx += 1

	def load_from_pickle(self, data_path):
		with open(data_path, 'rb') as fin:
			data = pickle.load(fin)
		self.idx = data.idx
		self.word2idx = data.word2idx
		self.idx2word = data.idx2word

	def __call__(self, word):
		if word not in self.word2idx:
			return self.word2idx['<unk>']
		return self.word2idx[word]

	def __len__(self):
		return len(self.word2idx)


class CocoVocabulary(object):
    def __init__(self):
        self.idx = 0
        self.word2idx = {}
        self.idx2word = {}

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if word not in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)


def main():
	counter = Counter()
	texts = []
	print(f'Creating vocab with threshold ', threshold)
	for split in ['val', 'test', 'train']:
		print('Split: ', split)
		with open(os.path.join(data_folder, split + '_STRCONTEXTS_' + dataset + '.json'), 'r') as f:
			cntxs = json.load(f)
			texts.extend(cntxs)
			print(f'Added {len(cntxs)} sections')
		with open(os.path.join(data_folder, split + '_STRCAPS_' + dataset + '.json'), 'r') as f:
			caps = json.load(f)
			texts.extend(caps)
			print(f'Added {len(caps)} captions')
		with open(os.path.join(data_folder, split + '_STRDESCS_' + dataset + '.json'), 'r') as f:
			descs = json.load(f)
			texts.extend(descs)
			print(f'Added {len(descs)} descriptions')
	print(f'Total {len(texts)} descriptions, sections and captions')

	for text in tqdm(texts):
		tokens = word_tokenize(text.lower()) # lower everything?
		counter.update(tokens)

	words = [word for word, cnt in counter.items() if cnt >= threshold]
	print('Vocab size: {}'.format(len(words)))

	vocab = Vocabulary()
	vocab.add_word('<pad>')
	vocab.add_word('<start>')
	vocab.add_word('<end>')
	vocab.add_word('<unk>')

	# Add words to the vocabulary.
	for word in words:
		vocab.add_word(word)

	if not os.path.isdir('./vocab'):
		os.makedirs('./vocab')
	with open(os.path.join('./vocab/', f'{dataset}_vocabpp_{threshold}.pkl'), 'wb') as f:
		pickle.dump(vocab, f, pickle.HIGHEST_PROTOCOL)
	print("Saved vocabulary file to ", os.path.join('./vocab/', f'{dataset}_vocabpp_{threshold}.pkl'))


if __name__ == '__main__':
	main()