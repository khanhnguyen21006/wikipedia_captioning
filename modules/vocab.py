import json
from tqdm import tqdm
import os
import pickle
from collections import Counter
from nltk.tokenize import word_tokenize


def parse(vocab_path):
	vocab_name = vocab_path.split('/')[-1].split('.')[0]
	assert len(vocab_name), 'vocab name must be in format dataset_vocab(pp)_threshold.pkl'
	dataset = vocab_name.split('_')[0]
	is_pp = vocab_name.split('_')[1] == 'vocabpp'
	if dataset == 'wit':
		data_folder = '/data/users/vkhanh/refined/' if is_pp else '/data/users/vkhanh/all/'
	elif dataset == 'coco':
		data_folder = '/data/users/vkhanh/coco/'
	else:
		data_folder = f'/data/users/vkhanh/{dataset}/'
	threshold = int(vocab_name.split('_')[2])
	return dataset, data_folder, is_pp, threshold


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

	def load_from_pickle(self, vocab_path):
		if os.path.exists(vocab_path):
			with open(vocab_path, 'rb') as f:
				vocab = pickle.load(f)
		else:
			dataset, data_folder, pp, threshold = parse(vocab_path)
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
				if pp:
					with open(os.path.join(data_folder, split + '_STRDESCS_' + dataset + '.json'), 'r') as f:
						descs = json.load(f)
						texts.extend(descs)
						print(f'Added {len(descs)} descriptions')
			print(f"Total {len(texts)} {'descriptions,' if pp else ''} sections and captions.")

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

			if not os.path.isdir('./modules/vocab'):
				os.makedirs('./modules/vocab')
			with open(os.path.join('./modules/vocab/', f"{dataset}_{'vocabpp' if pp else 'vocab'}_{threshold}.pkl"), 'wb') as f:
				pickle.dump(vocab, f, pickle.HIGHEST_PROTOCOL)
			print("Saved vocabulary file to ", os.path.join('./modules/vocab/', f"{dataset}_{'vocabpp' if pp else 'vocab'}_{threshold}.pkl"))

		self.idx = vocab.idx
		self.word2idx = vocab.word2idx
		self.idx2word = vocab.idx2word

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
