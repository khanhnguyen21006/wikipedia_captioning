import torch
from torchvision import transforms as t
from torch.nn.utils.rnn import pad_sequence
import transformers

from nltk.tokenize import word_tokenize

from .vocab import Vocabulary
from datasets import *
from utils import merge_padded_tensors

image_size = 256

def get_tokenizer(_name, path=None):
	if "gpt2" in _name:
		return GPT2Tokenizer(_name)
	elif "t5" in _name:
		return T5Tokenizer()
	elif "clip" in _name:
		return CLIPTokenizer(_name)
	elif _name == "roberta":
		return RoBERTaTokenizer()
	elif 'sentence-transformers' in _name:
		return SentenceTransformersTokenizer(_name)
	elif _name == "gru":
		return GRUTokenizer(path)
	else:
		raise ValueError(f"{_name} Tokenizer is not supported.")

def get_transform(method):
	if method == 'resnet_h5py':
		base_t = [
				t.ToTensor(),
				t.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
			]
		if image_size != 256:
			base_t = [t.Resize((image_size, image_size))] + base_t
		return {
			'train': t.Compose(base_t),
			'val': t.Compose(base_t)
		}
	elif method == 'resnet_pil':
		return {
			'train': t.Compose([
				t.Resize((image_size, image_size)),
				t.ToTensor(),
				t.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
			]),
			'val': t.Compose([
				t.Resize((image_size, image_size)),
				t.ToTensor(),
				t.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
			])
		}
	elif method == 'image_net':
		return {
			'train': t.Compose([
				t.RandomResizedCrop(224),
				t.RandomHorizontalFlip(),
				t.ToTensor(),
				t.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
			]),
			'val': t.Compose([
				t.Resize(256),
				t.CenterCrop(224),
				t.ToTensor(),
				t.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
			]),
		}
	elif method == 'vit':
		return {
			'train': t.Compose([
				t.Resize((224, 224)),
				t.ToTensor(),
				t.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
			]),
			'val': t.Compose([
				t.Resize((224, 224)),
				t.ToTensor(),
				t.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
			]),
		}
	elif method == 'clip_vit':
		return {
			'train': t.Compose([
				t.Resize(224),
				t.ToTensor(),
				t.CenterCrop(224),
				t.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
			]),
			'val': t.Compose([
				t.Resize(224),
				t.ToTensor(),
				t.CenterCrop(224),
				t.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
			]),
		}
	elif method == 'clip_vit_h5py':
		return {
			'train': t.Compose([
				t.ToTensor(),
				t.CenterCrop(224),
				t.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
			]),
			'val': t.Compose([
				t.ToTensor(),
				t.CenterCrop(224),
				t.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
			]),
		}
	else:
		raise f"{method} Image Transformation is not supported."

def get_dataset(name):
	if name == 'wit':
		return WitDataset
	elif name == 'coco':
		return CocoDataset
	elif name == 'wittoy':
		return WitToyDataset
	elif name == 'witpage':
		return WitPageDataset
	elif name == 'witretmulti':
		return WitRetMultiDataset
	elif name == 'wikiweb2m':
		return WikiWebDataset
	elif name == 'goodnews':
		return GoodNewsDataset
	# elif name == 'nytimes800k':
	# 	return NYTimesDataset
	if name == 'rlwit':
		return RLWitDataset
	else:
		raise Exception(f"{key} Dataset is not supported.")

# COMPOSITION = ['section_caption', 'section_prompt']
GPT2PP_PROMPT = ['section_caption', 'section_prompt']
#, 'description_caption', 'description_prompt','description_section_caption', 'description_section_prompt'

TEXT_ENCODER_KEYS = ['description', 'section', 'description_section', 'caption']
TEXT_ENCODER_NUMERICS = [
	'description_id', 'description_mask', 'section_id', 'section_mask',
	'description_section_id', 'description_section_mask'
]
TEXT_DECODER_KEYS = ['caption', 'prompt', 'section_caption', 'section_prompt'
		'description_caption', 'description_prompt', 'description_section_caption', 'description_section_prompt'
	]
TEXT_DECODER_NUMERICS = [
		'caption_id', 'caption_mask', 'prompt_id', 'prompt_mask',
		'section_caption_id', 'section_caption_mask', 'section_prompt_id', 'section_prompt_mask',
		'description_caption_id', 'description_prompt_id', 'description_caption_mask', 'description_prompt_mask',
		'description_section_caption_id', 'description_section_prompt_id',
		'description_section_caption_mask', 'description_section_prompt_mask',
	]
NUMERICS = ['image'] + TEXT_ENCODER_NUMERICS + TEXT_DECODER_NUMERICS

def get_dataset_hparams(_config):
	name, text_ml, encoder, decoder = _config["dataset"], _config["text_max_len"],\
									 _config["text_encoder"], _config["text_decoder"]
	if name == 'witpage':
		# assert _config["n_embed"] > 1, "WitPage dataset requires setting no. space > 1"
		pass

	has_encoder = encoder is not None
	has_decoder = decoder is not None
	use_gpt2_decoder = has_decoder and 'gpt2' in decoder
	use_t5_decoder = has_decoder and 't5' in decoder
	use_gpt2pp_decoder = has_decoder and 'gpt2++' == decoder
	use_adapter = (encoder == 't5-adapter') or (decoder == 'gpt2-adapter')

	vocab_path = _config["vocab_path"]
	enc_tokenizer = get_tokenizer(encoder if has_encoder else decoder, path=vocab_path)
	dec_tokenizer = get_tokenizer(decoder if has_decoder else encoder, path=vocab_path)

	ext_args = _config["extract_context"].split('_')
	metric, extract_context = None if ext_args[0] == 'None' else ext_args[0], '_'.join(ext_args[1:])
	wiki_context = _config['wiki_context']

	if not use_adapter and (use_gpt2_decoder or use_t5_decoder):  # for non-adapter models, images are processed separately
		if image_size == 256:
			text_ml = text_ml - 64
		elif image_size == 224:
			text_ml = text_ml - 49

	if name in ["wit", "witpage", "wikiweb2m"]:
		context_keys = ['description', 'section'] + (['caption'] if not has_decoder else []) \
							+ (GPT2PP_PROMPT if use_gpt2pp_decoder else [])
	else:
		context_keys = ['section'] + (['caption'] if not has_decoder else []) \
							+ (GPT2PP_PROMPT if use_gpt2pp_decoder else [])

	if has_decoder:
		target_keys = ['caption'] + (['prompt'] if use_gpt2_decoder else [])
	else:
		target_keys = []

	return {
		'enc_tokenizer': enc_tokenizer,
		'dec_tokenizer': dec_tokenizer,
		'context_keys': context_keys,
		'target_keys': target_keys,
		'text_max_len': text_ml,
		'num_space': _config["n_embed"],
		'extract_context': extract_context,
		'wiki_context': wiki_context,
		'metric': metric,
	}

class RoBERTaTokenizer():
	def __init__(self):
		roberta = torch.hub.load('pytorch/fairseq:main', 'roberta.base')
		self.bpe = roberta.bpe.bpe
		self.sd = roberta.task.source_dictionary

	def tokenize(self, texts, max_len):
		tokens = []
		for text in texts:
			bpe_tokens = []
			for t in self.bpe.re.findall(self.bpe.pat, text):
				bpe_t = ''.join(self.bpe.byte_encoder[b] for b in t.encode('utf-8'))
				bpe_ids = [self.bpe.encoder[bt] for bt in self.bpe.bpe(bpe_t).split(' ')]
				bpe_tokens.extend(bpe_ids)
			concat = ' '.join(map(str, bpe_tokens))
			words = re.compile(r"\s+").sub(" ", concat).strip().split()
			words = ['<s>'] + words[:max_len - 2] + ['</s>']
			tokens.append(torch.Tensor([self.sd.indices[w] for w in words]))
		tokens = pad_sequence(tokens, batch_first=True, padding_value=self.sd.indices['<pad>']).long()
		masks = (tokens != self.sd.indices['<pad>']).long()
		return tokens, masks

class GPT2Tokenizer():
	def __init__(self, name):
		if name == "gpt2++":
			self.tokenizer = transformers.GPT2Tokenizer.from_pretrained('gpt2',
				bos_token='<|startoftext|>', eos_token='<|endoftext|>', sep_token='<|sep|>', pad_token='<pad>'
			)
		else:
			self.tokenizer = transformers.GPT2Tokenizer.from_pretrained('gpt2',
				bos_token='<|startoftext|>', eos_token='<|endoftext|>', pad_token='<pad>')

	def get_length(self):
		return len(self.tokenizer)

	def tokenize(self, texts, max_len):
		sos, eos = self.tokenizer.bos_token, self.tokenizer.eos_token
		if any(isinstance(i, list) for i in texts):
			sep, pad = self.tokenizer.sep_token, self.tokenizer.pad_token_id
			cap_encodings = self.tokenizer([f'{sep}{t}{eos}' if t else sep for t in texts[-1]],
									return_tensors="pt", padding="longest", truncation=True, max_length=100)
			cntx = [sos + ''.join([f'{t_}' for t_ in t]) for t in zip(*texts[:-1])]
			cntx_encodings = self.tokenizer(cntx, return_tensors="pt", truncation=True, padding="longest",
												max_length=(max_len-len(cap_encodings['input_ids'][0])))
			tokens = merge_padded_tensors(cntx_encodings['input_ids'], cap_encodings['input_ids'], pad)
			cntx_masks = cntx_encodings['attention_mask'] * (-100)
			masks = merge_padded_tensors(cntx_masks, cap_encodings['attention_mask'])
		else:
			texts = [f'{sos}{t}{eos}' if t else sos for t in texts]
			encodings = self.tokenizer(texts, return_tensors="pt", padding="longest", truncation=True, max_length=max_len)
			tokens, masks = encodings['input_ids'], encodings['attention_mask']
		return tokens, masks

class AutoTokenizer():
	def __init__(self):
		self.tokenizer = None

	def get_length(self):
		return len(self.tokenizer)

	def tokenize(self, texts, max_len):
		encodings = self.tokenizer(texts, return_tensors="pt", padding="longest", truncation=True, max_length=max_len)
		tokens, masks = encodings['input_ids'], encodings['attention_mask']
		return tokens, masks

class T5Tokenizer(AutoTokenizer):
	def __init__(self):
		super().__init__()
		self.tokenizer = transformers.T5Tokenizer.from_pretrained('t5-base')

class CLIPTokenizer(AutoTokenizer):
	def __init__(self, name):
		super().__init__()
		self.tokenizer = transformers.CLIPTokenizer.from_pretrained(name)

	def tokenize(self, texts, max_len):
		encodings = self.tokenizer(texts, return_tensors="pt", padding="longest", truncation=True, max_length=max_len)
		tokens, masks = encodings['input_ids'], encodings['attention_mask']
		return tokens, masks

class SentenceTransformersTokenizer(AutoTokenizer):
	def __init__(self, name):
		super().__init__()
		self.tokenizer = transformers.AutoTokenizer.from_pretrained(name)

class GRUTokenizer():
	def __init__(self, path):
		self.tokenizer = Vocabulary()
		self.tokenizer.load_from_pickle(path)

	def tokenize(self, texts, max_len):
		tokens = []
		for text in texts:
			words = word_tokenize(str(text).lower())
			tokenized = [self.tokenizer('<start>')] + [self.tokenizer(w) for w in words] + [self.tokenizer('<end>')]
			tokens.append(torch.Tensor(tokenized))
		tokens = pad_sequence(tokens, batch_first=True, padding_value=self.tokenizer('<pad>')).long()
		masks = (tokens != self.tokenizer('<pad>')).long()
		return tokens, masks
