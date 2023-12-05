import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.functional import pad
import torchtext
from torchvision import models

from transformers import GPT2Config, GPT2LMHeadModel,\
						T5Config, T5ForConditionalGeneration,\
						ViTModel, CLIPModel,\
						AutoModel

from .data_pool import get_tokenizer, Vocabulary, pad_sequence, merge_padded_tensors
from utils import GPT2_ADAPTER_LAYERS, T5_ADAPTER_LAYERS

import random
import spacy
nlp = spacy.load("en_core_web_sm")

def get_image_encoder(_config):
	"""
	Image Encoders experimented:
		- ResNet pre-trained on ImageNet ('resnet152', 'resnet50')
		- ViT pre-trained on ImageNet ('google/vit-base-patch16-224', 'google/vit-base-patch16-224-in21k')
		- CLIP-ViT pre-trained on openai dataset ('openai/clip-vit-base-patch32', 'openai/clip-vit-base-patch16')
	"""
	_name = _config['image_encoder']
	embed_dim = _config['embed_dim']
	finetune = _config["image_encoder_finetune"]
	if 'resnet' in _name:
		model = ResNet(_name, embed_dim, _config['n_embed'], finetune)
		dim = model.d
	# Note: Huggingface's VisionModel already has a linear mapping for pooled feature
	elif 'google/vit' in _name:
		model = ViT(_name, embed_dim, finetune)
		dim = model.d
	elif 'openai/clip' in _name:
		model = CLIPImageEncoder(_name, embed_dim, finetune, _config["image_encoder_use_linear_layer"])
		dim = model.d
	else:
		raise ValueError(f"{_name} Image Encoder is not supported.")
	_config['image_encoder_dim'] = dim
	return model, dim

def get_text_encoder(_config):
	"""
	Text Encoders experimented:
		- RoBERTa pre-trained ('roberta-base')
		- SentenceTransformers pre-trained ('sentence-transformers/all-distilroberta-v1',
														 'sentence-transformers/sentence-t5')
	"""
	_name = _config['text_encoder']
	embed_dim = _config['embed_dim']
	finetune = _config["text_encoder_finetune"]
	if _name is None:
		return None, None
	if _name == 'roberta':
		model = RoBERTa(embed_dim, finetune)
		dim = model.d
	elif _name == "t5-adapter":
		model = T5Adapter(finetune)
		dim = T5Config.from_pretrained('t5-base').d_model
	elif 'sentence-transformers' in _name:
		model = SentenceTransformers(_name, embed_dim, finetune)
		dim = model.d
	elif 'openai/clip' in _name:
		model = CLIPTextEncoder(_name, embed_dim, finetune, _config["text_encoder_use_linear_layer"])
		dim = model.d
	elif _name == 'gru':
		model = GRU(**_config)
		dim = _config['text_dim']
	else:
		raise ValueError(f"{_name} Text Encoder is not supported.")
	_config['text_encoder_dim'] = dim
	return model, dim

def get_text_decoder(_config):
	_name = _config['text_decoder']
	finetune = _config["text_decoder_finetune"]
	if _name is None:
		return None, None
	if _name == 'gpt2':
		cfg = GPT2Config.from_pretrained('gpt2')
		model = GPT2(cfg, get_tokenizer(_name), finetune)
		dim = cfg.n_embd
	elif _name == 'gpt2++':
		assert _config['n_embed'] == 0, f"This model allows no pooling by design, invalid n_embed: {_config['n_embed']} "
		cfg = GPT2Config.from_pretrained('gpt2')
		model = GPT2pp(cfg, get_tokenizer(_name), finetune)
		dim = cfg.n_embd
	elif _name == 'gpt2-adapter':
		cfg = GPT2Config.from_pretrained('gpt2', add_cross_attention=True)
		model = GPT2Adapter(cfg, get_tokenizer(_name), finetune)
		dim = cfg.n_embd
	elif _name == 't5++':
		assert _config['n_embed'] == 0, f"This model allows no pooling by design, invalid n_embed: {_config['n_embed']} "
		model = T5(finetune)
		dim = T5Config.from_pretrained('t5-base').d_model
	elif _name == 'otp':
		pass
	elif _name == 'otp-adapter':
		pass
	else:
		raise ValueError(f"{_name} Text Decoder is not supported.")
	return model, dim

def init_weights(module):
	nn.init.xavier_uniform_(module.weight)
	nn.init.constant_(module.bias, 0.0)

def vilt_init_weights(module):
	if isinstance(module, (nn.Linear, nn.Embedding)):
		module.weight.data.normal_(mean=0.0, std=0.02)
	elif isinstance(module, nn.LayerNorm):
		module.weight.data.fill_(1.0)
		module.bias.data.zero_()

def set_finetune(module, ft):
	if module is None:
		return
	for param in module.parameters():
		param.requires_grad = ft

def mean_pool(X, mask):
	mask_expanded = mask.unsqueeze(-1).expand(X.size()).float()
	return torch.sum(X * mask_expanded, dim=1)/ mask_expanded.sum(dim=1)

class GPT2(nn.Module):
	def __init__(self, cfg, tokenizer, finetune):
		super(GPT2, self).__init__()
		self.gpt2 = GPT2LMHeadModel.from_pretrained("gpt2", config=cfg)
		self.gpt2.resize_token_embeddings(tokenizer.get_length())
		self.finetune_gpt2(finetune)

	def finetune_gpt2(self, ft):
		for n, p in self.gpt2.named_parameters():
			if not any([True if (l in n) else False for l in GPT2_ADAPTER_LAYERS]):
				p.requires_grad = ft

	def embed(self, X):
		ml = X.size(1)
		X = self.gpt2.transformer.wte(X)
		pos = torch.arange(0, ml, dtype=torch.long).to(X.device)
		pos = pos.unsqueeze(0).view(-1, ml)
		X = X + self.gpt2.transformer.wpe(pos)
		return X

	def forward(self, **kwargs):
		kwargs = self.prepare_forward(**kwargs)
		X = torch.cat([kwargs['X_in'], kwargs['X']], dim=1)
		X_mask = torch.cat([kwargs['in_mask'], kwargs['X_mask']], dim=1)
		X_label = torch.cat([kwargs['in_label'], kwargs['X_label']], dim=1)
		X = self.gpt2(inputs_embeds=X,
						attention_mask=X_mask,
						labels=X_label,
						inputs_embeds_as_hidden_states=True)
		return X, X_label

	def generate(self, **kwargs):
		kwargs = self.prepare_generate(**kwargs)
		X = kwargs['X']
		X_in = kwargs['X_in']

		generations = []
		tokenizer = kwargs['tokenizer']
		for i in range(X.size(0)):
			prompt = kwargs['X'][i].unsqueeze(0)
			lp, ml = len(prompt), kwargs['max_len']
			X_gen = self.gpt2.generate(
					prompt,
					max_length=lp+(100 if (ml-lp) > 100 else ml-lp),
					num_beams=5,
					num_return_sequences=1,
					early_stopping=True,
					bos_token_id=tokenizer.bos_token_id,
					pad_token_id=tokenizer.pad_token_id,
					eos_token_id=tokenizer.eos_token_id,
					forced_eos_token_id=tokenizer.eos_token_id,
					image_embeds=X_in[i].unsqueeze(0),
					context_embeds=X[i].unsqueeze(0),
					decoder_input_embeds_as_prompt=True,)
			X_gen_id = X_gen[0].tolist()
			start, end = X_gen_id.index(tokenizer.bos_token_id)+1, X_gen_id.index(tokenizer.eos_token_id)
			generations.append(tokenizer.decode(X_gen[0][start:end], skip_special_tokens=False).strip())
		return generations

	def prepare_forward(self, **kwargs):
		X = kwargs['caption']['caption_id']
		X_mask = kwargs['caption']['caption_mask'].bool()
		X_label = X.masked_fill(~X_mask, -100)
		X_image = kwargs['image']['embedding']
		X_text = kwargs['section']['embedding']
		X_in = torch.cat([X_image, X_text], dim=1)
		in_mask = torch.cat([kwargs['image']['mask'], kwargs['section']['mask']], dim=1)
		in_label = torch.ones_like(in_mask) * (-100)

		kwargs = {
			'X': X,
			'X_mask': X_mask,
			'X_label': X_label,
			'X_in': X_in,
			'in_mask': in_mask,
			'in_label': in_label,
		}
		return kwargs

	def prepare_generate(self, **kwargs):
		X = self.embed(kwargs['prompt']['prompt_id'])
		X_image = kwargs['image']['embedding']
		X_text = kwargs['section']['embedding'][kwargs['section']['mask']]
		X_in = torch.cat([X_image, X_text], dim=1)

		X = {
			'X': X,
			'X_in': X_in,
			'tokenizer': kwargs['dec_tokenizer'].tokenizer,
			'max_len': kwargs['text_max_len']
		}
		return kwargs

class GPT2pp(GPT2):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

	def forward(self, **kwargs):
		kwargs = self.prepare_forward(**kwargs)
		X_text = self.embed(kwargs['text_id'])
		X = torch.cat([kwargs['X_image'], X_text], dim=1)
		X_mask = torch.cat([kwargs['image_mask'], kwargs['text_mask']], dim=1)
		X_label = torch.cat([kwargs['image_label'], kwargs['text_label']], dim=1)
		X = self.gpt2(inputs_embeds=X,
						attention_mask=X_mask,
						labels=X_label,
						inputs_embeds_as_hidden_states=True)
		return X, X_label

	def generate(self, **kwargs):
		kwargs = self.prepare_generate(**kwargs)
		X_text = self.embed(kwargs['text_id'])
		X_image = kwargs['X_image']

		generations = []
		tokenizer = kwargs['tokenizer']
		for i in range(X_text.size(0)):
			prompt = kwargs['text_id'][i].unsqueeze(0)
			pl, ml = len(prompt), kwargs['max_len']
			X_gen = self.gpt2.generate(
					prompt,
					max_length=pl+(100 if (ml-pl) > 100 else ml-pl),
					num_beams=5,
					num_return_sequences=1,
					early_stopping=True,
					bos_token_id=tokenizer.bos_token_id,
					pad_token_id=tokenizer.pad_token_id,
					eos_token_id=tokenizer.eos_token_id,
					forced_eos_token_id=tokenizer.eos_token_id,
					image_embeds=X_image[i].unsqueeze(0),
					context_embeds=X_text[i].unsqueeze(0),
					decoder_input_embeds_as_prompt=True,)
			X_gen_id = X_gen[0].tolist()
			start, end = X_gen_id.index(tokenizer.bos_token_id)+1, X_gen_id.index(tokenizer.eos_token_id)
			generations.append(tokenizer.decode(X_gen[0][start:end], skip_special_tokens=False).strip())
		return generations

	def prepare_forward(self, **kwargs):
		text_id = kwargs['section_caption']['section_caption_id']
		text_mask = kwargs['section_caption']['section_caption_mask']
		X_label = torch.where((text_mask != 0) & (text_mask != -100), text_id, -100)
		text_mask[text_mask == -100] = 1
		X_image = kwargs['image']['embedding']
		bs, iml = X_image.size(0), X_image.size(1)
		image_mask = torch.ones((bs, iml), dtype=torch.long, device=X_image.device)
		image_label = image_mask * (-100)

		kwargs = {
			'text_id': text_id,
			'text_mask': text_mask,
			'X_image': X_image,
			'image_mask': image_mask,
			'image_label': image_label,
			'text_label': X_label,
		}
		return kwargs

	def prepare_generate(self, **kwargs):
		text_id = kwargs['section_prompt_id']
		X_image = kwargs['image']['embedding']

		kwargs = {
			'text_id': text_id,
			'X_image': X_image,
			'tokenizer': kwargs['dec_tokenizer'].tokenizer,
			'max_len': kwargs['text_max_len']
		}
		return kwargs

class GPT2Adapter(GPT2):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

	def forward(self, **kwargs):
		kwargs = self.prepare_forward(**kwargs)
		X_in, in_mask = kwargs['X_in'], kwargs['in_mask']
		X, X_mask, X_label = kwargs['X'], kwargs['X_mask'], kwargs['X_label']
		X = self.gpt2(X,
					attention_mask=X_mask,
					labels=X_label,
					encoder_hidden_states=X_in,
					encoder_attention_mask=in_mask,)
		return X, X_label

	def generate(self, **kwargs):
		kwargs = self.prepare_generate(**kwargs)
		X_in = kwargs['X_in']
		in_mask = kwargs['in_mask']

		generations = []
		tokenizer = kwargs['tokenizer']
		for i in range(X_in.size(0)):
			prompt = kwargs['X'][i].unsqueeze(0)
			lp, ml = len(prompt), kwargs['max_len']
			X_gen = self.gpt2.generate(
					prompt,
					max_length=100,
					num_beams=5,
					num_return_sequences=1,
					early_stopping=True,
					bos_token_id=tokenizer.bos_token_id,
					pad_token_id=tokenizer.pad_token_id,
					eos_token_id=tokenizer.eos_token_id,
					forced_eos_token_id=tokenizer.eos_token_id,
					encoder_hidden_states=kwargs['X_in'][i].unsqueeze(0),
					encoder_attention_mask=in_mask[i].unsqueeze(0),)
			X_gen_id = X_gen[0].tolist()
			start, end = X_gen_id.index(tokenizer.bos_token_id)+1, X_gen_id.index(tokenizer.eos_token_id)
			generations.append(tokenizer.decode(X_gen[0][start:end], skip_special_tokens=False).strip())
		return generations

	def prepare_generate(self, **kwargs):
		X = kwargs['prompt']['prompt_id']
		X_image = kwargs['image']['embedding']
		X_in = torch.cat([X_image, kwargs['section']['embedding']], dim=1)
		in_mask = torch.cat([kwargs['image']['mask'], kwargs['section']['mask']], dim=1)

		kwargs = {
			'X': X,
			'X_in': X_in,
			'in_mask': in_mask,
			'tokenizer': kwargs['dec_tokenizer'].tokenizer,
			'max_len': kwargs['text_max_len']
		}
		return kwargs

class T5(nn.Module):
	def __init__(self, finetune):
		super(T5, self).__init__()
		self.t5 = T5ForConditionalGeneration.from_pretrained("t5-base")
		self.finetune_t5(finetune)

	def finetune_t5(self, ft):
		for n, p in self.t5.named_parameters():
			# if ('decoder' not in n) or (not any([True if (l in n) else False for l in T5_ADAPTER_LAYERS])):
			# 	p.requires_grad = ft
			p.requires_grad = ft

	def forward(self, **kwargs):
		kwargs = self.prepare_forward(**kwargs)
		X_text = self.t5.shared(kwargs['text_id'])
		X_in = torch.cat([kwargs['X_image'], X_text], dim=1)
		in_mask = torch.cat([kwargs['image_mask'], kwargs['text_mask']], dim=1)
		X_label = kwargs['X_label']

		X = self.t5(inputs_embeds=X_in,
					attention_mask=in_mask,
					labels=X_label)
		return X, X_label

	def generate(self, **kwargs):
		kwargs = self.prepare_generate(**kwargs)
		X_text = self.t5.shared(kwargs['text_id'])
		X_in = torch.cat([kwargs['X_image'], X_text], dim=1)
		in_mask = torch.cat([kwargs['image_mask'], kwargs['text_mask']], dim=1)

		tokenizer = kwargs['tokenizer']
		X_gen = self.t5.generate(
				inputs_embeds=X_in,
				attention_mask=in_mask,
				max_length=100,
				num_beams=5,
				num_return_sequences=1,
				early_stopping=True,
				eos_token_id=tokenizer.eos_token_id,
				forced_eos_token_id=tokenizer.eos_token_id,)
		generations = [tokenizer.decode(g, skip_special_tokens=True) for g in X_gen]
		return generations

	def prepare_forward(self, **kwargs):
		text_id = kwargs['section']['section_id']
		text_mask = kwargs['section']['section_mask']

		pt_objective = kwargs['pt_objective']
		dec_tokenizer = kwargs['dec_tokenizer']
		dset = kwargs['dataset']
		if pt_objective is not None:
			assert pt_objective in {'T5', 'BERT', 'Full', 'MNEM'}
			caption_id, caption_mask = [], []
			X_label = []
			if pt_objective == 'T5':
				for _input_id in kwargs['caption']['caption_id']:
					_new_input, _new_label = self.t5_mask_random_span(dec_tokenizer, _input_id.tolist())
					caption_id.append(torch.LongTensor(_new_input))
					caption_mask.append(torch.LongTensor([1] * len(_new_input)))
					X_label.append(torch.LongTensor(_new_label))
			elif pt_objective == 'BERT' or pt_objective == 'Full':
				mask_token_id = 32099
				full_mask = pt_objective == 'Full'
				for _input_id in kwargs['caption']['caption_id']:
					_new_input, _new_label = self.t5_bert_masking(dec_tokenizer, _input_id, mask_token_id, full_mask=full_mask)
					caption_id.append(_new_input)
					caption_mask.append(torch.LongTensor([1] * len(_new_input)))
					X_label.append(_new_label)
			elif pt_objective == 'MNEM':
				mnem_token_id = 32099
				_, mnem_masks = self.mnem_masking(dec_tokenizer, kwargs['caption']['caption'], kwargs['caption']['caption_id'], mnem_token_id, dset)
				# pad a with 0 at both sides for edge cases when a starts or ends with 1
				diff = torch.diff(pad(mnem_masks, (1, 1), mode='constant'))
				for i, _input_id in enumerate(kwargs['caption']['caption_id']):
					_input_id = _input_id.tolist()
					_input_length = _input_id.index(dec_tokenizer.eos_token_id) + 1
					start, end = torch.nonzero(diff[i] == 1, as_tuple=True)[0].tolist(), (torch.nonzero(diff[i] == -1, as_tuple=True)[0] - 1).tolist()
					# -1 to make it consistent with gpt2 experiments
					_new_input, _new_label = self.t5_replace_span(dec_tokenizer, _input_id, _input_length, list(zip(start, end)), mnem=True)
					caption_id.append(torch.LongTensor(_new_input))
					caption_mask.append(torch.LongTensor([1] * len(_new_input)))
					X_label.append(torch.LongTensor(_new_label))
			caption_id = pad_sequence(caption_id, batch_first=True, padding_value=dec_tokenizer.pad_token_id)  # pad pad_id
			caption_mask = pad_sequence(caption_mask, batch_first=True)  # pad 0
			# Workaround: torch does not have deterministic implementation for this operation
			text_id = merge_padded_tensors(text_id.cpu(), caption_id.cpu(), dec_tokenizer.pad_token_id).to(text_id.device)
			text_mask = merge_padded_tensors(text_mask.cpu(), caption_mask.cpu()).to(text_mask.device)
			X_label = pad_sequence(X_label, batch_first=True, padding_value=-100).to(text_id.device)
		else:
			X_label = kwargs['caption']['caption_id']
			X_label[X_label == 0] = -100
		X_image = kwargs['image']['embedding']
		bs, iml = X_image.size(0), X_image.size(1)
		image_mask = torch.ones((bs, iml), dtype=torch.long, device=X_image.device)

		kwargs = {
			'text_id': text_id,
			'text_mask': text_mask,
			'X_image': X_image,
			'image_mask': image_mask,
			'X_label': X_label,
		}
		return kwargs

	def prepare_generate(self, **kwargs):
		text_id = kwargs['section']['section_id']
		text_mask = kwargs['section']['section_mask']
		X_image = kwargs['image']['embedding']
		bs, iml = X_image.size(0), X_image.size(1)
		image_mask = torch.ones((bs, iml), dtype=torch.long, device=X_image.device)

		kwargs = {
			'text_id': text_id,
			'text_mask': text_mask,
			'X_image': X_image,
			'image_mask': image_mask,
			'tokenizer': kwargs['dec_tokenizer'].tokenizer,
		}
		return kwargs

	def t5_mask_random_span(self, tokenizer, input_id, prob=0.15):
		# mask random spans in input with unique mask IDs then in target, correspond each mask ID with the span GT.
		mask_spans = []
		input_length = input_id.index(tokenizer.eos_token_id) + 1  # find the first occurrence of EOS token and therefore input length
		last_ind = (0 if input_id[0] != tokenizer.cls_token_id else 1)
		prob = prob * 0.5 
		while last_ind < input_length - 1:  # last token is EOS, which we don't want to mask
			if len(mask_spans) < 100 and random.random() < prob:
				start = last_ind
				end = last_ind + random.randint(1, 5)  # create a span of 1-to-5 tokens.
				end = min(end, input_length - 2)
				mask_spans.append([start, end])
				last_ind = end + 1
			else:
				last_ind += 1
		return self.t5_replace_span(tokenizer, input_id, input_length, mask_spans)

	def t5_bert_masking(self, tokenizer, input_id, mask, percent=0.15, full_mask=False):
		eos = tokenizer.eos_token_id
		input_length = input_id.tolist().index(eos) + 1
		new_input_id = input_id[:input_length].clone().detach()
		old_input_id = input_id[:input_length]
		rand = torch.rand(input_length).to(new_input_id.device)

		# create mask array
		mask_arr = (rand <= percent) * (old_input_id != eos)
		selection = torch.flatten(mask_arr.nonzero()).tolist()

		if full_mask:
			new_input_id[selection] = mask
		else:
			value_i = []
			for idx in selection:
				prob = random.random()
				if prob <= 0.8:
					value_i.append(mask)
				elif prob <= 0.9:
					value_i.append(random.randint(0, len(tokenizer)))
				else:
					value_i.append(new_input_id[idx])
			new_input_id[selection] = torch.LongTensor(value_i).to(new_input_id)

		assert new_input_id.size() == old_input_id.size()
		return new_input_id, old_input_id

	def mnem_masking(self, tokenizer, texts, input_ids, mask, dset, percent=0.8, model='T5', is_context=False):
		mnem_masks = []
		# We first compute the start and end points for each token.
		# End points are exclusive.
		for i, doc in enumerate(nlp.pipe(texts, disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer"])):
				
			tokens = tokenizer.convert_ids_to_tokens(input_ids[i])
			assert len(tokens) == len(input_ids[i])

			starts = []
			ends = []
			current = 0
			for t_id, token in enumerate(tokens):
				if t_id == 0:
					if token == '<SOS>':  # gpt-2 context input id starts with '<SOS>'
						starts.append(0)
						current += 0
						ends.append(0)
					elif model == 'T5':  # t5 caption/context input id both start with '_'
						if dset == 'goodnews' and (texts[i].startswith('\n') or texts[i].startswith('\n')):				
							starts.append(current)
							current += len(token)
							ends.append(current)
						else:
							starts.append(0)
							current += len(token) - 1
							ends.append(current)
					else:  # gpt-2 caption input id
						starts.append(current)
						current += len(token)
						ends.append(current)
				else:   
					if token == '<PAD>' or token == '<pad>':
						break
					starts.append(current)
					current += len(token) if token != '<unk>' and token != '<UNK>' else 1
					ends.append(current)

			mnem_mask = [0] * len(tokens)
			# Next we get the character positions of named entities
			for ent in doc.ents:
				if random.random() < percent:
					# A token is part of an entity if it lies strictly inside it
					for t, (start, end, token) in enumerate(zip(starts, ends, tokens)):
						entity_start = ent.start_char
						if token[0] == 'Ġ' or token[0] == '▁':
							entity_start -= 1
						entity_end = ent.end_char
						if not is_context:
							if dset in ['goodnews', 'nytimes800k']:
								cat_list = ['PERSON', 'ORG', 'GPE', 'NORP', 'LOC', 'EVENT']
							else:
								cat_list = ['PERSON', 'ORG', 'GPE']
						else:
							cat_list = ['PERSON', 'ORG', 'GPE', 'NORP', 'LOC', 'FAC', 'EVENT', 'PRODUCT']
						if start >= entity_start and end <= entity_end and ent.label_ in cat_list:
							mnem_mask[t] = 1
			mnem_masks.append(mnem_mask)
		mnem_masks = torch.BoolTensor(mnem_masks).to(input_ids)
		input_ids = input_ids.masked_fill(mnem_masks.bool(), mask)

		return input_ids, mnem_masks

	def t5_replace_span(self, tokenizer, input_id, input_length, mask_spans, mnem=False):
		lm_labels = []
		new_input_id = []
		mask_ID_counter = 0
		previous_e = 0
		for s, e in mask_spans:
			extra_id = tokenizer._convert_token_to_id("<extra_id_%d>" % mask_ID_counter)
			lm_labels.append(extra_id)
			lm_labels.extend(input_id[s:e + 1])
			new_input_id.extend(input_id[previous_e:s])
			new_input_id.append(extra_id)
			previous_e = e + 1
			mask_ID_counter += 1

		new_input_id += input_id[previous_e:input_length - 1]
		if not mnem:
			# add EOS token to lm_labels and new_inputs_list
			lm_labels = lm_labels[:int(len(input_id) * 0.25) - 1]  # make sure lm_labels is within max length limit and we use a lower limit for the lm_labels length
		lm_labels.append(tokenizer.eos_token_id)
		new_input_id = new_input_id[:len(input_id) - 1]
		new_input_id.append(tokenizer.eos_token_id)

		return new_input_id, lm_labels

class T5Adapter(T5):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

	def forward(self, **kwargs):
		kwargs = self.prepare_forward(**kwargs)
		X_in, X_label= kwargs['X_in'], kwargs['X_label']
		X = self.t5(encoder_outputs=(X_in,),
					attention_mask=kwargs['in_mask'],
					labels=X_label)
		return X, X_label
	def generate(self, **kwargs):
		kwargs = self.prepare_generate(**kwargs)
		tokenizer = kwargs['tokenizer']
		X_gen = self.t5.generate(
					encoder_outputs=(kwargs['X_in'],),
					attention_mask=kwargs['in_mask'],
					max_length=100,
					num_beams=5,
					num_return_sequences=1,
					early_stopping=True,
					eos_token_id=tokenizer.eos_token_id,
					forced_eos_token_id=tokenizer.eos_token_id,)
		generations = [tokenizer.decode(g, skip_special_tokens=True) for g in X_gen]
		return generations

	def prepare_forward(self, **kwargs):
		X_label = kwargs['caption']['caption_id']
		X_label[X_label == 0] = -100
		X_in = torch.cat([kwargs['image']['embedding'], kwargs['section']['embedding']], dim=1)
		in_mask = torch.cat([kwargs['image']['mask'], kwargs['section']['mask']], dim=1)
		kwargs = {
			'X_in': X_in,
			'in_mask': in_mask,
			'X_label': X_label
		}
		return kwargs

	def prepare_generate(self, **kwargs):
		X_in = torch.cat([kwargs['image']['embedding'], kwargs['section']['embedding']], dim=1)
		in_mask = torch.cat([kwargs['image']['mask'], kwargs['section']['mask']], dim=1)
		kwargs = {
			'X_in': X_in,
			'in_mask': in_mask,
			'tokenizer': kwargs['dec_tokenizer'].tokenizer,
		}
		return kwargs

class ResNet(nn.Module):
	def __init__(self, name, d_emb, n_emb, finetune):
		super(ResNet, self).__init__()
		self.resnet = getattr(models, name)(pretrained=True)
		self.d = self.resnet.fc.in_features

		self.pool = n_emb > 0
		if self.pool:
			self.avgpool = self.resnet.avgpool
		self.fc = nn.Sequential(
			nn.Linear(self.d, d_emb),
			nn.LayerNorm(d_emb),
			# nn.Dropout(),
			nn.GELU(),
		)
		vilt_init_weights(self.fc[0])

		self.resnet.avgpool = nn.Sequential()
		self.resnet.fc = nn.Sequential()
		set_finetune(self.resnet, finetune)

	def forward(self, X):
		f = X.size(-1) // 32
		X = self.resnet(X)
		X_fxf = X.view(-1, self.d, f, f)
		if self.pool:
			X_cls = self.fc(self.avgpool(X_fxf).view(-1, self.d))
			X = X_fxf.view(-1, self.d, f*f).transpose(1, 2)
			return X, X_cls
		else:
			X_fxf = X_fxf.view(-1, self.d, f*f).transpose(1, 2)
			X = self.fc(X_fxf)
			X_cls = torch.mean(X_fxf, dim=1)
			return X, X_cls

class ViT(nn.Module):
	def __init__(self, name, d_emb, finetune, use_linear=False):
		super(ViT, self).__init__()
		self.vit = ViTModel.from_pretrained(name)
		self.d = self.vit.config.hidden_size
		self.use_linear = use_linear
		if not use_linear:
			self.fc = nn.Sequential(
				nn.Linear(self.d, d_emb),
				nn.LayerNorm(d_emb),
				# nn.Dropout(),
				nn.GELU(),
			)
			vilt_init_weights(self.fc[0])
		else:
			self.d = self.vit.config.hidden_size
			assert self.d == d_emb
		set_finetune(self.vit, finetune)

	def forward(self, X):
		X = self.vit(X)
		if self.use_linear:
			X_cls = X.pooler_output
		else:
			X_cls = self.fc(X.pooler_output)
		X = X.last_hidden_state[:, 1:, :]
		return X, X_cls

class CLIPImageEncoder(nn.Module):
	def __init__(self, name, d_emb, finetune, use_linear=False):
		super(CLIPImageEncoder, self).__init__()
		clip = CLIPModel.from_pretrained(name)
		self.clip_image_encoder = clip.vision_model
		if not use_linear:
			self.d = self.clip_image_encoder.config.hidden_size
			self.fc = nn.Sequential(
				nn.Linear(self.d, d_emb),
				nn.LayerNorm(d_emb),
				# nn.Dropout(),
				nn.GELU(),
			)
			vilt_init_weights(self.fc[0])
		else:
			self.d = clip.config.projection_dim
			self.fc = clip.visual_projection
			assert self.d == d_emb
			set_finetune(clip.visual_projection, finetune)
			self.logit_scale = clip.logit_scale
		set_finetune(self.clip_image_encoder, finetune)

	def forward(self, X):
		X = self.clip_image_encoder(X)
		X_cls = self.fc(X.pooler_output)
		X = X.last_hidden_state[:, 1:, :]
		return X, X_cls

class CLIPTextEncoder(nn.Module):
	def __init__(self, name, d_emb, finetune, use_linear=False):
		super(CLIPTextEncoder, self).__init__()
		clip = CLIPModel.from_pretrained(name)
		self.clip_text_encoder = clip.text_model
		if not use_linear:
			self.d = self.clip_text_encoder.config.hidden_size
			self.fc = nn.Sequential(
				nn.Linear(self.d, d_emb),
				nn.LayerNorm(d_emb),
				# nn.Dropout(),
				nn.GELU(),
			)
			vilt_init_weights(self.fc[0])
		else:
			self.d = clip.config.projection_dim
			self.fc = clip.text_projection
			assert self.d == d_emb
			set_finetune(clip.text_projection, finetune)
		set_finetune(self.clip_text_encoder, finetune)

	def forward(self, X, **kwargs):
		X = self.clip_text_encoder(input_ids=X, attention_mask=kwargs['mask'])
		X_cls = self.fc(X.pooler_output)
		X = X.last_hidden_state[:, 1:, :]
		X_mask = kwargs['mask'].bool()
		return X, X_cls, X_mask

class SentenceTransformers(nn.Module):
	def __init__(self, name, d_emb, finetune):
		super(SentenceTransformers, self).__init__()
		self.model = AutoModel.from_pretrained(name)
		self.d = self.model.config.hidden_size
		self.fc = nn.Sequential(
			nn.Linear(self.d, d_emb),
			nn.LayerNorm(d_emb),
			# nn.Dropout(),
			nn.GELU(),
		)
		vilt_init_weights(self.fc[0])
		set_finetune(self.model, finetune)

	def forward(self, X, **kwargs):
		X = self.model(
			input_ids=X,
			attention_mask=kwargs['mask'],
		)
		X = X.last_hidden_state
		X_cls = mean_pool(X, kwargs['mask'])
		if kwargs['pool']:
			X_cls = self.fc(X_cls)
		else:
			X = self.fc(X)
			X_cls = mean_pool(X, kwargs['mask'])
		X_mask = kwargs['mask'].bool()
		return X, X_cls, X_mask

class RoBERTa(nn.Module):
	def __init__(self, d_emb, finetune):
		super(RoBERTa, self).__init__()
		self.roberta = torch.hub.load('pytorch/fairseq:main', 'roberta.large')
		self.d = self.roberta.cfg.model.encoder_embed_dim
		self.fc = nn.Sequential(
			nn.Linear(self.d, d_emb),
			nn.LayerNorm(d_emb),
			# nn.Dropout(),
			nn.GELU(),
		)
		vilt_init_weights(self.fc[0])
		set_finetune(self.roberta, finetune)

	def forward(self, X, **kwargs):
		X = self.roberta.extract_features(X, return_all_hiddens=False)
		X_cls = mean_pool(X, kwargs['mask'])
		if kwargs['pool']:
			X_cls = self.fc(X_cls)
		else:
			X = self.fc(X)
			X_cls = mean_pool(X, kwargs['mask'])
		X_mask = kwargs['mask'].bool()
		return X, X_cls, X_mask

class GRU(nn.Module):
	def __init__(self, **kwargs):
		super(GRU, self).__init__()

		text_emb = kwargs["text_embed"]
		text_dim = kwargs["text_dim"]
		d_emb = kwargs["embed_dim"]

		vocab = Vocabulary()
		vocab.load_from_pickle(kwargs["vocab_path"])
		word2idx = vocab.word2idx
		# Word embedding
		self.embed = nn.Embedding(len(word2idx), text_dim)
		self.embed.weight.requires_grad = kwargs["text_encoder_finetune"]
		if 'fasttext' == text_emb:
			wemb = torchtext.vocab.FastText(cache=kwargs['cache_dir'])
		elif 'glove' == text_emb:
			wemb = torchtext.vocab.GloVe(cache=kwargs['cache_dir'])
		else:
			raise ValueError(f"{text_emb} Text Embedding for GRU is not supported.")
		assert wemb.vectors.shape[1] == text_dim
		# quick-and-dirty trick to improve word-hit rate
		missing_words = []
		for word, idx in word2idx.items():
			if word not in wemb.stoi:
				word = word.replace('-', '').replace('.', '').replace("'", '')
				if '/' in word:
					word = word.split('/')[0]
			if word in wemb.stoi:
				self.embed.weight.data[idx] = wemb.vectors[wemb.stoi[word]]
			else:
				missing_words.append(word)
		print('Words: {}/{} found in vocabulary; {} words missing'.format(
				len(word2idx) - len(missing_words), len(word2idx), len(missing_words)))
		self.rnn = nn.GRU(text_dim, d_emb//2, bidirectional=True, batch_first=True)
		self.set_finetune(self.embed, kwargs['text_encoder_finetune'])

	def forward(self, X, **kwargs):
		X = self.embed(X)
		X_mask = kwargs['mask']
		X_len = torch.sum(X_mask, dim=-1)
		packed = pack_padded_sequence(X, X_len.cpu(), batch_first=True, enforce_sorted=False)
		if torch.cuda.device_count() > 1:
			self.rnn.flatten_parameters()
		X_rnn, _ = self.rnn(packed)
		padded = pad_packed_sequence(X_rnn, batch_first=True)
		X_cls = []
		for b in range(X.size(0)):
			X_cls.append(padded[0][b][X_len[b] - 1, :])
		X_cls = torch.stack(X_cls)
		X_mask = X_mask.bool()
		return X, X_cls, X_mask
