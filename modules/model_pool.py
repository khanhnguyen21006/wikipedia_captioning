import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torchtext
from torchvision import models

from transformers import GPT2Config, GPT2LMHeadModel,\
						T5Config, T5ForConditionalGeneration,\
						ViTModel, CLIPVisionModel,\
						AutoModel

from .data_pool import get_tokenizer, Vocabulary
from utils import GPT2_ADAPTER_LAYERS, T5_ADAPTER_LAYERS

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
		model = CLIPImageEncoder(_name, embed_dim, finetune)
		dim = model.d
	else:
		raise ValueError(f"{_name} Image Encoder is not supported.")
	_config['image_encoder_dim'] = dim
	return model, dim

def get_text_encoder(_config):
	"""
	Text Encoders experimented:
		- RoBERTa pre-trained ('roberta-base')
		- SentenceTransformers pre-trained ('sentence-transformers/all-distilroberta-v1', 'sentence-transformers/sentence-t5')
	"""
	_name = _config['text_encoder']
	embed_dim = _config['embed_dim']
	finetune = _config["text_encoder_finetune"]
	if _name == 'roberta':
		model = RoBERTa(embed_dim, finetune)
		dim = model.d
	elif _name == "t5-adapter":
		model = T5Adapter(finetune)
		dim = T5Config.from_pretrained('t5-base').d_model
	elif 'sentence-transformers' in _name:
		model = SentenceTransformers(_name, embed_dim, finetune)
		dim = model.d
	elif _name == 'gru':
		model = GRU(**_config)
		dim = _config['text_dim']
	elif _name is None:
		return None, None
	else:
		raise ValueError(f"{_name} Text Encoder is not supported.")
	_config['text_encoder_dim'] = dim
	return model, dim

def get_text_decoder(_config):
	_name = _config['text_decoder']
	finetune = _config["text_decoder_finetune"]
	if _name == 'gpt2':
		cfg = GPT2Config.from_pretrained('gpt2')
		model = GPT2(cfg, get_tokenizer(_name), finetune)
		dim = cfg.n_embd
	elif _name == 'gpt2++':
		cfg = GPT2Config.from_pretrained('gpt2')
		model = GPT2pp(cfg, get_tokenizer(_name), finetune)
		dim = cfg.n_embd
	elif _name == 'gpt2-adapter':
		cfg = GPT2Config.from_pretrained('gpt2', add_cross_attention=True)
		model = GPT2Adapter(cfg, get_tokenizer(_name), finetune)
		dim = cfg.n_embd
	elif _name == 't5++':
		model = T5(finetune)
		dim = T5Config.from_pretrained('t5-base').d_model
	elif _name == 'otp':
		pass
	elif _name == 'otp-adapter':
		pass
	elif _name is None:
		return None, None
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
		module.bias.data.zero_()
		module.weight.data.fill_(1.0)

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
		X = self.embed(kwargs['X'])
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
		bs, ml = X_in.size(0), X_in.size(1)
		in_mask = torch.ones((bs, ml), dtype=torch.long, device=X_in.device)
		in_label = in_mask * (-100)

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
		X = kwargs['prompt']['prompt_id']
		X_image = kwargs['image']['embedding']
		X_text = kwargs['section']['embedding']
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
		X, X_in, X_mask, X_label = kwargs['X'], kwargs['X_in'], kwargs['X_mask'], kwargs['X_label']
		X = self.gpt2(X,
					attention_mask=X_mask,
					labels=X_label,
					encoder_hidden_states=X_in)
		return X, X_label

	def generate(self, **kwargs):
		kwargs = self.prepare_generate(**kwargs)
		X_in = kwargs['X_in']

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
					encoder_hidden_states=kwargs['X_in'][i].unsqueeze(0))
			X_gen_id = X_gen[0].tolist()
			start, end = X_gen_id.index(tokenizer.bos_token_id)+1, X_gen_id.index(tokenizer.eos_token_id)
			generations.append(tokenizer.decode(X_gen[0][start:end], skip_special_tokens=False).strip())
		return generations

	def prepare_generate(self, **kwargs):
		X = kwargs['prompt']['prompt_id']
		X_in = torch.cat([kwargs['image']['embedding'], kwargs['section']['embedding']], dim=1)

		kwargs = {
			'X': X,
			'X_in': X_in,
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
		for n, p in self.t5.decoder.named_parameters():
			if ('decoder' not in n) or (not any([True if (l in n) else False for l in T5_ADAPTER_LAYERS])):
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

class T5Adapter(T5):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

	def forward(self, **kwargs):
		kwargs = self.prepare_forward(**kwargs)
		X_in, X_label= kwargs['X_in'], kwargs['X_label']
		X = self.t5(encoder_outputs=(X_in,),
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
		X_image = kwargs['image']['embedding']
		X_text = kwargs['section']['embedding']
		X_in = torch.cat([X_image, X_text], dim=1)

		kwargs = {
			'X_in': X_in,
			'X_label': X_label
		}
		return kwargs

	def prepare_generate(self, **kwargs):
		X_image = kwargs['image']['embedding']
		X_text = kwargs['section']['embedding']
		X_in = torch.cat([X_image, X_text], dim=1)
		bs, iml, tml = X_image.size(0), X_image.size(1), X_text.size(1)
		image_mask = torch.ones((bs, iml), dtype=torch.long, device=X_image.device)
		text_mask = torch.ones((bs, tml), dtype=torch.long, device=X_text.device)
		in_mask = torch.cat([image_mask, text_mask], dim=1)
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
	def __init__(self, name, d_emb, finetune):
		super(ViT, self).__init__()
		self.vit = ViTModel.from_pretrained(name)
		self.d = self.vit.config.hidden_size
		self.fc = nn.Sequential(
			nn.Linear(self.d, d_emb),
			nn.LayerNorm(d_emb),
			# nn.Dropout(),
			nn.GELU(),
		)
		vilt_init_weights(self.fc[0])
		set_finetune(self.vit, finetune)

	def forward(self, X):
		X = self.vit(X)
		X_cls = self.fc(X.pooler_output)
		X = X.last_hidden_state[:, 1:, :]
		return X, X_cls

class CLIPImageEncoder(nn.Module):
	def __init__(self, name, d_emb, finetune):
		super(CLIPImageEncoder, self).__init__()
		self.clip_image_encoder = CLIPVisionModel.from_pretrained(name)
		self.d = self.clip_image_encoder.config.hidden_size
		self.fc = nn.Sequential(
			nn.Linear(self.d, d_emb),
			nn.LayerNorm(d_emb),
			# nn.Dropout(),
			nn.GELU(),
		)
		vilt_init_weights(self.fc[0])
		set_finetune(self.clip_image_encoder, finetune)

	def forward(self, X):
		X = self.clip_image_encoder(X)
		X_cls = self.fc(X.pooler_output)
		X = X.last_hidden_state[:, 1:, :]
		return X, X_cls

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
		X_mask = ~kwargs['mask'].bool()
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
		X_mask = ~kwargs['mask'][:, 1:].bool()
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
		X_mask = ~X_mask.bool()
		return X, X_cls, X_mask
