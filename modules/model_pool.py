import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torchtext
from torchvision import models

from transformers import GPT2Config, GPT2Tokenizer, GPT2LMHeadModel,\
						T5Config, T5Tokenizer, T5ForConditionalGeneration,\
						ViTModel, CLIPVisionModel
from .vocab import Vocabulary
from utils import *

GPT2_ADAPTER_LAYERS = ['crossattention', 'ln_cross_attn']
T5_ADAPTER_LAYERS = ['EncDecAttention', 'layer.1.layer_norm', 'layer.2.layer_norm']  # NEED TO SPECIFY THIS

def get_tokenizer(model_name):
	if model_name == "gpt2++":
		return GPT2Tokenizer.from_pretrained('gpt2', bos_token='<|startoftext|>', eos_token='<|endoftext|>',\
													 sep_token='<|sep|>', pad_token='<pad>')
	elif "gpt2" in model_name:
		return GPT2Tokenizer.from_pretrained('gpt2', bos_token='<|startoftext|>', eos_token='<|endoftext|>',\
													 pad_token='<pad>')
	elif "t5" in model_name:
		return T5Tokenizer.from_pretrained('t5-base')
	elif model_name == "roberta":
		roberta = torch.hub.load('pytorch/fairseq:main', 'roberta.base')
		return RoBERTaTokenizer(roberta)
	elif model_name == "sbert":
		return
	elif model_name == "gru":
		vocab = Vocabulary()
		vocab.load_from_pickle(path)
		return vocab
	else:
		raise ValueError(f"{model_name} Tokenizer is not supported.")

def get_image_encoder(_config):
	"""
	Visual Encoders supported:
		- ResNet pre-trained on ImageNet
		- ViT pre-trained on ImageNet
		- CLIP-ViT pre-trained on openai dataset
	"""
	model_name = _config['image_encoder']
	embed_dim = _config['embed_dim']
	finetune = _config["image_encoder_finetune"]
	if 'resnet' in model_name:
		model = ResNet(model_name, embed_dim, _config['n_embeds'])
		dim = model.d
		return model, dim
	# Note: Huggingface's VisionModel already has a linear mapping for pooled feature
	elif 'vit' in model_name:
		model = ViTModel.from_pretrained(model_name)
		dim = model.config.hidden_size
	elif 'clip' in model_name:
		model = CLIPVisionModel.from_pretrained(model_name)
		dim = model.config.hidden_size
	else:
		raise ValueError(f"{model_name} Image Encoder is not supported.")
	set_finetune(model, finetune)
	return model, dim

def get_text_encoder(_config):
	model_name = _config['text_encoder']
	embed_dim = _config['embed_dim']
	finetune = _config["text_encoder_finetune"]
	if model_name == 'roberta' :
		model = RoBERTa(embed_dim)
		dim = model.d
	elif model_name == 'gpt2++':
		return None, GPT2Config.from_pretrained('gpt2').n_embd
	elif model_name == "t5++":
		return None, T5Config.from_pretrained('t5-base').d_model
	elif model_name == "t5-adapter":
		cfg = T5Config.from_pretrained('t5-base')
		model = T5Adapter(finetune=finetune)
		dim = cfg.d_model
		return model, dim
	elif model_name == 'sbert':
		pass
	elif model_name == 'st5':
		pass
	elif model_name == 'gru':
		model = GRU(embed_dim)
		dim = embed_dim
	else:
		raise ValueError(f"{model_name} Text Encoder is not supported.")
	set_finetune(model, finetune)
	return model, dim

def get_text_decoder(_config):
	model_name = _config['text_decoder']
	finetune = _config["text_decoder_finetune"]
	if model_name == 'gpt2':
		cfg = GPT2Config.from_pretrained('gpt2')
		model = GPT2(cfg, get_tokenizer(model_name), finetune=finetune)
		dim = cfg.n_embd
		return model, dim
	elif model_name == 'gpt2++':
		cfg = GPT2Config.from_pretrained('gpt2')
		model = GPT2pp(cfg, get_tokenizer(model_name), finetune=finetune)
		dim = cfg.n_embd
		return model, dim
	elif model_name == 'gpt2-adapter':
		cfg = GPT2Config.from_pretrained('gpt2', add_cross_attention=True)
		model = GPT2Adapter(cfg, get_tokenizer(model_name), finetune=finetune)
		dim = cfg.n_embd
		return model, dim
	elif model_name == 't5++':
		cfg = T5Config.from_pretrained('t5-base')
		model = T5(finetune=finetune)
		dim = cfg.d_model
		return model, dim
	elif model_name == "t5-adapter":
		return None, None
	elif model_name == 'otp':
		pass
	elif model_name == 'otp-adapter':
		pass
	else:
		raise ValueError(f"{model_name} Text Decoder is not supported.")
	set_finetune(model, finetune)
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
	for param in module.parameters():
		param.requires_grad = ft

class RoBERTaTokenizer():
	def __init__(self, roberta):
		self.bpe = roberta.bpe.bpe
		self.sd = roberta.task.source_dictionary

class GPT2(nn.Module):
	def __init__(self, cfg, tokenizer, finetune=False):
		super(GPT2, self).__init__()
		self.gpt2 = GPT2LMHeadModel.from_pretrained("gpt2", config=cfg)
		self.gpt2.resize_token_embeddings(len(tokenizer))
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
		X = torch.cat([kwargs['X_in'], kwargs['X']], dim=1)
		X_mask = torch.cat([kwargs['in_mask'], kwargs['X_mask']], dim=1)
		X_label = torch.cat([kwargs['in_label'], kwargs['X_label']], dim=1)
		X = self.gpt2(inputs_embeds=X, 
						attention_mask=X_mask, 
						labels=X_label, 
						inputs_embeds_as_hidden_states=True)
		return X

	def generate(self, **kwargs):
		X_t = self.embed(kwargs['X'])
		X_in = kwargs['X_in']

		generations = []
		tokenizer = kwargs['tokenizer']
		for i in range(X_t.size(0)):
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
					context_embeds=X_t[i].unsqueeze(0),
					decoder_input_embeds_as_prompt=True,)
			X_gen_ids = X_gen[0].tolist()
			start, end = X_gen_ids.index(tokenizer.bos_token_id)+1, X_gen_ids.index(tokenizer.eos_token_id)
			generations.append(tokenizer.decode(X_gen[0][start:end], skip_special_tokens=False).strip())
		return generations

class GPT2pp(GPT2):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

	def forward(self, **kwargs):
		X_t = self.embed(kwargs['txt_id'])

		X = torch.cat([kwargs['X_im'], X_t], dim=1)
		X_mask = torch.cat([kwargs['im_mask'], kwargs['txt_mask']], dim=1)
		X_label = torch.cat([kwargs['im_label'], kwargs['txt_label']], dim=1)
		X = self.gpt2(inputs_embeds=X,
						attention_mask=X_mask, 
						labels=X_label, 
						inputs_embeds_as_hidden_states=True)
		return X

	def generate(self, **kwargs):
		X_t = self.embed(kwargs['txt_id'])
		X_im = kwargs['X_im']

		generations = []
		tokenizer = kwargs['tokenizer']
		for i in range(X_t.size(0)):
			prompt = kwargs['txt_id'][i].unsqueeze(0)
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
					image_embeds=X_im[i].unsqueeze(0),
					context_embeds=X_t[i].unsqueeze(0),
					decoder_input_embeds_as_prompt=True,)
			X_gen_ids = X_gen[0].tolist()
			start, end = X_gen_ids.index(tokenizer.bos_token_id)+1, X_gen_ids.index(tokenizer.eos_token_id)
			generations.append(tokenizer.decode(X_gen[0][start:end], skip_special_tokens=False).strip())
		return generations

class GPT2Adapter(GPT2):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

	def forward(self, **kwargs):
		X = self.gpt2(kwargs['X'], 
					attention_mask=kwargs['X_mask'], 
					labels=kwargs['X_label'], 
					encoder_hidden_states=kwargs['X_in'])
		return X

	def generate(self, **kwargs):
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
			X_gen_ids = X_gen[0].tolist()
			start, end = X_gen_ids.index(tokenizer.bos_token_id)+1, X_gen_ids.index(tokenizer.eos_token_id)
			generations.append(tokenizer.decode(X_gen[0][start:end], skip_special_tokens=False).strip())
		return generations

class T5(nn.Module):
	def __init__(self, finetune=False):
		super(T5, self).__init__()
		self.t5 = T5ForConditionalGeneration.from_pretrained("t5-base")
		self.finetune_t5(finetune)

	def finetune_t5(self, ft):
		for n, p in self.t5.decoder.named_parameters():
			if ('decoder' not in n) or (not any([True if (l in n) else False for l in T5_ADAPTER_LAYERS])):
				p.requires_grad = ft

	def forward(self, **kwargs):
		X_t = self.t5.shared(kwargs['txt_id'])
		X = torch.cat([kwargs['X_im'], X_t], dim=1)
		X_mask = torch.cat([kwargs['im_mask'], kwargs['txt_mask']], dim=1)
		X_label = kwargs['X_label']

		X = self.t5(inputs_embeds=X,
					attention_mask=X_mask,
					labels=X_label)
		return X

	def generate(self, **kwargs):
		X_t = self.t5.shared(kwargs['txt_id'])
		X = torch.cat([kwargs['X_im'], X_t], dim=1)
		X_mask = torch.cat([kwargs['im_mask'], kwargs['txt_mask']], dim=1)

		tokenizer = kwargs['tokenizer']
		X_gen = self.t5.generate(
				inputs_embeds=X,
				attention_mask=X_mask,
				max_length=100,
				num_beams=5,
				num_return_sequences=1,
				early_stopping=True,
				eos_token_id=tokenizer.eos_token_id,
				forced_eos_token_id=tokenizer.eos_token_id,)
		generations = [tokenizer.decode(g, skip_special_tokens=True) for g in X_gen]
		return generations

class T5Adapter(T5):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

	def forward(self, **kwargs):
		X = self.t5(encoder_outputs=(kwargs['X_in'],),
					labels=kwargs['X_label'])
		return X
	def generate(self, **kwargs):
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

class ResNet(nn.Module):
	def __init__(self, name, d_emb, n_emb, finetune=False):
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
			X = self.fc(self.avgpool(X_fxf).view(-1, self.d))
			X_fxf = X_fxf.view(-1, self.d, f*f).transpose(1, 2)
		else:
			X_fxf = X_fxf.view(-1, self.d, f*f).transpose(1, 2)
			X = self.fc(X_fxf)
		return X, X_fxf

class RoBERTa(nn.Module):
	def __init__(self, d_emb):
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

	def forward(self, X, pool=False):
		X = self.roberta.extract_features(X, return_all_hiddens=False)
		if pool:
			X_cls = self.fc(X[:, 0, :]) # (b, d_im)
			return X, X_cls
		else:
			X = self.fc(X)
			return X, X[:, 0, :]

class GRU(nn.Module):
	def __init__(self, d_emb, text_embed='glove', cache_dir=None):
		super(GRU, self).__init__()
		if 'fasttext' == text_embed:
			self.wemb = torchtext.vocab.FastText(cache=cache_dir)
		elif 'glove' == text_embed:
			self.wemb = torchtext.vocab.GloVe(cache=cache_dir)
		else:
			raise ValueError(f"{text_embed} Text Embedding for GRU is not supported.")
		self.rnn = nn.GRU(300, d_emb//2, bidirectional=True, batch_first=True)
		
	def forward(self, X, X_mask):
		X = self.wemb(txt_id)
		X_len = X_mask.sum(dim=-1)

		# Forward propagate RNNs
		packed = pack_padded_sequence(X, X_len.cpu(), batch_first=True, enforce_sorted=False)
		if torch.cuda.device_count() > 1:
			self.rnn.flatten_parameters()
		rnn_out, _ = self.txt_encoder[1](packed)
		padded = pad_packed_sequence(rnn_out, batch_first=True)

		X_cls = []
		for b in range(X.size(0)):
			X_cls .append(padded[0][b][X_len[b] - 1, :])
		X_cls = torch.stack(X_cls)
		return X, X_cls