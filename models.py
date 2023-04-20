import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import *
import objectives

class Model(nn.Module):
	def __init__(self, _config):
		super(Model, self).__init__()

		self._config = _config

		self.im_encoder, d_im = get_image_encoder(_config)
		self.txt_encoder, d_txt = get_text_encoder(_config)
		self.txt_decoder, _ = get_text_decoder(_config)

		n_emb = _config['n_embeds']
		d_emb = _config['embed_dim']

		self.to_pool = n_emb > 0
		if self.to_pool:
			self.im_pooler = PIENet(n_emb, d_im, d_emb, d_im//2) 
			self.txt_pooler = PIENet(n_emb, d_txt, d_emb, d_txt//2)
		# import pudb; pu.db
		# self.fuser = None

	def encode(self, batch):
		X_encode = dict()

		X_image = self.encode_image(batch)
		X_section = self.encode_text(batch)

		X_encode.update({
			'image': X_image,
			'section': X_section,
			'caption': {
				'caption_id': batch['caption_id'],
				'caption_mask': batch['caption_mask']
			},
		})
		return X_encode

	def encode_image(self, batch):
		images = batch['image']
		m_name = self._config['image_encoder']

		X_encode = dict()
		if 'resnet' in m_name:
			X, X_fxf = self.im_encoder(images)
			if self.to_pool:
				X, ret = self.pool(self.im_pooler, X, X_fxf)
				X_encode.update(ret)  # (b, n_emb, d_im)
			X_encode['embedding'] = X  # (cls_emb+residual) (b, n_emb, d_im)
		elif 'vit' in m_name or 'clip' in m_name:
			X = self.im_encoder(images)
			if self.to_pool:
				X, ret = self.pool(self.im_pooler, X.pooler_output, X.last_hidden_state[:, 1:, :])
				X_encode.update(ret)  # (b, n_emb, d_im)
			X_encode['embedding'] = X # (b, n_emb, d_im)
		else:
			raise Exception(f"{m_name} Image Encoder is not supported.")
		return X_encode

	def encode_text(self, batch):
		txt_id, txt_mask = batch['section_id'], batch['section_mask']
		m_name = self._config['text_encoder']

		X_encode = {
			'section_id': txt_id,
			'section_mask': txt_mask
		}
		if m_name == 'roberta':
			X, X_cls = self.txt_encoder(txt_id, self.to_pool)
			if self.to_pool:
				X, ret = self.pool(self.txt_pooler, X_cls, X[:, 1:, :], mask=~txt_mask[:, 1:].bool())
				X_encode.update(ret)
			X_encode['embedding'] = X
		elif m_name == 'gpt2++':
			X_encode['section_caption_id'] = batch['section_caption_id']
			X_encode['section_caption_mask'] = batch['section_caption_mask']
		elif m_name == 't5++':
			pass
		elif m_name == 't5-adapter':
			txt_id = batch['section_id']
			txt_mask = batch['section_mask']

			X = self.txt_encoder.t5.encoder(
				input_ids=txt_id,
				attention_mask=txt_mask,
			)
			X = X.last_hidden_state
			X_cls = torch.mean(X, dim=1)
			if self.to_pool:
				X, ret = self.pool(self.txt_pooler, X_cls, X[:, 1:, :], mask=~txt_mask[:, 1:].bool())
				X_encode.update(ret)
			X_encode['embedding'] = X
		elif m_name == 'sbert':
			return
		elif m_name == 'gru':
			X, X_cls = self.txt_encoder(txt_id, txt_mask)
			if self.to_pool:
				X, ret = self.pool(self.txt_pooler, X_cls, X, mask=~txt_mask.bool())
				X_encode.update(ret)
			X_encode['embedding'] = X
		else:
			raise Exception(f"{name} Text Encoder is not supported.")
		return X_encode

	def pool(self, pooler, X_cls, X_seq, mask=None):
		X_pool = dict()
		X_pool['cls_embedding'] = X_cls

		X, attn, res = pooler(X_cls, X_seq, mask)
		X_pool['attention'] = attn # (b, n_emb)
		X_pool['residual'] = res # (b, n_emb, d_im)  # residual is not normalized at this point

		X = F.normalize(X, p=2, dim=-1)
		return X, X_pool

	def decode(self, X_encode):
		m_name = self._config['text_decoder']
		if m_name == 'gpt2++':
			txt_id = X_encode['section']['section_caption_id']
			txt_mask = X_encode['section']['section_caption_mask']
			txt_label = torch.where((txt_mask != 0) & (txt_mask != -100), txt_id, -100)
			txt_mask[txt_mask == -100] = 1

			X_im = X_encode['image']['embedding']
			bs, iml = X_im.size(0), X_im.size(1)
			im_mask = torch.ones((bs, iml), dtype=torch.long, device=X_im.device)
			im_label = im_mask * (-100)

			X = {
				'txt_id': txt_id,
				'txt_mask': txt_mask,
				'txt_label': txt_label,
				'X_im': X_im,
				'im_mask': im_mask,
				'im_label': im_label,
			}
			X_label = txt_label
			X_out = self.txt_decoder(**X)
		elif 'gpt2' in m_name:
			X = X_encode['caption']['caption_id']
			X_mask = X_encode['caption']['caption_mask'].bool()
			X_label = X.masked_fill(~X_mask, -100)

			X_in = torch.cat([X_encode['image']['embedding'], X_encode['section']['embedding']], dim=1)
			bs, ml = X_in.size(0), X_in.size(1)
			in_mask = torch.ones((bs, ml), dtype=torch.long, device=X_in.device)
			in_label = in_mask * (-100)

			X = {
				'X': X,
				'X_mask': X_mask,
				'X_label': X_label,
				'X_in': X_in,
				'in_mask': in_mask,
				'in_label': in_label,
			}
			X_out = self.txt_decoder(**X)
		elif m_name == 't5++':
			txt_id = X_encode['section']['section_id']
			txt_mask = X_encode['section']['section_mask']
			txt_label = X_encode['caption']['caption_id']
			txt_label[txt_label == 0] = -100

			X_im = X_encode['image']['embedding']
			bs, iml = X_im.size(0), X_im.size(1)
			im_mask = torch.ones((bs, iml), dtype=torch.long, device=X_im.device)
			
			X_label = txt_label
			X = {
				'txt_id': txt_id,
				'txt_mask': txt_mask,
				'X_im': X_im,
				'im_mask': im_mask,
				'X_label': X_label,
			}
			X_out = self.txt_decoder(**X)
		elif m_name == 't5-adapter':
			txt_label = X_encode['caption']['caption_id']
			txt_label[txt_label == 0] = -100

			X_in = torch.cat([X_encode['image']['embedding'], X_encode['section']['embedding']], dim=1)
			X_label = txt_label

			X = {
				'X_in': X_in,
				'X_label': X_label
			}
			X_out = self.txt_encoder(**X)
		else:
			raise Exception(f"{name} Text Decoder is not supported.")
		X_decode = {
			'loss': X_out['loss'],
			'logits': X_out['logits'][..., :-1, :].contiguous(),
			'label': X_label[..., 1:].contiguous(),
		}
		return X_decode

	def generate(self, batch, X_encode, tokenizer=None):
		m_name = self._config['text_decoder']
		X_generate = {k:v for k,v in batch.items() if k not in NUMERICS}
		if m_name == 'gpt2++':
			txt_id = batch['section_prompt_id']
			X_im = X_encode['image']['embedding']

			X = {
				'txt_id': txt_id,
				'X_im': X_im,
				'tokenizer': tokenizer,
				'max_len': get_collate_hparams(self._config)['text_ml']
			}
			generations = self.txt_decoder.generate(**X)
		elif m_name == 'gpt2' in m_name:
			X = batch['prompt_id']
			X_in = torch.cat([X_encode['image']['embedding'], X_encode['section']['embedding']], dim=1)

			X = {
				'X': X,
				'X_in': X_in,
				'tokenizer': tokenizer,
				'max_len': get_collate_hparams(self._config)['text_ml']
			}
			generations = self.txt_decoder.generate(**X)
		elif m_name == 'gpt2-adapter':
			X = batch['prompt_id']
			X_in = torch.cat([X_encode['image']['embedding'], X_encode['section']['embedding']], dim=1)

			X = {
				'X': X,
				'X_in': X_in,
				'tokenizer': tokenizer,
				'max_len': get_collate_hparams(self._config)['text_ml']
			}
			generations = self.txt_decoder.generate(**X)
		elif m_name == 't5++':
			txt_id = X_encode['section']['section_id']
			txt_mask = X_encode['section']['section_mask']

			X_im = X_encode['image']['embedding']
			bs, iml = X_im.size(0), X_im.size(1)
			im_mask = torch.ones((bs, iml), dtype=torch.long, device=X_im.device)

			X = {
				'txt_id': txt_id,
				'txt_mask': txt_mask,
				'X_im': X_im,
				'im_mask': im_mask,
				'tokenizer': tokenizer,
			}
			generations = self.txt_decoder.generate(**X)
		elif m_name == 't5-adapter':
			X_im = X_encode['image']['embedding']
			X_t = X_encode['section']['embedding']
			bs, iml, tml = X_im.size(0), X_im.size(1), X_t.size(1)
			im_mask = torch.ones((bs, iml), dtype=torch.long, device=X_im.device)
			txt_mask = torch.ones((bs, tml), dtype=torch.long, device=X_t.device)

			X_in = torch.cat([X_im, X_t], dim=1)
			in_mask = torch.cat([im_mask, txt_mask], dim=1)
			X = {
				'X_in': X_in,
				'in_mask': in_mask,
				'tokenizer': tokenizer,
			}
			generations = self.txt_encoder.generate(**X)
		else:
			raise Exception(f"{name} Text Decoder is not supported.")
		X_generate.update({'generated': generations})
		return X_generate
