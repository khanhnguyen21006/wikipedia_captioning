import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import *

class Model(nn.Module):
	def __init__(self, _config):
		super(Model, self).__init__()

		self._config = _config

		self.image_encoder, d_im = get_image_encoder(_config)
		self.text_encoder, d_txt = get_text_encoder(_config)
		self.text_decoder, _ = get_text_decoder(_config)

		n_emb = _config['n_embed']
		self.to_pool = n_emb > 0
		if self.to_pool:
			self.image_pooler = PCMENet(d_im, **_config)
			if self.text_encoder is not None:
				self.text_pooler = PCMENet(d_txt, **_config)
		# import pudb; pu.db
		# self.fuser = None

	def encode(self, batch):
		X_encode = dict()

		X_image = self.encode_image(batch)
		X_encode.update({'image': X_image})
		
		[
			X_encode.update({k: self.encode_text(batch, k)})
			for k in TEXT_ENCODER_KEYS if k in batch
		]

		[
			X_encode.update({k: {
				f'{k}_id': batch[f'{k}_id'], 
				f'{k}_mask': batch[f'{k}_mask']
			}})
			for k in TEXT_DECODER_KEYS if f'{k}_id' in batch
		]
		return X_encode

	def encode_image(self, batch):
		images = batch['image']
		X_encode = dict()

		X, X_cls = self.image_encoder(images)
		if self.to_pool:
			X, ret = self.image_pooler(X_cls, X)
			X_encode.update(ret)
		X_encode['embedding'] = X
		X_encode['cls_embedding'] = X_cls
		return X_encode

	def encode_text(self, batch, k):
		text_id, text_mask = batch[f'{k}_id'], batch[f'{k}_mask']
		X_encode = {
			f'{k}_id': text_id,
			f'{k}_mask': text_mask
		}
		if self.text_encoder is None:
			return X_encode
		elif isinstance(self.text_encoder, T5Adapter):
			X = self.text_encoder.t5.encoder(
				input_ids=text_id,
				attention_mask=text_mask,
			)
			X = X.last_hidden_state
			mask_expanded = text_mask.unsqueeze(-1).expand(X.size()).float()
			X_cls = torch.sum(X * mask_expanded, dim=1)/ mask_expanded.sum(dim=1)
			X_mask = ~text_mask.bool()
		else:
			X, X_cls, X_mask = self.text_encoder(text_id, mask=text_mask, pool=self.to_pool)
		if self.to_pool:
			X, ret = self.text_pooler(X_cls, X, mask=X_mask)
			X_encode.update(ret)
		X_encode['embedding'] = X
		X_encode['cls_embedding'] = X_cls
		return X_encode

	def decode(self, X_encode):
		if isinstance(self.text_encoder, T5Adapter):
			X_out, X_label = self.text_encoder(**X_encode)
		else:
			X_out, X_label = self.text_decoder(**X_encode)
		X_decode = {
			'loss': X_out['loss'],
			'logits': X_out['logits'][..., :-1, :].contiguous(),
			'label': X_label[..., 1:].contiguous(),
		}
		return X_decode

	def generate(self, batch, X_encode, **kwargs):
		X_generate = {k:v for k,v in batch.items() if k not in NUMERICS}
		if isinstance(self.text_encoder, T5Adapter):
			generations = self.text_encoder.generate(**X)
		else:
			generations = self.text_decoder.generate(**X_encode, **kwargs)
		X_generate['generated'] = generations
		return X_generate
