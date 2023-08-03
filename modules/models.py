import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_pool import *
from .data_pool import *
from .module_utils import *

from utils import *

def build_model(_config):
	model_type = _config["model"]
	if model_type == "multi_encoder_single_decoder":
		model = MESDModel(_config)
	elif model_type == "rl_t5_caption_model":
		model = RLT5CaptionModel(_config)
	else:
		raise ValueError(f"{model_type} Model is not supported.")
	return model

class MESDModel(nn.Module):
	def __init__(self, _config):
		super(MESDModel, self).__init__()

		self._config = _config

		self.image_encoder, d_im = get_image_encoder(_config)
		self.text_encoder, d_txt = get_text_encoder(_config)
		self.text_decoder, _ = get_text_decoder(_config)

		n_emb = _config['n_embed']
		self.pool = n_emb > 0
		if self.pool:
			self.image_pooler = get_image_pooler(d_im, _config)
			if self.text_encoder is not None:
				self.text_pooler = get_text_pooler(d_txt, _config)
		self.fuser = get_fuser(_config)

	def forward(self, batch):
		return self._encode(batch)

	def _encode(self, batch):
		X_encode = dict()

		X_image = self._encode_image(batch)
		X_encode.update({'image': X_image})

		[
			X_encode.update({k: self._encode_text(batch, k)})
			for k in TEXT_ENCODER_KEYS if k in batch
		]
		if self.fuser is not None:
			X_encode = self.fuser(X_encode)
		if self.text_decoder is not None:
			[
				X_encode.update({k: {
					f'{k}_id': batch[f'{k}_id'], 
					f'{k}_mask': batch[f'{k}_mask']
				}})
				for k in TEXT_DECODER_KEYS if f'{k}_id' in batch
			]
		return X_encode

	def _encode_image(self, batch):
		images = batch['image']
		X_encode = dict()
		X, X_cls = self.image_encoder(images)
		if self.pool:
			X, ret = self.image_pooler(X_cls, X)
			X_encode.update(ret)
		X_encode['embedding'] = X
		X_encode['cls_embedding'] = X_cls
		X_encode['mask'] = torch.ones(X.shape[:2], device=X.device).long()
		return X_encode

	def _encode_text(self, batch, k):
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
			X_mask = text_mask.bool()
		else:
			X, X_cls, X_mask = self.text_encoder(text_id, mask=text_mask, pool=self.pool)
		if self.pool:
			X, ret = self.text_pooler(X_cls, X, mask=X_mask)
			X_mask = torch.ones(X.shape[:2], device=X.device).long()
			X_encode.update(ret)
		X_encode['embedding'] = X
		X_encode['cls_embedding'] = X_cls
		X_encode['mask'] = X_mask.long()
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

class RLT5CaptionModel(nn.Module):
	def __init__(self, _config):
		super(RLT5CaptionModel, self).__init__()

		self._config = _config

		if self._config["clip_ckpt"] is not None:
			cf_copy = self._config.copy()
			cf_copy.update({"model": 'multi_encoder_single_decoder', "text_decoder": None})

			pt_clip = build_model(cf_copy)
			clip_ckpt = torch.load(self._config["clip_ckpt"], map_location='cpu')
			clip_state_dict = clip_ckpt['state_dict']
			pt_clip.load_state_dict(clip_state_dict, strict=False)

			self.image_encoder = pt_clip.image_encoder
			self.text_encoder = pt_clip.text_encoder

			del pt_clip
			set_finetune(self.image_encoder, self._config["image_encoder_finetune"])
			set_finetune(self.text_encoder, self._config["text_encoder_finetune"])
		else:
			self.image_encoder, _ = get_image_encoder(_config)  # CLIP IE
			self.text_encoder, _ = get_text_encoder(_config)  # CLIP TE
		self.text_decoder, d_emb = get_text_decoder(_config)  # T5

		self.lin = nn.Sequential(
			nn.Linear(d_emb, d_emb),
			nn.LayerNorm(d_emb),
			# nn.Dropout(),
			nn.GELU(),
		)
		vilt_init_weights(self.lin[0])

	def forward(self, batch, sc_flag, **kwargs):
		X_encode = self._encode(batch)
		if sc_flag:
			import pudb; pu.db
			if kwargs["cider_baseline"] == "greedy":
				self.eval()
				with torch.no_grad():
					sample_inputs = self._prepare_sample_inputs(**X_encode)
					baseline_seq, _ = self._sample(greedy=True, **sample_inputs, **kwargs)
			else:
				assert kwargs['cider_baseline'] in ["description", "caption"]
				baseline_seq = batch[f"{kwargs['cider_baseline']}_id"]

			self.train()
			sample_inputs = self._prepare_sample_inputs(sample_n=kwargs["sample_n"], **X_encode)
			sample_seq, sample_logprobs = self._sample(**sample_inputs, **kwargs)

			outputs = {
				"baseline_seq": baseline_seq,
				"sample_seq": sample_seq,
				"sample_logprobs": sample_logprobs,
				"caption": batch["caption"], # this will be both CIDEr&CLIP(ground truth)
				"caption_id": batch["caption_id"],
				"caption_mask": batch["caption_mask"],
				f"{kwargs['cider_baseline']}": batch[f"{kwargs['cider_baseline']}"], # this will be CIDEr(baseline)
				f"{kwargs['cider_baseline']}_id": batch[f"{kwargs['cider_baseline']}_id"],
				f"{kwargs['cider_baseline']}_mask": batch[f"{kwargs['cider_baseline']}_mask"],
				"image_cls": X_encode["image"]["cls_embedding"],
			}
			if kwargs['clip_baseline'] != "":
				assert kwargs['clip_baseline'] in ["description", "caption"]
				outputs.update({
					f"{kwargs['clip_baseline']}": batch[f"{kwargs['clip_baseline']}"], # this will be CLIP(baseline)
					f"{kwargs['clip_baseline']}_id": batch[f"{kwargs['clip_baseline']}_id"],
					f"{kwargs['clip_baseline']}_mask": batch[f"{kwargs['clip_baseline']}_mask"],
				})
		else:
			X_out, X_label = self.text_decoder(**X_encode)
			outputs = {
				'loss': X_out['loss'],
				'logits': X_out['logits'][..., :-1, :].contiguous(),
			}

		return outputs

	def _encode(self, batch):
		X_encode = dict()
		X_image, cls_image = self.image_encoder(batch['image'])
		X_image = self.lin(X_image)
		X_encode.update({"image": {
			"embedding": X_image,
			"cls_embedding": cls_image,
			"mask": torch.ones(X_image.shape[:2], device=X_image.device).long(),
		}})

		# X_description, cls_description, mask_description = self.text_encoder(batch["description_id"], mask=batch["description_mask"])
		# X_encode.update({"description": {
		# 	"embedding": X_description,
		# 	"cls_embedding": cls_description,
		# 	"mask": mask_description,
		# }})
				
		for k in ["section", "caption", "prompt"]:
			if f"{k}_id" in batch:
				X_encode.update({k: {
					f"{k}_id": batch[f'{k}_id'], 
					f"{k}_mask": batch[f'{k}_mask']
				}})
		return X_encode

	def _sample(self, greedy=False, **kwargs):
		# 1. Set generation parameters (bos, pad, eos, n_beam)
		# 2. Set output from encoder
		# 3. Define other model kwargs
		# 4. Define decoder input
		# 5. run search
		input_ids = kwargs["decoder_input_ids"]
		_bs = input_ids.size(0)

		unfinished = input_ids.new(_bs).fill_(1)
		# cur_len = input_ids.shape[-1]
		max_length = kwargs["sample_max_len"]

		dec_tokenizer = kwargs["dec_tokenizer"].tokenizer
		pad_token_id = dec_tokenizer.pad_token_id
		eos_token_id = dec_tokenizer.eos_token_id

		next_logits, next_log_probs = (), ()

		while True:
			model_inputs = self._prepare_model_inputs(**kwargs)

			outputs = self.text_decoder.t5(
				**model_inputs,
				return_dict=True,
			)

			next_token_logits = outputs.logits[:, -1, :]
			next_logits += (next_token_logits,)

			# missing a processing part (from raw logis to distribution) next_tokens_scores = logits_processor(input_ids, next_token_logits)
			if not greedy:
				probs = F.softmax(next_token_logits, dim=-1)
				next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
				next_log_probs += (torch.log(probs),)
			else:
				next_tokens = torch.argmax(next_token_logits, dim=-1)
				log_probs = F.log_softmax(next_token_logits, dim=-1)
				next_log_probs += (log_probs,)

			next_tokens = next_tokens * unfinished + pad_token_id * (1 - unfinished)

			input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)

			if "past_key_values" in outputs:
				model_inputs["past_key_values"] = outputs.past_key_values
			# cur_len = cur_len + 1

			if eos_token_id is not None:
				unfinished = unfinished.mul((next_tokens != eos_token_id).long())

			if unfinished.max() == 0 or input_ids.size(-1) >= max_length:
				break

		return input_ids, next_log_probs

	def _prepare_sample_inputs(self, sample_n=1, past=None, **kwargs):
		text_id, mask_text, X_image, mask_image = *(
			repeat_tensor_batch_dim(sample_n, x) 
			for x in [kwargs["section"]["section_id"], kwargs["section"]["section_mask"],\
						kwargs["image"]["embedding"], kwargs["image"]["mask"]]
		),

		X_section = self.text_decoder.t5.shared(text_id)
		X_input = torch.cat([X_image, X_section], dim=1)
		mask_input = torch.cat([mask_image, mask_text], dim=1)
		encoder_outputs = self.text_decoder.t5.encoder(
								inputs_embeds=X_input, 
								attention_mask=mask_input,
								output_attentions=False, 
								output_hidden_states=False, 
								return_dict=True)

		_bs = X_section.size(0)
		input_ids = torch.zeros((_bs, 1), dtype=torch.long, device=X_section.device)
		ret_dict = {
			"decoder_input_ids": input_ids,
			"attention_mask": mask_input,
			"encoder_outputs": encoder_outputs,
			'head_mask': None, 
			'decoder_head_mask': None, 
			'cross_attn_head_mask': None,
			"past_key_values": past,
		}
		return ret_dict

	def _prepare_model_inputs(self, **kwargs):
		return {
			"decoder_input_ids": kwargs["decoder_input_ids"],
			"attention_mask": kwargs["attention_mask"],
			"use_cache": kwargs["use_cache"],
			"encoder_outputs": kwargs["encoder_outputs"],
			"head_mask": kwargs["head_mask"], 
			"decoder_head_mask": kwargs["decoder_head_mask"], 
			"cross_attn_head_mask": kwargs["cross_attn_head_mask"],
			"past_key_values": kwargs["past_key_values"],
		}