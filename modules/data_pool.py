from torchvision import transforms as t
from datasets import WitDataset, CocoDataset, WitToyDataset, GoodNewsDataset

image_size = 256

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
	elif name == 'wit_toy':
		return WitToyDataset
	elif name == 'goodnews':
		return GoodNewsDataset
	# elif name == 'nytimes800k':
	# 	return NYTimesDataset
	else:
		raise Exception(f"{key} Dataset is not supported.")

COMPOSITION = ['section_caption', 'section_prompt']
WIT_COMPOSITION = ['section_caption', 'section_prompt']  # 'section_caption', 'description_caption', 'description_section'
NUMERICS = ['image', 'description_id', 'description_mask', 'section_id', 'section_mask',\
				 'caption_id', 'caption_mask', 'prompt_id', 'prompt_mask']
def get_collate_hparams(_config):
	name, text_ml, encoder, decoder = _config["dataset"], _config["text_max_len"],\
									 _config["text_encoder"], _config["text_decoder"]
	
	is_gpt2_decoder = 'gpt2' in decoder
	is_gpt2pp_encoder = 'gpt2++' == encoder
	use_adapter = 'adapter' in encoder

	if not use_adapter: # for non-adapter models, images are processed separately
		if image_size == 256:
			text_ml = text_ml - 64
		elif image_size == 224:
			text_ml = text_ml - 49

	if name == "wit":
		context_keys = ['description', 'section'] \
						+ (WIT_COMPOSITION if is_gpt2pp_encoder else [])
	else:
		context_keys = ['section'] \
						+ (COMPOSITION if is_gpt2pp_encoder else [])
	target_keys = ['caption'] + (['prompt'] if is_gpt2_decoder else [])

	return {'context_keys': context_keys, 'target_keys': target_keys, 'text_ml': text_ml}
