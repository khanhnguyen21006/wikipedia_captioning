import torch

from pycocoevalcap.bleu.bleu_scorer import BleuScorer
from pycocoevalcap.cider.cider_scorer import CiderScorer
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from .clipscore import clip, extract_all_images, get_clip_score, get_refonlyclipscore

import re, os, json, h5py, glob
import string, types, hashlib
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

import spacy
from spacy.tokens import Doc
from nltk.tokenize import word_tokenize

nlp = spacy.load("en_core_web_sm")

def caption_wrapup(outs, _config):
	rank = torch.distributed.get_rank()

	path = os.path.join(_config['result_dir'], 'inference', _config['expt_name'], 'caption')
	os.makedirs(path, exist_ok=True)

	curr_rank = os.path.join(path, f'generations_{rank}.jsonl')
	if os.path.exists(curr_rank):
		os.remove(curr_rank)
	with open(curr_rank, 'a+') as f:
		flatten = list()
		for out in outs:
			flatten = sum([[[k]*len(v), v] for k,v in out.items()], [])
			assert len(flatten) == 2*len(list(out.items()))
			for _t in zip(*flatten):
				f.write(f'{json.dumps({_t[i*2]:_t[i*2+1] for i in range(len(_t)//2)})}\n')

	torch.distributed.barrier()

	if rank == 0:
		jsonl = list()
		all_ranks = list(glob.glob(path + "/generations_*.jsonl"))
		for r in all_ranks:
			with open(r, "r") as f:
				jsonl.extend([jl for jl in list(f)])

		# For WIT, compute captioning metrics with pycoco scripts, while P&R similar to Tell
		if _config['dataset'] in ['wit', 'rlwit']:
			metrics = create_pycoco_files(jsonl, path)
			precision, recall = precison_recall_tell(jsonl)
			metrics.update({
				'Entity all - precision': precision,
				'Entity all - recall': recall
			})
		elif _config['dataset'] in ['goodnews', 'nytimes800k']:
			metrics = all_metrics_tell(jsonl, _config['data_folder'])
		else:
			raise ValueError(f"Invalid dataset: {_config['dataset']}.")
		metrics.update(clip_score(jsonl, _config['dataset'], _config['data_folder']))
		print(metrics)

		with open(os.path.join(path, 'metrics.json'), 'w') as f:
			json.dump(metrics, f)

	torch.distributed.barrier()
	os.remove(curr_rank)

def compute_entity(ent, c):
	caption_ents = ent['caption_ents']
	generated_ents = ent['generated_ents']

	c['n_caption_ents'] += len(caption_ents)
	c['n_gen_ents'] += len(generated_ents)
	for g_ent in generated_ents:
		if contain_entity(caption_ents, g_ent):
			c['n_gen_ent_matches'] += 1
	for c_ent in caption_ents:
		if contain_entity(generated_ents, c_ent):
			c['n_caption_ent_matches'] += 1
	return c

def contain_entity(ents, target):
	for ent in ents:
		if ent['text'] == target['text'] and ent['label'] == target['label']:
			return True
	return False

def get_entity(doc):
	ents = []
	for ent in doc.ents:
		ents.append({
			'text': ent.text,
			'label': ent.label_,
			'tokens': [{'text': tok.text, 'pos': tok.pos_} for tok in ent],
		})
	return ents

# Patch meteor scorer. See https://github.com/tylin/coco-caption/issues/25
def _stat(self, hypothesis_str, reference_list):
	# SCORE ||| reference 1 words ||| reference n words ||| hypothesis words
	hypothesis_str = hypothesis_str.replace('|||', '').replace('  ', ' ')
	score_line = ' ||| '.join(
		('SCORE', ' ||| '.join(reference_list), hypothesis_str))
	score_line = score_line.replace('\n', '').replace('\r', '')
	self.meteor_p.stdin.write('{}\n'.format(score_line).encode())
	self.meteor_p.stdin.flush()
	return self.meteor_p.stdout.readline().decode().strip()

# https://stackoverflow.com/questions/19790188/expanding-english-language-contractions-in-python
def decontracted(phrase):
	# specific
	phrase = re.sub(r"won\s*[\’\'\"]+\s*t", "will not", phrase)
	phrase = re.sub(r"can\s*[\’\'\"]+\s*t", "can not", phrase)
	# general
	phrase = re.sub(r"n\s*[\’\'\"]+\s*t", " not", phrase)
	phrase = re.sub(r"\s*[\’\'\"]+\s*re", " are", phrase)
	phrase = re.sub(r"\s*[\’\'\"]+\s*s", " is", phrase)
	phrase = re.sub(r"\s*[\’\'\"]+\s*d", " would", phrase)
	phrase = re.sub(r"\s*[\’\'\"]+\s*ll", " will", phrase)
	phrase = re.sub(r"\s*[\’\'\"]+\s*t", " not", phrase)
	phrase = re.sub(r"\s*[\’\'\"]+\s*ve", " have", phrase)
	phrase = re.sub(r"\s*[\’\'\"]+\s*m", " am", phrase)
	return phrase

def spacize(text):
	key = hashlib.sha256(text.encode('utf-8')).hexdigest()
	return Doc(nlp.vocab).from_bytes(nlp(text).to_bytes())

def jaccard(str1, str2):
	str2 = set(str2)
	if len(str2) == 0:
		return 0
	str1 = set(str1)
	return len(str2.intersection(str1)) / len(str2)

def precison_recall_tell(jsonl):
	count = 0
	ent_counter = defaultdict(int)

	for jline in tqdm(jsonl):
		jline = json.loads(jline)
		try:
			c_doc = spacize(jline['caption'])
			g_doc = nlp(jline['generated'])
			caption_ents = get_entity(c_doc)
			generated_ents = get_entity(g_doc)
		except ValueError:
			count+=1
			print(count)
			continue
		ent = {
			'caption_ents': caption_ents,
			'generated_ents': generated_ents,
		}
		compute_entity(ent, ent_counter)

	precision = {
		'count': ent_counter['n_gen_ent_matches'],
		'total': ent_counter['n_gen_ents'],
		'percentage': ent_counter['n_gen_ent_matches'] / ent_counter['n_gen_ents'],
	}
	recall = {
		'count': ent_counter['n_caption_ent_matches'],
		'total': ent_counter['n_caption_ents'],
		'percentage': ent_counter['n_caption_ent_matches'] / ent_counter['n_caption_ents'],
	}
	return precision, recall

def all_metrics_tell(jsonl, data_folder):
	bleu_scorer = BleuScorer(n=4)
	rouge_scorer = Rouge()
	rouge_scores = []
	cider_scorer = CiderScorer(n=4, sigma=6.0)
	meteor_scorer = Meteor()
	meteor_scorer._stat = types.MethodType(_stat, meteor_scorer)
	meteor_scores = []
	eval_line = 'EVAL'
	meteor_scorer.lock.acquire()

	count = 0
	ent_counter = defaultdict(int)

	# Use Tell's output as reference
	tell = dict()
	tell_jsonl = list(open(os.path.join(data_folder, 'generations.jsonl'), "r"))
	for tell_jline in tqdm(tell_jsonl):
		tell_jline = json.loads(tell_jline)
		tell.update(
			{
				tell_jline['image_path'].split('/')[-1].split('.')[0]: {
					'caption': tell_jline['raw_caption'],
					# 'caption_NEs': tell_jline['caption_ents'],
				}
			}
		)

	for jline in tqdm(jsonl):
		jline = json.loads(jline)
		if jline['image_id'] not in tell:
			continue

		caption = jline['caption']
		generated = jline['generated']
		caption = re.sub(r'[^\w\s]', '', caption)
		try:
			generated = re.sub(r'[^\w\s]', '', generated)
		except TypeError:
			continue

		bleu_scorer += (generated, [caption])
		rouge_score = rouge_scorer.calc_score([generated], [caption])
		rouge_scores.append(rouge_score)
		cider_scorer += (generated, [caption])
		stat = meteor_scorer._stat(generated, [caption])
		eval_line += ' ||| {}'.format(stat)
		count += 1

		c_doc = spacize(jline['caption'])
		g_doc = nlp(jline['generated'])
		caption_ents = get_entity(c_doc)
		generated_ents = get_entity(g_doc)

		ent = {
			'caption_ents': caption_ents,
			'generated_ents': generated_ents,
		}
		compute_entity(ent, ent_counter)

	meteor_scorer.meteor_p.stdin.write('{}\n'.format(eval_line).encode())
	meteor_scorer.meteor_p.stdin.flush()
	for _ in range(count):
		meteor_scores.append(float(meteor_scorer.meteor_p.stdout.readline().strip()))
	meteor_score = float(meteor_scorer.meteor_p.stdout.readline().strip())
	meteor_scorer.lock.release()

	blue_score, _ = bleu_scorer.compute_score(option='closest')
	rouge_score = np.mean(np.array(rouge_scores))
	cider_score, _ = cider_scorer.compute_score()

	metrics = {
		'BLEU-1': blue_score[0],
		'BLEU-2': blue_score[1],
		'BLEU-3': blue_score[2],
		'BLEU-4': blue_score[3],
		'ROUGE': rouge_score,
		'CIDEr': cider_score,
		'METEOR': meteor_score,
		'Entity all - precision': {
			'count': ent_counter['n_gen_ent_matches'],
			'total': ent_counter['n_gen_ents'],
			'percentage': ent_counter['n_gen_ent_matches'] / ent_counter['n_gen_ents'],
		},
		'Entity all - recall': {
			'count': ent_counter['n_caption_ent_matches'],
			'total': ent_counter['n_caption_ents'],
			'percentage': ent_counter['n_caption_ent_matches'] / ent_counter['n_caption_ents'],
		},
	}
	return metrics

def create_pycoco_files(jsonl, path, split=False):
	def convert_format(d):
		d['references'][u'images'] = [dict({u'id': k}, **v) for k, v in d['references'][u'images'].items()]
		d['references'][u'annotations'] = [dict({u'id': k}, **v) for k, v in d['references'][u'annotations'].items()]
		d['hypotheses'] = [dict({u'image_id': k}, **v) for k, v in d['hypotheses'].items()]
		return d

	ret = {
		'references': {u'info': {}, u'images': dict(), u'licenses': {}, u'type': u'captions', u'annotations': dict()},
		'hypotheses': dict(),
		'inference': dict(),
	}
	easy = {
		'references': {u'info': {}, u'images': dict(), u'licenses': {}, u'type': u'captions', u'annotations': dict()},
		'hypotheses': dict(),
	}
	hard = {
		'references': {u'info': {}, u'images': dict(), u'licenses': {}, u'type': u'captions', u'annotations': dict()},
		'hypotheses': dict(),
	}

	for idx, jline in tqdm(enumerate(jsonl)):
		jline = json.loads(jline)
		im_id = jline['image_id']
		image_url = jline['image_url'] if jline.get('', None) else ''
		reference = jline['caption']
		generated = jline['generated']
		context = jline['section']

		ret['references'][u'images'][im_id] = {u'license': 3, u'file_name': str(idx)}
		ret['references'][u'annotations'][im_id] = {u'image_id': im_id, u'caption': reference.strip()}
		ret['hypotheses'][im_id] = {u'caption': generated.strip()}

		r_tokens = [word for word in word_tokenize(reference) if not re.fullmatch('[' + string.punctuation + ']+', word)]
		g_tokens = [word for word in word_tokenize(generated) if not re.fullmatch('[' + string.punctuation + ']+', word)]
		c_tokens = [word for word in word_tokenize(context) if not re.fullmatch('[' + string.punctuation + ']+', word)]

		rc_overlap = jaccard(r_tokens, c_tokens)
		gc_overlap = jaccard(g_tokens, c_tokens)

		ret['inference'][im_id] = {
			'context': context,
			'reference': reference,
			'generated': generated,
			'r_length': len(r_tokens),
			'g_length': len(g_tokens),
			'rc_jaccard': rc_overlap,
			'gc_jaccard': gc_overlap,
			'image_url': image_url,
		}

		if rc_overlap >= 0.5:
			easy['references'][u'images'][im_id] = {u'license': 3, u'file_name': str(idx)}
			easy['references'][u'annotations'][im_id] = {u'image_id': im_id, u'caption': reference.strip()}
			easy['hypotheses'][im_id] = {u'caption': generated.strip()}  # u'image_id'
		else:
			hard['references'][u'images'][im_id] = {u'license': 3, u'file_name': str(idx)}
			hard['references'][u'annotations'][im_id] = {u'image_id': im_id, u'caption': reference.strip()}
			hard['hypotheses'][im_id] = {u'caption': generated.strip()}

	convert_format(ret)
	assert len(ret['references'][u'annotations']) == len(ret['hypotheses'])
	with open(os.path.join(path, 'references.json'), 'w') as f:
		json.dump(ret['references'], f)
	with open(os.path.join(path, 'generated.json'), 'w') as f:
		json.dump(ret['hypotheses'], f)

	df = pd.DataFrame([dict({'image_id': k}, **v) for k,v in ret['inference'].items()])
	df.to_csv(os.path.join(path, 'inference.csv'), index=False)
	print('Saving files to : ', path)

	if split:
		convert_format(easy)
		assert len(easy['references'][u'annotations']) == len(easy['hypotheses'])
		with open(os.path.join(path, 'easy_references.json'), 'w') as f:
			json.dump(easy['references'], f)
		with open(os.path.join(path, 'easy_generated.json'), 'w') as f:
			json.dump(easy['hypotheses'], f)

		convert_format(hard)
		assert len(hard['references'][u'annotations']) == len(hard['hypotheses'])
		with open(os.path.join(path, 'hard_references.json'), 'w') as f:
			json.dump(hard['references'], f)
		with open(os.path.join(path, 'hard_generated.json'), 'w') as f:
			json.dump(hard['hypotheses'], f)
	returned_metrics = dict({
			'average caption length': df.r_length.mean(),
			'average generated length': df.g_length.mean(),
			'average caption jaccard': df.rc_jaccard.mean(),
			'average generated jaccard': df.gc_jaccard.mean()
		})
	return returned_metrics

def clip_score(jsonl, ds_name, ds_path, return_per_instance_scores=False):
	test_images = h5py.File(os.path.join(ds_path, f'test_IMAGES_{ds_name}.hdf5'), 'r')['images']
	test_image_ids = json.load(open(os.path.join(ds_path, f'test_IMAGEIDS_{ds_name}.json'), 'r'))

	images, image_ids, candidates, references = [], [], [], []
	for jline in tqdm(jsonl):
		jline = json.loads(jline)
		images.append(test_images[test_image_ids.index(jline['image_id'])])
		image_ids.append(jline['image_id'])
		candidates.append(jline['generated'])
		references.append([jline['caption']])
	# import pudb; pu.db
	# images = np.stack(images)

	device = "cuda" if torch.cuda.is_available() else "cpu"
	model, transform = clip.load("ViT-B/32", device=device, jit=False)
	model.eval()

	image_feats = extract_all_images(images, model, device, batch_size=64, num_workers=8)

	# get image-text clipscore
	_, per_instance_image_text, candidate_feats = get_clip_score(model, image_feats, candidates, device)

	# get text-text clipscore
	_, per_instance_text_text = get_refonlyclipscore(model, references, candidate_feats, device)

	# F-score
	refclipscores = 2 * per_instance_image_text * per_instance_text_text / (per_instance_image_text + per_instance_text_text)
	per_instance_scores = {image_id: {'CLIPScore': float(clipscore), 'RefCLIPScore': float(refclipscore)}
						for image_id, clipscore, refclipscore in zip(image_ids, per_instance_image_text, refclipscores)}
	final_scores = {
		'CLIPScore': np.mean([s['CLIPScore'] for s in per_instance_scores.values()]),
		'RefCLIPScore': np.mean([s['RefCLIPScore'] for s in per_instance_scores.values()])
	}
	if return_per_instance_scores:
		return final_scores, per_instance_scores
	else:
		return final_scores

def test():
	pass

if __name__ == '__main__':
	test()
