import torch

from pycocoevalcap.bleu.bleu_scorer import BleuScorer
from pycocoevalcap.cider.cider_scorer import CiderScorer
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge

import re
import os
import sys
import json
import glob
import string
import types
import hashlib
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict

import spacy
from spacy.tokens import Doc
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nlp = spacy.load("en_core_web_sm")

def utility_wrapup(outs, _config):
	caption_wrapup(outs, _config)
	if _config['run_retrieval']:
		retrieve_wrapup(outs, _config)

def caption_wrapup(outs, _config):
	rank = torch.distributed.get_rank()

	path = os.path.join(_config['result_dir'], 'inference', _config['expt_name'], 'caption')
	os.makedirs(path, exist_ok=True)

	curr_rank = os.path.join(path, f'generations_{rank}.jsonl')
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
		if _config['dataset'] == 'wit':
			create_pycoco_files(jsonl, path)
			precision, recall = precison_recall_tell(jsonl)
			metrics = {
				'Entity all - precision': precision,
				'Entity all - recall': recall
			}
		elif _config['dataset'] == 'goodnews' or _config['dataset'] == 'nytimes800k':
			metrics = all_metrics_tell(jsonl, _config['data_folder'])
		else:
			raise ValueError(f"Invalid {_config['dataset']} dataset.")
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
	for tell_jl in tqdm(tell_jsonl):
		tell_jl = json.loads(tell_jl)
		tell.update(
			{
				tell_jl['image_path'].split('/')[-1].split('.')[0]: {
					'caption': tell_jl['raw_caption'],
					# 'caption_NEs': tell_jl['caption_ents'],
				}
			}
		)

	for jl in tqdm(jsonl):
		jl = json.loads(jl)
		if jl['image_id'] not in tell:
			continue

		caption = jl['caption']
		generated = jl['generated']
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

		c_doc = spacize(jl['caption'])
		g_doc = nlp(jl['generated'])
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
	ret = {
		'references': {u'info': {}, u'images': [], u'licenses': {}, u'type': u'captions', u'annotations': []},
		'hypotheses': list(),
		'inference': list(),
	}
	easy = {
		'references': {u'info': {}, u'images': [], u'licenses': {}, u'type': u'captions', u'annotations': []},
		'hypotheses': list(),
		'inference': list(),
	}
	hard = {
		'references': {u'info': {}, u'images': [], u'licenses': {}, u'type': u'captions', u'annotations': []},
		'hypotheses': list(),
		'inference': list(),
	}

	for idx, jline in tqdm(enumerate(jsonl)):
		jline = json.loads(jline)
		im_id = jline['image_id']
		image_url = jline['image_url'] if jline.get('', None) else ''
		reference = jline['caption']
		generated = jline['generated']
		context = jline['section']

		ret['references'][u'images'].append({u'license': 3, u'file_name': str(idx), u'id': im_id})
		ret['references'][u'annotations'].append({u'image_id': im_id, u'id': im_id, u'caption': reference.strip()})
		ret['hypotheses'].append({u'caption': generated.strip(), u'image_id': im_id})
			
		r_tokens = [word for word in word_tokenize(reference) if not re.fullmatch('[' + string.punctuation + ']+', word)]
		g_tokens = [word for word in word_tokenize(generated) if not re.fullmatch('[' + string.punctuation + ']+', word)]
		c_tokens = [word for word in word_tokenize(context) if not re.fullmatch('[' + string.punctuation + ']+', word)]
		
		rc_overlap = jaccard(r_tokens, c_tokens)
		gc_overlap = jaccard(g_tokens, c_tokens)

		ret['inference'].append({
			'image_id': im_id,
			'context': context,
			'reference': reference,
			'generated': generated,
			'r_length': len(r_tokens),
			'g_length': len(g_tokens),
			'rc_jaccard': rc_overlap,
			'gc_jaccard': gc_overlap,
			'image_url': image_url,
		})

		if rc_overlap >= 0.5:
			easy['references'][u'images'].append({u'license': 3, u'file_name': str(idx), u'id': im_id})
			easy['references'][u'annotations'].append({u'image_id': im_id, u'id': im_id, u'caption': reference.strip()})
			easy['hypotheses'].append({u'caption': generated.strip(), u'image_id': im_id})
		else:
			hard['references'][u'images'].append({u'license': 3, u'file_name': str(idx), u'id': im_id})
			hard['references'][u'annotations'].append({u'image_id': im_id, u'id': im_id, u'caption': reference.strip()})
			hard['hypotheses'].append({u'caption': generated.strip(), u'image_id': im_id})

	assert len(ret['references'][u'annotations']) == len(ret['hypotheses'])
	with open(os.path.join(path, 'references.json'), 'w') as f:
		json.dump(ret['references'], f)
	with open(os.path.join(path, 'generated.json'), 'w') as f:
		json.dump(ret['hypotheses'], f)

	df = pd.DataFrame(ret['inference'])
	df.to_csv(os.path.join(path, 'inference.csv'), index=False)
	print('Saving files to : ', path)

	if split:
		assert len(easy['references'][u'annotations']) == len(easy['hypotheses'])
		with open(os.path.join(path, 'easy_references.json'), 'w') as f:
			json.dump(easy['references'], f)
		with open(os.path.join(path, 'easy_generated.json'), 'w') as f:
			json.dump(easy['hypotheses'], f)

		assert len(hard['references'][u'annotations']) == len(hard['hypotheses'])
		with open(os.path.join(path, 'hard_references.json'), 'w') as f:
			json.dump(hard['references'], f)
		with open(os.path.join(path, 'hard_generated.json'), 'w') as f:
			json.dump(hard['hypotheses'], f)

def test():
	pass

if __name__ == '__main__':
	test()