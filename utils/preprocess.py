import os, json, base64, io, glob
import string, re
import argparse
import h5py
import cv2
import spacy

from PIL import Image
import pandas as pd
import numpy as np

from tqdm import tqdm
from nltk.tokenize import sent_tokenize, word_tokenize

def preprocess_wit(data_dir, save_dir, lang='en'):
	####### SAVE WIT dataframe for each SPLIT #######
	split_dict = {
		'train': [],
		'val': [],
		'test': [],
	}
	# NOTE: Please follow the folder structure as specified in the WIT dataset
	# The official val/test split was not released at the time of this work
	image_dict = {
		'train': ([], os.path.join(data_dir, 'train', 'image_data_train/image_pixels/*.csv.gz')),
		'val': ([], ''), 
		'test': ([], ''),
	}

	for _split in ['train']:
		split_folder = os.path.join(data_dir, _split)
		for infile in tqdm(glob.glob(d_folder + "/*.gz"), desc='reading data'):
			data = pd.read_csv(infile, sep='\t')
			data = data[data.language == lang]
			# store DataFrame in list
			split_dict[_split].append(data)

		# see pd.concat documentation for more info
		split_dict[_split] = pd.concat(split_dict[_split])

		# write DataFrame 
		# split_dict[_split].to_pickle(os.path.join(data_dir, _split, f'wit_{lang}_{_split}.pkl'))
		print(f'Total {_split} data points: {len(split_dict[_split])}')
		
		for _im_file in tqdm(glob.glob(image_dict[_split][1]), desc='reading image'):
			images = pd.read_csv(_im_file, sep='\t', names=['image_url', 'b64_bytes', 'metadata_url'])
			image_dict[_split][0].append(images)

		split_images = pd.concat(image_dict[_split][0])
		image_dict[_split][0] = split_images
		print(f'Total {_split} image data: {len(split_images)}')


	####### PREPROCESS text/image data for each SPLIT #######
	for _split in ['train']:
		split_data = split_dict[_split]
		split_images = image_dict[_split][0]
		print(f'{_split} data points: ', len(split_data))
		unique_images = split_data['image_url'].value_counts()
		print(f'{_split} unique images: ', len(unique_images))
		
		# exclude redundant fields
		split_data = split_data.drop(['original_height', 'original_width', 'is_main_image', 
											'attribution_passes_lang_id','page_changed_recently'], axis=1)

		# remove entries that have NaN caption or context 
		split_data = split_data[(split_data['caption_reference_description'].notna()) 
								  & (split_data['context_section_description'].notna())]
		print(f'{_split} data after NaN filtered: ', len(split_data))

		
		merged = split_data.merge(split_images, on='image_url', how='left')
		print(f'{_split} data after merged: ', len(merged))
		merged = merged[~(merged.image_url.str.endswith('.gif')) & ~(merged.image_url.str.endswith('.svg'))]
		print(f'{_split} data after uncommon format filtered: ', len(merged))
		merged.dropna(subset=['b64_bytes'], inplace=True)
		print(f'{_split} data after merged: {len(merged)}, saving to pickle file...')
		merged.to_pickle(os.path.join(data_dir, _split, f'wit_{lang}_merged.pkl'))

	####### CREATING train/val/tests split from WIT TRAINING data #######
	np.random.seed(2610)
	sampled_df = merged

	test_urls = np.random.choice(sampled_df.image_url.value_counts().index.tolist(), 20000, replace=False)
	test_df = sampled_df[sampled_df.image_url.isin(test_urls)]
	sampled_df = sampled_df[~sampled_df.image_url.isin(test_urls)]

	val_urls = np.random.choice(sampled_df.image_url.value_counts().index.tolist(), 8000, replace=False)
	val_df = sampled_df[sampled_df.image_url.isin(val_urls)]
	sampled_df = sampled_df[~sampled_df.image_url.isin(val_urls)]

	train_df = sampled_df
	print('test ', len(test_df))
	print('val ', len(val_df))
	print('train ', len(train_df))

	print('Creating dataset from WIT')
	for _split, _df  in [('val', val_df), ('test', test_df), ('train', train_df)]:
		print(f'{_split} data...')

		images, im_ids, im_urls = [], [], []
		str_descs, str_secs, str_caps = [], [], []
		for _ind, _row in tqdm(enumerate(_df.iterrows())):	  
			try:
				images.append(process_image(_row.b64_bytes, _row.image_url))
			except Exception as e:
				print(f"Exception {e}, image: {_row.image_url}")
				continue

			desc = str(_row.caption_attribution_description)
			sec = str(_row.context_section_description)
			cap = str(_row.caption_reference_description)
			str_desc = clean_and_tokenize(desc)[0]
			str_sec = clean_and_tokenize(sec)[0]
			str_cap = clean_and_tokenize(cap)[0]
	
			im_ids.append(_ind)
			im_urls.append(_row.image_url)
			str_descs.append(str_desc)
			str_secs.append(str_sec)
			str_caps.append(str_cap)

		with h5py.File(os.path.join(save_dir, _split + '_IMAGES_wit.hdf5'), 'a') as h:
			images_h5py = h.create_dataset('images', (len(images), 3, 256, 256), dtype='uint8')
			for _ind, _im in tqdm(enumerate(images), desc=f'writing {_split} data'):
				images_h5py[_ind] = _im

		assert len(images_h5py) == len(im_ids) == len(im_urls) == len(str_descs) == len(str_secs) == len(str_caps)

		with open(os.path.join(save_dir, 'all', f'{_split}_IMAGEIDS_wit.json'), 'w') as j:
			json.dump(im_ids, j)
		with open(os.path.join(save_dir, 'all', f'{_split}_IMAGEURLS_wit.json'), 'w') as j:
			json.dump(im_urls, j)
		with open(os.path.join(save_dir, 'all', f'{_split}_STRDESCS_wit.json'), 'w') as j:
			json.dump(str_descs, j)
		with open(os.path.join(save_dir, 'all', f'{_split}_STRCONTEXTS_wit.json'), 'w') as j:
			json.dump(str_secs, j)
		with open(os.path.join(save_dir, 'all', f'{_split}_STRCAPS_wit.json'), 'w') as j:
			json.dump(str_caps, j)

def preprocess_goodnews(data_dir, save_dir):
	for _split in ['train', 'val', 'test']:
		with open(os.path.join(data_dir, _split.upper() + '_RAWSTRARTICLES.json'), 'r') as f:
			sections = json.load(f)
		with open(os.path.join(data_dir, _split.upper() + '_RAWSTRCAPS.json'), 'r') as f:
			captions = json.load(f)
		with open(os.path.join(data_dir, _split.upper() + '_IMAGEIDS.json'), 'r') as f:
			ids = json.load(f)
		with open(os.path.join(data_dir, _split.upper() + '_ARTICLEURLS.json'), 'r') as f:
			urls = json.load(f)
		h = h5py.File(os.path.join(data_dir, _split.upper() + '_IMAGES.hdf5'), 'r')
		images = h['images']

		all_data = []
		cleaned_secs = []
		cleaned_caps = []
		for it, (sec, cap) in tqdm(enumerate(zip(sections, captions))):
			merged_sec, tokens_sec = clean_and_tokenize(sec)
			merged_cap, tokens_cap = clean_and_tokenize(cap)
			all_data.append(it)
			cleaned_secs.append(merged_sec)
			cleaned_caps.append(merged_cap)
		print(f'All: {len(all_data)}')

		with h5py.File(os.path.join(save_dir, 'all', _split + '_IMAGES_goodnews.hdf5'), 'a') as h:
			images_h5py = h.create_dataset('images', (len(images), 3, 256, 256), dtype='uint8')
			# Create dataset inside HDF5 file to store images
			print(f"\nReading {_split.upper()} images, sections and captions, storing to file...\n")
			
			for i, _im in enumerate(images):
				images_h5py[i] = _im

		with open(os.path.join(save_dir, 'all', _split + '_STRCONTEXTS_goodnews.json'), 'w') as j:
			json.dump(sections, j)
		with open(os.path.join(save_dir, 'all', _split + '_STRCAPS_goodnews.json'), 'w') as j:
			json.dump(captions, j)
		# with open(os.path.join(save_dir, 'all', _split + '_STRCONTEXTS_goodnews.json'), 'w') as j:
		# 	json.dump(cleaned_secs, j)
		# with open(os.path.join(save_dir, 'all', _split + '_STRCAPS_goodnews.json'), 'w') as j:
		# 	json.dump(cleaned_caps, j)
		with open(os.path.join(save_dir, 'all', _split + '_IMAGEIDS_goodnews.json'), 'w') as j:
			json.dump(ids, j)
		with open(os.path.join(save_dir, 'all', _split + '_IMAGEURLS_goodnews.json'), 'w') as j:
			json.dump(urls, j)

def preprocess_nytimes800k(data_dir, save_dir):
	from pymongo import MongoClient
	client = MongoClient(host='localhost', port=27017)

	for _split in ['train', 'valid', 'test']:
	    print(f"Processing {_split} data...")
	    db = client.nytimes

	    count = 0
	    image_paths, articles, captions = [], [], []  
	    image_ids, image_urls = [], []
	    for article in tqdm(db.articles.find({'split': _split})):
	        count += 1
	        sections = article['parsed_section']
	        paragraphs = []
	        for section in sections:
	            if section['type'] == 'paragraph':
	                paragraphs.append(section['text'])
	        article_text = '\n'.join(paragraphs).strip()
	        article_text, _ = clean_and_tokenize(article_text)
	        
	        # no of captions is as many as no of images
	        for pos in article['image_positions']:
	            image_path = os.path.join(data_folder, f"{sections[pos]['hash']}.jpg")
	            image = process_image(image_path, sections[pos]['hash'])
	            if image is None:
	                continue
	            image_paths.append(image_path)
	            articles.append(article_text)
	            caption_text, _ = clean_and_tokenize(sections[pos]['text'].strip())
	            captions.append(caption_text)
	            image_ids.append(sections[pos]['hash'])
	            image_urls.append(article['web_url'])
	    print(f'Number of {_split} articles : {count}')

	    split_df = pd.DataFrame(data={'image_id':image_ids, 'article': articles, 
	    								'caption': captions, 'url':image_urls, 'path': image_paths})
	    split_df = split_df.drop_duplicates(subset=['image_id', 'article', 'caption'])
	    
	    with h5py.File(os.path.join(save_dir, 'all', _split + '_IMAGES_nytimes800k.hdf5'), 'a') as h:
	        images = h.create_dataset('images', (len(split_df), 3, 256, 256), dtype='uint8')
	        for i, (idx, row) in tqdm(enumerate(split_df.iterrows())):
	            images[i] = process_image(row.path, row.image_id)
	            # # You can also load the pre-computed FaceNet embeddings of the faces in the image
	            # facenet_embeds = sections[pos]['facenet_details']['embeddings']

	            # # Object embeddings are stored in a separate collection due to a size limit in mongo
	            # obj = db.objects.find_one({'_id': sections[pos]['hash']})
	            # object_embeds = obj['object_features']

	        # Sanity check
	        assert len(split_df) == len(images)
	        print(f'Final {_split} data points: {len(images)}')
	        with open(os.path.join(save_dir, category, _split + '_STRCONTEXTS_nytimes800k.json'), 'w') as j:
	            json.dump(split_df.article.tolist(), j)
	        with open(os.path.join(save_dir, category, _split + '_STRCAPS_nytimes800k.json'), 'w') as j:
	            json.dump(split_df.caption.tolist(), j)
	        with open(os.path.join(save_dir, category, _split + '_IMAGEIDS_nytimes800k.json'), 'w') as j:
	            json.dump(split_df.image_id.tolist(), j)
	        with open(os.path.join(save_dir, category, _split + '_IMAGEURLS_nytimes800k.json'), 'w') as j:
	            json.dump(split_df.url.tolist(), j)

def process_image(im_b64, img_url):
	try:
		base64_decoded = base64.b64decode(im_b64)
		img = Image.open(io.BytesIO(base64_decoded))
		img = img.convert('RGB')
		save_img = img
		img = np.array(img)
		if len(img.shape) == 2:
			img = img[:, :, np.newaxis]
			img = np.concatenate([img, img, img], axis=2)
		img = np.array(Image.fromarray(img).resize((256, 256)))
		if len(img.shape) > 2 and img.shape[2] == 4:
			# convert the image from RGBA2RGB for .png image
			img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
	except TypeError as e:
		print(f'{e} at image {img_url}')
		img = np.array(Image.fromarray((img * 255).astype(np.uint8)).resize((256, 256)))
	except Exception as e:
		print(f"An exception occurred {e} at image {img_url}")
		return
	img = img.transpose(2, 0, 1)
	assert img.shape == (3, 256, 256)
	assert np.max(img) <= 255

	return img

def view_image(im_arr):
	img = Image.fromarray(np.moveaxis(im_arr,  0, -1))
	img = img.convert('RGB')
	img.show()

def clean_and_tokenize(text):
	out = ''
	tokens = []
	for word in word_tokenize(text):
		if word == '.' or word == ',' or not re.fullmatch('[' + string.punctuation + ']+', word):
			tokens.append(word)
	for token in tokens:
		if token == '.' or token == ',':
			out += token
		else:
			out += ' ' + token
	return out.strip(), tokens

FUNC_DICT = {
	'wit': preprocess_wit,
	'goodnews': preprocess_goodnews,
}

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Pre-processing data for Wiki Caption task, support WIT/GoodNews datasets.')
	parser.add_argument('--dset', type=str, help='dataset')
	parser.add_argument('--data_dir', type=str, default='', help='path to the original data')
	parser.add_argument('--save_dir', type=str, default='', help='path to save Wiki Caption data')
	args = parser.parse_args()

	dset = args.dset
	data_dir = args.data_dir
	save_dir = args.save_dir

	FUNC_DICT[dset](data_dir, save_dir)