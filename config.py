from sacred import Experiment

ex = Experiment("Wit")


def _loss_names(d):
	ret = {
		"lm": 0,
		"wd": 0,
		"div": 0,
	}
	ret.update(d)
	return ret


@ex.config
def config():
	expt_name = "Wit"
	seed = 0

	dataset = 'wit'
	data_folder = '/data/users/vkhanh/all'
	transform = 'resnet_h5py'

	losses = _loss_names({"lm": 1})
	batch_size = 256  # accumulated batch size.

	# default config values
	image_encoder = 'resnet152'
	text_encoder = 'roberta'
	text_decoder = 'gpt2'
	embed_dim = 768  # == decoder_dim, since text decoder has no linear projection
	image_encoder_finetune = False
	text_encoder_finetune = False
	text_decoder_finetune = False
	text_max_len = 512
	text_embed = 'glove'  # for RNNs

	n_embeds = 8
	wd_lambda = 1
	div_lambda = 1

	# Optimizer Setting
	optimizer = "adamp"
	learning_rate = 1e-4
	weight_decay = 0.01	 # diff
	lr_scheduler = 'with_warmup'
	max_epoch = 100
	end_lr = 0
	warmup_steps = 2500
	decay_power = 1
	gradient_clip_val = 0
	run_retrieval = False

	# PL Trainer Setting
	ckpt_path = None
	fast_dev_run = False
	val_check_interval = 0.5
	test = False
	distributed = False
	accelerator = "gpu"
	save_top_k = 1
	num_sanity_val_steps = 2

	# below params varies with the environment
	vocab_path = './modules/vocab/wit_vocab_100.pkl'
	cache_dir = './.cache'
	result_dir = "result"
	per_gpu_batchsize = 2
	num_gpus = 1
	num_nodes = 1
	load_path = ''
	num_workers = 0
	precision = 16


@ex.named_config
def cluster():
	data_folder = '/data/users/vkhanh/all'
	num_workers = 8


@ex.named_config
def dist():
	distributed = True
	accelerator = "ddp"
	num_gpus = 2


@ex.named_config
def wit():
	dataset = 'wit'
	data_folder = '/data/users/vkhanh/refined'


@ex.named_config
def goodnews():
	dataset = 'goodnews'
	data_folder = '/data/users/vkhanh/goodnews/data/goodnews_jaccard/all'


@ex.named_config
def prelim():
	expt_name = "prelim"
	losses = _loss_names({"lm": 1}) # , "wd": 1, "div": 1

	# text_encoder = 't5-adapter'
	# text_decoder = 't5-adapter'
	# text_encoder_finetune = True
	# precision = 32

	# n_embeds = 16
	# wd_lambda = 1
	# div_lambda = 1

	# text_encoder = 'roberta'
	# text_decoder = 'gpt2-adapter'
	# text_decoder_finetune = True
	# # precision = 32

	# n_embeds = 0
	# wd_lambda = 1
	# div_lambda = 1

	# text_encoder = 't5++'
	# text_decoder = 't5++'
	# text_decoder_finetune = True
	# precision = 32

	# n_embeds = 0
	# wd_lambda = 1
	# div_lambda = 1

	text_encoder = 'gpt2++'
	text_decoder = 'gpt2++'
	text_decoder_finetune = True
	# precision = 32

	n_embeds = 0
	wd_lambda = 1
	div_lambda = 1

	# transform = 'resnet_h5py'
	# text_max_len = 512
	# per_gpu_batchsize = 32

	


@ex.named_config
def eval():
	expt_name = "eval" 
	# load_path = ''
	test = True


@ex.named_config
def gpt2_adapter():
	expt_name = "gpt2_adapter"
	losses = _loss_names({"lm": 1, "wd": 1, "div": 1})
	image_encoder = 'openai/clip-vit-base-patch16'
	text_encoder = 'roberta'
	text_decoder = 'gpt2-adapter'
	image_encoder_finetune = False
	text_encoder_finetune = False
	text_decoder_finetune = False

	n_embeds = 8
	wd_lambda = 10
	div_lambda = 10

	transform = 'clip_vit_h5py'
	text_max_len = 512
	per_gpu_batchsize = 32


@ex.named_config
def t5_adapter():
	expt_name = "t5_adapter"
	losses = _loss_names({"lm": 1, "wd": 1, "div": 1})
	image_encoder = 'openai/clip-vit-base-patch16'
	text_encoder = 't5-adapter'
	text_decoder = 't5-adapter'
	image_encoder_finetune = False
	text_encoder_finetune = False
	text_decoder_finetune = False
	precision = 32

	n_embeds = 8
	wd_lambda = 10
	div_lambda = 10

	transform = 'clip_vit_h5py'
	text_max_len = 512
	per_gpu_batchsize = 32