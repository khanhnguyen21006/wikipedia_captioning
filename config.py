from sacred import Experiment

ex = Experiment("Wit")


def _loss_names(d):
	ret = {
		"lm": 0,
		"wd": 0,
		"div": 0,
		"mmd": 0,
		"pe": 0,
		"de": 0,
		"mmpe": 0,
		"vib": 0,
		"se": 0,
		"ms": 0,
		"pm": 0,
		"clip": 0,
		"cider": 0,
		"clips": 0,
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
	extract_context = ''
	context_source = 'section'

	losses = _loss_names({"lm": 1})
	batch_size = 256  # accumulated batch size.

	# default config values
	model = 'multi_encoder_single_decoder' # default
	image_encoder = 'resnet152'  # mandatory
	text_encoder = None  # optional 'roberta'|'gru'|'sbert'|'st5'|'t5-adapter'
	text_decoder = None  # optional 'gpt-2'|'t5'
	image_pooling = 'pcme'  # optional 'pcme'|'slot'|'gaussian'
	text_pooling = 'pcme'  #
	fuse = None
	embed_dim = 768  # == decoder_dim, since text decoder has no linear projection
	image_encoder_finetune = False
	text_encoder_finetune = False
	text_decoder_finetune = False
	image_encoder_use_linear_layer = False
	text_encoder_use_linear_layer = False
	text_max_len = 512
	text_embed = 'glove'  # for RNNs
	text_dim = 300  # for RNNs

	n_embed = 8
	prob_embed = False
	pe_scale = 1.5
	pe_shift = 1.
	wd_lambda = 1
	div_lambda = 1
	mmd_lambda = 1
	vib_lambda = 1
	mmpe_l2_lambda = 1
	se_match = 'multi_instance'
	margin = 0.2  # for set embedding
	hard_mining = False  # for set embedding
	chamfer_alpha = 1

	self_critical_after = -1
	scst_batchsize = 0
	sample_max_len = 50  # for RL experiments
	sample_n = 5
	# num_beam = 1
	cider_baseline = "" # "greedy"|"description"|"caption"
	clip_baseline = "" # "description"|"caption"
	cider_lambda = 0.
	clip_lambda = 0.
	clip_ckpt = None

	# Optimizer Setting
	optimizer = "adamp"
	learning_rate = 1e-4
	weight_decay = 0.01
	lr_multiplier = 1.
	lr_scheduler = 'with_warmup'
	max_epoch = 100
	end_lr = 0
	warmup_steps = 2500
	decay_power = 1
	gradient_clip_val = 0
	run_caption = False
	run_retrieve = False
	retrieval_testset = ''
	source_to_target = {'source': [], 'target': ''}
	multi_query = None  # 'addition'|'multiplication'
	eval_method = ''

	# PL Trainer Setting
	ckpt_path = None
	fast_dev_run = False
	val_check_interval = 0.5
	test = False
	distributed = False
	accelerator = "gpu"
	save_top_k = 1
	num_sanity_val_steps = 2

	# Environment hyperparams
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
def witpage():
	dataset = 'witpage'
	data_folder = '/data/users/vkhanh/refined'


@ex.named_config
def goodnews():
	dataset = 'goodnews'
	data_folder = '/data/users/vkhanh/goodnews/data/goodnews_jaccard/all'


@ex.named_config
def prelim():
	expt_name = "prelim"
	losses = _loss_names({"lm": 1}) # , "wd": 1, "div": 1

	# text_encoder = None
	# text_decoder = 't5++'
	# text_decoder_finetune = True
	# precision = 32

	# n_embed = 0
	# wd_lambda = 1
	# div_lambda = 1

	text_encoder = None
	text_decoder = 'gpt2++'
	text_decoder_finetune = True

	n_embed = 0
	wd_lambda = 1
	div_lambda = 1

	transform = 'resnet_h5py'
	text_max_len = 512
	per_gpu_batchsize = 32


@ex.named_config
def eval():
	expt_name = "eval"
	# load_path = ''
	test = True
	run_caption = True

	# run_retrieve = True
	# retrieval_testset = 'test_5k_RET' # 'test_5k_multi_RET', 'test_5k_RET'
	# eval_method =  'clip' # 'match_sentence_max_2', 'clip'
	num_gpus = 1


@ex.named_config
def gpt2_adapter():
	expt_name = "gpt2_adapter"
	losses = _loss_names({"lm": 1, "wd": 1, "div": 1})
	image_encoder = 'openai/clip-vit-base-patch32'
	text_encoder = 'roberta'
	text_decoder = 'gpt2-adapter'
	image_encoder_finetune = False
	text_encoder_finetune = False
	text_decoder_finetune = False

	n_embed = 8
	wd_lambda = 10
	div_lambda = 10

	transform = 'clip_vit_h5py'
	text_max_len = 512
	per_gpu_batchsize = 32

	lr_multiplier = 1.


@ex.named_config
def t5_adapter():
	expt_name = "t5_adapter"
	losses = _loss_names({"lm": 1, "wd": 1, "div": 1})
	image_encoder = 'openai/clip-vit-base-patch32'
	text_encoder = 't5-adapter'
	text_decoder = None
	image_encoder_finetune = False
	text_encoder_finetune = False
	text_decoder_finetune = False
	precision = 32

	n_embed = 8
	wd_lambda = 10
	div_lambda = 10

	transform = 'clip_vit_h5py'
	text_max_len = 512
	per_gpu_batchsize = 32

	lr_multiplier = 1.

@ex.named_config
def prob_embed():
	expt_name = "prob_embed"
	# losses = _loss_names({"pe": 1, "vib": 1})
	losses = _loss_names({"de": 1})
	image_encoder = 'openai/clip-vit-base-patch32'
	text_encoder = 'sentence-transformers/all-distilroberta-v1'
	text_decoder = None
	image_encoder_finetune = False
	text_encoder_finetune = False
	text_decoder_finetune = False
	embed_dim = 1024

	# n_embed = 8
	# prob_embed = True
	n_embed = 1
	prob_embed = False
	# pe_scale = 1.5
	# pe_shift = 1.
	# vib_lambda = 0.00001
	multi_query = None
	# source_to_target = {'source': ['image', 'description'], 'target': 'section'}
	source_to_target = {'source': ['image'], 'target': 'section'}
	# extract_context = 'cider_by_cap_3'

	optimizer = "adamp"
	learning_rate = 2e-4
	weight_decay = 0.0001
	lr_scheduler = 'cosine_annealing'

	transform = 'clip_vit_h5py'
	text_max_len = 512
	max_epoch = 30
	per_gpu_batchsize = 128


@ex.named_config
def set_embed_slot_chamfer():
	expt_name = "set_embed_slot_chamfer"
	losses = _loss_names({"se": 1, "div": 1, "mmd": 1,})
	image_encoder = 'openai/clip-vit-base-patch32'
	text_encoder = 'sentence-transformers/all-distilroberta-v1'
	text_decoder = None
	image_encoder_finetune = False
	text_encoder_finetune = False
	text_decoder_finetune = False
	embed_dim = 1024
	image_pooling = 'slot'
	text_pooling = 'slot'

	n_embed = 4
	prob_embed = False
	mmd_lambda = 0.2
	div_lambda = 1
	se_match = 'smooth_chamfer'
	chamfer_alpha = 16
	margin = 0.3
	hard_mining = False

	multi_query = None
	source_to_target = {'source': ['image'], 'target': 'section'}

	optimizer = "adamw"
	learning_rate = 1e-4
	weight_decay = 0.0001
	lr_scheduler = 'with_warmup'

	transform = 'clip_vit_h5py'
	text_max_len = 512
	max_epoch = 30
	per_gpu_batchsize = 128


@ex.named_config
def multi_space_embed():
	expt_name = "multi_space_embed"
	losses = _loss_names({"ms": 1, "div": 1})
	# losses = _loss_names({"ms": 1})
	image_encoder = 'openai/clip-vit-base-patch32'
	text_encoder = 'sentence-transformers/all-distilroberta-v1'
	text_decoder = None
	image_encoder_finetune = False
	text_encoder_finetune = False
	text_decoder_finetune = False
	embed_dim = 1024
	# image_pooling = 'pcme'
	# text_pooling = 'pcme'
	image_pooling = 'slot'
	text_pooling = 'slot'

	# n_embed = 1
	n_embed = 3
	prob_embed = False
	div_lambda = 1
	# source_to_target = {'source': ['image', 'description'], 'target': 'section'}
	source_to_target = {'source': ['image'], 'target': 'section'}

	optimizer = "adamp"
	learning_rate = 2e-4
	weight_decay = 0.0001
	lr_scheduler = 'cosine_annealing'

	transform = 'clip_vit_h5py'
	text_max_len = 512
	max_epoch = 30
	# per_gpu_batchsize = 128
	per_gpu_batchsize = 64

@ex.named_config
def page_mining():
	expt_name = "page_margin"
	# losses = _loss_names({"pe": 1, "vib": 1})
	losses = _loss_names({"pm": 1})
	image_encoder = 'openai/clip-vit-base-patch32'
	text_encoder = 'sentence-transformers/all-distilroberta-v1'
	text_decoder = None
	image_encoder_finetune = False
	text_encoder_finetune = False
	text_decoder_finetune = False
	embed_dim = 1024

	n_embed = 1
	source_to_target = {'source': ['image'], 'target': 'section'}
	page_margin = 0.1
	margin = 0.25

	optimizer = "adamp"
	learning_rate = 2e-4
	weight_decay = 0.0001
	lr_scheduler = 'cosine_annealing'

	transform = 'clip_vit_h5py'
	text_max_len = 512
	max_epoch = 30
	per_gpu_batchsize = 128


@ex.named_config
def finetune_clip():
	expt_name = "finetune_clip"
	losses = _loss_names({"clip": 1})
	image_encoder = 'openai/clip-vit-base-patch32'
	text_encoder = 'openai/clip-vit-base-patch32'
	text_decoder = None
	image_encoder_finetune = True
	text_encoder_finetune = True

	# image_encoder_use_linear_layer = False
	# text_encoder_use_linear_layer = False
	# embed_dim = 1024
	image_encoder_use_linear_layer = True
	text_encoder_use_linear_layer = True
	embed_dim = 512

	n_embed = 0
	multi_query = None
	source_to_target = {'source': ['image'], 'target': 'section'}
	extract_context = ''
	# extract_context = 'None_random_2'

	optimizer = "adamw"
	learning_rate = 1e-7
	weight_decay = 0.01
	lr_scheduler = 'with_warmup'

	transform = 'clip_vit_h5py'
	text_max_len = 77
	max_epoch = 30
	batch_size = 512
	per_gpu_batchsize = 256


@ex.named_config
def scst_t5_clip_caption_cider_clips():
	expt_name = "scst_t5_clip_caption_cider_clips"
	model = "rl_t5_caption_model"
	dataset = 'rlwit'
	losses = _loss_names({"lm": 1, "cider": 1, "clips": 1})
	image_encoder = 'openai/clip-vit-base-patch32'
	text_encoder = 'openai/clip-vit-base-patch32'
	text_decoder = 't5++'
	image_encoder_finetune = False
	text_encoder_finetune = False
	text_decoder_finetune = True
	image_encoder_use_linear_layer = True
	text_encoder_use_linear_layer = True
	embed_dim = 512

	n_embed = 0

	optimizer = "adamw"
	learning_rate = 1e-4
	weight_decay = 0.01
	lr_scheduler = 'with_warmup'
	max_epoch = 10
	warmup_steps = 500

	transform = 'clip_vit_h5py'
	text_max_len = 512
	per_gpu_batchsize = 32
	precision = 32

	self_critical_after = 5
	scst_batchsize = 8
	sample_max_len = 20
	sample_n = 3
	cider_baseline = "greedy"
	clip_baseline = "caption"
	cider_lambda = 1.
	clip_lambda = 3.
	clip_ckpt = 'result/clip_i2d_1e7_seed0/version_0/checkpoints/last.ckpt'

@ex.named_config
def scst_t5_clip_caption_cider():
	expt_name = "scst_t5_clip_caption_cider"
	model = "rl_t5_caption_model"
	dataset = 'rlwit'
	losses = _loss_names({"lm": 1, "cider": 1})
	image_encoder = 'openai/clip-vit-base-patch32'
	text_encoder = None
	text_decoder = 't5++'
	image_encoder_finetune = False
	text_encoder_finetune = False
	text_decoder_finetune = True

	embed_dim = 768
	n_embed = 0

	optimizer = "adamw"
	learning_rate = 1e-4
	weight_decay = 0.01
	lr_scheduler = 'with_warmup'
	max_epoch = 10
	warmup_steps = 500

	transform = 'clip_vit_h5py'
	text_max_len = 512
	per_gpu_batchsize = 32
	precision = 32

	self_critical_after = 5
	scst_batchsize = 8
	sample_max_len = 20
	sample_n = 3
	cider_baseline = "greedy"
	cider_lambda = 1.

