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
	wiki_context = False

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
	scst_on_hard = False
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
	gradient_clip_val = 0.
	gradient_clip_algo = None

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
def gpt2pp():
	expt_name = "gpt2pp"
	losses = _loss_names({"lm": 1}) # , "wd": 1, "div": 1

	text_encoder = None
	text_decoder = 'gpt2++'
	text_decoder_finetune = True
	image_pooling = None 
	text_pooling = None

	n_embed = 0
	max_epoch = 200

	transform = 'resnet_h5py'
	text_max_len = 512
	per_gpu_batchsize = 16
	val_check_interval = 0.25
	batch_size = 64


@ex.named_config
def t5pp():
	expt_name = "t5pp"
	losses = _loss_names({"lm": 1}) # , "wd": 1, "div": 1

	image_encoder = 'resnet152'
	text_encoder = None
	text_decoder = 't5++'
	text_decoder_finetune = True
	precision = 32
	image_pooling = None 
	text_pooling = None

	n_embed = 0
	max_epoch = 200

	transform = 'resnet_h5py'
	text_max_len = 512
	per_gpu_batchsize = 16
	val_check_interval = 0.25
	batch_size = 64


@ex.named_config
def caption_eval():
	expt_name = "caption_eval"
	# load_path = ''
	test = True
	run_caption = True


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
