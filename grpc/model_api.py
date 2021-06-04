import contextlib
import glob
import os

import torch

import nemo.collections.asr as nemo_asr
from nemo.utils import logging, model_utils
logging.setLevel(logging.CRITICAL)

# setup AMP (optional)
if torch.cuda.is_available() and hasattr(torch.cuda, 'amp') and hasattr(torch.cuda.amp, 'autocast'):
    logging.info("AMP enabled!\n")
    autocast = torch.cuda.amp.autocast
else:
    @contextlib.contextmanager
    def autocast():
        yield

MODEL_CACHE = {}

# Special tags for fallbacks / user notifications
TAG_ERROR_DURING_TRANSCRIPTION = "<ERROR_DURING_TRANSCRIPTION>"


def get_model_names():
    # Populate local copy of models
    local_model_paths = glob.glob(os.path.join('models', "**", "*.nemo"), recursive=True)
    local_model_names = list(sorted([os.path.basename(path) for path in local_model_paths]))

    # Populate with pretrained checkpoint list
    nemo_model_names = set()
    for model_info in nemo_asr.models.ASRModel.list_available_models():
        for superclass in model_info.class_.mro():
            if 'CTC' in superclass.__name__ or 'RNNT' in superclass.__name__:
                nemo_model_names.add(model_info.pretrained_model_name)
                break
    nemo_model_names = list(sorted(nemo_model_names))
    return nemo_model_names, local_model_names


def initialize_model(model_name):
    # load model
    if model_name not in MODEL_CACHE:
        if '.nemo' in model_name:
            # use local model
            model_name_no_ext = os.path.splitext(model_name)[0]
            model_path = os.path.join('models', model_name_no_ext, model_name)

            # Extract config
            model_cfg = nemo_asr.models.ASRModel.restore_from(restore_path=model_path, return_config=True)
            classpath = model_cfg.target  # original class path
            imported_class = model_utils.import_class_by_path(classpath)  # type: ASRModel
            logging.info(f"Restoring local model : {imported_class.__name__}")

            # load model from checkpoint
            model = imported_class.restore_from(restore_path=model_path, map_location='cpu')  # type: ASRModel

        else:
            # use pretrained model
            model = nemo_asr.models.ASRModel.from_pretrained(model_name, map_location='cpu')

        model.freeze()

        # cache model
        MODEL_CACHE[model_name] = model

    model = MODEL_CACHE[model_name]
    return model


def transcribe_all(filepaths, model_name, use_gpu_if_available=True):
    # instantiate model
    if model_name in MODEL_CACHE:
        model = MODEL_CACHE[model_name]
    else:
        model = initialize_model(model_name)

    if torch.cuda.is_available() and use_gpu_if_available:
        model = model.cuda()

    # transcribe audio
    logging.info("Begin transcribing audio...")
    try:
        with autocast():
            with torch.no_grad():
                transcriptions = model.transcribe(filepaths, batch_size=32)

    except RuntimeError:
        # Purge the cache to clear some memory
        MODEL_CACHE.clear()
        logging.info("Ran out of memory on device - performing inference on CPU for now")
        try:
            model = model.cpu()
            with torch.no_grad():
                transcriptions = model.transcribe(filepaths, batch_size=32)
        except Exception as e:
            logging.info(f"Exception {e} occured while attemting to transcribe audio. Returning error message")
            return TAG_ERROR_DURING_TRANSCRIPTION
    logging.info(f"Finished transcribing {len(filepaths)} files !")

    # If RNNT models transcribe, they return a tuple (greedy, beam_scores)
    if type(transcriptions[0]) == list and len(transcriptions) == 2:
        # get greedy transcriptions only
        transcriptions = transcriptions[0]

    # Force onto CPU
    model = model.cpu()
    MODEL_CACHE[model_name] = model

    return transcriptions


from nemo.utils import logging
logging.setLevel(logging.ERROR)
from streaming_asr import StreamingASR, AudioDataLayer, DataLoader, preprocessor_normalization
from omegaconf import OmegaConf
import copy
import nemo.collections.asr as nemo_asr


def init_streaming_mdl(MDL, frame_len=1, frame_overlap=2, offset=4):

    # asr model
    asr_model = nemo_asr.models.EncDecCTCModel.from_pretrained(MDL)
    cfg = copy.deepcopy(asr_model._cfg)

    #preprocessor normalization
    cfg = preprocessor_normalization(cfg)
    asr_model.preprocessor = asr_model.from_config_dict(cfg.preprocessor)

    # inference mode
    asr_model.eval()
    asr_model = asr_model.to(asr_model.device)
    asr = StreamingASR(asr_model, cfg,
			frame_len=frame_len, frame_overlap=frame_overlap,
			offset=offset)

    asr.reset()
    return(asr)
