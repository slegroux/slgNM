#!/usr/bin/env python

# Added support for lm rescoring 
# (c) 2021 Sylvain Le Groux <slegroux@ccrma.stanford.edu>

"""
This script serves three goals:
    (1) Demonstrate how to use NeMo Models outside of PytorchLightning
    (2) Shows example of batch ASR inference
    (3) Serves as CI test for pre-trained checkpoint
"""

from argparse import ArgumentParser
import os
import torch
from nemo.collections.asr.metrics.wer import WER, word_error_rate
from nemo.collections.asr.models import EncDecCTCModel
from nemo.collections.asr.modules import BeamSearchDecoderWithLM
from nemo.utils import logging
from IPython import embed


try:
    from torch.cuda.amp import autocast
except ImportError:
    from contextlib import contextmanager

    @contextmanager
    def autocast(enabled=None):
        yield

# lm_path = "/home/syl20/data/en/librispeech/lm/3-gram.pruned.1e-7.arpa"
# lm_path = "/home/syl20/data/en/librispeech/lm/3gram_libri_lower.arpa"

can_gpu = torch.cuda.is_available()

def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--asr_model", type=str, default="QuartzNet15x5Base-En", required=True, help="Pass: 'QuartzNet15x5Base-En'",
    )
    parser.add_argument("--dataset", type=str, required=True, help="path to evaluation data")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--wer_tolerance", type=float, default=1.0, help="used by test")
    parser.add_argument(
        "--normalize_text", default=True, type=bool, help="Normalize transcripts or not. Set to False for non-English."
    )
    parser.add_argument("--lm_path", type=str, required=True, help="path to language model")
    parser.add_argument("--beam_width", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=2)
    parser.add_argument("--beta", type=float, default=1.5)

    args = parser.parse_args()
    return(args)


def main():
    args = get_args()

    torch.set_grad_enabled(False)

    if args.asr_model.endswith('.nemo'):
        logging.info(f"Using local ASR model from {args.asr_model}")
        asr_model = EncDecCTCModel.restore_from(restore_path=args.asr_model)
    else:
        logging.info(f"Using NGC cloud ASR model {args.asr_model}")
        asr_model = EncDecCTCModel.from_pretrained(model_name=args.asr_model)

    asr_model.setup_test_data(
        test_data_config={
            'sample_rate': 16000,
            'manifest_filepath': args.dataset,
            'labels': asr_model.decoder.vocabulary,
            'batch_size': args.batch_size,
            'normalize_transcripts': args.normalize_text,
        }
    )

    if can_gpu:
        asr_model = asr_model.cuda()

    asr_model.eval()
    labels_map = dict([(i, asr_model.decoder.vocabulary[i]) for i in range(len(asr_model.decoder.vocabulary))])

    # wer
    wer = WER(vocabulary=asr_model.decoder.vocabulary)
    hypotheses = []
    references = []

    # beam search
    beam_search_lm = BeamSearchDecoderWithLM(
        vocab=asr_model.cfg.decoder.vocabulary,
        beam_width=args.beam_width,
        alpha=args.alpha,
        beta=args.beta,
        lm_path=args.lm_path,
        num_cpus=max(os.cpu_count(), 1),
        input_tensor=True,
    )

    for test_batch in asr_model.test_dataloader():
        if can_gpu:
            test_batch = [x.cuda() for x in test_batch]
        with autocast():
            log_probs, encoded_len, greedy_predictions = asr_model(
                input_signal=test_batch[0], input_signal_length=test_batch[1]
            )

        beam_predictions = beam_search_lm.forward(log_probs=log_probs, log_probs_length=encoded_len)
        # beam_results, beam_scores, timesteps, out_lens = beam_search_lm.decode(log_probs)
        hypotheses += [pred[0][1] for pred in beam_predictions]

        for batch_ind in range(greedy_predictions.shape[0]):
            seq_len = test_batch[3][batch_ind].cpu().detach().numpy()
            seq_ids = test_batch[2][batch_ind].cpu().detach().numpy()
            reference = ''.join([labels_map[c] for c in seq_ids[0:seq_len]])
            references.append(reference)
        del test_batch
    
    wer_value = word_error_rate(hypotheses=hypotheses, references=references)
    if wer_value > args.wer_tolerance:
        raise ValueError(f"Got WER of {wer_value}. It was higher than {args.wer_tolerance}")
    logging.info(f'Got WER of {wer_value}. Tolerance was {args.wer_tolerance}')


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
