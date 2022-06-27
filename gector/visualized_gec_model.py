"""Wrapper of AllenNLP model. Fixes errors based on model predictions"""
import logging
import os
import sys
from time import time

import torch
from allennlp.data.dataset import Batch
from allennlp.data.fields import TextField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.nn import util

from gector.bert_token_embedder import PretrainedBertEmbedder
from gector.seq2labels_model import Seq2Labels
from gector.tokenizer_indexer import PretrainedBertIndexer
from utils.helpers import PAD, UNK, get_target_sent_by_edits, START_TOKEN
from utils.helpers import get_weights_name

logging.getLogger("werkzeug").setLevel(logging.ERROR)
logger = logging.getLogger(__file__)

from gector.gec_model import GecBERTModel


class VisualizedGecBERTModel(GecBERTModel):
    def __init__(self, vocab_path=None, model_paths=None,
                 weigths=None,
                 max_len=50,
                 min_len=3,
                 lowercase_tokens=False,
                 log=False,
                 iterations=3,
                 model_name='roberta',
                 special_tokens_fix=1,
                 is_ensemble=True,
                 min_error_probability=0.0,
                 confidence=0,
                 del_confidence=0,
                 resolve_cycles=False,
                 ):
        super().__init__(vocab_path, model_paths,
                 weigths,
                 max_len,
                 min_len,
                 lowercase_tokens,
                 log,
                 iterations,
                 model_name,
                 special_tokens_fix,
                 is_ensemble,
                 min_error_probability,
                 confidence,
                 del_confidence,
                 resolve_cycles,
                 )

    def postprocess_batch(self, batch, all_probabilities, all_idxs,
                          error_probs):
        all_results = []
        noop_index = self.vocab.get_token_index("$KEEP", "labels")
        all_sugg_tokens = []
        for tokens, probabilities, idxs, error_prob in zip(batch,
                                                           all_probabilities,
                                                           all_idxs,
                                                           error_probs):
            length = min(len(tokens), self.max_len)
            edits = []
            sugg_tokens = []

            # skip whole sentences if there no errors
            if max(idxs) == 0:
                all_results.append(tokens)
                all_sugg_tokens.append(['' for i in range(len(tokens))])
                continue

            # skip whole sentence if probability of correctness is not high
            if error_prob < self.min_error_probability:
                all_results.append(tokens)
                all_sugg_tokens.append(['' for i in range(len(tokens))])
                continue

            for i in range(length + 1):
                # because of START token
                if i == 0:
                    token = START_TOKEN
                else:
                    token = tokens[i - 1]
                # skip if there is no error
                if idxs[i] == noop_index:
                    if i != 0:
                        sugg_tokens.append('')
                    continue

                sugg_token = self.vocab.get_token_from_index(idxs[i],
                                                             namespace='labels')
                action = self.get_token_action(token, i, probabilities[i],
                                               sugg_token)
                if not action:
                    if i != 0:
                        sugg_tokens.append('')
                    continue
                if i!= 0:
                    sugg_tokens.append(sugg_token)
                edits.append(action)
            all_sugg_tokens.append(sugg_tokens)
            all_results.append(get_target_sent_by_edits(tokens, edits))
        return all_results, all_sugg_tokens

    def handle_batch(self, full_batch):
        """
        Handle batch of requests.
        """
        final_batch = full_batch[:]
        batch_size = len(full_batch)
        prev_preds_dict = {i: [final_batch[i]] for i in range(len(final_batch))}
        short_ids = [i for i in range(len(full_batch))
                     if len(full_batch[i]) < self.min_len]
        pred_ids = [i for i in range(len(full_batch)) if i not in short_ids]
        total_updates = 0
        all_iter_orig_batch = [[] for _ in range(len(full_batch))]
        all_iter_tag_batch = [[] for _ in range(len(full_batch))]
        for n_iter in range(self.iterations):
            prev_pred_ids = pred_ids
            orig_batch = [final_batch[i] for i in pred_ids]

            sequences = self.preprocess(orig_batch)
            if not sequences:
                break
            probabilities, idxs, error_probs = self.predict(sequences)

            pred_batch, sugg_tokens = self.postprocess_batch(orig_batch, probabilities,
                                                idxs, error_probs)
            if self.log:
                print(f"Iteration {n_iter + 1}. Predicted {round(100*len(pred_ids)/batch_size, 1)}% of sentences.")

            for i, orig_id in enumerate(pred_ids):
                all_iter_orig_batch[orig_id].append(orig_batch[i])
                all_iter_tag_batch[orig_id].append(sugg_tokens[i])

            final_batch, pred_ids, cnt = \
                self.update_final_batch(final_batch, pred_ids, pred_batch,
                                        prev_preds_dict)
            total_updates += cnt

            if not pred_ids:
                break

        return final_batch, total_updates, all_iter_orig_batch, all_iter_tag_batch
