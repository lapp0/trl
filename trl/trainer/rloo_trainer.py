from copy import deepcopy
import gc
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple, Union, Callable, Any
import warnings
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.state import AcceleratorState
from accelerate.utils import broadcast
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import (
    DataCollatorWithPadding,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
    PreTrainedTokenizerBase
)

from ..models import SUPPORTED_ARCHITECTURES, create_reference_model, PreTrainedModelWrapper

from . import PolicyTrainerBase, PolicyTrainerArguments

from ..import_utils import is_peft_available


INVALID_LOGPROB = 1.0


@dataclass
class RLOOConfig(PolicyTrainerArguments):
    cliprange: float = 0.2
    """the clip range"""
    kl_coef: float = 0.10
    """the KL coefficient"""
    rloo_k: int = 2
    """REINFORCE Leave-One-Out (RLOO) number of online samples per prompt"""


def first_true_indices(bools, dtype=torch.long):
    """
    Takes an N-dimensional bool tensor and returns an (N-1)-dimensional tensor of integers giving
    the position of the first True in each "row".

    Returns the length of the rows (bools.size(-1)) if no element is True in a given row.
    """
    row_len = bools.size(-1)
    zero_or_index = row_len * (~bools).type(dtype) + torch.arange(row_len, dtype=dtype, device=bools.device)
    return torch.min(zero_or_index, dim=-1).values


class RLOOTrainer(PolicyTrainerBase):
    _tag_names = ["trl", "rloo"]

    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        https://github.com/huggingface/transformers/blob/8c12690cecbb97e187861e386f7a0ac790e4236c/src/transformers/trainer.py#L3112
        """
        model.train()

        inputs = self._prepare_inputs(inputs)
        queries = inputs["input_ids"].to(self.accelerator.device)
        queries = queries.repeat(self.args.rloo_k, 1)

        context_length = queries.shape[1]

        with self.cast_model_ctx():
            with torch.no_grad(), self.time_metric_ctx("calc_advantages"):
                # PR TODO: refactor into a function shared by ppov2 which calculates sequences and logprobs
                #          see DPOTrainer.concatenated_forward

                with self.time_metric_ctx("generate"):
                    query_responses, logits = self.generate(
                        model,
                        queries,
                        self.train_generation_config,
                    )
                responses = query_responses[:, context_length:]
                all_logprobs = F.log_softmax(logits, dim=-1)
                logprobs = torch.gather(all_logprobs, 2, responses.unsqueeze(-1)).squeeze(-1)
                del logits, all_logprobs


                with self.time_metric_ctx("ref_model_forward"):
                    with self.ref_model_mgr as ref_model:
                        ref_output_logits = self.forward(ref_model, query_responses).logits
                ref_logits = ref_output_logits[:, context_length - 1 : -1]
                ref_logits /= self.args.temperature + 1e-7
                ref_all_logprobs = F.log_softmax(ref_logits, dim=-1)
                ref_logprobs = torch.gather(ref_all_logprobs, 2, responses.unsqueeze(-1)).squeeze(-1)
                del ref_output_logits, ref_logits, ref_all_logprobs
                torch.cuda.empty_cache()

                # Response Processing 1. truncate response after the
                # first occurrence of `truncate_token_id`
                postprocessed_responses = responses
                if self.args.truncate_token_id:
                    postprocessed_responses = self.truncate_response(responses)

                # Response Processing 2. run reward model on the truncated responses
                postprocessed_query_responses = torch.cat((queries, postprocessed_responses), 1)
                sequence_lengths = first_true_indices(postprocessed_responses == self.tokenizer.pad_token_id) - 1

                # clear cache before get_reward call
                gc.collect()
                torch.cuda.empty_cache()

                with self.time_metric_ctx("get_reward"):
                    _, scores, _ = self.get_reward(
                        self.reward_model,
                        postprocessed_query_responses,
                        context_length
                    )
                torch.cuda.empty_cache()

                # Response Processing 3. filter response. Ensure that the sample contains truncate_token_id
                # responses not passing that filter will receive a low (fixed) score
                # only query humans on responses that pass that filter
                if self.args.non_eos_penalty:
                    contain_eos_token = torch.any(postprocessed_responses == self.tokenizer.eos_token_id, dim=-1)
                    scores = torch.where(contain_eos_token, scores, torch.full_like(scores, self.args.penalty_reward_value))
                    # PR TODO: remove this debug statement
                    self.accelerator.print(f"{scores=}, {(contain_eos_token.sum() / len(contain_eos_token))=}")

                # be very careful with `padding_mask`;
                # see https://excalidraw.com/#json=LWnzG4w2k5DjF_EOL_xPt,e2w3a-hFJ_gX5vOfeyXGTw
                response_idxs = torch.arange(responses.shape[1], device=responses.device).repeat(responses.shape[0], 1)
                padding_mask = response_idxs > sequence_lengths.unsqueeze(1)
                logprobs = torch.masked_fill(logprobs, padding_mask, INVALID_LOGPROB)
                ref_logprobs = torch.masked_fill(ref_logprobs, padding_mask, INVALID_LOGPROB)

                print("Log Probabilities - logprobs Min:", logprobs.min().item(), "Max:", logprobs.max().item(), "Contains NaN or Inf:", torch.isnan(logprobs).any().item() or torch.isinf(logprobs).any().item())
                print("Log Probabilities - ref_logprobs Min:", ref_logprobs.min().item(), "Max:", ref_logprobs.max().item(), "Contains NaN or Inf:", torch.isnan(ref_logprobs).any().item() or torch.isinf(ref_logprobs).any().item())

                # 4. compute rewards
                kl = logprobs - ref_logprobs
                print("Log Probabilities - kl Min:", ref_logprobs.min().item(), "Max:", ref_logprobs.max().item(), "Contains NaN or Inf:", torch.isnan(ref_logprobs).any().item() or torch.isinf(ref_logprobs).any().item())

                non_score_reward = (-self.args.kl_coef * kl).sum(1)
                rlhf_reward = scores + non_score_reward.unsqueeze(1)

                # we generated `self.args.rloo_k` many responses per prompt
                # now we can implement the RLOO loss by subtracting the reward of
                # a response by the average rewards of other `rloo_k - 1` responses
                advantages = torch.zeros_like(rlhf_reward)
                for i in range(0, len(advantages)):
                    other_response_rlhf_rewards = []
                    for j in range(0, len(advantages)):
                        if i != j:
                            other_response_rlhf_rewards.append(rlhf_reward[j])
                    advantages[i] = rlhf_reward[i] - torch.stack(other_response_rlhf_rewards).mean(0)
                torch.cuda.empty_cache()

            with self.time_metric_ctx("calc_loss"):
                # PR TODO: remove this assertion when stable
                # ensure gradients can be set
                assert model.training, "model is incorrectly in eval mode"
                assert torch.is_grad_enabled(), "grad is disabled, but we need to calculate grad"

                # calculate gradients and loss
                with self.time_metric_ctx("model_forward"):
                    output = self.forward(model, query_responses)
                logits = output.logits[:, context_length - 1 : -1]
                logits /= self.args.temperature + 1e-7

                print("Response Generation - logits Min:", logits.min().item(), "Max:", logits.max().item(), "Std Dev:", logits.std().item(), "Contains NaN or Inf:", torch.isnan(logits).any().item() or torch.isinf(logits).any().item())
                print("Log Probability Extraction - responses Min ID:", responses.min().item(), "Max ID:", responses.max().item(), "Contains Invalid IDs:", torch.any(responses < 0).item() or torch.any(responses >= logits.size(-1)).item())

                new_all_logprobs = F.log_softmax(logits, dim=-1)
                new_logprobs = torch.gather(new_all_logprobs, 2, responses.unsqueeze(-1)).squeeze(-1)
                new_logprobs = torch.masked_fill(
                    new_logprobs, padding_mask, INVALID_LOGPROB
                )
                new_ratio = (new_logprobs - logprobs).exp()
                logprobs_diff = new_logprobs.sum(1) - logprobs.sum(1)
                ratio = torch.exp(logprobs_diff)
                pg_losses = -advantages * ratio
                pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - self.args.cliprange, 1.0 + self.args.cliprange)
                pg_loss_max = torch.max(pg_losses, pg_losses2)
                pg_loss = pg_loss_max.mean()
                pg_clipfrac = (pg_losses2 > pg_losses).float().mean()

                print("Gradient Calc - new_logprobs Min:", new_logprobs.min().item(), "Max:", new_logprobs.max().item(), "Contains NaN:", torch.isnan(new_logprobs).any().item())
                print("Gradient Calc - logprobs Min:", logprobs.min().item(), "Max:", logprobs.max().item(), "Contains NaN:", torch.isnan(logprobs).any().item())
                print("Gradient Calc - ratio Min:", ratio.min().item(), "Max:", ratio.max().item(), "Contains NaN or Inf:", torch.isnan(ratio).any().item() or torch.isinf(ratio).any().item())
                print("Gradient Calc - pg_losses Min:", pg_losses.min().item(), "Max:", pg_losses.max().item(), "Contains NaN or Inf:", torch.isnan(pg_losses).any().item() or torch.isinf(pg_losses).any().item())
                print("Gradient Calc - pg_loss Value:", pg_loss.item(), "Contains NaN or Inf:", torch.isnan(pg_loss).any().item() or torch.isinf(pg_loss).any().item())


            # log metrics
            with torch.no_grad():
                prob_dist = torch.nn.functional.softmax(logits, dim=-1)
                entropy = torch.logsumexp(logits, dim=-1) - torch.sum(prob_dist * logits, dim=-1)
                approxkl = 0.5 * (logprobs_diff**2).mean()

                rlhf_reward_mean = self.accelerator.gather(rlhf_reward).mean().item()
                # PR TODO: this is from original, but maybe it should be logged somewhere?
                #self.accelerator.print(f"{rlhf_reward_mean=}")
                mean_kl = kl.sum(1).mean()
                mean_entropy = (-logprobs).sum(1).mean()
                # PR TODO: why is this metric removed in the original
                # mean_non_score_reward = non_score_reward.sum(1).mean()
                # "objective/non_score_reward"


            metrics = {
                "objective/kl": self.accelerator.gather(mean_kl).mean().item(),
                "objective/entropy": self.accelerator.gather(mean_entropy).mean().item(),
                "objective/rlhf_reward": self.accelerator.gather(rlhf_reward).mean().item(),
                "objective/scores": self.accelerator.gather(scores.mean()).mean().item(),
                "policy/approxkl_avg": self.accelerator.gather(approxkl).mean().item(),
                "policy/clipfrac_avg": self.accelerator.gather(pg_clipfrac).mean().item(),
                "loss/policy_avg": self.accelerator.gather(pg_loss).mean().item(),
                # PR TODO: this isn't calculated in the original
                #"loss/value_avg": self.accelerator.gather(vf_loss_stats).mean().item(),
                #"val/clipfrac_avg": self.accelerator.gather(vf_clipfrac_stats).mean().item(),

                # PR TODO: how does this differ from mean_entropy
                #"policy/entropy_avg": self.accelerator.gather(entropy).mean().item(),
                "val/ratio": self.accelerator.gather(new_ratio).mean().item(),

                # PR TODO
                #"val/ratio_var": self.accelerator.gather(ratio_stats).var().item(),
                "val/num_eos_tokens": (responses == self.tokenizer.eos_token_id).sum().item(),
            }

            self.store_metrics(metrics)

            loss = pg_loss.to(self.args.device)

            # PR TODO: delete the commented if it truly is what's detaching the graph
            # it probably isn't a problem, I saw issues with LoRA_MLPBackward w/ Unsloth
            """
            del (
                output, logits, new_all_logprobs, new_logprobs,
                logprobs_diff, ratio, pg_losses, pg_losses2,
                pg_loss, pg_clipfrac, prob_dist, entropy, approxkl,
                kl, mean_kl, mean_entropy, scores
            )
            torch.cuda.empty_cache()
            """

            if return_outputs:
                return (loss, metrics)
            return loss

if __name__ == "__main__":
    pass
