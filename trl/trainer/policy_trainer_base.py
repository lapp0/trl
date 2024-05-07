from accelerate.utils import is_deepspeed_available
from collections import defaultdict
from contextlib import nullcontext, contextmanager
from copy import deepcopy
from dataclasses import dataclass
from tqdm import tqdm
from typing import Dict, Literal, Optional, Tuple, Union, Callable, Any
import gc
import inspect
import torch
import torch.nn as nn
import warnings

from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    GenerationConfig,
    PreTrainedModel,
    Trainer,
    TrainingArguments,
    PreTrainedTokenizerBase
)

# PR TODO: Determine if this is necessary
# from trl.models.utils import unwrap_model_for_generation

from ..core import logprobs_from_logits
from ..models import SUPPORTED_ARCHITECTURES, create_reference_model, PreTrainedModelWrapper
from .utils import disable_dropout_in_model, peft_module_casting_to_bf16
from ..import_utils import is_peft_available


if is_peft_available():
    from peft import PeftModel, get_peft_model, prepare_model_for_kbit_training

if is_deepspeed_available():
    import deepspeed


@dataclass
class PolicyTrainerArguments(TrainingArguments):
    response_length: int = 53
    """the length of the response"""
    truncate_token: Optional[Literal["eos"]] = None
    """the truncate token"""
    truncate_token_id: Optional[int] = None
    """the truncation token id"""
    temperature: float = 1.0
    """the sampling temperature"""
    penalty_reward_value: int = -1
    """the reward value for responses that do not contain `truncate_token_id`"""
    non_eos_penalty: bool = False
    """whether to penalize responses that do not contain `truncate_token_id`"""

    update_generation_steps: Optional[int] = 64
    """Number of steps between updating the generation model. If None, once per epoch"""


class fast_eval_mode:
    """
    Convert to model.eval(), then revert to previous state

    Behavior
    - DOESNT disable grad
    - Disable dropout layers
    - Freeze BatchNorm
    """
    def __init__(self, model):
        self.model = model

    def __enter__(self):
        self.was_training = self.model.training
        if self.was_training:
            self.model.eval()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.was_training:
            self.model.train()


def prepare_model_and_ref_model(
        model: Optional[Union[PreTrainedModel, nn.Module, str]],
        ref_model: Optional[Union[PreTrainedModel, nn.Module, str]],
        model_init_kwargs: Optional[Dict],
        ref_model_init_kwargs: Optional[Dict],
        peft_config: Optional[Dict],
        force_use_ref_model: bool,
        args: Optional[TrainingArguments],
):
    """
    Adapted from dpo_trainer.py
    Allow user to pass a model or model URI + init kwargs + optional peft_config
    Return a fully initialized model and ref_model
    """
    if model_init_kwargs is None:
        model_init_kwargs = {}
    elif not isinstance(model, str):
        raise ValueError("You passed model_init_kwargs to the trainer. But model is already instantiated.")

    if ref_model_init_kwargs is None:
        ref_model_init_kwargs = {}
    elif not isinstance(ref_model, str):
        raise ValueError(
            "You passed ref_model_init_kwargs to the trainer. But your ref_model is already instantiated."
        )

    if isinstance(model, str):
        warnings.warn(
            "You passed a model_id to the trainer. This will automatically create an "
            "`AutoModelForCausalLM` or a `PeftModel` (if you passed a `peft_config`) for you."
        )
        model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)

    if isinstance(ref_model, str):
        warnings.warn(
            "You passed a ref model_id to the trainer. This will automatically create an "
            "`AutoModelForCausalLM`"
        )
        ref_model = AutoModelForCausalLM.from_pretrained(ref_model, **ref_model_init_kwargs)

    if not is_peft_available() and peft_config is not None:
        raise ValueError(
            "PEFT is not installed and you passed a `peft_config` in the trainer's kwargs, please install it to use the PEFT models"
        )
    elif is_peft_available() and peft_config is not None:
        # if model is a peft model and we have a peft_config, we merge and unload it first
        if isinstance(model, PeftModel):
            model = model.merge_and_unload()

        if ref_model is not None and not force_use_ref_model:
            raise ValueError(
                "You passed both a ref_model and a peft_config. For training PEFT adapters there is no need to pass a reference"
                " model. Please pass `ref_model=None` in case you want to train PEFT adapters, or pass a ref_model with `force_use_ref_model=True` in trainer's init."
                " if you want to use a different ref_model."
            )

        if getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_loaded_in_4bit", False):
            _support_gc_kwargs = hasattr(
                args, "gradient_checkpointing_kwargs"
            ) and "gradient_checkpointing_kwargs" in list(
                inspect.signature(prepare_model_for_kbit_training).parameters
            )

            prepare_model_kwargs = {"use_gradient_checkpointing": args.gradient_checkpointing}

            if _support_gc_kwargs:
                prepare_model_kwargs["gradient_checkpointing_kwargs"] = args.gradient_checkpointing_kwargs

            model = prepare_model_for_kbit_training(model, **prepare_model_kwargs)

        # PR TODO: Ensure I can remove this, as it only applies to older versions of transformers
        """
        elif getattr(args, "gradient_checkpointing", False):
            # For backward compatibility with older versions of transformers
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            else:

                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)

                model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
        """
        # get peft model with the given config
        model = get_peft_model(model, peft_config)
        if args.bf16 and getattr(model, "is_loaded_in_4bit", False):
            peft_module_casting_to_bf16(model)

    # PR TODO: Ensure I can remove this, as it only applies to older versions of transformers
    """
    # For models that use gradient_checkpointing, we need to attach a hook that enables input
    # to explicitly have `requires_grad=True`, otherwise training will either silently
    # fail or completely fail.
    elif getattr(args, "gradient_checkpointing", False):
        # For backward compatibility with older versions of transformers
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    """

    return model, ref_model


# PR TODO: maybe this isn't necessary? This may be handled already by the accelerator, as it prepares
#          any model type object in Accelerator._prepare_deepspeed
# https://github.com/huggingface/accelerate/blob/main/src/accelerate/accelerator.py#L1530
def _prepare_deepspeed(self, accelerator, model: PreTrainedModelWrapper, evaluation_mode):
    # Adapted from accelerate: https://github.com/huggingface/accelerate/blob/739b135f8367becb67ffaada12fe76e3aa60fefd/src/accelerate/accelerator.py#L1473
    deepspeed_plugin = accelerator.state.deepspeed_plugin
    config_kwargs = deepcopy(deepspeed_plugin.deepspeed_config)

    if model is not None:
        if hasattr(model, "config"):
            hidden_size = (
                max(model.config.hidden_sizes)
                if getattr(model.config, "hidden_sizes", None)
                else getattr(model.config, "hidden_size", None)
            )
            if hidden_size is not None and config_kwargs["zero_optimization"]["stage"] == 3:
                # Note that `stage3_prefetch_bucket_size` can produce DeepSpeed messages like: `Invalidate trace cache @ step 0: expected module 1, but got module 0`
                # This is expected and is not an error, see: https://github.com/microsoft/DeepSpeed/discussions/4081
                config_kwargs.update(
                    {
                        "zero_optimization.reduce_bucket_size": hidden_size * hidden_size,
                        "zero_optimization.stage3_param_persistence_threshold": 10 * hidden_size,
                        "zero_optimization.stage3_prefetch_bucket_size": 0.9 * hidden_size * hidden_size,
                    }
                )

    # If ZeRO-3 is used, we shard both the active and reference model.
    # Otherwise, we assume the reference model fits in memory and is initialized on each device with ZeRO disabled (stage 0)
    if config_kwargs["zero_optimization"]["stage"] != 3:
        config_kwargs["zero_optimization"]["stage"] = 0
    model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
    if evaluation_mode:
        model.eval()
    return model


def _prepare_multigpu(model, accelerator, is_deepspeed_enabled: bool):
    if model is None:
        return model
    elif is_deepspeed_enabled:
        return _prepare_deepspeed(
            accelerator,
            model,
            evaluation_mode=True
        )
    else:
        return accelerator.prepare_model(
            model,
            evaluation_mode=True
        )


def cuda_gc(func):
    def wrapper(*args, **kwargs):
        # Run the function
        result = func(*args, **kwargs)
        # Perform cleanup actions
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return result
    return wrapper


class DynamicDataLoader:
    """
    Augment the base_dataloader with num_steps lookahead
    """
    def __init__(
            self,
            base_dataloader,
            mutate_fn,
            num_steps,
    ):
        self.base_dataloader = base_dataloader
        self.mutate_fn = mutate_fn
        self.num_steps = num_steps

    def __iter__(self):
        batch_buffer = []
        for batch in self.base_dataloader:
            batch_buffer.append(batch)
            if len(batch_buffer) >= self.num_steps:
                # When the buffer reaches the specified number of steps, apply the mutation function
                batch_buffer = self.mutate_fn(batch_buffer)
                # Yield batches one by one from the mutated buffer
                while batch_buffer:
                    yield batch_buffer.pop(0)
                # reset tqdm while not changing position in dataloader

        # If there are any remaining batches after the last full set, yield them as well
        while batch_buffer:
            yield batch_buffer.pop(0)

    def __len__(self):
        return len(self.base_dataloader)


class ReferenceModelManager:
    """
    Context manager to prepare and manage the reference model.
    - If it doesn't exist create a reference model
      - OR use the base model with adapters disabled if base model uses PEFT
    - Distribute the model to the accelerator
    """
    def __init__(
            self,
            accelerator,
            ref_model: Optional[PreTrainedModelWrapper] = None,
            model: Optional[PreTrainedModelWrapper] = None,
            is_deepspeed_enabled: bool = False
    ):
        self.accelerator = accelerator
        if not is_peft_available():
            self.is_peft_model = False
        else:
            self.is_peft_model = (
                getattr(model, "is_peft_model", False) or isinstance(model, PeftModel)
            )

        if isinstance(ref_model, SUPPORTED_ARCHITECTURES):
            self.ref_model = ref_model
        elif ref_model is None and not self.is_peft_model:
            self.ref_model = create_reference_model(model)
        elif self.is_peft_model:
            self.ref_model = None
            self.model = model
        else:
            raise ValueError(
                f"ref_model must be a PreTrainedModelWrapper or `None` "
                f"got {type(ref_model)} "
                f"- supported architectures are: {SUPPORTED_ARCHITECTURES} "
            )

        if self.ref_model is not None and not self.is_peft_model:
            self.ref_model = _prepare_multigpu(self.ref_model, self.accelerator, is_deepspeed_enabled)

    @contextmanager
    def __call__(self):
        if self.ref_model is not None:
            yield self.ref_model
        elif self.is_peft_model:
            with self.model.disable_adapter():
                yield self.model
        else:
            raise ValueError


class PolicyTrainerBase(Trainer):
    """
    Base class for implementing a policy training algorithm.
    # PR TODO: document arguments
    """
    def __init__(
            self,
            model: Optional[PreTrainedModelWrapper],
            ref_model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
            args: Optional[TrainingArguments] = None,
            train_dataset: Optional[Dataset] = None,
            reward_model: Optional[PreTrainedModelWrapper] = None,
            reward_fn: Callable = None,
            train_generation_config: Optional[GenerationConfig] = None,
            eval_generation_config: Optional[GenerationConfig] = None,
            tokenizer: Optional[PreTrainedTokenizerBase] = None,
            model_init_kwargs: Optional[Dict] = None,
            ref_model_init_kwargs: Optional[Dict] = None,
            peft_config: Optional = None,
            force_use_ref_model: bool = False,
            **kwargs
    ) -> None:

        model, ref_model = prepare_model_and_ref_model(
            model=model,
            ref_model=ref_model,
            model_init_kwargs=model_init_kwargs,
            ref_model_init_kwargs=ref_model_init_kwargs,
            peft_config=peft_config,
            force_use_ref_model=force_use_ref_model,
            args=args,
        )

        # Disable dropout ensures logprobs during generation aren't different from forward pass
        # https://github.com/huggingface/trl/pull/1586#discussion_r1579533825
        for m in [model, ref_model, reward_model]:
            if m is not None:
                disable_dropout_in_model(m)

        # PR TODO: subclass with RewardTrainerBase which accepts a reward_model or reward_fn
        #          remove the below from this class
        assert (reward_model is not None) != (reward_fn is not None), "Must set either reward_model or reward_fn, but not both"
        if reward_model is not None and "score" not in dir(reward_model):
            raise TypeError(f"Reward model of type {type(reward_model)} has no score function.")
        self.reward_model = reward_model
        self.reward_fn = reward_fn

        # prepare generation config
        default_generation_config = GenerationConfig(
            max_new_tokens=args.response_length,
            min_new_tokens=args.response_length,
            temperature=(args.temperature + 1e-7),
            top_k=0.0,
            top_p=1.0,
            do_sample=True,
        )
        self.train_generation_config = train_generation_config or default_generation_config
        self.eval_generation_config = eval_generation_config or default_generation_config

        # force disable `pad_token_id` and `eos_token_id` because we just want to
        # generate tokens without truncation / padding
        model.generation_config.eos_token_id = None
        model.generation_config.pad_token_id = None

        if args.truncate_token and args.truncate_token == "eos":
            args.truncate_token_id = tokenizer.eos_token_id

        # handle casting self.model
        if args.bf16 and getattr(model, "is_loaded_in_4bit", False):
            self.cast_model_ctx = lambda: torch.cuda.amp.autocast(dtype=torch.bfloat16)
        else:
            self.cast_model_ctx = nullcontext

        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
            **kwargs,
        )

        self.ref_model_mgr = ReferenceModelManager(
            self.accelerator,
            ref_model=ref_model,
            model=model,
            is_deepspeed_enabled=self.is_deepspeed_enabled,
        )

        # Add tags for models that have been loaded with the correct transformers version
        if hasattr(self.model, "add_model_tags"):
            self.model.add_model_tags(self._tag_names)

        # PR TOOD: accelerate with reward model
        #self.reward_model.to(self.accelerator.device)
        #self.reward_model = _prepare_multigpu(
        #    self.reward_model,
        #    self.accelerator,
        #    self.is_deepspeed_enabled
        #)

        self._stored_metrics = defaultdict(lambda: defaultdict(list))

    @cuda_gc
    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:

        with self.cast_model_ctx():
            loss, metrics = self.get_batch_loss_metrics(model, inputs)

        loss = loss.to(self.args.device)

        self.store_metrics(metrics, train_eval="train")

        if return_outputs:
            return (loss, metrics)
        return loss

    @cuda_gc
    def generate_batch_extras(self, model, input_ids):
        # PR TODO: generation_batch_size
        queries = input_ids.to(self.accelerator.device)
        context_length = queries.shape[1]
        with torch.no_grad(), self.cast_model_ctx():
            query_responses = self.generate(
                self.model,
                queries,
                self.train_generation_config,
            )
            responses = query_responses[:, context_length:]
            generation_logits, generation_logprobs = self.calc_logprobs(
                model, query_responses, context_length
            )
        return {
            "queries": queries.detach().cpu(),
            "query_responses": query_responses.detach().cpu(),
            "responses": responses.detach().cpu(),
            "generation_logits": generation_logits.detach().cpu(),
            "generation_logprobs": generation_logprobs.detach().cpu(),
        }

    @cuda_gc
    def calc_logprobs(self, model, query_responses, context_length):
        responses = query_responses[:, context_length:]
        output_logits = self.forward(model, query_responses).logits
        response_logits = output_logits[:, context_length - 1: -1]
        response_logits /= max(self.args.temperature, 1e-7)
        response_logprobs = logprobs_from_logits(response_logits, responses, gather=True)
        return response_logits, response_logprobs

    def get_train_dataloader(self):
        import pdb;pdb.set_trace()
        dataloader = super().get_train_dataloader()

        # PR TODO: generation_batch_size
        def mutate_fn(batches):
            for batch in tqdm(batches, desc="generating batch extras"):
                batch_extras = self.generate_batch_extras(
                    self.model, batch["input_ids"]
                )
                batch.update(batch_extras)
            return batches
        return DynamicDataLoader(
            dataloader,
            mutate_fn,
            num_steps=self.args.update_generation_steps * self.args.gradient_accumulation_steps
        )

    @staticmethod
    def _disable_dropout(model):
        if model is None:
            return
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0

    def generate(self, model, queries, generation_config, requires_grad=True):
        """generate in a way that does not affect padding tokens"""
        with fast_eval_mode(model):
            context_length = queries.shape[1]
            attention_mask = queries != self.tokenizer.pad_token_id
            input_ids = torch.masked_fill(queries, ~attention_mask, 0)
            output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=generation_config,
                return_dict_in_generate=True,
                # PR TODO: use_cache is faster, but what are the risks? review. use_cache is necessary for forward() but is it necessary for generate
                # use_cache=False,
            )
            # PR TODO: only return the response not anything else
        query_responses = torch.cat((queries, output.sequences[:, context_length:]), dim=1)
        return query_responses

    def forward(self, model, query_responses):
        attention_mask = query_responses != self.tokenizer.pad_token_id
        input_ids = torch.masked_fill(query_responses, ~attention_mask, 0)
        return model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            output_hidden_states=True,
            use_cache=False,
        )

    @cuda_gc
    def get_reward(self, reward_model, query_responses, context_length):
        attention_mask = query_responses != self.tokenizer.pad_token_id

        input_ids = torch.masked_fill(query_responses, ~attention_mask, 0)

        output = reward_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            output_hidden_states=True,
        )
        reward_logits = reward_model.score(output.hidden_states[-1])
        sequence_lengths = (
            self.first_true_indices(
                query_responses[:, context_length:] == self.tokenizer.pad_token_id
            ) - 1 + context_length
        )

        # https://github.com/huggingface/transformers/blob/dc68a39c8111217683bf49a4912d0c9018bab33d/src/transformers/models/gpt2/modeling_gpt2.py#L1454
        return (
            reward_logits,
            reward_logits[
                torch.arange(reward_logits.size(0), device=reward_logits.device),
                sequence_lengths,
            ].squeeze(-1),
            sequence_lengths,
        )

    def truncate_response(self, responses):
        trunc_idxs = self.first_true_indices(responses == self.args.truncate_token_id).unsqueeze(-1)
        new_size = [1] * (len(responses.size()) - 1) + [responses.shape[1]]
        idxs = torch.arange(responses.shape[1], device=responses.device).view(*new_size)
        postprocessed_responses = torch.masked_fill(
            responses, idxs > trunc_idxs,
            self.tokenizer.pad_token_id
        )
        return postprocessed_responses

    @staticmethod
    def first_true_indices(bools, dtype=torch.long):
        """
        Takes an N-dimensional bool tensor and returns an (N-1)-dimensional tensor of integers giving
        the position of the first True in each "row".

        Returns the length of the rows (bools.size(-1)) if no element is True in a given row.
        """
        row_len = bools.size(-1)
        zero_or_index = (
            row_len * (~bools).type(dtype) +
            torch.arange(row_len, dtype=dtype, device=bools.device)
        )
        return torch.min(zero_or_index, dim=-1).values

    def store_metrics(
            self,
            metrics: Dict[str, float],
            train_eval: Literal["train", "eval"] = "train"
    ) -> None:
        for key, value in metrics.items():
            self._stored_metrics[train_eval][key].append(value)

    def log(self, logs: Dict[str, float]) -> None:
        """
        Log `logs` on the various objects watching training, including stored metrics.

        Args:
            logs (`Dict[str, float]`):
                The values to log.
        """
        # logs either has 'loss' or 'eval_loss'
        train_eval = "train" if "loss" in logs else "eval"
        # Add averaged stored metrics to logs
        for key, metrics in self._stored_metrics[train_eval].items():
            logs[key] = torch.tensor(metrics).to(dtype=torch.float32).mean().item()
        del self._stored_metrics[train_eval]
        return super().log(logs)

    def time_metric_ctx(self, timer_name: str):
        from time import perf_counter
        timer_metric_name = f"timer/{timer_name}"

        class catchtime:
            def __enter__(s):
                s.start = perf_counter()

            def __exit__(s, type, value, traceback):
                runtime = perf_counter() - s.start
                self.store_metrics({timer_metric_name: runtime})

        return catchtime()

    def training_step(self, *args, **kwargs):
        """time logged training step"""
        with self.time_metric_ctx("training_step"):
            return super().training_step(*args, **kwargs)
