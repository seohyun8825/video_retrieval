# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/trainer_seq2seq.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
from types import MethodType
from typing import TYPE_CHECKING, Any, Optional, Union
from pathlib import Path

import numpy as np
import torch
from transformers import Seq2SeqTrainer
from typing_extensions import override

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from ...extras.packages import is_transformers_version_greater_than
from ..callbacks import SaveProcessorCallback
from ..fp8_utils import configure_fp8_environment, verify_fp8_status
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler


if TYPE_CHECKING:
    from torch.utils.data import Dataset
    from transformers import PreTrainedTokenizer, ProcessorMixin
    from transformers.trainer import PredictionOutput

    from ...hparams import FinetuningArguments, ModelArguments


logger = logging.get_logger(__name__)


class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    r"""Inherits Seq2SeqTrainer to compute generative metrics such as BLEU and ROUGE."""

    def __init__(
        self,
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"],
        model_args: Optional["ModelArguments"] = None,
        gen_kwargs: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        # Configure FP8 environment if enabled
        if model_args is not None and model_args.fp8:
            configure_fp8_environment(model_args)
        if is_transformers_version_greater_than("4.46"):
            kwargs["processing_class"] = kwargs.pop("tokenizer")
        else:
            self.processing_class: PreTrainedTokenizer = kwargs.get("tokenizer")

        super().__init__(**kwargs)
        if processor is not None:
            # avoid wrong loss under gradient accumulation
            # https://github.com/huggingface/transformers/pull/36044#issuecomment-2746657112
            self.model_accepts_loss_kwargs = False

        self.finetuning_args = finetuning_args
        if gen_kwargs is not None:
            # https://github.com/huggingface/transformers/blob/v4.45.0/src/transformers/trainer_seq2seq.py#L287
            self._gen_kwargs = gen_kwargs

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

        if finetuning_args.use_dft_loss:
            from ..trainer_utils import dft_loss_func

            self.compute_loss_func = dft_loss_func

        # Verify FP8 status after trainer initialization (accelerator should be available)
        if model_args is not None and model_args.fp8 and hasattr(self, "accelerator"):
            verify_fp8_status(self.accelerator, model_args)

        # Debug: first-N sample token analyzer (enabled by env LMF_ANALYZE_TOKEN=1)
        self._debug_token_log_count: int = 0
        self._debug_token_log_max: int = int(os.environ.get("LMF_ANALYZE_MAX_SAMPLES", 3))

    @override
    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    @override
    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    @override
    def _get_train_sampler(self, *args, **kwargs) -> Optional["torch.utils.data.Sampler"]:
        if self.finetuning_args.disable_shuffling:
            return torch.utils.data.SequentialSampler(self.train_dataset)

        return super()._get_train_sampler(*args, **kwargs)

    @override
    def compute_loss(self, model, inputs, *args, **kwargs):
        # Optional lightweight debug logging before forward
        self._maybe_log_first_samples(inputs)
        # Strip debug-only keys so model forward doesn't receive them
        inputs.pop("__debug_vidlens", None)
        inputs.pop("__debug_video_token_id", None)
        inputs.pop("__debug_vision_bos_id", None)
        inputs.pop("__debug_vision_eos_id", None)
        return super().compute_loss(model, inputs, *args, **kwargs)

    def _maybe_log_first_samples(self, inputs: dict[str, Any]) -> None:
        if self._debug_token_log_count >= self._debug_token_log_max:
            return
        if os.environ.get("LMF_ANALYZE_TOKEN", "0") != "1":
            return
        if not self.is_world_process_zero():
            return

        try:
            input_ids = inputs.get("input_ids")
            if input_ids is None:
                return
            bsz = input_ids.size(0)

            # Prepare output directory
            out_dir = Path(self.args.output_dir) / "output_debug"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / "first3_from_trainer.jsonl"

            # Resolve video token ids (may be absent)
            vtok_id = inputs.get("__debug_video_token_id")
            vbos_id = inputs.get("__debug_vision_bos_id")
            veos_id = inputs.get("__debug_vision_eos_id")
            vtok_id = int(vtok_id.item()) if torch.is_tensor(vtok_id) else (int(vtok_id) if vtok_id is not None else None)
            vbos_id = int(vbos_id.item()) if torch.is_tensor(vbos_id) else (int(vbos_id) if vbos_id is not None else None)
            veos_id = int(veos_id.item()) if torch.is_tensor(veos_id) else (int(veos_id) if veos_id is not None else None)

            # Video grid per batch (concatenated across samples)
            vg = inputs.get("video_grid_thw")
            if vg is not None and torch.is_tensor(vg):
                vg_list = vg.detach().cpu().tolist()
            elif isinstance(vg, list):
                vg_list = vg
            else:
                vg_list = []

            # Number of videos per sample to slice vg_list
            vidlens_t = inputs.get("__debug_vidlens")
            if vidlens_t is not None and torch.is_tensor(vidlens_t):
                vidlens = vidlens_t.detach().cpu().tolist()
            elif isinstance(vidlens_t, list):
                vidlens = vidlens_t
            else:
                vidlens = [0] * bsz

            # Iterate samples in this batch, log until reaching max
            vpos = 0
            logs = []
            for i in range(bsz):
                if self._debug_token_log_count >= self._debug_token_log_max:
                    break

                ids_i = input_ids[i].detach().cpu().tolist()
                total_tokens = int(len(ids_i))

                # Prefer counting by explicit token ids; fallback to grid-based estimate
                def _count_id(arr, tok):
                    return int(sum(1 for x in arr if x == tok)) if tok is not None else 0

                video_visual_tokens = _count_id(ids_i, vtok_id)
                wrap_bos = _count_id(ids_i, vbos_id)
                wrap_eos = _count_id(ids_i, veos_id)
                wrap_tokens = wrap_bos + wrap_eos

                # Slice per-sample video grids and derive frames and grid(H,W)
                nvid = int(vidlens[i]) if i < len(vidlens) else 0
                sample_vg = vg_list[vpos : vpos + nvid] if nvid > 0 else []
                vpos += nvid
                video_frames = [int(t[0]) for t in sample_vg] if sample_vg else []
                video_grid_hw = [[int(t[1]), int(t[2])] for t in sample_vg] if sample_vg else []

                # Fallback if token IDs are unavailable or zero-count: estimate from grids
                if (vtok_id is None or video_visual_tokens == 0) and sample_vg:
                    merge_size = 2  # default for Qwen family
                    def _per_video_counts(thw):
                        T, H, W = int(thw[0]), int(thw[1]), int(thw[2])
                        seqlen_per_frame = (H // merge_size) * (W // merge_size)
                        visual = T * seqlen_per_frame
                        wrap = 2 * T
                        return visual, wrap
                    visual_sum, wrap_sum = 0, 0
                    for thw in sample_vg:
                        v, w = _per_video_counts(thw)
                        visual_sum += v
                        wrap_sum += w
                    video_visual_tokens = int(visual_sum)
                    wrap_tokens = int(wrap_sum)

                # Estimate text tokens as remainder after removing visual+wrap
                total_text_tokens = max(0, total_tokens - video_visual_tokens - wrap_tokens)

                logs.append(
                    {
                        "batch_index": int(getattr(self.state, "global_step", 0)),
                        "sample_in_batch": i,
                        "total_text_tokens": total_text_tokens,
                        "video_visual_tokens": int(video_visual_tokens),
                        "wrap_tokens": int(wrap_tokens),
                        "video_total_tokens_no_timestamps": int(video_visual_tokens + wrap_tokens),
                        "video_frames": video_frames,
                        "video_grid_hw": video_grid_hw,
                        "input_ids_total_len": total_tokens,
                    }
                )

                self._debug_token_log_count += 1

            if logs:
                with open(out_path, "a", encoding="utf-8") as w:
                    for rec in logs:
                        w.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception as e:  # never break training
            logger.warning_rank0_once(f"[analyze_token] debug logging failed: {e}")

    @override
    def prediction_step(
        self,
        model: "torch.nn.Module",
        inputs: dict[str, Union["torch.Tensor", Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
        **gen_kwargs,
    ) -> tuple[Optional[float], Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        r"""Remove the prompt part in the generated tokens.

        Subclass and override to inject custom behavior.
        """
        if self.args.predict_with_generate:  # do not pass labels to model when generate
            labels = inputs.pop("labels", None)
        else:
            labels = inputs.get("labels")

        loss, generated_tokens, _ = super().prediction_step(
            model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys, **gen_kwargs
        )
        if generated_tokens is not None and self.args.predict_with_generate:
            generated_tokens[:, : inputs["input_ids"].size(-1)] = self.processing_class.pad_token_id
            generated_tokens = generated_tokens.contiguous()

        return loss, generated_tokens, labels

    def save_predictions(
        self, dataset: "Dataset", predict_results: "PredictionOutput", skip_special_tokens: bool = True
    ) -> None:
        r"""Save model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info_rank0(f"Saving prediction results to {output_prediction_file}")

        labels = np.where(
            predict_results.label_ids != IGNORE_INDEX, predict_results.label_ids, self.processing_class.pad_token_id
        )
        preds = np.where(
            predict_results.predictions != IGNORE_INDEX,
            predict_results.predictions,
            self.processing_class.pad_token_id,
        )

        for i in range(len(preds)):
            pad_len = np.nonzero(preds[i] != self.processing_class.pad_token_id)[0]
            if len(pad_len):  # move pad token to last
                preds[i] = np.concatenate((preds[i][pad_len[0] :], preds[i][: pad_len[0]]), axis=-1)

        decoded_inputs = self.processing_class.batch_decode(dataset["input_ids"], skip_special_tokens=False)
        decoded_preds = self.processing_class.batch_decode(preds, skip_special_tokens=skip_special_tokens)
        decoded_labels = self.processing_class.batch_decode(labels, skip_special_tokens=skip_special_tokens)

        with open(output_prediction_file, "w", encoding="utf-8") as f:
            for text, pred, label in zip(decoded_inputs, decoded_preds, decoded_labels):
                f.write(json.dumps({"prompt": text, "predict": pred, "label": label}, ensure_ascii=False) + "\n")
