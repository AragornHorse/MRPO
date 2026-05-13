import math
from trl import GRPOConfig, GRPOTrainer
import torch
from dataclasses import dataclass, field
import utils
from transformers import is_wandb_available
import torch.nn as nn


if is_wandb_available():
    import wandb


def nanmin(tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute the minimum value of a tensor, ignoring NaNs. This function only supports 1D tensors.

    Args:
        tensor (`torch.Tensor`): Input tensor of shape `(N,)`.

    Returns:
        `torch.Tensor`: Minimum value of the tensor, ignoring NaNs. Returns NaN if all values are NaN.
    """
    if torch.isnan(tensor).all():
        return torch.tensor(float("nan"), dtype=tensor.dtype, device=tensor.device)
    return torch.min(tensor[~torch.isnan(tensor)])


def nanmax(tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute the maximum value of a tensor, ignoring NaNs. This function only supports 1D tensors.

    Args:
        tensor (`torch.Tensor`): Input tensor of shape `(N,)`.

    Returns:
        `torch.Tensor`: Maximum value of the tensor, ignoring NaNs. Returns NaN if all values are NaN.
    """
    if torch.isnan(tensor).all():
        return torch.tensor(float("nan"), dtype=tensor.dtype, device=tensor.device)
    return torch.max(tensor[~torch.isnan(tensor)])



@dataclass
class MRPOConfig(GRPOConfig):
    loss_func: str = field(
        default='l1',
        metadata={
            "help": "Loss function used for MRPO"
        }
    )

    loss_clip_lower: float = field(
        default=0.2,
        metadata={
            "help": ""
        }
    )

    loss_clip_upper: float = field(
        default=0.28,
        metadata={
            "help": ""
        }
    )


failure_prompts = [
]


class MRTrainer(GRPOTrainer):

    def _compute_loss(self, model, inputs):
        # Compute the per-token log probabilities for the model
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

        per_token_logps = self._get_per_token_logps(model, input_ids, attention_mask, logits_to_keep)

        # Compute the KL divergence between the model and the reference model
        if self.beta != 0.0:
            with torch.no_grad():
                ref_per_token_logps = inputs["ref_per_token_logps"]
                per_token_kl = (
                        torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
                )

        # Compute the loss
        advantages = inputs["advantages"]

        old_per_token_logps = (
            per_token_logps.detach() if inputs["old_per_token_logps"] is None else inputs["old_per_token_logps"]
        )

        # MRPO-G
        if self.args.loss_func == 'l1':
            coef_1 = torch.exp(per_token_logps - old_per_token_logps)
            coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)

            # Two-sided clipping
            if self.args.delta is not None:
                coef_1 = torch.clamp(coef_1, max=self.args.delta)

            per_token_loss1 = coef_1 * advantages.unsqueeze(1)
            per_token_loss2 = coef_2 * advantages.unsqueeze(1)
            per_token_loss = -torch.min(per_token_loss1, per_token_loss2)

            per_token_l1 = nn.L1Loss()(
                utils.ratio(per_token_logps, old_per_token_logps) * completion_mask,
                utils.ratio(ref_per_token_logps, old_per_token_logps) * completion_mask
            ) / torch.clamp(completion_mask.float().mean(), 0.01, 1.)

            loss = ((per_token_loss * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)).mean()
            loss = loss + self.beta * per_token_l1

        # MRPO-R
        elif self.args.loss_func == 'l1_log':
            with torch.no_grad():
                coef_1 = torch.exp(per_token_logps - old_per_token_logps)
                coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
            ratio = per_token_logps - old_per_token_logps
            clipped_ratio = torch.clamp(ratio, math.log(1 - self.args.loss_clip_lower), math.log(1 + self.args.loss_clip_upper))
            per_token_loss = torch.min(ratio * advantages[..., None], clipped_ratio * advantages[..., None])

            per_token_l1 = nn.L1Loss()(
                utils.ratio(per_token_logps, old_per_token_logps) * completion_mask,
                utils.ratio(ref_per_token_logps, old_per_token_logps) * completion_mask
            ) / torch.clamp(completion_mask.float().mean(), 0.01, 1.)

            loss = -((per_token_loss * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)).mean()
            loss = loss + self.beta * per_token_l1

        else:
            raise ValueError(f"Unknown loss_func: {self.args.loss_func}")


        # Log the metrics
        mode = "train" if self.model.training else "eval"

        if self.beta != 0.0:
            mean_kl = (per_token_kl * completion_mask).sum() / completion_mask.sum()
            self._metrics[mode]["kl"].append(self.accelerator.gather(mean_kl).nanmean().item())

        # Compute the clipped probability ratios
        is_low_clipped = (coef_1 < 1 - self.epsilon_low) & (advantages.unsqueeze(1) < 0)
        is_high_clipped = (coef_1 > 1 + self.epsilon_high) & (advantages.unsqueeze(1) > 0)
        is_region_clipped = is_low_clipped | is_high_clipped

        low_clip = (is_low_clipped * completion_mask).sum() / completion_mask.sum()
        high_clip = (is_high_clipped * completion_mask).sum() / completion_mask.sum()
        clip_ratio = (is_region_clipped * completion_mask).sum() / completion_mask.sum()

        gathered_low_clip = self.accelerator.gather(low_clip)
        self._metrics[mode]["clip_ratio/low_mean"].append(gathered_low_clip.nanmean().item())
        self._metrics[mode]["clip_ratio/low_min"].append(nanmin(gathered_low_clip).item())
        gathered_high_clip = self.accelerator.gather(high_clip)
        self._metrics[mode]["clip_ratio/high_mean"].append(gathered_high_clip.nanmean().item())
        self._metrics[mode]["clip_ratio/high_max"].append(nanmax(gathered_high_clip).item())
        gathered_clip_ratio = self.accelerator.gather(clip_ratio)
        self._metrics[mode]["clip_ratio/region_mean"].append(gathered_clip_ratio.nanmean().item())
        return loss

