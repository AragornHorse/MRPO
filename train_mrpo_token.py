from datasets import load_dataset
import argparse
from peft import LoraConfig
from mr_trainer_token import MRTrainer, MRPOConfig
from dataset_pretreat import get_map, get_split, get_drop_cols
from transformers import AutoTokenizer


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str)
parser.add_argument("--base-model", type=str)
parser.add_argument("--reward-model", type=str)
parser.add_argument("--output-dir", type=str)
parser.add_argument("--lr", type=float, default=1e-5)
parser.add_argument("--n-epoch", type=int, default=1)
parser.add_argument("--grad-acc-step", type=int, default=32)
parser.add_argument("--num-generation", type=int, default=8)
parser.add_argument("--per-device-batch-size", type=int, default=1)
parser.add_argument("--lora-r", type=int, default=8)
parser.add_argument("--lora-alpha", type=int, default=16)
parser.add_argument("--max-prompt-len", type=int, default=256)
parser.add_argument("--max-completion-len", type=int, default=512)
parser.add_argument("--beta", type=float, default=0.01)
parser.add_argument("--log-step", type=int, default=10)
parser.add_argument("--save-step", type=int, default=10)
parser.add_argument("--checkpoint", type=str, default=None)

parser.add_argument("--loss-func", type=str, default="l1")
parser.add_argument("--loss-clip-lower", type=float, default=1e-3)
parser.add_argument("--loss-clip-upper", type=float, default=30)
parser.add_argument("--generation-batch-size", type=int, default=48)
args = parser.parse_args()

print(args.__dict__)

dataset = load_dataset(
    args.dataset, split=get_split(args.dataset)
)

tokenizer = AutoTokenizer.from_pretrained(args.base_model)

map_func = get_map(args.dataset, tokenizer=tokenizer)
if map_func is not None:
    dataset = dataset.map(map_func, batched=True)

drop_cols = get_drop_cols(args.dataset)
if drop_cols is not None:
    dataset = dataset.remove_columns(drop_cols)

training_args = MRPOConfig(
    output_dir=args.output_dir, logging_steps=args.log_step, beta=args.beta, learning_rate=args.lr, num_train_epochs=args.n_epoch,
    gradient_accumulation_steps=args.grad_acc_step, num_generations=args.num_generation,
    per_device_train_batch_size=args.per_device_batch_size, max_prompt_length=args.max_prompt_len,
    max_completion_length=args.max_completion_len, save_steps=args.save_step, bf16=True, bf16_full_eval=True,
    loss_func=args.loss_func, generation_batch_size=args.generation_batch_size, loss_clip_lower=args.loss_clip_lower,
    loss_clip_upper=args.loss_clip_upper
)

trainer = MRTrainer(
    model=args.base_model,
    reward_funcs=args.reward_model,
    args=training_args,
    train_dataset=dataset,
    peft_config=LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
)

train_result = trainer.train(resume_from_checkpoint=args.checkpoint)
print(train_result)


