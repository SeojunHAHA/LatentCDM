import torch
from accelerate import PartialState
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def load_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    return tokenizer


def load_model(model_cfg, training_cfg):
    quant_config = None
    if model_cfg.quant_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    device_map = {"": PartialState().process_index}
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg.model_name,
        quantization_config=quant_config,
        device_map=device_map,
        torch_dtype=torch.bfloat16 if training_cfg.bf16 else torch.float16,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )

    if model_cfg.quant_4bit:
        model = prepare_model_for_kbit_training(model)

    if model_cfg.use_lora:
        model.enable_input_require_grads()
        lora_config = LoraConfig(
            r=model_cfg.lora_r,
            lora_alpha=model_cfg.lora_alpha,
            target_modules=list(model_cfg.lora_target_modules),
            lora_dropout=model_cfg.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

    if training_cfg.gradient_checkpointing:
        model.config.use_cache = False

    return model
