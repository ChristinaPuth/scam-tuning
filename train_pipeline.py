# test_saved_model.py

from src.config import create_default_configs
from src.model_loader import ModelLoader
from src.inference import ModelInference
from src.data_processor import DataProcessor

from pathlib import Path

model_dir = "output/20250627_141934/lora_unsloth_DeepSeek_R1_Distill_Qwen_14B_unsloth_bnb_4bit"

model_config, lora_config, training_config, data_config, inference_config, save_config = create_default_configs()


loader = ModelLoader(model_config, lora_config)
model, tokenizer = loader.load_pretrained_lora(model_dir)


inference = ModelInference(model, tokenizer, inference_config)


processor = DataProcessor(data_config, tokenizer)
scam_sample, legit_sample = processor.sample_comparison_data(random_state=inference_config.sample_random_state)

inference.run_sample_comparison(scam_sample, legit_sample, model_name=Path(model_dir).name)
