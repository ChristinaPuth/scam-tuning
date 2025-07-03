import torch
from unsloth import FastLanguageModel
from typing import Tuple
from .config import ModelConfig, LoRAConfig

class ModelLoader:
    """Handles model loading and LoRA adapter setup"""
    
    def __init__(self, model_config: ModelConfig, lora_config: LoRAConfig):
        self.model_config = model_config
        self.lora_config = lora_config
        self.model = None
        self.tokenizer = None
    
    def setup_gpu_buffers(self) -> None:
        """Setup GPU memory buffers for optimization"""
        dtype = self.model_config.dtype
        n_gpus = torch.cuda.device_count()
        
        GPU_BUFFERS = tuple([
            torch.empty(2*256*2048, dtype=dtype, device=f"cuda:{i}") 
            for i in range(n_gpus)
        ])
        
        print(f"Set up GPU buffers for {n_gpus} GPUs")
    
    def load_base_model(self) -> Tuple[object, object]:
        """Load the base model and tokenizer"""
        print(f"Loading model: {self.model_config.model_name}")
        
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_config.model_name,
            max_seq_length=self.model_config.max_seq_length,
            load_in_4bit=self.model_config.load_in_4bit,
            load_in_8bit=self.model_config.load_in_8bit,
            full_finetuning=self.model_config.full_finetuning,
            token=self.model_config.token,
            dtype=self.model_config.dtype,
            attn_implementation="eager",
        )
        
        print("Base model loaded successfully")
        return self.model, self.tokenizer
    
    def setup_lora(self) -> object:
        """Setup LoRA adapters on the loaded model"""
        if self.model is None:
            raise ValueError("Model must be loaded before setting up LoRA")
        
        print("Setting up LoRA adapters...")
        
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=self.lora_config.r,
            target_modules=self.lora_config.target_modules,
            lora_alpha=self.lora_config.lora_alpha,
            lora_dropout=self.lora_config.lora_dropout,
            bias=self.lora_config.bias,
            use_gradient_checkpointing=self.lora_config.use_gradient_checkpointing,
            random_state=self.lora_config.random_state,
            use_rslora=self.lora_config.use_rslora,
            loftq_config=self.lora_config.loftq_config,
        )
        
        print("LoRA adapters configured successfully")
        return self.model
    
    def load_and_setup(self) -> Tuple[object, object]:
        """Complete model loading and LoRA setup pipeline"""
        self.setup_gpu_buffers()
        self.load_base_model()
        self.setup_lora()
        
        return self.model, self.tokenizer
    
    def load_pretrained_lora(self, lora_path: str) -> Tuple[object, object]:
        """Load a previously saved LoRA model"""
        print(f"Loading LoRA model from: {lora_path}")
        
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=lora_path,
            max_seq_length=self.model_config.max_seq_length,
            load_in_4bit=self.model_config.load_in_4bit,
            attn_implementation="eager",
        )
        
        print("LoRA model loaded successfully")
        return self.model, self.tokenizer 