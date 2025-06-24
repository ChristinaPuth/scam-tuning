import torch
from trl import SFTTrainer, SFTConfig
from datasets import Dataset
from typing import Dict, Any, Optional
from .config import TrainingConfig

class ModelTrainer:
    """Handles model training using SFTTrainer"""
    
    def __init__(self, training_config: TrainingConfig):
        self.training_config = training_config
        self.trainer = None
        self.training_stats = None
    
    def setup_trainer(self, model, tokenizer, train_dataset: Dataset, eval_dataset: Optional[Dataset] = None) -> SFTTrainer:
        """Setup SFTTrainer with given model and dataset"""
        print("Setting up SFT trainer...")
        
        sft_config = SFTConfig(
            dataset_text_field=self.training_config.dataset_text_field,
            per_device_train_batch_size=self.training_config.per_device_train_batch_size,
            gradient_accumulation_steps=self.training_config.gradient_accumulation_steps,
            warmup_steps=self.training_config.warmup_steps,
            max_steps=self.training_config.max_steps,
            learning_rate=self.training_config.learning_rate,
            logging_steps=self.training_config.logging_steps,
            optim=self.training_config.optim,
            weight_decay=self.training_config.weight_decay,
            lr_scheduler_type=self.training_config.lr_scheduler_type,
            seed=self.training_config.seed,
            report_to=self.training_config.report_to,
        )
        
        self.trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=sft_config,
        )
        
        print("SFT trainer configured successfully")
        return self.trainer
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Get current GPU memory statistics"""
        gpu_stats = torch.cuda.get_device_properties(0)
        current_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
        
        return {
            "gpu_name": gpu_stats.name,
            "current_memory_gb": current_memory,
            "max_memory_gb": max_memory,
            "memory_usage_percent": round(current_memory / max_memory * 100, 3)
        }
    
    def log_memory_stats(self, stage: str = "current") -> None:
        """Log memory statistics"""
        stats = self.get_memory_stats()
        print(f"\n{stage.upper()} MEMORY STATS:")
        print(f"GPU: {stats['gpu_name']}")
        print(f"Current memory: {stats['current_memory_gb']} GB")
        print(f"Max memory: {stats['max_memory_gb']} GB")
        print(f"Usage: {stats['memory_usage_percent']}%")
        print("-" * 50)
    
    def train(self, resume_from_checkpoint: bool = False) -> Dict[str, Any]:
        """Execute training"""
        if self.trainer is None:
            raise ValueError("Trainer must be setup before training")
        
        print("Starting training...")
        self.log_memory_stats("pre-training")
        
        # Record start memory
        start_memory = self.get_memory_stats()["current_memory_gb"]
        
        # Train the model
        if resume_from_checkpoint:
            self.training_stats = self.trainer.train(resume_from_checkpoint=True)
        else:
            self.training_stats = self.trainer.train()
        
        # Log final memory stats
        self.log_memory_stats("post-training")
        
        # Calculate and log training statistics
        self._log_training_summary(start_memory)
        
        return self.training_stats
    
    def _log_training_summary(self, start_memory: float) -> None:
        """Log training summary statistics"""
        if self.training_stats is None:
            return
        
        final_stats = self.get_memory_stats()
        used_memory_for_training = round(final_stats["current_memory_gb"] - start_memory, 3)
        training_memory_percent = round(used_memory_for_training / final_stats["max_memory_gb"] * 100, 3)
        
        metrics = self.training_stats.metrics
        
        print(f"\nTRAINING SUMMARY:")
        print(f"Training time: {metrics['train_runtime']:.2f} seconds ({metrics['train_runtime']/60:.2f} minutes)")
        print(f"Peak memory usage: {final_stats['current_memory_gb']} GB")
        print(f"Memory used for training: {used_memory_for_training} GB")
        print(f"Training memory percentage: {training_memory_percent}%")
        
        if 'train_loss' in metrics:
            print(f"Final training loss: {metrics['train_loss']:.4f}")
        if 'train_samples_per_second' in metrics:
            print(f"Training speed: {metrics['train_samples_per_second']:.2f} samples/second")
        
        print("-" * 50)
    
    def save_trainer_state(self, output_dir: str) -> None:
        """Save trainer state for resuming"""
        if self.trainer is None:
            raise ValueError("No trainer to save")
        
        self.trainer.save_state()
        print(f"Trainer state saved")
    
    def get_training_stats(self) -> Optional[Dict[str, Any]]:
        """Get training statistics"""
        return self.training_stats 