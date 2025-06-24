#!/usr/bin/env python3
"""
Fine-tuning Pipeline for Scam Detection
Based on the Unsloth Qwen3 14B Reasoning Conversational notebook
"""

import argparse
import os
import sys
from typing import Optional

from .config import (
    ModelConfig, LoRAConfig, TrainingConfig, 
    DataConfig, InferenceConfig, SaveConfig
)
from .model_loader import ModelLoader
from .data_processor import DataProcessor
from .trainer import ModelTrainer
from .inference import ModelInference
from .model_saver import ModelSaver

class FineTuningPipeline:
    """Main fine-tuning pipeline orchestrator"""
    
    def __init__(self, 
                 model_config: ModelConfig,
                 lora_config: LoRAConfig,
                 training_config: TrainingConfig,
                 data_config: DataConfig,
                 inference_config: InferenceConfig,
                 save_config: SaveConfig):
        
        self.model_config = model_config
        self.lora_config = lora_config
        self.training_config = training_config
        self.data_config = data_config
        self.inference_config = inference_config
        self.save_config = save_config
        
        # Initialize components
        self.model_loader = ModelLoader(model_config, lora_config)
        self.trainer = ModelTrainer(training_config)
        
        # These will be initialized after model loading
        self.data_processor = None
        self.inference = None
        self.model_saver = None
        
        # Model and tokenizer placeholders
        self.model = None
        self.tokenizer = None
    
    
    def load_model(self) -> None:
        """Load and setup the model"""
        print("=" * 60)
        print("LOADING MODEL")
        print("=" * 60)
        
        self.model, self.tokenizer = self.model_loader.load_and_setup()
        
        # Initialize components that need the tokenizer
        self.data_processor = DataProcessor(self.data_config, self.tokenizer)
        self.inference = ModelInference(self.model, self.tokenizer, self.inference_config)
        self.model_saver = ModelSaver(
            self.model, 
            self.tokenizer, 
            self.save_config, 
            model_name=self.model_config.model_name
        )
    
    def prepare_data(self) -> None:
        """Prepare training data"""
        print("=" * 60)
        print("PREPARING DATA")
        print("=" * 60)
        
        self.train_dataset = self.data_processor.create_training_dataset()
        
        # Preview a sample
        print("\nPreviewing training sample:")
        self.data_processor.preview_sample(self.train_dataset, 0)
    
    def train_model(self, resume_from_checkpoint: bool = False) -> None:
        """Train the model"""
        print("=" * 60)
        print("TRAINING MODEL")
        print("=" * 60)
        
        # Setup trainer
        self.trainer.setup_trainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=self.train_dataset
        )
        
        # Train
        training_stats = self.trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        
        return training_stats
    

    
    def save_model(self) -> dict:
        """Save the trained model"""
        print("=" * 60)
        print("SAVING MODEL")
        print("=" * 60)
        
        # Set training metadata before saving
        training_config_dict = {
            "model_config": self.model_config.__dict__,
            "lora_config": self.lora_config.__dict__,
            "training_config": self.training_config.__dict__,
            "data_config": self.data_config.__dict__,
            "inference_config": self.inference_config.__dict__,
        }
        
        training_stats = self.trainer.get_training_stats()
        self.model_saver.set_training_metadata(training_config_dict, training_stats)
        
        return self.model_saver.save_all_formats()
    
    def run_full_pipeline(self, resume_from_checkpoint: bool = False) -> dict:
        """Run the complete fine-tuning pipeline"""
        results = {}

        
        try:
            
            # Load model
            self.load_model()
            
            # Prepare data
            self.prepare_data()
            
            # Train
            training_stats = self.train_model(resume_from_checkpoint=resume_from_checkpoint)
            results['training_stats'] = training_stats
            
            # Save model
            save_results = self.save_model()
            results['save_results'] = save_results
            
            return results
            
        except Exception as e:
            print(f"Pipeline failed with error: {e}")
            raise e
    
    def run_interactive_mode(self) -> None:
        """Run interactive classification mode"""
        if self.model is None or self.tokenizer is None:
            print("Loading model for interactive mode...")
            self.load_model()
        
        self.inference.interactive_classification()

def create_default_configs():
    """Create default configuration objects"""
    model_config = ModelConfig()
    lora_config = LoRAConfig()
    training_config = TrainingConfig()
    data_config = DataConfig()
    inference_config = InferenceConfig()
    save_config = SaveConfig()
    
    return (model_config, lora_config, training_config, 
            data_config, inference_config, save_config)

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Modular Fine-tuning Pipeline")
    parser.add_argument("--mode", choices=["train", "inference", "interactive"], 
                       default="train", help="Pipeline mode")
    parser.add_argument("--resume", action="store_true", 
                       help="Resume training from checkpoint")
    parser.add_argument("--data-path", type=str, 
                       help="Path to training data CSV file")
    parser.add_argument("--output-dir", type=str,
                       help="Output directory for saved models")
    parser.add_argument("--max-steps", type=int,
                       help="Maximum training steps")
    parser.add_argument("--learning-rate", type=float,
                       help="Learning rate")
    
    args = parser.parse_args()
    
    # Create default configs
    configs = create_default_configs()
    model_config, lora_config, training_config, data_config, inference_config, save_config = configs
    
    # Override configs with command line arguments
    if args.data_path:
        data_config.data_path = args.data_path
    if args.output_dir:
        save_config.base_output_dir = args.output_dir
    if args.max_steps:
        training_config.max_steps = args.max_steps
    if args.learning_rate:
        training_config.learning_rate = args.learning_rate
    
    # Create pipeline
    pipeline = FineTuningPipeline(
        model_config=model_config,
        lora_config=lora_config,
        training_config=training_config,
        data_config=data_config,
        inference_config=inference_config,
        save_config=save_config
    )
    
    # Run based on mode
    if args.mode == "train":
        results = pipeline.run_full_pipeline(resume_from_checkpoint=args.resume)
        print("Training completed. Results:", results)
    
    elif args.mode == "inference":
        pipeline.load_model()
        pipeline.run_interactive_mode()
    
    elif args.mode == "interactive":
        pipeline.run_interactive_mode()

if __name__ == "__main__":
    main() 