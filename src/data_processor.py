import pandas as pd
import datasets
from datasets import Dataset
from typing import List, Dict, Any, Tuple
import random
from .config import DataConfig

class DataProcessor:
    """Handles data loading and conversation formatting for scam detection"""
    
    def __init__(self, data_config: DataConfig, tokenizer):
        self.data_config = data_config
        self.tokenizer = tokenizer
    
    def load_raw_data(self) -> pd.DataFrame:
        """Load raw data from CSV file"""
        print(f"Loading data from: {self.data_config.data_path}")
        
        train_df = pd.read_csv(self.data_config.data_path)
        print(f"Loaded {len(train_df)} samples")
        
        return train_df
    
    def generate_conversation(self, examples: Dict[str, List]) -> Dict[str, List]:
        """Convert raw examples into conversational format"""
        contents = examples["original_content"]
        classes = examples["class"]
        explanations = examples["explanation"]
        conversations = []
        
        for content, class_label, explanation in zip(contents, classes, explanations):
            conversation = [
                {
                    "role": "system", 
                    "content": "You are an AI assistant specialised in detecting scam or legitimate content."
                },
                {
                    "role": "user", 
                    "content": f"Here is a content that you need to classify: {content}"
                },
                {
                    "role": "assistant", 
                    "content": f"The correct answer is: class {class_label} Here is a concise explanation: {explanation}"
                },
            ]
            conversations.append(conversation)
        
        return {"conversations": conversations}
    
    def format_conversations(self, dataset: Dataset) -> List[str]:
        """Apply chat template to conversations"""
        print("Formatting conversations with chat template...")
        
        conversation_dataset = dataset.map(
            self.generate_conversation, 
            batched=True
        )
        
        formatted_conversations = self.tokenizer.apply_chat_template(
            conversation_dataset["conversations"],
            tokenize=False,
        )
        
        print(f"Formatted {len(formatted_conversations)} conversations")
        return formatted_conversations
    
    def create_training_dataset(self) -> Dataset:
        """Create complete training dataset pipeline"""
        # Load raw data
        train_df = self.load_raw_data()
        
        # Convert to Dataset
        dataset = Dataset.from_pandas(train_df, preserve_index=False)
        print(f"Created dataset with {len(dataset)} samples")
        
        # Format conversations
        formatted_conversations = self.format_conversations(dataset)
        
        # Create final dataset
        data = pd.Series(formatted_conversations, name="text")
        final_dataset = Dataset.from_pandas(pd.DataFrame(data))
        
        # Shuffle dataset
        final_dataset = final_dataset.shuffle(seed=self.data_config.shuffle_seed)
        
        print(f"Final training dataset created with {len(final_dataset)} samples")
        return final_dataset
    
    def preview_sample(self, dataset: Dataset, index: int = 0) -> str:
        """Preview a sample from the dataset"""
        if index >= len(dataset):
            raise ValueError(f"Index {index} out of range for dataset of size {len(dataset)}")
        
        sample = dataset[index]["text"]
        print(f"Sample {index}:")
        print("-" * 50)
        print(sample)
        print("-" * 50)
        
        return sample
    
    def sample_comparison_data(self, random_state: int = 42) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Sample one Scam and one Legitimate example for comparison"""
        # Load raw data
        train_df = self.load_raw_data()
        
        # Set random seed for reproducible sampling
        random.seed(random_state)
        
        # Filter by class
        scam_samples = train_df[train_df['class'] == 'Scam']
        legitimate_samples = train_df[train_df['class'] == 'Legitimate']
        
        if len(scam_samples) == 0:
            raise ValueError("No Scam samples found in dataset")
        if len(legitimate_samples) == 0:
            raise ValueError("No Legitimate samples found in dataset")
        
        # Sample one from each class
        scam_sample = scam_samples.sample(n=1, random_state=random_state).iloc[0]
        legitimate_sample = legitimate_samples.sample(n=1, random_state=random_state).iloc[0]
        
        # Convert to dictionaries with required fields
        scam_data = {
            'content': scam_sample['original_content'],
            'class': scam_sample['class'],
            'explanation': scam_sample['explanation']
        }
        
        legitimate_data = {
            'content': legitimate_sample['original_content'],
            'class': legitimate_sample['class'],
            'explanation': legitimate_sample['explanation']
        }
        
        print(f"Sampled comparison data (random_state={random_state}):")
        print(f"- Scam samples available: {len(scam_samples)}")
        print(f"- Legitimate samples available: {len(legitimate_samples)}")
        
        return scam_data, legitimate_data 