import torch
from transformers import TextStreamer
from typing import List, Dict, Any, Optional
from .config import InferenceConfig

class ModelInference:
    """Handles model inference and text generation"""
    
    def __init__(self, model, tokenizer, inference_config: InferenceConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.inference_config = inference_config
    
    def create_scam_detection_message(self, content: str) -> List[Dict[str, str]]:
        """Create a properly formatted message for scam detection"""
        return [
            {
                "role": "user", 
                "content": f"please classify the following content as scam or legitimate: {content}"
            }
        ]
    
    def format_messages(self, messages: List[Dict[str, str]], add_generation_prompt: bool = True) -> str:
        """Format messages using the chat template"""
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=self.inference_config.enable_thinking,
        )
        return text
    
    def generate_response(self, 
                         messages: List[Dict[str, str]], 
                         max_new_tokens: Optional[int] = None,
                         temperature: Optional[float] = None,
                         top_p: Optional[float] = None,
                         top_k: Optional[int] = None,
                         use_streamer: bool = True,
                         return_full_text: bool = False) -> str:
        """Generate response from formatted messages"""
        
        # Use provided values or fall back to config defaults
        max_new_tokens = max_new_tokens or self.inference_config.max_new_tokens
        temperature = temperature or self.inference_config.temperature
        top_p = top_p or self.inference_config.top_p
        top_k = top_k or self.inference_config.top_k
        
        # Format the input
        text = self.format_messages(messages)
        
        # Tokenize input
        inputs = self.tokenizer(text, return_tensors="pt").to("cuda")
        
        # Setup streamer if requested
        streamer = TextStreamer(self.tokenizer, skip_prompt=True) if use_streamer else None
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                streamer=streamer,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode the response
        if return_full_text:
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        else:
            # Only return the generated part (skip the input)
            generated_tokens = outputs[0][len(inputs.input_ids[0]):]
            response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return response
    
    def classify_sample_data(self, sample_data: Dict[str, Any], use_streamer: bool = True) -> str:
        """Classify a sample data dictionary and display information"""
        content = sample_data['content']
        true_class = sample_data['class']
        explanation = sample_data['explanation']
        
        print(f"\n{'='*60}")
        print(f"SAMPLE CLASSIFICATION - TRUE CLASS: {true_class}")
        print(f"{'='*60}")
        print(f"Content: {content[:200]}{'...' if len(content) > 200 else ''}")
        print(f"\nTrue Class: {true_class}")
        print(f"Explanation: {explanation[:300]}{'...' if len(explanation) > 300 else ''}")
        print(f"\n{'-'*50}")
        print("MODEL PREDICTION:")
        print(f"{'-'*50}")
        
        messages = self.create_scam_detection_message(content)
        response = self.generate_response(
            messages=messages,
            use_streamer=use_streamer,
            return_full_text=False
        )
        
        return response
    
    def run_sample_comparison(self, scam_data: Dict[str, Any], legitimate_data: Dict[str, Any], model_name: str = "Model") -> None:
        """Run inference on sample data and display results"""
        print(f"\n{'='*70}")
        print(f"SAMPLE COMPARISON - {model_name}")
        print(f"{'='*70}")
        
        # Test on Scam sample
        print(f"\n🚨 TESTING ON SCAM SAMPLE")
        scam_response = self.classify_sample_data(scam_data, use_streamer=True)
        
        # Test on Legitimate sample
        print(f"\n✅ TESTING ON LEGITIMATE SAMPLE")
        legitimate_response = self.classify_sample_data(legitimate_data, use_streamer=True)
        
        print(f"\n{'='*70}")
        print(f"COMPARISON SUMMARY - {model_name}")
        print(f"{'='*70}")
        print(f"Scam Sample - True: Scam | Predicted: {scam_response[:100]}{'...' if len(scam_response) > 100 else ''}")
        print(f"Legitimate Sample - True: Legitimate | Predicted: {legitimate_response[:100]}{'...' if len(legitimate_response) > 100 else ''}")
        print(f"{'='*70}")
    
    def classify_content(self, content: str, use_streamer: bool = True) -> str:
        """Classify content as scam or legitimate"""
        messages = self.create_scam_detection_message(content)
        
        print(f"Classifying content: {content[:100]}...")
        print("-" * 50)
        
        response = self.generate_response(
            messages=messages,
            use_streamer=use_streamer,
            return_full_text=False
        )
        
        return response
    
    def interactive_classification(self) -> None:
        """Interactive mode for content classification"""
        print("Interactive Scam Detection Mode")
        print("Enter 'quit' to exit")
        print("-" * 50)
        
        while True:
            try:
                content = input("\nEnter content to classify: ")
                
                if content.lower() in ['quit', 'exit', 'q']:
                    print("Exiting interactive mode...")
                    break
                
                if not content.strip():
                    print("Please enter some content to classify.")
                    continue
                
                self.classify_content(content, use_streamer=True)
                
            except KeyboardInterrupt:
                print("\nExiting interactive mode...")
                break
            except Exception as e:
                print(f"Error during classification: {e}")
    
    def batch_classify(self, contents: List[str], use_streamer: bool = False) -> List[str]:
        """Classify multiple contents in batch"""
        results = []
        
        print(f"Classifying {len(contents)} contents...")
        
        for i, content in enumerate(contents):
            print(f"\nProcessing {i+1}/{len(contents)}")
            try:
                response = self.classify_content(content, use_streamer=use_streamer)
                results.append(response)
            except Exception as e:
                print(f"Error classifying content {i+1}: {e}")
                results.append(f"Error: {e}")
        
        return results
    
    def test_model(self) -> None:
        """Test model functionality with interactive classification"""
        print("Model testing redirected to interactive classification mode...")
        print("Use the sample comparison feature for systematic testing.")
        print("=" * 60)
        self.interactive_classification() 