import transformers
import torch

class LLaMAProcessor:
    def __init__(self, model_id="meta-llama/Meta-Llama-3.1-8B-Instruct"):
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )
    
    def process_text(self, messages):
        outputs = self.pipeline(
            messages,
            max_new_tokens=256,
        )
        return outputs[0]["generated_text"]

    def analyze_sentiment(self, text):
        messages = [
            {"role": "system", "content": "You are a financial analyst. Analyze the sentiment of the following text and respond with either 'positive', 'negative', or 'neutral'."},
            {"role": "user", "content": text},
        ]
        sentiment = self.process_text(messages)
        return sentiment.strip().lower()

    def extract_key_topics(self, text):
        messages = [
            {"role": "system", "content": "You are a financial analyst. Extract the key topics from the following text and list them as comma-separated values."},
            {"role": "user", "content": text},
        ]
        topics = self.process_text(messages)
        return [topic.strip() for topic in topics.split(',')]