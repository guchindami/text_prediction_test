from flask import Flask, request, jsonify, send_from_directory
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F
from functools import lru_cache
import re

app = Flask(__name__)

class TextSuggester:
    def __init__(self, model_name='gpt2-medium'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.eval()

    @lru_cache(maxsize=1000)
    def suggest(self, prompt, num_suggestions=5, context_length=20):
        # Clean and tokenize the prompt
        clean_prompt = re.sub(r'\s+', ' ', prompt).strip()
        tokens = self.tokenizer.encode(clean_prompt, return_tensors='pt')
        
        # Get the last 'context_length' tokens
        if tokens.shape[1] > context_length:
            input_ids = tokens[:, -context_length:]
        else:
            input_ids = tokens

        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)

            # Get top k token indices
            top_k_probs, top_k_indices = torch.topk(probs, 100)  # Get more candidates

            suggestions = []
            for prob, idx in zip(top_k_probs[0], top_k_indices[0]):
                token = self.tokenizer.decode([idx])
                if self.is_valid_suggestion(token, clean_prompt):
                    suggestions.append((token.strip(), prob.item()))
                if len(suggestions) == num_suggestions:
                    break

        # Sort suggestions by probability
        suggestions.sort(key=lambda x: x[1], reverse=True)

        return [word for word, _ in suggestions]

    def is_valid_suggestion(self, token, prompt):
        # Remove leading space if present
        token = token.lstrip()
        
        # Check if the token is a single word (no spaces)
        if ' ' in token:
            return False
        
        # Check if it's an actual word (contains at least one letter)
        if not re.search(r'[a-zA-Z]', token):
            return False
        
        # Check if it's not just a punctuation mark
        if token in '.,!?;:':
            return False
        
        # Check if it's not already the last word in the prompt
        last_word = prompt.split()[-1] if prompt else ''
        if token.lower() == last_word.lower():
            return False
        
        return True

suggester = TextSuggester()

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/suggest', methods=['POST'])
def suggest():
    data = request.json
    prompt = data.get('prompt', '')
    suggestions = suggester.suggest(prompt)
    return jsonify({'suggestions': suggestions})

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')
    