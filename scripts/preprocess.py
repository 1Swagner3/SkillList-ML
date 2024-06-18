import re

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    return text
