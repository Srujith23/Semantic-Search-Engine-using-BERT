import re
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', "", text)
    text = " ".join(text.split())
    return text