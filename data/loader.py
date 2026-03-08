import re
from sklearn.datasets import fetch_20newsgroups
def clean_text(text):
    # remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    # remove URLs
    text = re.sub(r'http\S+', '', text)
    # remove quoted replies (lines starting with >)
    text = re.sub(r'^>.*$', '', text, flags=re.MULTILINE)
    # normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    # limit document length (prevents extremely long posts slowing embeddings)
    text = text[:2000]
    return text.strip()
def load_dataset():
    dataset = fetch_20newsgroups(
        subset="all",
        remove=("headers", "footers", "quotes")
    )
    docs = dataset.data
    cleaned_docs = []
    for doc in docs:
        cleaned = clean_text(doc)
        # ignore very short documents
        if len(cleaned) > 50:
            cleaned_docs.append(cleaned)
    return cleaned_docs