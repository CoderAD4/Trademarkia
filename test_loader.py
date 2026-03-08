from data.loader import load_dataset
docs = load_dataset()
print("Number of documents:", len(docs))
print("Sample document:\n")
print(docs[0][:500])