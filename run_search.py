from search_engine import SearchEngine


engine = SearchEngine()


while True:

    query = input("\nEnter query (or 'exit'): ")

    if query == "exit":
        break

    results = engine.search(query)

    print("\nTop Results:\n")

    for i, doc in enumerate(results):

        print(f"Result {i+1}:\n")

        print(doc[:500])
        print("\n---------------------")