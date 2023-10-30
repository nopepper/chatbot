from haystack.document_stores import InMemoryDocumentStore
from haystack.utils import build_pipeline, add_example_data, print_answers

provider = "openai"
with open("openai.txt", encoding='utf-8') as f:
    API_KEY = f.read().strip()

document_store = InMemoryDocumentStore(use_bm25=True)
add_example_data(document_store, "./data/")

# Build a pipeline with a Retriever to get relevant documents to the query and a PromptNode interacting with LLMs using a custom prompt.
pipeline = build_pipeline(provider, API_KEY, document_store)

while True:
    question = input("Question: ")
    response = pipeline.run(question)
    print_answers(response, details="medium")