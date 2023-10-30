from haystack.utils import print_answers
from modules.approaches.agent import make_agent
from modules.retrievers.embedding import get_retriever

if __name__ == "__main__":
    retriever = get_retriever(force_rebuild=False)
    agent = make_agent(retriever)

    while True:
        question = input("Question: ")
        response = agent.run(question)
        print_answers(response, "minimum")