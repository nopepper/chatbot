import logging

logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
logging.getLogger("haystack").setLevel(logging.INFO)

from ..utils.human_loop import HumanInLoopNode

provider = "openai"
with open("openai.txt", encoding='utf-8') as f:
    api_key = f.read().strip()

from haystack.agents import Agent
from haystack.nodes import PromptNode, PromptTemplate, DocumentMerger, BaseRetriever, AnswerParser
from haystack.agents import Tool
from haystack.nodes import PromptNode
from haystack.agents.utils import conversational_agent_parameter_resolver
from haystack.pipelines import ExtractiveQAPipeline, Pipeline

question_answering_with_references = PromptTemplate("deepset/question-answering-with-references", output_parser=AnswerParser(reference_pattern=r"Document\[(\d+)\]"))

def make_agent(retrieval_pipe: BaseRetriever) -> Pipeline:
    prompt_node = PromptNode(
        "gpt-3.5-turbo",
        api_key=api_key,
        max_length=256,
        default_prompt_template=question_answering_with_references,
        model_kwargs={"temperature": 0.2},
    )

    pipe = Pipeline()

    pipe.add_node(component=retrieval_pipe, name="Retriever", inputs=["Query"])
    pipe.add_node(component=prompt_node, name="prompt_node", inputs=["Retriever"])

    return pipe