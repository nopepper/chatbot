from haystack.document_stores import InMemoryDocumentStore
from haystack.utils import build_pipeline, add_example_data, print_answers

provider = "openai"
with open("openai.txt", encoding='utf-8') as f:
    API_KEY = f.read().strip()

document_store = InMemoryDocumentStore(use_bm25=True)
add_example_data(document_store, "./data/")

from haystack.nodes import PromptNode, PromptTemplate, AnswerParser, BM25Retriever
from haystack.pipelines import Pipeline

retriever = BM25Retriever(document_store=document_store, top_k=3)

prompt_template = PromptTemplate(
    prompt="""
    Answer the question truthfully based solely on the given documents. If the documents do not contain the answer to the question, say that answering is not possible given the available information. Your answer should be no longer than 50 words.
    Documents:{join(documents)}
    Question:{query}
    Answer:
    """,
    output_parser=AnswerParser(),
)

prompt_node = PromptNode(
    model_name_or_path="gpt-3.5-turbo", api_key=API_KEY, default_prompt_template=prompt_template
)

generative_pipeline = Pipeline()
generative_pipeline.add_node(component=retriever, name="retriever", inputs=["Query"])
generative_pipeline.add_node(component=prompt_node, name="prompt_node", inputs=["retriever"])

from haystack.agents import Tool

search_tool = Tool(
    name="obsidian_search",
    pipeline_or_node=generative_pipeline,
    description="useful for when you need to answer questions about obsidian",
    output_variable="answers",
)

from haystack.nodes import PromptNode

agent_prompt_node = PromptNode(
    "gpt-3.5-turbo",
    api_key=API_KEY,
    max_length=256,
    stop_words=["Observation:"],
    model_kwargs={"temperature": 0.5},
)

from haystack.agents.memory import ConversationSummaryMemory
from haystack.nodes import PromptNode

memory_prompt_node = PromptNode(
    "philschmid/flan-t5-base-samsum", max_length=256, model_kwargs={"task_name": "text2text-generation"}, use_gpu=True
)
memory = ConversationSummaryMemory(memory_prompt_node, prompt_template="{chat_transcript}")

agent_prompt = """
In the following conversation, a human user interacts with an AI Agent. The human user poses questions, and the AI Agent goes through several steps to provide well-informed answers.
If the AI Agent knows the answer, the response begins with "Final Answer:" on a new line.
If the AI Agent is uncertain or concerned that the information may be outdated or inaccurate, it must use the available tools to find the most up-to-date information. The AI has access to these tools:
{tool_names_with_descriptions}

The following is the previous conversation between a human and an AI:
{memory}

AI Agent responses must start with one of the following:

Thought: [AI Agent's reasoning process]
Tool: {tool_names} (on a new line) Tool Input: [input for the selected tool WITHOUT quotation marks and on a new line] (These must always be provided together and on separate lines.)
Final Answer: [final answer to the human user's question]
When selecting a tool, the AI Agent must provide both the "Tool:" and "Tool Input:" pair in the same response, but on separate lines. "Observation:" marks the beginning of a tool's result, and the AI Agent trusts these results.

If the AI Agent cannot find a specific answer after exhausting available tools and approaches, its final answer will be a question to ask the human user.

Question: {query}
Thought:
{transcript}
"""

from haystack.agents import AgentStep, Agent
from haystack.agents.utils import conversational_agent_parameter_resolver

from haystack.agents.base import Agent, ToolsManager

conversational_agent = Agent(
    agent_prompt_node,
    prompt_template=agent_prompt,
    prompt_parameters_resolver=conversational_agent_parameter_resolver,
    memory=memory,
    tools_manager=ToolsManager([search_tool]),
)

while True:
    question = input("Question: ")
    response = conversational_agent.run(question)
    print(response)