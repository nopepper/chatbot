import logging
from .utils.human_loop import HumanInLoopNode

logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
logging.getLogger("haystack").setLevel(logging.INFO)

from haystack.document_stores import InMemoryDocumentStore

from haystack.utils import build_pipeline, add_example_data, print_answers

provider = "openai"
with open("openai.txt", encoding='utf-8') as f:
    api_key = f.read().strip()



from haystack.agents import Agent
from haystack.agents.conversational import ConversationalAgent
from haystack.nodes import PromptNode, PromptTemplate

template = PromptTemplate(prompt="""In the following conversation, a human user interacts with an AI Agent. The human user poses questions, and the AI Agent goes through several steps to provide well-informed answers.
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

If the AI Agent cannot find a specific answer after exhausting available tools and approaches, it answers with Final Answer: inconclusive

Question: {query}
Thought:
{transcript}""")

from haystack.agents import Tool
search_tool = Tool(
    name="obsidian_vault_search",
    pipeline_or_node=presidents_qa,
    description="search through the human's obsidian vault",
    output_variable="answers",
)

human_in_loop_node = HumanInLoopNode(sensory_memory=[])
human_tool = Tool(
    name="human_tool",
    pipeline_or_node=human_in_loop_node,
    description="ask the human for additional information or clarification",
    output_variable="answers",
)

from haystack.nodes import PromptNode

agent_prompt_node = PromptNode(
    "gpt-3.5-turbo",
    api_key=api_key,
    max_length=256,
    stop_words=["Observation:"],
    model_kwargs={"temperature": 0.5},
)

from haystack.agents.utils import conversational_agent_parameter_resolver
from haystack.agents.memory import ConversationSummaryMemory
from haystack.nodes import PromptNode

memory_prompt_node = PromptNode(
    "philschmid/flan-t5-base-samsum", max_length=256, model_kwargs={"task_name": "text2text-generation"}
)
memory = ConversationSummaryMemory(memory_prompt_node, prompt_template="{chat_transcript}")

agent = Agent(prompt_node=agent_prompt_node, prompt_template=template, memory=memory, prompt_parameters_resolver=conversational_agent_parameter_resolver)
agent.add_tool(search_tool)
agent.add_tool(human_tool)

from haystack.utils import print_answers

while True:
    question = input("Question: ")
    response = agent.run(question)
    print_answers(response, "minimum")