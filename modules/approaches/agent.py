import logging

logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
logging.getLogger("haystack").setLevel(logging.INFO)

from ..utils.human_loop import HumanInLoopNode

provider = "openai"
with open("openai.txt", encoding='utf-8') as f:
    api_key = f.read().strip()

from haystack.agents import Agent
from haystack.nodes import PromptNode, DocumentMerger, BaseRetriever
from haystack.agents import Tool
from haystack.pipelines import Pipeline
from haystack.agents.utils import react_parameter_resolver

template = """In the following conversation, a human user interacts with an AI Agent. The human user poses questions, and the AI Agent goes through several steps to provide well-informed answers.
If the AI Agent knows the answer, the response begins with "Final Answer:" on a new line.
If the AI Agent is uncertain or concerned that the information may be outdated or inaccurate, it must use the available tools to find the most up-to-date information. The AI has access to these tools:
{tool_names_with_descriptions}

AI Agent responses must start with one of the following:

1. Thought: [AI Agent's reasoning process]
2. Tool: {tool_names} (on a new line) Tool Input: [input for the selected tool WITHOUT quotation marks and on a new line] (These must always be provided together and on separate lines.)
3. Final Answer: [final answer to the human user's question]
When selecting a tool, the AI Agent must provide both the "Tool:" and "Tool Input:" pair in the same response, but on separate lines. "Observation:" marks the beginning of a tool's result, and the AI Agent trusts these results.

The agent gives very individualized answers to the human user by relying on the search tool.
If the AI Agent cannot find a specific answer after exhausting available tools and approaches, it answers with Final Answer: inconclusive

Question: {query}
Thought:
{transcript}"""

def make_agent(retriever: BaseRetriever) -> Agent:
    pipe = Pipeline()
    pipe.add_node(component=retriever, name="retriever", inputs=["Query"])
    pipe.add_node(component=DocumentMerger("\n(END OF DOCUMENT)\n"), name="merger", inputs=["retriever"])

    search_tool = Tool(
        name="search_tool",
        pipeline_or_node=pipe,
        description="universal search tool connected to the human's data and the internet. use with simple queries",
        output_variable="documents",
    )

    human_in_loop_node = HumanInLoopNode(sensory_memory=[])
    human_tool = Tool(
        name="human_tool",
        pipeline_or_node=human_in_loop_node,
        description="ask the human for additional information or clarification",
        output_variable="answers",
    )

    agent_prompt_node = PromptNode(
        "gpt-3.5-turbo",
        api_key=api_key,
        max_length=256,
        stop_words=["Observation:"],
        model_kwargs={"temperature": 0},
    )

    agent = Agent(prompt_node=agent_prompt_node, prompt_template=template, prompt_parameters_resolver=react_parameter_resolver)

    agent.add_tool(search_tool)
    #agent.add_tool(human_tool)

    return agent