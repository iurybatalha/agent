from textwrap import dedent

from agno.agent import Agent
from agno.models.ollama import OllamaTools
from agno.storage.sqlite import SqliteStorage
from agno.playground import Playground
from agno.tools.reasoning import ReasoningTools
from agno.memory.v2.memory import Memory
from agno.memory.v2.db.sqlite import SqliteMemoryDb
from agno.tools.arxiv import ArxivTools

## agent storage
agent_storage: str = "tmp/agents.db"
memory = Memory(
    # Use any model for creating and managing memories
    model=OllamaTools(id="llama3.1:8b"),
    # Store memories in a SQLite database
    db=SqliteMemoryDb(table_name="user_memories", db_file=agent_storage),
    # We disable deletion by default, enable it if needed
    delete_memories=True,
    clear_memories=True,
)

Agent_ai = Agent(
    model=OllamaTools(
        id="llama3.2",
        #id="deepseek-r1:8b",
        #max_tokens=4096,
        ), 
    tools=[
        ReasoningTools(add_instructions=True),
        ArxivTools(),
    ],
    markdown=True,
    name="Reaserch",
    user_id="Reaserch_arxiv",
    memory=memory,
    enable_agentic_memory=True,
    debug_mode=True,
    read_tool_call_history = True,
    storage=SqliteStorage(table_name="Speed_Agent", db_file=agent_storage),
    add_history_to_messages=True,
    show_tool_calls=True,
    instructions=dedent("""\
        You are an academic research assistant with expertise in analyzing scientific papers. Your task is to process metadata from arXiv and cross-reference it with Google Scholar to enrich the information. Follow these instructions carefully:

        1. **Filter Criteria**:
        - Only consider papers published in the last **5 years**.
        - Sort the papers by **publication date**, from **oldest to newest**.

        2. **For each paper**, extract and present the following information in **Markdown format**:
        - **Title**
        - **Authors**
        - **Publication Date**
        - **arXiv Link**
        - **PDF Link**
        - **Abstract**
        - **Comments** (if available)

        3. **Cross-reference** each paper with **Google Scholar**:
        - Search for the paper using its title.
        - If found, include:
            - **Google Scholar citation count**
            - **Google Scholar link**
            - **Related works or citations** (if available)

        4. **Formatting**:
        - Use clear Markdown headers and bullet points.
        - Group papers by year if possible.
        - Ensure links are clickable.

        5. **Tone and Style**:
        - Be precise and concise.
        - Avoid unnecessary commentary.
        - Focus on factual and structured output.

        Your goal is to provide a clean, organized, and research-ready summary of the papers, enriched with citation data from Google Scholar.

    """),
    )

playground_app = Playground(agents=[Agent_ai])
app = playground_app.get_app()

if __name__ == "__main__":
    playground_app.serve("Agent_reaserch:app", reload=True)




