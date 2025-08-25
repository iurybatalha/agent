import typer
from pathlib import Path
from typing import Optional
from rich.prompt import Prompt

from agno.agent import Agent
from agno.models.huggingface import HuggingFace
from agno.knowledge.pdf_url import PDFUrlKnowledgeBase
from agno.knowledge.csv import CSVKnowledgeBase, CSVReader
from agno.knowledge.pdf import PDFKnowledgeBase, PDFReader
from agno.vectordb.lancedb import LanceDb
from agno.vectordb.search import SearchType
from agno.embedder.huggingface import HuggingfaceCustomEmbedder
from agno.embedder.ollama import OllamaEmbedder
from agno.models.ollama import OllamaTools
import lancedb


## check if the DB ecxist
# whitch archive type
type = "csv"
path_db1 = Path("/tmp/lancedb/diff.lance")
db_name = "diff"
print(path_db1.exists)

#db_name = "table_model"
if path_db1.exists():
    print("if 1")
    # Knowledge Base
    vector_db = lancedb.connect("/tmp/lancedb/diff")
    print(vector_db)
    #vector_db = LanceDb(
    #    table_name=db_name,
    #    connect = db_connection,
    #    search_type=SearchType.vector
    #)
    if 'pdf' == type:
        knowledge_base = PDFKnowledgeBase(
            #urls=["https://agno-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"],
            path="Knowledge/Semantic_Successive_Refinement_A_Generative_AI-Aided_Semantic_Communication_Framework.pdf",
            #path="knowlegde/survey.pdf",
            vector_db=vector_db,
            reader=PDFReader(chunk=True),
            SearchType=SearchType.vector
        )

    elif 'csv' == type:
        # Knowledge Base
        knowledge_base = CSVKnowledgeBase(
            #urls=["https://agno-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"],
            path="Knowledge/Products_Modem_Management_Upgrade_Team.csv",
            #path="knowlegde/survey.pdf",
            vector_db=vector_db,
            reader=CSVReader(chunk=True),
            SearchType=SearchType.vector
        )
else:
    print("if else")
    vector_db = LanceDb(
        table_name=db_name,
        uri="/tmp/lancedb",
        embedder=OllamaEmbedder(id="nomic-embed-text", dimensions=768),
        search_type=SearchType.keyword,
    )
    if 'pdf' == type:
        knowledge_base = PDFKnowledgeBase(
            #urls=["https://agno-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"],
            path="Knowledge/Semantic_Successive_Refinement_A_Generative_AI-Aided_Semantic_Communication_Framework.pdf",
            #path="knowlegde/survey.pdf",
            vector_db=vector_db,
            reader=PDFReader(chunk=True),
        )
    elif 'csv' == type:
        # Knowledge Base
        knowledge_base = CSVKnowledgeBase(
            #urls=["https://agno-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"],
            path="Knowledge/Products_Modem_Management_Upgrade_Team.csv",
            #path="knowlegde/survey.pdf",
            vector_db=vector_db,
            reader=CSVReader(chunk=True),
        )
#knowledge_base.load(recreate=False)

def lancedb_agent(user: str = "MotoAI"):
    run_id: Optional[str] = None

    agent = Agent(
        #run_id=run_id,
        model=OllamaTools(
            id="llama3.1:8b",

            #max_tokens=4096,
            #base_url="https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct",
            #max_retries=3,
        ), 
        markdown=True,
        name="MotoAi",
        user_id=user,
        knowledge=knowledge_base,
        show_tool_calls=True,
        debug_mode=True,
    )

    if run_id is None:
        run_id = agent.run_id
        print(f"Started Run: {run_id}\n")
    else:
        print(f"Continuing Run: {run_id}\n")

    while True:
        message = Prompt.ask(f"[bold] :sunglasses: {user} [/bold]")
        if message in ("exit", "bye"):
            break
        agent.print_response(message)

if __name__ == "__main__":
    # Comment out after first run
    knowledge_base.load(recreate=True)

    typer.run(lancedb_agent())
