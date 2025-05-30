from dotenv import load_dotenv, find_dotenv
from smolagents import (
    CodeAgent,
    DuckDuckGoSearchTool,
    InferenceClientModel,
    VisitWebpageTool,
    WikipediaSearchTool,
    FinalAnswerTool,
    PythonInterpreterTool,
    Tool,
    ToolCallingAgent,
)
from huggingface_hub import model_info, InferenceClient
from PIL import Image
from io import BytesIO
import base64
from pathlib import Path
from src.insurance_assistants.complex_rag import RAG

_ = load_dotenv(dotenv_path=find_dotenv())
rag_app = RAG()
rag_app.vectordb_id = "policy_wordings"

class InsuranceInfoRetriever(Tool):
    name = "InsuranceInfoRetriever"
    description = "Retrieves information from insurance documents."
    inputs = {
        "query": {"type": "string", "description": "The query to search for."},
    }
    output_type = "string"

    def forward(self, query: str) -> str:
        client = InferenceClient(
            provider="hyperbolic",
            bill_to="VitalNest",
        )
        results = rag_app.search_documents(query)
        img_paths = [Path(res[0]) for res in results]

        grouped_images = [rag_app.encode_image_to_base64(pth) for pth in img_paths]
        chat_template = [
            {
                "role": "system",
                "content": """You find answers from the relevant documents. Answer only 
        from these documents. If answer isn't available return 'Question cannot be answered based
        on the documents provided.' """,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image}"},
                    }
                    for image in grouped_images
                ]
                + [{"type": "text", "text": query}],
            },
        ]
        completion = client.chat.completions.create(
            model="Qwen/Qwen2.5-VL-72B-Instruct",
            messages=chat_template,
            temperature=0.1,
            max_tokens=10_000,
        )
        answer = completion.choices[0].message.content
        if answer:
            answer += f"The information was retrived from the following documents: {img_paths}"
        return answer if answer else ""

insurance_agent = CodeAgent(
    tools=[InsuranceInfoRetriever(), FinalAnswerTool()],
    model=InferenceClientModel(bill_to="VitalNest", temperature=0.1),
    additional_authorized_imports=["os", "requests", "bs4", "pil", "base64", "io"],
    max_steps=1,
    # planning_interval=2,
    verbosity_level=0,
    name="insurance_agent",
    description="You answer health insurance questions based on the InsuranceInfoRetriever "
    "tool. All health insurance questions must be answered by you.",
)
websearch_agent = ToolCallingAgent(
    model=InferenceClientModel(
        model_id="Qwen/Qwen3-30B-A3B", bill_to="VitalNest", temperature=0.1
    ),
    tools=[
        VisitWebpageTool(max_output_length=20_000),
        DuckDuckGoSearchTool(max_results=5),
        FinalAnswerTool(),
    ],
    max_steps=4,
    verbosity_level=0,
    name="web_search_agent",
    planning_interval=2,
    description="Searches the web with a particular query.",
)

wikipedia_agent = ToolCallingAgent(
    model=InferenceClientModel(
        model_id="Qwen/Qwen3-30B-A3B", bill_to="VitalNest", temperature=0.1
    ),
    tools=[
        WikipediaSearchTool(user_agent="WikiAssistant (merlin@example.com)"),
        FinalAnswerTool(),
    ],
    max_steps=3,
    verbosity_level=0,
    name="wikipedia_agent",
    description="Searches Wikipedia for a topic.",
)

manager_agent = CodeAgent(
    tools=[FinalAnswerTool(), PythonInterpreterTool()],
    additional_authorized_imports=["os"],
    model=InferenceClientModel(
        model_id="Qwen/Qwen3-235B-A22B",
        bill_to="VitalNest",
        temperature=0.1,
    ),  # "Qwen/Qwen3-30B-A3B"
    managed_agents=[websearch_agent, wikipedia_agent, insurance_agent],
    max_steps=10,
    planning_interval=2,
    verbosity_level=-1,
    add_base_tools=True,
    name="Versatile_Multi_Agent",
    description="Answer health insurance related questions from pre-defined set of "
    "health insurance documents, search wikipedia and the web for general information.",
)