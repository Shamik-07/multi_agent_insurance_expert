from pyprojroot import find_root, has_file

PROJECT_ROOT_DIR = find_root(criterion=has_file("README.md"))

PROMPT = """
You are a smart assistant designed to answer questions about a PDF document.
You are given relevant information in the form of PDF pages. Use them to construct a short response to the question, and cite your sources (page numbers, etc).
If it is not possible to answer using the provided pages, do not attempt to provide an answer and simply say the answer is not present within the documents.
Give detailed and extensive answers, only containing info in the pages you are given.
You can answer using information contained in plots and figures if necessary.
Answer in the same language as the query.

Query: {query}
PDF pages:
"""

PRIMARY_HEADING = """# An Agentic RAG for Health Insurance Documents
#### This agent answers health insurance related questions from pre-ingested set \
of health insurance documents, search wikipedia, search the web \
and execute basic python code. \

The pre-ingested health insurance documents can be viewed under `PDF Viewer`.
"""

PROMPT_PREFIX = """\n
Apart from all the above instructions that we have given to you, FOLLOW the Additional Instructions below:
```
For any health insurance related queries, always use the `insurance_agent` first and return the results.
You are allowed to rephrase any query and detail it if required. When in doubt always ask the user a follow up question.
Don't assume anything.
```

"""