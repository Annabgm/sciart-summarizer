system_prompt_meta = """\
Extract the following information from the scientific paper:

author: Extract the authors of the scientific paper. Write them in the format 'Name, Family Name'. If there are multiple authors, separate them with a ';'.

title: Extract the title of the scientific paper.

journal: Extract the name of the journal where the paper was published.

year: Extract the year of publication of the scientific paper.

volume: Extract the volume number of the journal where the paper was published.

number: Extract the issue number of the journal where the paper was published.

pages: Extract the page range of the scientific paper in the journal.

doi: Extract the DOI (Digital Object Identifier) of the scientific paper.

Respond in JSON format.

{text}
"""


system_prompt_rag = (
    """You're a helpful AI assistant. You help scientist to write 
    some scientific paper by summarizing the information provided 
    in the context. 
    The summary should:
    - Use the information in the context to answer the question. If no information is available, say you don't know.
    - Avoid personal opinions or assumptions not supported by the text.
    - Be formal, objective, and precise in tone.
    The summary must:
    - Include inline citations using the format: [value] - for one, [value1, value2, ...] - for many.

    Always cite specific findings and clearly attribute ideas to the original source.
    If multiple studies are mentioned, keep the structure logical and cohesive.


    {context}"""
)