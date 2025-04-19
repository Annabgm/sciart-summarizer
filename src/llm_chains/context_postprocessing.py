from collections import defaultdict
from typing_extensions import List, TypedDict

from langchain_core.documents import Document


def format_docs_with_id(docs: list[Document]) -> str:
    template = "Source ID: {num}\nHash: {hash}\nAuthors: {author}\nArticle Snippet: {content}"
    formatted = []
    ids = 1
    id_dict = defaultdict(int)
    for doc in docs:
        if doc.metadata["hash"] not in id_dict:
            id_dict[doc.metadata["hash"]] = ids
            formatted.append(
                template.format(
                    num=ids, 
                    hash=doc.metadata["hash"], 
                    author=doc.metadata["author"], 
                    content=doc.page_content
                )
            )
            ids += 1
        else:
            formatted.append(
                template.format(
                    num=id_dict[doc.metadata["hash"]], 
                    hash=doc.metadata["hash"], 
                    author=doc.metadata["author"], 
                    content=doc.page_content
                )
            )
    return "\n\n" + "\n\n".join(formatted)
