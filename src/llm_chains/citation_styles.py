from pydantic import BaseModel, Field

from citeproc import CitationStylesStyle, CitationStylesBibliography
from citeproc import Citation, CitationItem, formatter
from citeproc.source.json import CiteProcJSON

from langchain_core.documents import Document

from .objects import QuotedAnswer, LlmCitation


class SummaryCitation(BaseModel):
    """
    Class to represent a summary with citations.
    """
    question: str
    context: dict[str, Document]
    answer: str
    citations: list[LlmCitation]

    @classmethod
    def parse_summary(cls, response: dict[str, str | list[Document] | QuotedAnswer]) -> "SummaryCitation":
        """
        Parses the response from the LLM and returns a SummaryCitation object.
        """
        question = response["question"]
        context = {i.metadata["hash"]:i for i in response["context"]}
        answer = response["answer"].answer
        citations = response["answer"].citations
        return cls(question=question, context=context, answer=answer, citations=citations)
    
    def style(self, style_name: str = 'harvard1') -> str:
        """
        Returns a string representation of the SummaryCitation object.
        """
        # Format the citations using the default style
        bibliography = self.format_citations(style_name)
        citations = "\n".join([str(item) for item in bibliography.bibliography()])
        return f"Summary: \n\n{self.answer}\n\n\n\nCitations: \n\n{citations}"

    def format_citations(self, style_name: str = 'harvard1') -> str:
        """
        Formats the citations in the specified style.
        """

        def warn(citation_item):
            print("WARNING: Reference with key '{}' not found in the bibliography."
                .format(citation_item.key))
        
        # We need to retrive paper metadata from the database
        # and format it in a way that citeproc can understand
        csl_entries = []
        for entry in self.citations:
            cit_meta = self.context[entry.hash].metadata
            csl_entries.append({
                "id": entry.hash,
                "type": "article-journal",
                "title": cit_meta.get("title"),
                "author": [
                    {"given": name.split()[0], "family": name.split()[-1]}
                    for name in cit_meta.get("author").split(";")
                ],
                "issued": {"date-parts": [[int(cit_meta.get("year"))]]},
                "container-title": cit_meta.get("journal"),
                "volume": cit_meta.get("volume"),
                "issue": cit_meta.get("number"),
                "page": cit_meta.get("pages"),
                "DOI": cit_meta.get("doi")
            })

        # Step 3: Load CSL style and generate bibliography
        bib_source = CiteProcJSON(csl_entries)
        style = CitationStylesStyle(style_name, validate=False)
        bibliography = CitationStylesBibliography(style, bib_source, formatter.plain)
        citation = Citation([CitationItem(entry["id"]) for entry in csl_entries])
        bibliography.register(citation)
        bibliography.cite(citation, warn)
        return bibliography
