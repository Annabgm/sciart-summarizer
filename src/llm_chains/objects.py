from pydantic import BaseModel, Field
from typing_extensions import List, TypedDict

from langchain_core.documents import Document


class Bibcitation(BaseModel):
    author: str = Field(
        ...,
        description="Extract the authors of the scientific paper. \
                Write them in the format 'Name, Family Name'. \
                If there are multiple authors, separate them with a ';'.",
    )
    title: str = Field(
        ...,
        description="Extract the title of the scientific paper.",
    )
    journal: str = Field(
        ...,
        description="Extract the name of the journal where the paper was published.",
    )
    year: str = Field(
        ...,
        description="Extract the year of publication of the scientific paper.",
    )
    volume: str = Field(
        ...,
        description="Extract the volume number of the journal where the paper was published.",
    )
    number: str = Field(
        ...,
        description="Extract the issue number of the journal where the paper was published.",
    )
    pages: str = Field(
        ...,
        description="Extract the page range of the scientific paper in the journal.",
    )
    doi: str = Field(
        ...,
        description="Extract the DOI (Digital Object Identifier) of the scientific paper.",
    )


class LlmCitation(BaseModel):
    source_id: int = Field(
        ...,
        description="The integer ID of a SPECIFIC source which justifies the answer.",
    )
    hash: str = Field(
        ...,
        description="The hashes of a SPECIFIC source which justifies the answer.",
    )


class QuotedAnswer(BaseModel):
    """Make a summary based only on the given sources, and cite the sources used."""

    answer: str = Field(
        ...,
        description="The summary, which is based only on the given sources.",
    )
    citations: list[LlmCitation] = Field(
        ...,
        description="Citations from the given sources that justify the answer.",
    )


class State(TypedDict):
    question: str
    context: List[Document]
    answer: QuotedAnswer