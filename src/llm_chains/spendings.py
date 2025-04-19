from pydantic import BaseModel, Field
from datetime import datetime
import pandas as pd


class SpendingsMeta(BaseModel):
    total_tokens: int
    prompt_tokens: int
    completion_tokens: int
    total_cost: float

    @classmethod
    def from_api_response(cls, sm: dict):
        return cls(
            total_tokens=sm.total_tokens if hasattr(sm, "total_tokens") else 0.,
            prompt_tokens=sm.prompt_tokens if hasattr(sm, "total_cost") else 0.,
            completion_tokens=sm.completion_tokens if hasattr(sm, "total_cost") else 0.,
            total_cost=sm.total_cost if hasattr(sm, "total_cost") else 0.,
        )


class Spendings(BaseModel):
    cost: SpendingsMeta
    timestamp: datetime = Field(default_factory=datetime.now)


class SpendingClient(BaseModel):
    client_name: str
    spendings: list[Spendings] = Field(default_factory=list)

    def add_spending(self, spending: Spendings):
        self.spendings.append(spending)


def spend_helper(spending_client: SpendingClient):
    """
    Helper function to format the spending information.
    """
    result = []
    for data in spending_client.spendings:
        date = data.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        result_loc = data.cost.__dict__
        result_loc['timestamp'] = date
        result.append(result_loc)
    df = pd.DataFrame(result)
    df.columns = df.columns.str.replace("_", " ").str.capitalize()
    return df