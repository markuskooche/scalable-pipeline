from pydantic import BaseModel, Field

class InputData(BaseModel):
    foo: str = Field()

