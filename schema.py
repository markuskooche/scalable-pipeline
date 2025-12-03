from pydantic import BaseModel, Field


class PredictPayload(BaseModel):
    age: int = Field(examples=[18])
    workclass: str = Field(examples=['Private'])
    fnlgt: int = Field(examples=[309634])
    education: str = Field(examples=['11th'])
    education_num: int = Field(examples=[7])
    marital_status: str = Field(examples=['Never-married'])
    occupation: str = Field(examples=['Other-service'])
    relationship: str = Field(examples=['Own-child'])
    race: str = Field(examples=['White'])
    sex: str = Field(examples=['Female'])
    capital_gain: int = Field(examples=[0])
    capital_loss: int = Field(examples=[0])
    hours_per_week: int = Field(examples=[22])
    native_country: str = Field(examples=['United-States'])
