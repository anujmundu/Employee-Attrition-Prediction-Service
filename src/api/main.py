from fastapi import FastAPI
from pydantic import BaseModel, Field
from src.monitoring import init_db, log_prediction
from src.predict import predict

app = FastAPI()

@app.on_event("startup")
def startup():
    init_db()

class EmployeeInput(BaseModel):
    Age: int = Field(ge=18, le=65)
    DistanceFromHome: int = Field(ge=0, le=100)
    Education: int = Field(ge=1, le=5)
    EnvironmentSatisfaction: int = Field(ge=1, le=4)
    JobSatisfaction: int = Field(ge=1, le=4)
    MonthlyIncome: int = Field(ge=0, le=500000)
    NumCompaniesWorked: int = Field(ge=0, le=20)
    WorkLifeBalance: int = Field(ge=1, le=4)
    YearsAtCompany: int = Field(ge=0, le=50)
    Department: str
    EducationField: str
    MaritalStatus: str


@app.post("/v1/predict")
def predict_attrition(data: EmployeeInput):
    result = predict(data.dict())
    log_prediction(data.dict(), result)
    return result
