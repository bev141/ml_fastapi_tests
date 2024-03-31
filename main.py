from fastapi import FastAPI
from transformers import pipeline
from pydantic import BaseModel


class Item(BaseModel):
    text: str


app = FastAPI()


@app.get("/")
def root():
    return {"message": "Hello World"}


@app.post("/predict/")
def predict(item: Item):
    """
    Predict the tone of English text
    """
    classifier = pipeline("sentiment-analysis")
    return classifier(item.text)[0]
