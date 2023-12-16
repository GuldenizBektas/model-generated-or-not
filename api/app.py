from fastapi import Body, Depends
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
import joblib

app = FastAPI()

# Load your SVC model
model = joblib.load("svc_model.joblib")

class InputData(BaseModel):
    text: str

def remove_empty_lines(text):
    cleaned_text = text.replace('\n', ' ').strip()
    return cleaned_text

@app.post("/predict")
def predict(data: InputData):
    text = remove_empty_lines(data.text)
    # Perform text classification using the loaded model
    result = model.predict([text])[0]
    return {"result": int(result)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)     