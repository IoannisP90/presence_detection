from datetime import datetime as dt
import pandas as pd
from fastapi import FastAPI

import uvicorn

app = FastAPI()
user_prob_df = pd.read_csv("./model/hourly_user_probabilities.csv")


@app.get("/")
async def root():
    return {"presence": {"present": "test", "probability": "test"}}


@app.get("hour_ahead_api/{user_id}/{date}/{hour}")
def get_post(user_id: str, date: str, hour: int):
    # ToDo: Other endpoint for determining hour ahead presence probability, given that we know current
    # temperature and electricity.
    return {"presence": {"present": "test", "probability": "test"}}


@app.get("/{user_id}/{date}/{hour}")
def get_post(user_id: str, date: str, hour: int):
    dt.fromisoformat(date)
    prob = user_prob_df[
        (user_prob_df["user"] == user_id) & (user_prob_df["hour"] == hour)
    ]["prob_presence"].iloc[0]
    present = prob > 0.5
    return {"presence": {"present": str(present), "probability": str(round(prob, 2))}}


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000)
