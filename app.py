from fastapi import FastAPI
from main import graph

app = FastAPI()
@app.get("/")
def read_root():
    print(123)
    return {"Hello": "World"}

@app.get("/summarizer")
def summarize(video_url : str):
    print("[debug] /summarizer request received", flush=True)
    print("[debug] starting graph.invoke", flush=True)
    result = graph.invoke(input={"video_url": video_url})
    print(f"[debug] graph.invoke completed; result fields={list(result.keys())}", flush=True)
    return result
