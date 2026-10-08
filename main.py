from pydantic import BaseModel, Field
from dotenv import load_dotenv
from typing import Optional
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
import truststore
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_community.tools import YouTubeSearchTool
from langgraph.graph import START, END, StateGraph
import httpx
import ssl
import truststore
load_dotenv() # 讀取專案中的 .env 檔案，並把其中的設定載入環境變數

http_client = httpx.Client(
    verify=truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
)

llm = ChatGroq(
    model="qwen/qwen3.8-27b",
    temperature=0.8,
    max_tokens = 400,
    http_client=http_client
)

class GraphState(BaseModel):
    video_url : str = Field(description="The url of the youtube video")
    video_id : Optional[str] = Field( default = None, description = "The id of the youtube video")
    transcription : Optional[str] = Field( default = None, description = "The transcription of the youtube video")
    summary : Optional[str] = Field( default = None, description = "The summary of the youtube video")
    keywords : Optional[str] = Field( default = None, description = "The keywords of the youtube video")
    questions : Optional[str] = Field( default = None, description = "Questions generated from the youtube video")
    next_steps : Optional[str] = Field( default = None, description = "Next steps based on the youtube video")
    

class ExtractVideoID(BaseModel):
    video_id : str= Field( default = None, description = "The id of the youtube video")
    
def extract_video_id(state: GraphState):
    print("[debug] extract_video_id: start", flush=True)
    video_url = state.video_url
    template = PromptTemplate(
        input_variables=["video_url"],
        template='''
        Extract the video ID from the following YouTube URL: {video_url}
        Return only the video ID.'''
    )
    llm_with_structured_output = llm.with_structured_output(ExtractVideoID)
    chain = template | llm_with_structured_output
    result = chain.invoke({"video_url": video_url})
    print(f"[debug] extract_video_id: completed; video_id={result.video_id!r}", flush=True)
    return {"video_id": result.video_id}

def extract_transcript(state: GraphState):
    print("[debug] extract_transcript: start", flush=True)
    video_id = state.video_id
    print(f"[debug] extract_transcript: video_id={video_id!r}", flush=True)
    ytt_api = YouTubeTranscriptApi()
    print(f"[debug] fetch video_id={video_id!r}", flush=True)
    fetched_transcript = ytt_api.fetch(video_id)
    print(f"[debug] extract_transcript: fetched_transcript={fetched_transcript}", flush=True)
    transcript_text = ""
    for snippet in fetched_transcript:
        transcript_text += " " + snippet.text
    print(f"[debug] extract_transcript: completed; chars={len(transcript_text)}", flush=True)
    return {"transcription": transcript_text}

def summarize_transcript(state:GraphState): # 這裡的 state 已經包含了 transcription
    print("[debug] summarize_transcript: start", flush=True)
    transcript = state.transcription
    print(f"[debug] summarize_transcript: transcript_chars={len(transcript or '')}", flush=True)
    template = PromptTemplate(
        template = '''
        Summarize the following transcript: {transcript}
        ''',
        input_variables=["transcript"] # 宣告 template 會用到這個欄位
    )
    chain = template | llm
    result = chain.invoke({"transcript": transcript})
    print("[debug] summarize_transcript: completed", flush=True)
    return {"summary": result.content}
    

def generation_questions(state: GraphState): # 這裡的 state 已經包含了 summary
    print("[debug] generation_questions: start", flush=True)
    summary = state.summary
    template = PromptTemplate(
        template = '''
        Generate 5 questions based on the following summary: {summary}
        ''',
        input_variables=["summary"] # 宣告 template 會用到這個欄位
    )
    chain = template | llm # 將 template 與 llm 組合成一個 橋梁
    result = chain.invoke({"summary": summary}) # 準備發射請求給 llm
    print("[debug] generation_questions: completed", flush=True)
    return {"questions": result.content}


def next_steps(state: GraphState): # 這裡的 state 已經包含了 summary
    print("[debug] next_steps: start", flush=True)
    summary = state.summary 
    template = PromptTemplate(
        template = '''
        Generate next steps based on the following summary: {summary}
        ''',
        input_variables=["summary"] # 宣告 template 會用到這個欄位
    )
    chain = template | llm
    result = chain.invoke({"summary": summary})
    print("[debug] next_steps: completed", flush=True)
    return {"next_steps": result.content}


def find_keywords(state:GraphState):
    print("[debug] find_keywords: start", flush=True)
    transcript = state.transcription
    print(f"[debug] find_keywords: transcript_chars={len(transcript or '')}", flush=True)
    template = PromptTemplate(
        template= '''
        Extract the keywords from the following transcript: {transcript}
        The keywords should be a single word or a short phrase that best represents the content of the transcript.
        ''',
        input_variables=["transcript"]
    )
    chain = template | llm # 把實際的 data of transcription + keywords template 組合成一個請求
    result = chain.invoke({"transcript": transcript}) # 準備發射請求給 llm
    print("[debug] find_keywords: completed", flush=True)
    return {"keywords": result.content} # 回傳從 llm 得到的 keywords 

def video_suggestion(state:GraphState):
    print("[debug] video_suggestion: start", flush=True)
    keywords = state.keywords
    print(f"[debug] video_suggestion: keywords={keywords!r}", flush=True)
    tool = YouTubeSearchTool() # 這個是 LangChain 社群提供的 YouTube 搜尋工具
    search_query = keywords.replace(",", " ").strip()
    video_suggestions = tool.run(search_query) # 透過 tool 發射請求並取得影片建議
    print(f"[debug] video_suggestion: tool result type={type(video_suggestions).__name__}", flush=True)
    return {"video_suggestions": video_suggestions} # 回傳從 tool 得到的影片建議


builder = StateGraph(GraphState) # 初始化一個 GraphBuilder 實例


builder.add_node("extract_video_id", extract_video_id)
builder.add_node("extract_transcript", extract_transcript)
builder.add_node("summarize_transcript", summarize_transcript)
builder.add_node("generation_questions", generation_questions)
builder.add_node("next_steps", next_steps)
builder.add_node("find_keywords", find_keywords)
builder.add_node("video_suggestion", video_suggestion)


builder.add_edge(START, "extract_video_id")
builder.add_edge("extract_video_id", "extract_transcript")
builder.add_edge("extract_transcript", "summarize_transcript")
builder.add_edge("summarize_transcript", "generation_questions")
builder.add_edge("summarize_transcript", "next_steps")
builder.add_edge("summarize_transcript", "find_keywords")
builder.add_edge("find_keywords", "video_suggestion")
builder.add_edge("generation_questions", END)
builder.add_edge("next_steps", END)
builder.add_edge("video_suggestion", END)

graph = builder.compile() # 把 builder 裡設定好的節點和執行路線，編譯成可以執行的 graph 物件
# langgraph.json 文件告訴 LangGraph 工具要去哪裡找到 graph 物件。