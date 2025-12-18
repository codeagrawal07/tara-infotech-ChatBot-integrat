from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from backenChatBot import chatBot  # Import your compiled graph
from langchain_core.messages import HumanMessage

app = FastAPI()

# IMPORTANT: Allow your website to talk to this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, replace with your website URL
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    thread_id: str

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        config = {"configurable": {"thread_id": request.thread_id}}
        initial_state = {"messages": [HumanMessage(content=request.message)]}
        
        # Invoke the chatbot
        result = chatBot.invoke(initial_state, config=config)
        
        # Get the last AI response
        ai_response = result["messages"][-1].content
        return {"response": ai_response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)