from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from app.agent import ask_agent

app = FastAPI(title="Sam’s Club Merchant AI Chatbot")

class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    query: str
    insight: str

@app.get("/", response_class=HTMLResponse)
def read_root():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Sam’s Club Merchant AI Chatbot</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            #chat { border: 1px solid #ccc; padding: 10px; height: 300px; overflow-y: scroll; }
            #message { width: 80%; padding: 5px; }
            button { padding: 5px 10px; }
        </style>
    </head>
    <body>
        <h1>Sam’s Club Merchant AI Chatbot</h1>
        <div id="chat"></div>
        <input type="text" id="message" placeholder="Ask a question...">
        <button onclick="sendMessage()">Send</button>
        <script>
            async function sendMessage() {
                const message = document.getElementById('message').value;
                if (!message) return;
                document.getElementById('chat').innerHTML += '<p><strong>You:</strong> ' + message + '</p>';
                document.getElementById('message').value = '';
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ query: message })
                });
                const data = await response.json();
                document.getElementById('chat').innerHTML += '<p><strong>AI:</strong> ' + data.insight + '</p>';
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    print(f"API received query: {request.query}")
    response = ask_agent(request.query)
    print(f"API sending response: {response}")
    return {
        "query": request.query,
        "insight": response
    }
