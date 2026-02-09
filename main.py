import uvicorn
import time
import threading
import os

if __name__ == "__main__":
    print("Starting the Sam’s Club Merchant AI Chatbot server...")
    print("The browser will open automatically to http://localhost:8003")

    def open_browser():
        time.sleep(5)  # Wait for server to start
        os.system('start "" "http://localhost:8003"')

    threading.Thread(target=open_browser, daemon=True).start()
    uvicorn.run("app.api:app", host="127.0.0.1", port=8003, reload=False)
