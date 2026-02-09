from dotenv import load_dotenv
import os
import re

from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.agents.factory import create_agent
from langchain_core.tools import tool
from typing import Optional
from app.tools import sales_summary, recent_drop, markdown_impact, holiday_impact

# Explicitly load api.env
load_dotenv(dotenv_path="api.env")

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)

@tool
def sales_summary_tool():
    """Get total and average weekly sales for all stores"""
    return sales_summary()

@tool
def recent_drop_tool():
    """Detect if sales dropped in the last 2 weeks overall"""
    return recent_drop()

@tool
def markdown_impact_tool():
    """Analyze impact of markdown promotions on sales overall"""
    return markdown_impact()

@tool
def holiday_impact_tool():
    """Compare holiday vs non-holiday sales overall"""
    return holiday_impact()

tools = [sales_summary_tool, recent_drop_tool, markdown_impact_tool, holiday_impact_tool]

SYSTEM_PROMPT = """
You are an AI assistant helping Sam's Club merchants analyze sales data.
Provide clear, concise answers based on the data.
Use the tools to gather information and answer questions directly.
Keep responses brief and to the point.
"""

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=SYSTEM_PROMPT,
)

def _extract_text_from_result(result):
    """
    Robustly extract a human-readable string from various agent return shapes.
    Handles:
      - plain string
      - dicts with keys: 'output', 'text', 'message', 'result'
      - dict with 'messages' list (last message may contain 'content' or 'text')
      - other dicts => str(result)
    """
    if isinstance(result, str):
        return result

    if result is None:
        return "No response from agent."

    try:
        if isinstance(result, dict):
            for k in ("output", "text", "message", "result"):
                if k in result and isinstance(result[k], (str, int, float)):
                    return str(result[k])

            if "messages" in result and isinstance(result["messages"], (list, tuple)) and len(result["messages"]) > 0:
                last = result["messages"][-1]
                if isinstance(last, dict):
                    for kk in ("content", "text", "message"):
                        if kk in last and isinstance(last[kk], (str, int, float)):
                            return str(last[kk])
                    if isinstance(last, str):
                        return last
                else:
                    return str(last)

            for k, v in result.items():
                if isinstance(v, (str, int, float)):
                    return str(v)

            return str(result)
    except Exception:
        return str(result)

def _is_direct_tool_query(query: str) -> Optional[str]:
    """
    Return the tool name to call directly for common queries to avoid hitting model token limits.
    Recognizes:
      - sales summary
      - recent drop or sales drop
      - markdown impact
      - holiday impact
    """
    q = query.lower()
    if re.search(r"\b(sales summary|give me sales summary|summary of sales|total sales)\b", q):
        return "sales_summary"
    if re.search(r"\b(recent drop|sales dropped|drop in sales|sales drop)\b", q):
        return "recent_drop"
    if re.search(r"\b(markdown|markdowns|promotion impact)\b", q):
        return "markdown_impact"
    if re.search(r"\b(holiday impact|holidays|holiday sales)\b", q):
        return "holiday_impact"
    return None

def ask_agent(query: str):
    print(f"Received query: {query}")

    # Short-circuit to local tools for well-known, data-backed requests
    direct_tool = _is_direct_tool_query(query)
    if direct_tool:
        try:
            if direct_tool == "sales_summary":
                out = sales_summary()
            elif direct_tool == "recent_drop":
                out = recent_drop()
            elif direct_tool == "markdown_impact":
                out = markdown_impact()
            elif direct_tool == "holiday_impact":
                out = holiday_impact()
            else:
                out = None

            if out is not None:
                print(f"Handled locally with tool '{direct_tool}': {out}")
                return out
        except Exception as e:
            print(f"Local tool '{direct_tool}' error: {e}")
            # Fall through to try the agent if local tool fails

    # Otherwise invoke the agent (LLM). If the LLM errors due to token limits, fallback to a local concise response.
    try:
        result = agent.invoke({"input": query})
        print(f"Agent raw result: {result}")
        if isinstance(result, dict):
            try:
                print(f"Result keys: {list(result.keys())}")
            except Exception:
                pass
        extracted = _extract_text_from_result(result)
        print(f"Extracted response: {extracted}")
        return extracted
    except Exception as e:
        err_str = str(e)
        print(f"Error in agent: {err_str}")

        # Detect Groq token / rate-limit errors and provide a graceful fallback
        if "rate_limit_exceeded" in err_str or "Request too large" in err_str or "Request too large for model" in err_str or "413" in err_str or "tokens per minute" in err_str or "Requested" in err_str:
            # If the user asked for a sales-summary-like item, give a concise local summary
            if direct_tool == "sales_summary" or re.search(r"\b(sales summary|summary of sales|total sales)\b", query.lower()):
                try:
                    fallback = sales_summary()
                    return f"[Fallback: model unavailable due to token limits] {fallback}"
                except Exception as e2:
                    print(f"Fallback local sales_summary failed: {e2}")
                    return "Model request too large and local fallback failed."
            # For other cases, attempt a conservative local fallback: short sales summary
            try:
                fallback = sales_summary()
                return f"[Fallback: model unavailable due to token limits] Returning concise sales summary: {fallback}"
            except Exception as e2:
                print(f"Fallback local summary failed: {e2}")
                return "Model request too large and local fallback failed."

        # Generic error fallback
        return f"Error: {err_str}"
