"""StudyBuddy ka phone kit — planner server (MCP) se connection.

Factory llm.py jaisi hi hai, bas async: pehli call par phone lagti hai,
uske baad wahi connection dobara use hota hai (cache).
"""
import sys

from langchain_mcp_adapters.client import MultiServerMCPClient

_client = None
_tools = None


def text_nikalo(raw):
    """Parcel kholne wala: MCP ka jawab blocks-list mein aata hai, text jor kar wapis."""
    if isinstance(raw, str):
        return raw
    return "\n".join(b["text"] for b in raw if isinstance(b, dict) and b.get("type") == "text")


async def planner_tools():
    """Pehli call par planner server se connect ho kar tools laao, phir cache se."""
    global _client, _tools
    if _tools is None:
        _client = MultiServerMCPClient(
            {
                "planner": {
                    "command": sys.executable,
                    "args": ["-m", "studybuddy.planner_server"],
                    "transport": "stdio",
                }
            }
        )
        _tools = await _client.get_tools()
    return _tools
