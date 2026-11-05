"""MCP connection proof: StudyBuddy ka pehla phone-call planner server se."""
import asyncio
import sys

from langchain_mcp_adapters.client import MultiServerMCPClient


def text_nikalo(raw):
    """adapters 0.3.x: jawab list of blocks aata hai [{type: 'text', text: ...}] — text nikalna zaroori."""
    if isinstance(raw, str):
        return raw
    return "\n".join(b["text"] for b in raw if isinstance(b, dict) and b.get("type") == "text")


async def main():
    client = MultiServerMCPClient(
        {
            "planner": {
                "command": sys.executable,
                "args": ["-m", "studybuddy.planner_server"],
                "transport": "stdio",
            }
        }
    )
    print("=== Test 1: connection + tool listing ===")
    tools = await client.get_tools()
    print("Server se aaye tools:", [t.name for t in tools])

    print("=== Test 2: plan banao ===")
    plan = text_nikalo(await tools[0].ainvoke({"topic": "RAG", "days": 3}))
    print(plan)

    print("=== Test 3: bada input — cap check ===")
    plan2 = text_nikalo(await tools[0].ainvoke({"topic": "MCP", "days": 99}))
    print("Day 14 tak hi gaya?", "Day 14" in plan2 and "Day 15" not in plan2)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    asyncio.run(main())
