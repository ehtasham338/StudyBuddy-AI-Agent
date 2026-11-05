
"""
MCP Planner Server — StudyBuddy's external consultant (separate process).

This server listens through stdio (communication channel).
StudyBuddy connects to this server to request tools.

WARNING:
Never use print() in this file because stdout is reserved
for MCP protocol communication. Using print() may corrupt
the communication between the MCP client and server.
"""

from mcp.server.fastmcp import FastMCP


# Create the MCP server instance.
mcp = FastMCP("planner")


@mcp.tool()
def make_study_plan(topic: str, days: int) -> str:
    """
    Create a simple day-by-day study plan.

    Args:
        topic: The subject or topic to study.
        days: The number of days for the study plan.

    Returns:
        A readable day-by-day study plan.
    """

    # Restrict the study plan duration to 1–14 days.
    days = max(1, min(int(days), 14))

    # Define the study activities.
    activities = [
        "Read and understand the topic",
        "Create study notes",
        "Solve practice questions",
        "Review and revise the topic"
    ]

    # Initialize the study plan with its heading.
    lines = [f"Study Plan: {topic} ({days} days)"]

    # Generate a study activity for each day.
    for day in range(1, days + 1):

        activity = activities[(day - 1) % len(activities)]

        lines.append(
            f"Day {day}: {activity} — 45 minutes"
        )

    # Add a self-assessment task at the end.
    lines.append(
        f"On Day {days}, take a short self-test on {topic}."
    )

    # Return the complete study plan as a formatted string.
    return "\n".join(lines)


# Start the MCP server when this file is executed directly.
if __name__ == "__main__":
    mcp.run(transport="stdio")
