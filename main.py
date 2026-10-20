from agents.data_agent import data_agent
from langchain_core.messages import HumanMessage


def get_final_answer(response: dict) -> str:
    """Extracts the final human-readable answer from the data agent response."""
    messages = response.get("messages", [])
    if not messages:
        return "No response received."

    last_item = messages[-1]

    # If the sub-agent response dict was appended
    if isinstance(last_item, dict):
        if last_item.get("final_answer"):
            return last_item["final_answer"]
        sub_messages = last_item.get("messages", [])
        if sub_messages:
            last_sub = sub_messages[-1]
            return getattr(last_sub, "content", str(last_sub))

    # If a message object was appended directly
    if hasattr(last_item, "content"):
        return last_item.content

    return str(last_item)


if __name__ == "__main__":
    query = (
        "What are the different types of Payment Methods we have in our database?"
    )

    print(f"\nUser Query: {query}")
    print("Processing with Data Agent (routing and executing)...\n")

    response = data_agent.invoke(
        {
            "messages": [HumanMessage(content=query)],
            "route_response": ""
        }
    )

    route = response.get("route_response", "Unknown")
    answer = get_final_answer(response)

    print(f"--- [Routed to: {route.upper()}] ---")
    print(f"Final Answer:\n{answer}\n")