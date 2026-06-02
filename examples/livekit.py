"""LiveKit example: wire a DialogMachine into a LiveKit Agent.

Run with ``pip install superdialog[livekit]`` and a configured LiveKit
project. This is a structural example — it does not call out to a real
LiveKit room.
"""

from __future__ import annotations

from superdialog import DialogMachine
from superdialog.adapters.livekit import DialogMachineLLM
from superdialog.flow.models import ConversationFlow, FlowNode


def build_dialog_machine() -> DialogMachine:
    flow = ConversationFlow(
        id="demo",
        initial_node="welcome",
        nodes=[
            FlowNode(id="welcome", static_text="Hi, how can I help?", is_final=True)
        ],
    )
    return DialogMachine(flow=flow, llm="openai/gpt-5.1")


def main() -> None:
    dm = build_dialog_machine()
    llm = DialogMachineLLM(dm)
    # LiveKit wiring (pseudocode -- requires a real session/agent):
    #     from livekit.agents import Agent, AgentSession
    #     agent = Agent(llm=llm)
    #     await AgentSession().start(agent=agent)
    print(f"Built DialogMachineLLM for agent: {llm}")


if __name__ == "__main__":
    main()
