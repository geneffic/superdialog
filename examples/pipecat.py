"""PipeCat example: drop a DialogMachine into a PipeCat pipeline.

Run with ``pip install superdialog[pipecat]``.
"""

from __future__ import annotations

from superdialog import DialogMachine
from superdialog.adapters.pipecat import make_processor
from superdialog.flow.models import ConversationFlow, FlowNode


def main() -> None:
    flow = ConversationFlow(
        id="demo",
        initial_node="welcome",
        nodes=[FlowNode(id="welcome", static_text="Welcome.", is_final=True)],
    )
    dm = DialogMachine(flow=flow, llm="openai/gpt-5.1")
    processor = make_processor(dm)
    # Wire `processor` into a PipeCat Pipeline -- see PipeCat docs for the
    # full Pipeline()/PipelineRunner() construction.
    print(f"Built PipeCat processor: {processor}")


if __name__ == "__main__":
    main()
