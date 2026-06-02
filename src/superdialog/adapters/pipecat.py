"""PipeCat adapter: a ``FrameProcessor`` that runs any superdialog :class:`Agent`.

PipeCat plumbs frames between processors; we accept ``TextFrame`` items
on the inbound side, drive a single ``Agent.turn(text)``, and emit
``TextFrame`` items on the outbound side. Subclassing ``FrameProcessor``
is deferred so importing this module without the ``pipecat`` extra is
safe; only :func:`make_processor` (and class instantiation) requires it.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from superdialog.agent import Agent
else:
    Agent = Any

logger = logging.getLogger(__name__)


def _require_pipecat() -> tuple[Any, Any]:
    """Return ``(FrameProcessor, TextFrame)`` or raise a friendly error."""
    try:
        from pipecat.frames.frames import TextFrame  # type: ignore
        from pipecat.processors.frame_processor import FrameProcessor  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "DialogMachineProcessor requires the pipecat extra: "
            "`pip install superdialog[pipecat]`"
        ) from e
    return FrameProcessor, TextFrame


def make_processor(agent: Agent) -> Any:
    """Return a PipeCat ``FrameProcessor`` bound to ``agent``.

    Accepts any superdialog :class:`Agent` (``DialogMachine``,
    ``LLMAgent``, ``LangChainAgent``, or a custom implementation).

    PipeCat's processor API has rotated between releases; rather than
    bake a specific signature into our import graph we synthesise a
    subclass at call time from whatever ``FrameProcessor`` is installed.
    """
    FrameProcessor, TextFrame = _require_pipecat()

    class DialogMachineProcessor(FrameProcessor):  # type: ignore[misc, valid-type]
        """Concrete processor driving ``agent.turn`` per text frame."""

        def __init__(self) -> None:
            super().__init__()
            self._agent = agent

        async def process_frame(
            self,
            frame: Any,
            direction: Any = None,
        ) -> None:
            # Defer to parent for upstream/control frames it knows about.
            parent_process = getattr(super(), "process_frame", None)
            if parent_process is not None:
                try:
                    await parent_process(frame, direction)
                except TypeError:
                    await parent_process(frame)
            if not isinstance(frame, TextFrame):
                return
            user_text = getattr(frame, "text", "") or ""
            if not user_text:
                return
            turn = await self._agent.turn(user_text)
            push = getattr(self, "push_frame", None)
            if push is None:
                return
            try:
                await push(TextFrame(turn.text), direction)
            except TypeError:
                await push(TextFrame(turn.text))

    return DialogMachineProcessor()


__all__ = ["make_processor"]
