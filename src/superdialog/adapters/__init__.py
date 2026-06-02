"""Host adapters for :class:`superdialog.DialogMachine`.

Each adapter is gated behind its own optional extra so the core package
stays import-light:

* :mod:`superdialog.adapters.livekit`   -- ``pip install superdialog[livekit]``
* :mod:`superdialog.adapters.pipecat`   -- ``pip install superdialog[pipecat]``
* :mod:`superdialog.adapters.fastapi`   -- ``pip install superdialog[fastapi]``
* :mod:`superdialog.adapters.websocket` -- ``pip install superdialog[ws]``

Import the submodule you need; this package intentionally does not eagerly
import any adapter to keep the dependency surface minimal.
"""
