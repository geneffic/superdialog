"""Tests for the RuntimeAdapter protocol."""

from superdialog.machine.adapters.base import RuntimeAdapter
from superdialog.machine.testing.mock_adapter import (
    MockAdapter,
    MockAdapterWithCriteria,
)


class TestRuntimeAdapterProtocol:
    """Verify that mock adapters satisfy the RuntimeAdapter protocol."""

    def test_mock_adapter_is_runtime_adapter(self) -> None:
        adapter = MockAdapter(edge_sequence=["e1"])
        assert isinstance(adapter, RuntimeAdapter)

    def test_mock_adapter_with_criteria_is_runtime_adapter(
        self,
    ) -> None:
        adapter = MockAdapterWithCriteria(edge_id="e1")
        assert isinstance(adapter, RuntimeAdapter)

    def test_non_adapter_fails_isinstance(self) -> None:
        """A plain object should not satisfy RuntimeAdapter."""

        class NotAnAdapter:
            pass

        assert not isinstance(NotAnAdapter(), RuntimeAdapter)
