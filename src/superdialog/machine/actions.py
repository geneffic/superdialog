"""ActionExecutor — executes flow actions (webhooks) and stores results."""

from __future__ import annotations

import logging
import traceback
from typing import Any

from superdialog.flow.models import ActionTriggerType, CustomAction
from superdialog.machine._lang_util import save_failed_execution_log
from superdialog.machine.models import ActionRecord, FlowContext

logger = logging.getLogger(__name__)


class ActionExecutor:
    """Executes flow actions and stores results in context."""

    def __init__(
        self,
        adapter: Any,
        action_map: dict[str, CustomAction],
    ) -> None:
        self._adapter = adapter
        self._action_map = action_map

    async def execute(
        self,
        actions: list[Any],
        context: FlowContext,
        trigger_type: ActionTriggerType | None = None,
    ) -> list[str]:
        """Execute actions matching trigger type.

        Returns list of action IDs that fired.
        """
        print(
            f"[TRACK] ActionExecutor.execute START - actions count: {len(actions)}, trigger_type: {trigger_type}"
        )
        print(
            f"[TRACK] ActionExecutor.execute - userdata keys: {list(context.userdata.keys()) if context.userdata else 'None'}"
        )
        fired: list[str] = []
        to_execute = actions
        if trigger_type is not None:
            to_execute = [a for a in actions if a.trigger_type == trigger_type]

        for action_trigger in to_execute:
            action = self._action_map.get(action_trigger.action_id)
            if not action:
                logger.warning(
                    "Action '%s' not found in action_map",
                    action_trigger.action_id,
                )
                continue
            try:
                print(
                    f"[TRACK] ActionExecutor - executing action: {action.id}, userdata keys: {list(context.userdata.keys()) if context.userdata else 'None'}"
                )
                result = await self._adapter.execute_action(action, context.userdata)
                fired.append(action.id)

                # Strip internal metadata keys before storing into userdata
                rendered_url = ""
                method_str = ""
                if isinstance(result, dict):
                    rendered_url = result.pop("_rendered_url", "")
                    method_str = result.pop("_method", "")

                print(f"[TRACK] ActionExecutor - action {action.id} result: {result}")
                if result is not None and action.store_response_as:
                    print(
                        f"[TRACK] ActionExecutor - storing result as: {action.store_response_as}, result: {result}"
                    )
                    context.userdata[action.store_response_as] = result
                    context.data.merge(
                        {action.store_response_as: result},
                        source=f"action:{action.id}",
                    )
                    print(
                        f"[TRACK] ActionExecutor - AFTER STORE - userdata keys: {list(context.userdata.keys())}"
                    )
                # Apply env_updates: extract values from result and store in
                # userdata so templates like {{ACCESS_TOKEN}} resolve correctly
                # in all subsequent actions within this session.
                if result is not None and action.env_updates:
                    for update in action.env_updates:
                        try:
                            value = result
                            for key in update.result_path.split("."):
                                value = value[key]
                            context.userdata[update.env_key] = str(value)
                            logger.info(
                                "[ActionExecutor] env_update applied: %s = <token> (from %s)",
                                update.env_key,
                                update.result_path,
                            )
                        except (KeyError, TypeError, IndexError):
                            logger.warning(
                                "[ActionExecutor] env_update: could not resolve '%s' "
                                "from action '%s' result",
                                update.result_path,
                                action.id,
                            )

                # Record API call in action_log for traversal history
                if result is not None:
                    context.action_log.append(ActionRecord(
                        action_id=action.id,
                        node_id=context.current_node_id,
                        trigger=trigger_type.value if trigger_type is not None else "unknown",
                        url=rendered_url,
                        method=method_str,
                        status=result.get("status", 0),
                        success=bool(result.get("success", False)),
                        result_data=result.get("data", {}) if isinstance(result.get("data"), dict) else {},
                    ))
            except Exception as exc:
                logger.error(
                    "Action '%s' failed (non-fatal): %s",
                    action.id,
                    exc,
                )
                await save_failed_execution_log(
                    task_id=(context.userdata or {}).get("task_id"),
                    step="flow_action_failed",
                    location="ActionExecutor.execute",
                    error_message=(f"{type(exc).__name__}: {exc}".strip()),
                    data={
                        "action_id": action.id,
                        "trigger_type": (
                            str(trigger_type) if trigger_type is not None else None
                        ),
                        "node": context.current_node_id,
                    },
                    traceback_str=traceback.format_exc(),
                )
                fired.append(action.id)
        return fired
