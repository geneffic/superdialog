"""``create_dialog_flow`` -- two-phase LLM bootstrap of a flow.

Phase 1: Generate a rich, structured system prompt from the user's description.
Phase 2: Generate the ConversationFlow (nodes + edges) from that system prompt.
"""

from __future__ import annotations

import json
import re
from typing import Any

from ..llm.resolver import resolve_llm
from .models import ConversationFlow as Flow

# ── Phase 1: System prompt generation ─────────────────────────────────────────

_SYSTEM_PROMPT_GEN = """
You are a voice AI agent prompt architect. Given a description of an agent,
produce a professional, production-ready system prompt as plain text.

═══════════════════════════════════════
PHASE 1 — EXTRACT CORE IDENTITY
═══════════════════════════════════════
From the description extract:
  • Agent name (infer a fitting name if not stated)
  • Company / brand name
  • Role / use-case (e.g. "card support inbound", "booking outbound")
  • Language(s): English / Hindi / Hinglish / etc. (infer from context)

═══════════════════════════════════════
PHASE 2 — SELECT RELEVANT SECTIONS
═══════════════════════════════════════
Include ONLY sections relevant to this agent. Skip empty ones.

  [Identity]          Who the agent is: name, company, role
  [Objective]         Primary goal of the conversation
  [Persona & Tone]    Personality, speaking style, emotional register
  [Language]          Language rules, script switching, number pronunciation
  [Response Rules]    Length, pacing, turn-taking, interruption handling
  [Business Details]  Products, services, processes the agent needs to know
  [Greeting Script]   Exact opening lines in all relevant languages
  [FAQs]              Common questions + scripted answers (Q: / A: format)
  [Rebuttals]         Objection handling (Caller says: / You say: format)
  [Escalation]        When and how to transfer to a human
  [Never Do]          Absolute prohibitions (at minimum: never guarantee
                      outcomes, never share personal data)
  [Out of Scope]      What to deflect and how

Add extra sections if the description clearly requires them.

═══════════════════════════════════════
PHASE 3 — WRITE EACH SECTION
═══════════════════════════════════════
• Use second-person: "You are...", "Your role is...", "Never..."
• Be specific — use actual agent name, company name, scripts from description
• Voice AI context: short sentences, natural speech patterns
• FAQs: Q: / A: per question
• Rebuttals: Caller says: / You say: per objection

═══════════════════════════════════════
PHASE 4 — SELF-REVIEW BEFORE RETURNING
═══════════════════════════════════════
Check every item. Fix before returning:
  □ Pronunciation rules present? (numbers as words, brand names)
  □ Interruption handling? ("If caller interrupts, stop and listen")
  □ Language switching logic? (if multi-language)
  □ Out-of-scope fallback?
  □ Never-Do list?
  □ Greeting script with actual opening lines?
  □ Financial/health/legal domain → compliance disclaimer section?

Fix every missing item. Do NOT return until all checks pass.

═══════════════════════════════════════
PHASE 5 — RENDER
═══════════════════════════════════════
rendered_prompt: Join sections using this exact format —

[Identity]

<content>

[Objective]

<content>

... (one blank line between heading and content, two newlines between sections)

first_message: The exact opening line(s) the agent speaks on a call.
  Extract from Greeting Script section if you wrote one.
  If not, synthesize a natural opening from Identity + Objective.
  Include multi-language variants if the agent is multi-lingual.

═══════════════════════════════════════
OUTPUT FORMAT
═══════════════════════════════════════
Return a single valid JSON object with exactly these two keys:
  "rendered_prompt": "<full prompt text with [Heading] sections>",
  "first_message":   "<exact opening line the agent speaks>"

No markdown fences. No commentary. No extra keys.
"""

# ── Phase 2: Flow generation ───────────────────────────────────────────────────

_FLOW_GEN = """
You are a dialog-flow generator. Given an agent system prompt, emit a single
JSON object that matches the ConversationFlow schema exactly.

This schema supports full production flows: HTTP actions fired on node entry,
router nodes for silent branching, edge input_schema for passing data between
nodes, global_edges for universal triggers, and env_vars for secrets/config.

════════════════════════════════════════
FULL JSON SCHEMA — EVERY FIELD EXPLAINED
════════════════════════════════════════
{
  "system_prompt": "<copy the full system prompt text here verbatim>",
  "initial_node": "greeting",

  "env_vars": [
    {"key": "API_BASE_URL", "value": "https://api.example.com/v1"},
    {"key": "CLIENT_ID",    "value": ""},
    {"key": "CLIENT_SECRET","value": ""},
    {"key": "ACCESS_TOKEN", "value": ""}
  ],

  "global_actions": [
    {
      "id": "action-auth",
      "name": "Authenticate",
      "description": "Get bearer token",
      "method": "POST",
      "url": "{{API_BASE_URL}}/auth/token",
      "headers": [{"key": "Content-Type", "value": "application/json"}],
      "body": {"client_id": "{{CLIENT_ID}}", "client_secret": "{{CLIENT_SECRET}}"},
      "timeout": 30,
      "run_once": true,
      "store_response_as": "auth_result",
      "env_updates": [{"env_key": "ACCESS_TOKEN", "result_path": "data.access_token"}]
    },
    {
      "id": "action-get-data",
      "name": "Fetch data",
      "description": "GET example resource",
      "method": "GET",
      "url": "{{API_BASE_URL}}/resource/{{resource_id}}",
      "headers": [
        {"key": "Authorization", "value": "Bearer {{ACCESS_TOKEN}}"},
        {"key": "Content-Type",  "value": "application/json"}
      ],
      "timeout": 30,
      "run_once": false,
      "store_response_as": "resource_result"
    }
  ],

  "global_edges": [
    {
      "id": "global_goodbye",
      "condition": "Caller says goodbye, bye, thank you and goodbye, wants to end the call",
      "target_node_id": "call_end"
    }
  ],

  "nodes": [
    {
      "id": "greeting",
      "name": "Greeting",
      "node_type": "instruction",
      "instruction": "<greeting speech + routing logic>",
      "is_final": false,
      "interruptible": true,
      "actions": [
        {"id": "na-greeting-auth", "trigger": "on_enter", "action_id": "action-auth"}
      ],
      "edges": [
        {
          "id": "greeting_to_main",
          "condition": "Caller responds and wants to proceed",
          "target_node_id": "router_intent"
        },
        {
          "id": "greeting_to_end",
          "condition": "Caller refuses, wrong person, or silent",
          "target_node_id": "call_end"
        }
      ]
    },
    {
      "id": "router_intent",
      "name": "Intent Router",
      "node_type": "router",
      "instruction": "Do not output any text. Do not generate any speech. Silently classify caller intent and call the appropriate edge tool immediately. Zero words before or after the tool call.",
      "is_final": false,
      "interruptible": false,
      "edges": [
        {
          "id": "router_to_booking",
          "condition": "Caller wants to make a booking",
          "target_node_id": "collect_details",
          "input_schema": {
            "type": "object",
            "properties": {
              "date":    {"type": "string", "description": "Date if mentioned"},
              "players": {"type": "string", "description": "Number of players if mentioned"}
            }
          }
        },
        {
          "id": "router_to_other",
          "condition": "Caller has a different request",
          "target_node_id": "other_handler"
        }
      ]
    },
    {
      "id": "collect_details",
      "name": "Collect Details",
      "node_type": "instruction",
      "instruction": "<collect missing fields one at a time: date, time, players>",
      "is_final": false,
      "interruptible": true,
      "actions": [
        {"id": "na-collect-data", "trigger": "on_enter", "action_id": "action-get-data"}
      ],
      "edges": [
        {
          "id": "collect_to_confirm",
          "condition": "All required details collected",
          "target_node_id": "confirm_summary",
          "input_schema": {
            "type": "object",
            "properties": {
              "date":      {"type": "string", "description": "Booking date YYYY-MM-DD"},
              "time":      {"type": "string", "description": "Preferred time HH:MM"},
              "players":   {"type": "string", "description": "Number of players"},
              "item_id":   {"type": "string", "description": "ID of selected item"}
            },
            "required": ["date", "time", "players", "item_id"]
          }
        },
        {
          "id": "collect_to_end",
          "condition": "Caller cancels or wants to stop",
          "target_node_id": "call_end"
        }
      ]
    },
    {
      "id": "confirm_summary",
      "name": "Confirm Summary",
      "node_type": "instruction",
      "instruction": "<read back full summary, ask for confirmation before finalising>",
      "is_final": false,
      "interruptible": true,
      "edges": [
        {
          "id": "confirm_to_done",
          "condition": "Caller confirms",
          "target_node_id": "call_end"
        },
        {
          "id": "confirm_to_collect",
          "condition": "Caller wants to change something",
          "target_node_id": "collect_details"
        }
      ]
    },
    {
      "id": "other_handler",
      "name": "Other Query",
      "node_type": "instruction",
      "instruction": "<handle the request or redirect back to main task>",
      "is_final": false,
      "interruptible": true,
      "edges": [
        {
          "id": "other_to_end",
          "condition": "Query answered or caller is done",
          "target_node_id": "call_end"
        },
        {
          "id": "other_to_main",
          "condition": "Caller wants to proceed with main task",
          "target_node_id": "collect_details"
        }
      ]
    },
    {
      "id": "call_end",
      "name": "Call End",
      "node_type": "instruction",
      "instruction": "Thank the caller warmly and say goodbye.",
      "is_final": true,
      "interruptible": false,
      "edges": []
    }
  ]
}

════════════════════════
NODE COUNT GUIDE
════════════════════════
Simple flow  (greeting → task → end):                3–5 nodes
Medium flow  (multi-step + verification + task):     8–15 nodes
Complex flow (auth + data lookup + multi-path task): 15–30 nodes

Always include: greeting node + at least one is_final=true terminal node.

═══════════════════════
HARD RULES
═══════════════════════
1.  Output ONLY valid JSON — no markdown, no commentary.
2.  edges MUST live INSIDE each node object.
3.  Every non-final node MUST have at least 2 edges (primary + exit).
4.  Every edge target_node_id MUST exactly match an existing node id.
5.  Final nodes MUST have edges=[].
6.  Every non-final node must eventually reach a final node.
7.  Every node id (except initial_node) must appear as a target at least once.
8.  Edge id format: {source_node_id}_to_{target_node_id}
    Multiple edges from same source: add short descriptor suffix.
9.  global_actions ids referenced in node actions[] MUST exist in global_actions[].
10. store_response_as key becomes the Jinja2 variable name in instructions.
    Use snake_case (e.g. "auth_result", "courses_result").
11. env_vars CLIENT_ID / CLIENT_SECRET / ACCESS_TOKEN: leave value="" (filled at runtime).
12. global_edges fire from ANY node — use only for truly universal triggers
    (goodbye, escalation). Do NOT duplicate in individual node edges.

═══════════════════════
NODE TYPE RULES
═══════════════════════
"instruction" — agent speaks + waits for caller response. Has normal edges.
"router"      — SILENT branching node. instruction MUST start with
                "Do not output any text. Do not generate any speech."
                Used immediately after greeting or after data loads.
                Reads userdata (API results) and routes with zero speech.
"static"      — fixed scripted text (use static_text field, not instruction).
"final"       — terminal; is_final=true; edges=[].

═══════════════════════
ACTION RULES
═══════════════════════
- global_actions: define ALL HTTP calls here (auth, data fetch, submit, etc.)
- node actions[]: reference action ids with trigger "on_enter"
  (fires before the node speaks or routes)
- run_once: true — auth and one-time lookups; prevents duplicate API calls
- env_updates: use to extract tokens/ids from responses into env_vars
  so subsequent actions can use {{ACCESS_TOKEN}} etc.
- store_response_as: use descriptive names — results are available in
  node instructions as Jinja2 variables e.g. {{auth_result.success}},
  {{courses_result.data.courses}}

═══════════════════════
EDGE INPUT_SCHEMA RULES
═══════════════════════
Add input_schema to edges when the TARGET node needs data that the SOURCE
node collected (course_id, booking_id, date, player count, etc.).
This data is passed as edge payload and available in the target node's userdata.
Use "required" array for fields that MUST be present.

═══════════════════════
NODE INSTRUCTION RULES
═══════════════════════
Each instruction node instruction must include:
  1. Available API result variables (e.g. auth_result.success={{auth_result.success|default(false)}})
  2. What the agent speaks at this stage
  3. What the agent listens for
  4. Routing: which edge fires under which condition (reference edge ids by name)

system_prompt → global persona/language/never-do. Do NOT repeat in node instructions.
Each node instruction → ONLY content specific to that stage.

═════════════════
FIRST MESSAGE RULE
═════════════════
If FIRST MESSAGE is provided AND substantive (>20 chars, contains the agent or
company name, is recognisably the actual greeting): use it verbatim as the
primary "Speak:" line in the greeting node's instruction.
If it is short, a placeholder, or empty: IGNORE it and synthesise from source.
"""

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?```", re.DOTALL)


def _extract_json(raw: str) -> str:
    match = _JSON_FENCE_RE.search(raw)
    if match:
        return match.group(1).strip()
    # strip leading/trailing non-JSON text before first {
    start = raw.find("{")
    if start != -1:
        return raw[start:].strip()
    return raw.strip()


def _build_phase2_user_message(system_prompt: str, first_message: str) -> str:
    """Build Phase 2 user message, mirroring core's _build_flow_user_message."""
    parts = [f"## AGENT SYSTEM PROMPT\n\n{system_prompt}"]
    if first_message:
        parts.append(f"## FIRST MESSAGE\n\n{first_message}")
    return "\n\n---\n\n".join(parts)


async def create_dialog_flow(prompt: str, llm: str, **kwargs: Any) -> Flow:
    """Bootstrap a ConversationFlow from a natural-language description.

    Two-phase process:
      1. Generate a rich system prompt (agent persona, FAQs, rebuttals, etc.)
         Returns JSON with ``rendered_prompt`` and ``first_message``.
      2. Generate flow nodes + edges from that system prompt + first_message.

    Args:
        prompt: Plain-English description of the agent and conversation.
        llm:    LiteLLM model URI (e.g. "openai/gpt-4.1").

    Returns:
        A validated ConversationFlow instance.
    """
    provider = resolve_llm(llm)

    # ── Phase 1: system prompt ─────────────────────────────────────────────────
    phase1_messages: list[dict[str, Any]] = [
        {"role": "system", "content": _SYSTEM_PROMPT_GEN},
        {"role": "user", "content": prompt},
    ]
    phase1_result = await provider.complete(
        phase1_messages,
        response_format={"type": "json_object"},
        **kwargs,
    )
    # Parse structured Phase 1 output; fall back to plain text for legacy responses.
    try:
        phase1_data = json.loads(_extract_json(phase1_result.text))
        system_prompt_text = phase1_data.get("rendered_prompt", "").strip()
        first_message = phase1_data.get("first_message", "").strip()
        if not system_prompt_text:
            raise ValueError("rendered_prompt empty")
    except Exception:
        system_prompt_text = phase1_result.text.strip()
        first_message = ""

    # ── Phase 2: flow nodes + edges ────────────────────────────────────────────
    phase2_messages: list[dict[str, Any]] = [
        {"role": "system", "content": _FLOW_GEN},
        {"role": "user", "content": _build_phase2_user_message(system_prompt_text, first_message)},
    ]
    phase2_result = await provider.complete(
        phase2_messages,
        response_format={"type": "json_object"},
        **kwargs,
    )
    payload = _extract_json(phase2_result.text)
    data = json.loads(payload)
    return Flow.model_validate(data)