# Playbook engine

A **Playbook** declares a conversation as journeys of checkpoints (goal, typed
slots, guidance prose, advance rules) plus a process layer (tools, pipelines,
handlers, interrupts, policies). At runtime a fast **Talker** LLM streams every
spoken turn while an async **Director** extracts slots, judges advancement, and
runs tools — both over an append-only, event-sourced log that doubles as the
audit/replay artifact. Legacy flow graphs compile down losslessly via
`compile_flow`.

## A minimal playbook

```yaml
persona: "You are a booking assistant."
journeys:
  booking:
    checkpoints:
      - id: collect
        goal: "Have city and date"
        slots:
          city: {type: str, required: true}
          date: {type: date, required: true}
        guidance: "Collect naturally."
        advance_when:
          - {when: "details complete", judge: llm, to: booking.confirm,
             requires: [city, date]}
      - id: confirm
        gate: hard
        say_verbatim: "Your booking is held."
        pipeline: confirm_and_hold
        advance_when:
          - {when: "pipeline.ok", judge: expr, to: booking.close}
      - id: close
        terminal: true
        outcome: confirmed
tools:
  - id: hold_slot
    method: POST
    url: "{{ env.API_BASE_URL }}/slots/hold"
    store_response_as: hold_result
pipelines:
  - id: confirm_and_hold
    steps:
      - tool: hold_slot
        on: {ok: continue, failed: {retry: 1, on_exhaust: booking.collect}}
```

## Usage

```python
from superdialog.playbook import Playbook, PlaybookAgent
from superdialog.playbook.toolexec import httpx_http

agent = PlaybookAgent(
    playbook=Playbook.load("booking.yaml"),
    talker_llm=talker,      # stream(messages) -> AsyncIterator[str]
    director_llm=director,  # async complete(messages) -> str
    http=httpx_http,
)
result = await agent.turn("hello")
```

Provider adapters: the Director wants plain text — wrap a real provider with
`(await provider.complete(messages)).text`; the Talker wants raw tokens —
yield `chunk.text` from `provider.stream(messages)`.

## Compiling legacy flows

```python
from superdialog.playbook import compile_flow, coverage_report

pb = compile_flow(flow)               # ConversationFlow -> Playbook
report = coverage_report(flow, pb)    # proves every node/edge/action mapped
```

Design rationale and the full architecture live in
`docs/plans/2026-06-10-checkpoint-compound-architecture-design.md`.
