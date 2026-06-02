"""
superdialog tester.

    python chat.py               → chat with last generated flow
    python chat.py --generate    → generate new flow from GENERATE_PROMPT,
                                   save to FLOW_PATH, then chat with it

Workflow:
  1. Edit GENERATE_PROMPT below
  2. Run: python chat.py --generate
     → new flow JSON saved to FLOW_PATH (overwrites)
     → chat starts immediately
  3. Next time, just: python chat.py   (reuses saved flow, no LLM generate call)
"""

from __future__ import annotations
import asyncio, os, sys, textwrap, argparse, json
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv

load_dotenv("/home/ankit/Unpod/super-sanyam/.env")

from superdialog.traversal import build_traversal, save_traversal

# ── CONFIGURE HERE ─────────────────────────────────────────────────────────────
MODEL = "openai/gpt-4.1-mini"

# --- generated flow always saved/loaded here ---
FLOW_PATH        = "/home/ankit/Downloads/flow_golf_ai_updated.json"
SYSTEM_PROMPT_PATH = "/home/ankit/Unpod/super-sanyam/super/superdialog/generated_system_prompt.txt"
TRAVERSAL_DIR = Path(__file__).parent / "src/superdialog/traversal"

# --- describe the agent you want to build ---
GENERATE_PROMPT = """
A voice tee-time booking agent for GolfAI TeeTime named Arjun (male).

On start: greet the caller with time-of-day salutation (Good morning / Good afternoon /
Good evening). Returning callers (with booking history) get a personalised greeting
mentioning their last course and are offered to rebook or go somewhere new.
New callers get a standard welcome.

Core booking flow:
1. Collect: city or specific course name.
2. If multiple courses in city, list them and ask caller to choose.
3. Collect: booking date, preferred tee time (range 6 AM – 5:45 PM), number of players (max 4).
4. Check availability for the selected course, date, and time window.
5. Present nearest available slot and price. Confirm full summary (course, date, time, players, total price).
6. Inform caller that payment link has been sent to their registered email and they have 7 minutes to pay.
7. End call after payment instruction — do not wait for payment.

Additional paths:
- Caller wants to rebook the same course as last time → skip city/course collection, ask only for date, time, players.
- Caller wants to check an existing booking → ask for booking reference, read back details.
- Caller wants to cancel a booking → ask for reference, confirm cancellation.
- Caller asks about course details (pricing, facilities, policy) → provide details, then offer to book.
- Caller asks which cities have courses → list available cities.
- Caller is silent after greeting → ask "Hello, can you hear me?"; if still silent → end call.
- Caller says call back later → collect preferred callback time, confirm, end call.
- Caller says goodbye → immediately end call politely without asking further questions.

Language: detect caller's language and respond in the same language throughout.
Support Hinglish (Hindi-English mix). Agent (Arjun) uses masculine Hindi verb forms for himself
but gender-neutral forms when addressing the caller.
Numbers are always spoken as words (e.g. four thousand eight hundred, not 4800).
Never reveal internal IDs (course_id, booking_id) to the caller.
Never re-greet after the greeting node.
"""
# ───────────────────────────────────────────────────────────────────────────────

R="\033[0m"; B="\033[1m"; DIM="\033[2m"
GRN="\033[92m"; CYN="\033[96m"; YLW="\033[93m"
GRY="\033[90m"; RED="\033[91m"; WHT="\033[97m"

W = min(os.get_terminal_size().columns if sys.stdout.isatty() else 88, 100)

def hr(c="─"): print(GRY + c*W + R)
def wrap(text, pad=6):
    p = " " * pad
    return textwrap.fill(text.strip(), width=W-pad, initial_indent=p, subsequent_indent=p)


def print_msg(msg: dict):
    role, content, n = msg.get("role"), msg.get("content","").strip(), msg.get("_node","")
    if not content or role == "system":
        return
    if role == "user":
        print(f"  {B}{GRN}You{R}")
        print(wrap(content)); print()
    elif role == "assistant":
        print(f"  {B}{CYN}Bot{R}  {GRY}[{n}]{R}")
        for line in content.split("\n"):
            if line.strip(): print(wrap(line))
        print()


def print_node_status(node: str):
    hr()
    print(f"  {DIM}node: {B}{WHT}{node}{R}  {GRY}│  quit = exit{R}")
    hr()




async def generate_and_save():
    """Generate flow from GENERATE_PROMPT, save to FLOW_PATH."""
    from superdialog import create_dialog_flow

    hr("═")
    print(f"{B}{WHT}  Generating flow...{R}  {DIM}{MODEL}{R}"); hr("═"); print()
    print(f"{YLW}  Prompt:{R}")
    for line in GENERATE_PROMPT.strip().split("\n"):
        if line.strip(): print(f"    {DIM}{line.strip()}{R}")
    print(f"\n  {DIM}Calling LLM...{R}", flush=True)

    flow = await create_dialog_flow(prompt=GENERATE_PROMPT.strip(), llm=MODEL)

    flow.save(FLOW_PATH)
    with open(SYSTEM_PROMPT_PATH, "w") as f:
        f.write(flow.system_prompt)
    print(f"  {GRN}✓ System prompt saved:{R}  {B}{SYSTEM_PROMPT_PATH}{R}")

    node_count = len(flow.nodes)
    edge_count = sum(len(n.edges) for n in flow.nodes)
    print(f"\n  {GRN}✓ Flow generated + saved:{R}  {B}{FLOW_PATH}{R}")
    print(f"  {B}{node_count} nodes{R}  {GRY}│{R}  {B}{edge_count} edges{R}\n")
    print(f"  {DIM}Nodes:{R}")
    for n in flow.nodes:
        star = f"{GRN}★{R}" if n.id == flow.initial_node else " "
        edges_preview = ", ".join(e.id for e in n.edges[:3])
        if len(n.edges) > 3: edges_preview += "..."
        print(f"    {star} {B}{n.id}{R}  {GRY}→ [{edges_preview}]{R}")
    print()
    input(f"  {DIM}Press Enter to start chat...{R} ")


async def chat(flow_path: str):
    from superdialog import DialogMachine, Flow

    if not os.path.exists(flow_path):
        print(f"{RED}Flow not found: {flow_path}{R}"); sys.exit(1)

    flow    = Flow.load(flow_path)
    machine = DialogMachine(flow=flow, llm=MODEL)
    source  = os.path.basename(flow_path)
    chat_turns: list[dict] = []
    started_at = datetime.now(timezone.utc)

    hr("═")
    print(f"{B}{WHT}  superdialog tester{R}  {GRY}│{R}  {CYN}{source}{R}  {GRY}│{R}  {DIM}model: {MODEL}{R}")
    hr("═"); print()

    try:
        first = await machine.start()
    except Exception as e:
        print(f"{RED}Failed to start: {e}{R}"); sys.exit(1)

    node = machine.state["node_id"]
    if first.text:
        msg = {"role": "assistant", "content": first.text, "_node": node}
        print_msg(msg)
    chat_turns.append({
        "step": 1,
        "bot": first.text or "",
        "user": None,
        "node": node,
        "ts": datetime.now(timezone.utc).isoformat(),
    })

    while True:
        print_node_status(node)

        if machine._machine and machine._machine.is_complete:
            print(f"\n  {B}{GRN}✓ Conversation complete.{R}\n"); break

        try:
            raw = input(f"\n  {B}{GRN}You ›{R} ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n\n  {DIM}[exited]{R}\n"); break

        if raw.lower() in {"quit","exit","q","/quit"}:
            print(f"\n  {DIM}[exited]{R}\n"); break
        if not raw:
            continue

        print()
        print_msg({"role": "user", "content": raw})

        try:
            turn = await machine.turn(raw)
        except Exception as e:
            print_msg({"role": "assistant", "content": f"[ERROR: {e}]", "_node": node})
            continue

        node = machine.state["node_id"]
        if turn.text:
            print_msg({"role": "assistant", "content": turn.text, "_node": node})
        chat_turns.append({
            "step": len(chat_turns) + 1,
            "bot": turn.text or "",
            "user": raw,
            "node": node,
            "ts": datetime.now(timezone.utc).isoformat(),
        })

    # Save traversal after loop exits
    if chat_turns:
        try:
            traversal = build_traversal(
                machine, chat_turns, flow, source, MODEL, started_at
            )
            saved_path = save_traversal(traversal, TRAVERSAL_DIR)
            print(f"\n  {GRN}✓ Traversal saved:{R}  {B}{saved_path}{R}")
            print(f"  {DIM}  {len(traversal['traversal'])} steps  │  "
                  f"{sum(1 for n in traversal['graph']['nodes'] if n['visited'])} nodes visited{R}\n")
        except Exception as e:
            print(f"\n  {YLW}Warning: traversal save failed: {e}{R}\n")


async def main_async(generate: bool):
    import asyncio as _aio
    if generate:
        await generate_and_save()
    elif not os.path.exists(FLOW_PATH):
        print(f"{YLW}No flow found at:{R}  {FLOW_PATH}")
        print(f"{DIM}Run with --generate first to create one.{R}")
        sys.exit(1)
    await chat(FLOW_PATH)
    await _aio.sleep(0.15)  # let SSL connections flush before loop closes


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--generate", action="store_true",
                   help="Generate new flow from GENERATE_PROMPT, save, then chat")
    args = p.parse_args()

    if not os.environ.get("OPENAI_API_KEY") and MODEL.startswith("openai/"):
        print(f"{YLW}Warning: OPENAI_API_KEY not set{R}")

    asyncio.run(main_async(args.generate))


if __name__ == "__main__":
    main()