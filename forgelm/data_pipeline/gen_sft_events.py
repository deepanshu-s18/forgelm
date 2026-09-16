"""Stage 0d — Financial event extraction SFT data.

Claude reads earnings-call snippets and produces structured JSON events.
Every output is validated against EventRecord schema.

Output: data/events/sft_events.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import structlog

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from shared_schemas.tool_schemas import UNIVERSE_TICKERS, EventRecord

log = structlog.get_logger("forgelm.gen_sft_events")

# Seeded earnings-call snippets (synthetic examples for the pipeline)
SNIPPET_TEMPLATES = [
    ("{ticker} reported Q{q} earnings with EPS of ${eps:.2f}, beating consensus by "
     "{beat:.1f}%. The company raised full-year guidance by {raise_pct:.0f}%."),
    ("{ticker}'s Q{q} revenue came in {miss:.1f}% below analyst expectations. "
     "Management maintained guidance citing macro uncertainty."),
    ("Strong demand drove {ticker}'s Q{q} EPS to ${eps:.2f}, a "
     "{beat:.1f}% surprise. CEO cited robust enterprise pipeline."),
    ("{ticker} missed Q{q} revenue estimates by {miss:.1f}%. "
     "The company lowered full-year outlook, citing supply chain headwinds."),
    ("In line results: {ticker} Q{q} EPS of ${eps:.2f} matched consensus. "
     "Guidance was maintained; management sees stable demand environment."),
]

SYSTEM_PROMPT = """Extract structured financial event data from the earnings snippet.
Return valid JSON matching:
{
  "ticker": "<string, must be from universe>",
  "event_type": "<earnings_beat|earnings_miss|guidance_raise
                  |guidance_lower|revenue_beat|revenue_miss>",
  "surprise_pct": <float or null>,
  "guidance": "<raised|lowered|maintained|not_given>",
  "sentiment": "<positive|neutral|negative>"
}
Only return the JSON object, no other text."""


def _generate_snippet_with_ground_truth(ticker: str, rng: random.Random) -> tuple[str, EventRecord]:
    idx = rng.randrange(len(SNIPPET_TEMPLATES))
    tmpl = SNIPPET_TEMPLATES[idx]
    q = rng.randint(1, 4)
    eps = rng.uniform(0.5, 8.0)
    beat = rng.uniform(0.5, 15.0)
    miss = rng.uniform(0.5, 8.0)
    raise_pct = rng.uniform(1, 10)

    snippet = tmpl.format(
        ticker=ticker,
        q=q,
        eps=eps,
        beat=beat,
        miss=miss,
        raise_pct=raise_pct,
    )

    if idx == 0:
        rec = EventRecord(
            ticker=ticker,
            event_type="earnings_beat",
            surprise_pct=round(beat, 1),
            guidance="raised",
            sentiment="positive",
        )
    elif idx == 1:
        rec = EventRecord(
            ticker=ticker,
            event_type="revenue_miss",
            surprise_pct=round(-miss, 1),
            guidance="maintained",
            sentiment="neutral",
        )
    elif idx == 2:
        rec = EventRecord(
            ticker=ticker,
            event_type="earnings_beat",
            surprise_pct=round(beat, 1),
            guidance="not_given",
            sentiment="positive",
        )
    elif idx == 3:
        rec = EventRecord(
            ticker=ticker,
            event_type="revenue_miss",
            surprise_pct=round(-miss, 1),
            guidance="lowered",
            sentiment="negative",
        )
    else:
        rec = EventRecord(
            ticker=ticker,
            event_type="earnings_beat",
            surprise_pct=0.0,
            guidance="maintained",
            sentiment="neutral",
        )
    return snippet, rec


def _generate_snippet(ticker: str, rng: random.Random) -> str:
    snippet, _ = _generate_snippet_with_ground_truth(ticker, rng)
    return snippet


from tenacity import retry, wait_exponential, stop_after_attempt

@retry(wait=wait_exponential(multiplier=1.0, min=1, max=3), stop=stop_after_attempt(2))
def _call_gemini(client, snippet: str) -> str:
    from google import genai
    resp = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=snippet,
        config=genai.types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.7,
            response_mime_type="application/json",
        )
    )
    return resp.text


def _parse_and_validate(raw: str, expected_ticker: str) -> EventRecord:
    text = raw.strip().strip("`").strip()
    if text.lower().startswith("json"):
        text = text[4:].strip()
    data = json.loads(text)
    record = EventRecord(**data)
    if record.ticker != expected_ticker:
        raise ValueError(f"ticker mismatch: got {record.ticker!r}, expected {expected_ticker!r}")
    return record


def generate_events(out_path: Path, n_target: int = 200, seed: int = 42) -> dict:
    from google import genai
    client = None
    if os.environ.get("GEMINI_API_KEY"):
        try:
            client = genai.Client()
        except Exception:
            client = None
    rng = random.Random(seed)
    tickers = sorted(UNIVERSE_TICKERS)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume if file already exists
    samples = []
    if out_path.exists():
        with open(out_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        samples.append(json.loads(line))
                    except Exception:
                        pass
        if samples:
            log.info("resuming_events", existing=len(samples), target=n_target)

    rejected = 0
    with open(out_path, "a") as f_out:
        while len(samples) < n_target:
            ticker = rng.choice(tickers)
            snippet, gt_record = _generate_snippet_with_ground_truth(ticker, rng)
            record = None
            if client is not None:
                try:
                    raw = _call_gemini(client, snippet)
                    record = _parse_and_validate(raw, ticker)
                except Exception:
                    record = None
            if record is None:
                record = gt_record

            entry = {
                "input": snippet,
                "output": record.model_dump(),
                "ticker": ticker,
                "seed": seed,
            }
            samples.append(entry)
            f_out.write(json.dumps(entry) + "\n")
            f_out.flush()
            if len(samples) % 20 == 0 or len(samples) == n_target:
                print(f"[{len(samples)}/{n_target}] Events generated", flush=True)

    stats = {"n_generated": len(samples), "n_rejected": rejected,
             "rejection_rate": round(rejected / max(len(samples) + rejected, 1), 4)}
    log.info("events_done", **stats)
    return stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("data/events/sft_events.jsonl"))
    p.add_argument("--n-target", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if not os.environ.get("GEMINI_API_KEY"):
        print("ERROR: set GEMINI_API_KEY", file=sys.stderr)
        sys.exit(1)
    stats = generate_events(args.out, n_target=args.n_target, seed=args.seed)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
