"""Stage 0a — Download and clean EDGAR 10-K/10-Q filings.

Fetches MD&A (Item 7) and Risk Factors (Item 1A) sections for ~100
companies, 2015-2024. Outputs chunked, deduplicated text to
data/corpus/ with a DVC-compatible manifest.

Usage:
    python -m forgelm.data_pipeline.download_edgar \
        --tickers AAPL MSFT GOOGL ... \
        --out data/corpus \
        --max-filings 200

SEC EDGAR requirements:
    - Rate limit: <= 10 req/s (hardcoded 0.12s sleep between requests)
    - User-Agent: must include name + contact email (set EDGAR_USER_AGENT env var)
      Format: "YourName/1.0 (your@email.com)"
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path

import requests
import structlog

log = structlog.get_logger("forgelm.download_edgar")

EDGAR_BASE = "https://efts.sec.gov/LATEST/search-index"
EDGAR_FULL = "https://data.sec.gov/submissions"
EDGAR_SEARCH = "https://efts.sec.gov/LATEST/search-index?q=%22{ticker}%22&dateRange=custom&startdt={start}&enddt={end}&forms={form}"

_CHUNK_TOKENS = 2048  # target chunk size (approximate, in whitespace-tokens)
_RATE_SLEEP = 0.12    # 1/10 req/s with margin

# Boilerplate patterns to strip (SEC XBRL/header noise)
_STRIP_RE = re.compile(
    r"(<[^>]+>|&[a-z]+;|\bixbrl\b|\bXBRL\b|Item\s+\d+[A-Z]?\.\s*$)",
    re.MULTILINE | re.IGNORECASE,
)
_WHITESPACE_RE = re.compile(r"\s{3,}")

# Sections we want
SECTIONS = {
    "item_1a": re.compile(r"Item\s+1A[\.\s]+Risk\s+Factor", re.IGNORECASE),
    "item_7":  re.compile(r"Item\s+7[\.\s]+Management.s\s+Discussion", re.IGNORECASE),
}


def _user_agent() -> str:
    ua = os.environ.get("EDGAR_USER_AGENT", "AlphaForgeResearch/1.0 (research@example.com)")
    return ua


def _get(url: str) -> requests.Response:
    """Rate-limited GET with required SEC User-Agent header."""
    time.sleep(_RATE_SLEEP)
    resp = requests.get(url, headers={"User-Agent": _user_agent()}, timeout=30)
    resp.raise_for_status()
    return resp


def _clean(text: str) -> str:
    text = _STRIP_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()


def _chunk(text: str, chunk_tokens: int = _CHUNK_TOKENS) -> list[str]:
    """Split text into ~chunk_tokens whitespace-token chunks with 10% overlap."""
    words = text.split()
    if not words:
        return []
    step = int(chunk_tokens * 0.9)
    chunks = []
    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + chunk_tokens])
        if len(chunk.split()) >= 50:  # drop tiny tail chunks
            chunks.append(chunk)
    return chunks


def _dedup(chunks: list[str]) -> list[str]:
    """Remove near-duplicate chunks (8-gram overlap > 80%)."""
    seen: set[frozenset] = set()
    out = []
    for c in chunks:
        grams = frozenset(zip(*[c.split()[i:] for i in range(8)]))
        if not any(len(grams & s) / max(len(grams), 1) > 0.8 for s in seen):
            seen.add(grams)
            out.append(c)
    return out


def fetch_cik(ticker: str) -> str | None:
    """Resolve ticker → CIK via SEC company tickers JSON."""
    resp = _get("https://www.sec.gov/files/company_tickers.json")
    data = resp.json()
    for entry in data.values():
        if entry.get("ticker", "").upper() == ticker.upper():
            return str(entry["cik_str"]).zfill(10)
    return None


def fetch_filings(cik: str, form: str = "10-K", start: str = "2015-01-01",
                  end: str = "2024-12-31", max_n: int = 20) -> list[dict]:
    """Return up to max_n filing metadata dicts for a CIK."""
    url = f"{EDGAR_FULL}/CIK{cik}.json"
    resp = _get(url)
    data = resp.json()
    filings = data.get("filings", {}).get("recent", {})
    forms = filings.get("form", [])
    dates = filings.get("filingDate", [])
    accessions = filings.get("accessionNumber", [])
    results = []
    for f, d, a in zip(forms, dates, accessions):
        if f == form and start <= d <= end:
            results.append({"form": f, "date": d, "accession": a.replace("-", "")})
        if len(results) >= max_n:
            break
    return results


def fetch_text(cik: str, accession: str) -> str:
    """Download and concatenate the primary document text for a filing."""
    idx_url = (
        f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/"
        f"{accession}/index.json"
    )
    try:
        resp = _get(idx_url)
        idx = resp.json()
    except Exception:
        return ""
    
    # primary document: largest .htm file in the filing index
    docs = []
    for doc in idx.get("directory", {}).get("item", []):
        name = doc.get("name", "")
        if name.endswith(".htm") or name.endswith(".html"):
            size_str = doc.get("size", "0")
            size = int(size_str) if size_str else 0
            docs.append((size, name))
            
    if not docs:
        return ""
        
    docs.sort(reverse=True)
    largest_name = docs[0][1]
    
    doc_url = (
        f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/"
        f"{accession}/{largest_name}"
    )
    try:
        return _get(doc_url).text
    except Exception:
        return ""


def extract_sections(raw_html: str) -> dict[str, str]:
    """Extract target sections from raw HTML text."""
    # Strip HTML tags for text extraction
    text = re.sub(r"<[^>]+>", " ", raw_html)
    text = re.sub(r"&nbsp;", " ", text)
    text = _clean(text)
    out = {}
    lines = text.split("\n")
    for sec_name, pattern in SECTIONS.items():
        for i, line in enumerate(lines):
            if pattern.search(line):
                # grab up to 8000 words after the section header
                section_text = " ".join(lines[i : i + 400])
                out[sec_name] = _clean(section_text)
                break
    return out


def process_ticker(ticker: str, out_dir: Path, max_filings: int = 20) -> list[dict]:
    """Fetch, clean, chunk, and save corpus chunks for one ticker."""
    log.info("processing_ticker", ticker=ticker)
    cik = fetch_cik(ticker)
    if not cik:
        log.warning("cik_not_found", ticker=ticker)
        return []
    manifest_rows = []
    for form in ("10-K", "10-Q"):
        filings = fetch_filings(cik, form=form, max_n=max_filings)
        for f in filings:
            raw = fetch_text(cik, f["accession"])
            if not raw:
                continue
            sections = extract_sections(raw)
            for sec_name, text in sections.items():
                chunks = _dedup(_chunk(text))
                for j, chunk in enumerate(chunks):
                    cs = hashlib.sha256(chunk.encode()).hexdigest()[:12]
                    fname = f"{ticker}_{f['date']}_{sec_name}_{j}_{cs}.txt"
                    (out_dir / fname).write_text(chunk)
                    manifest_rows.append({
                        "ticker": ticker, "date": f["date"], "form": form,
                        "section": sec_name, "chunk_idx": j, "file": fname,
                        "tokens_approx": len(chunk.split()),
                        "checksum": cs,
                    })
    return manifest_rows


def main():
    p = argparse.ArgumentParser(description="Download and clean EDGAR filings for ForgeLM DAPT")
    p.add_argument("--tickers", nargs="+", default=["AAPL", "MSFT", "GOOGL", "JPM", "XOM"])
    p.add_argument("--out", default="data/corpus", type=Path)
    p.add_argument("--max-filings", type=int, default=20)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = []
    for ticker in args.tickers:
        try:
            rows = process_ticker(ticker, args.out, args.max_filings)
            manifest.extend(rows)
            log.info("ticker_done", ticker=ticker, chunks=len(rows))
        except Exception as e:
            log.error("ticker_failed", ticker=ticker, error=str(e))
    manifest_path = args.out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Corpus: {len(manifest)} chunks → {manifest_path}")


if __name__ == "__main__":
    main()
