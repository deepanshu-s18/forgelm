import requests

from forgelm.data_pipeline.download_edgar import extract_sections


def custom_fetch_text(cik: str, accession: str) -> str:
    idx_url = (
        f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/"
        f"{accession}/index.json"
    )
    resp = requests.get(idx_url, headers={"User-Agent": "Deepanshu Singh/1.0 (deepanshu.singh@example.com)"})
    idx = resp.json()

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
    print("Fetching doc:", doc_url)
    resp = requests.get(doc_url, headers={"User-Agent": "Deepanshu Singh/1.0 (deepanshu.singh@example.com)"})
    return resp.text

cik = "0000320193"
accession = "000032019324000123"
raw = custom_fetch_text(cik, accession)
print("Raw doc length:", len(raw))
sections = extract_sections(raw)
print("Sections found:", list(sections.keys()))
print("Item 1A length:", len(sections.get("item_1a", "")))
print("Item 7 length:", len(sections.get("item_7", "")))
