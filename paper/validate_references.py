"""
Validate preprint references against CrossRef and Semantic Scholar.

Usage:
  python paper/validate_references.py
"""

import json
import re
import time
import urllib.request
import urllib.parse
import urllib.error
from difflib import SequenceMatcher

REFS = [
    {
        "id": 1,
        "first_author": "Oren",
        "title": "Proving test set contamination in black box language models",
        "journal": "ICLR",
        "volume": "",
        "pages": "",
        "year": 2024,
    },
    {
        "id": 2,
        "first_author": "Golchin",
        "title": "Time travel in LLMs: Tracing data contamination in large language models",
        "journal": "ICLR",
        "volume": "",
        "pages": "",
        "year": 2024,
    },
    {
        "id": 3,
        "first_author": "Sainz",
        "title": "NLP evaluation in trouble: On the need to measure LLM data contamination for each benchmark",
        "journal": "Findings of EMNLP",
        "volume": "",
        "pages": "",
        "year": 2023,
    },
    {
        "id": 4,
        "first_author": "Wang",
        "title": "A causal view of entity bias in (large) language models",
        "journal": "Findings of EMNLP",
        "volume": "",
        "pages": "",
        "year": 2023,
    },
    {
        "id": 5,
        "first_author": "Choi",
        "title": "When identity skews debate: Anonymization for bias-reduced multi-agent reasoning",
        "journal": "arXiv",
        "volume": "",
        "pages": "2510.07517",
        "year": 2025,
    },
    {
        "id": 6,
        "first_author": "Hermann",
        "title": "Beware of data leakage from protein LLM pretraining",
        "journal": "MLCB, PMLR",
        "volume": "261",
        "pages": "",
        "year": 2024,
    },
    {
        "id": 7,
        "first_author": "Hu",
        "title": "Evaluation of large language models for discovery of gene set function",
        "journal": "Nature Methods",
        "volume": "22",
        "pages": "82-91",
        "year": 2025,
    },
    {
        "id": 8,
        "first_author": "Lin",
        "title": "Evolutionary-scale prediction of atomic-level protein structure with a language model",
        "journal": "Science",
        "volume": "379",
        "pages": "1123-1130",
        "year": 2023,
    },
    {
        "id": 9,
        "first_author": "Theodoris",
        "title": "Transfer learning enables predictions in network biology",
        "journal": "Nature",
        "volume": "618",
        "pages": "616-624",
        "year": 2023,
    },
    {
        "id": 10,
        "first_author": "Elnaggar",
        "title": "ProtTrans: Toward understanding the language of life through self-supervised learning",
        "journal": "IEEE TPAMI",
        "volume": "44",
        "pages": "7112-7127",
        "year": 2022,
    },
    {
        "id": 11,
        "first_author": "Gebru",
        "title": "Datasheets for datasets",
        "journal": "Communications of the ACM",
        "volume": "64",
        "pages": "86-92",
        "year": 2021,
    },
    {
        "id": 12,
        "first_author": "Boiko",
        "title": "Autonomous chemical research with large language models",
        "journal": "Nature",
        "volume": "624",
        "pages": "570-578",
        "year": 2023,
    },
    {
        "id": 13,
        "first_author": "Wang",
        "title": "Scientific discovery in the age of artificial intelligence",
        "journal": "Nature",
        "volume": "620",
        "pages": "47-60",
        "year": 2023,
    },
    {
        "id": 14,
        "first_author": "Ochoa",
        "title": "The next-generation Open Targets Platform: reimagined, redesigned, rebuilt",
        "journal": "Nucleic Acids Research",
        "volume": "51",
        "pages": "D1353-D1359",
        "year": 2023,
    },
    {
        "id": 15,
        "first_author": "Dempster",
        "title": "Chronos: a cell population dynamics model of CRISPR experiments that improves inference of gene fitness effects",
        "journal": "Genome Biology",
        "volume": "22",
        "pages": "343",
        "year": 2021,
    },
    {
        "id": 16,
        "first_author": "Subramanian",
        "title": "A next generation connectivity map: L1000 platform and the first 1,000,000 profiles",
        "journal": "Cell",
        "volume": "171",
        "pages": "1437-1452",
        "year": 2017,
    },
    {
        "id": 17,
        "first_author": "GTEx Consortium",
        "title": "The GTEx Consortium atlas of genetic regulatory effects across human tissues",
        "journal": "Science",
        "volume": "369",
        "pages": "1318-1330",
        "year": 2020,
    },
    {
        "id": 18,
        "first_author": "Hoadley",
        "title": "Cell-of-origin patterns dominate the molecular classification of 10,000 tumors from 33 types of cancer",
        "journal": "Cell",
        "volume": "173",
        "pages": "291-304",
        "year": 2018,
    },
    {
        "id": 19,
        "first_author": "Newman",
        "title": "Robust enumeration of cell subsets from tissue expression profiles",
        "journal": "Nature Methods",
        "volume": "12",
        "pages": "453-457",
        "year": 2015,
    },
]

CROSSREF_URL = "https://api.crossref.org/works"
SEMSCHOLAR_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
HEADERS = {"User-Agent": "ReferenceValidator/1.0 (mailto:mcuccarese@gmail.com)"}


def query_crossref(title, rows=1):
    params = urllib.parse.urlencode({
        "query.title": title,
        "rows": rows,
        "select": "DOI,title,author,container-title,volume,page,published-print,published-online,issued",
    })
    url = f"{CROSSREF_URL}?{params}"
    req = urllib.request.Request(url, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
        items = data.get("message", {}).get("items", [])
        if not items:
            return None
        return items[0]
    except Exception as e:
        return {"error": str(e)}


def query_semantic_scholar(title):
    params = urllib.parse.urlencode({
        "query": title,
        "limit": 1,
        "fields": "title,authors,year,externalIds,venue,publicationVenue",
    })
    url = f"{SEMSCHOLAR_URL}?{params}"
    req = urllib.request.Request(url, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
        papers = data.get("data", [])
        if not papers:
            return None
        return papers[0]
    except Exception as e:
        return {"error": str(e)}


def title_similarity(a, b):
    a = re.sub(r"[^a-z0-9 ]", "", a.lower().strip())
    b = re.sub(r"[^a-z0-9 ]", "", b.lower().strip())
    return SequenceMatcher(None, a, b).ratio()


def extract_year(item):
    for field in ["published-print", "published-online", "issued"]:
        dp = item.get(field, {}).get("date-parts", [[]])
        if dp and dp[0] and dp[0][0]:
            return dp[0][0]
    return None


def extract_first_author(item):
    authors = item.get("author", [])
    if not authors:
        return "?"
    return authors[0].get("family", authors[0].get("name", "?"))


def extract_all_authors(item):
    return [a.get("family", a.get("name", "?")) for a in item.get("author", [])]


def normalize_pages(p):
    if not p:
        return ""
    return re.sub(r"[–—]", "-", str(p)).strip()


def validate_all():
    results = []
    for ref in REFS:
        rid = ref["id"]
        print(f"[{rid:2d}] Querying: {ref['first_author']} — {ref['title'][:50]}...")

        cr = query_crossref(ref["title"])
        time.sleep(0.5)

        issues = []
        status = "OK"
        doi = ""
        source = "CrossRef"
        cr_title = ""
        cr_author = ""
        cr_year = None
        cr_journal = ""
        cr_volume = ""
        cr_pages = ""
        cr_authors_all = []
        tsim = 0.0

        if cr and "error" not in cr:
            cr_titles = cr.get("title", [""])
            cr_title = cr_titles[0] if cr_titles else ""
            tsim = title_similarity(ref["title"], cr_title)
            doi = cr.get("DOI", "")
            cr_author = extract_first_author(cr)
            cr_authors_all = extract_all_authors(cr)
            cr_year = extract_year(cr)
            cr_journals = cr.get("container-title", [""])
            cr_journal = cr_journals[0] if cr_journals else ""
            cr_volume = cr.get("volume", "")
            cr_pages = normalize_pages(cr.get("page", ""))

            if tsim < 0.80:
                issues.append(f"Title match low ({tsim:.0%}): '{cr_title[:60]}'")
                status = "WARN"
            if ref["first_author"].lower() not in [a.lower() for a in cr_authors_all]:
                if ref["first_author"] not in ("GTEx Consortium", "CZI Single-Cell Biology Program"):
                    issues.append(f"Author mismatch: ours='{ref['first_author']}', CR='{cr_author}'")
                    status = "WARN"
            if cr_year and cr_year != ref["year"]:
                issues.append(f"Year mismatch: ours={ref['year']}, CR={cr_year}")
                status = "WARN"
            if ref["volume"] and cr_volume and str(ref["volume"]) != str(cr_volume):
                issues.append(f"Volume mismatch: ours={ref['volume']}, CR={cr_volume}")
                status = "WARN"
            our_pages = normalize_pages(ref["pages"])
            if our_pages and cr_pages and our_pages != cr_pages:
                if not cr_pages.startswith(our_pages):
                    issues.append(f"Pages mismatch: ours={our_pages}, CR={cr_pages}")
                    status = "WARN"
        else:
            ss = query_semantic_scholar(ref["title"])
            time.sleep(1.0)
            source = "SemScholar"
            if ss and "error" not in ss:
                cr_title = ss.get("title", "")
                tsim = title_similarity(ref["title"], cr_title)
                cr_year = ss.get("year")
                ss_authors = ss.get("authors", [])
                cr_author = ss_authors[0]["name"].split()[-1] if ss_authors else "?"
                cr_authors_all = [a["name"].split()[-1] for a in ss_authors]
                ext = ss.get("externalIds", {})
                doi = ext.get("DOI", ext.get("ArXiv", ""))
                if tsim < 0.80:
                    issues.append(f"Title match low ({tsim:.0%}): '{cr_title[:60]}'")
                    status = "WARN"
                if cr_year and cr_year != ref["year"]:
                    issues.append(f"Year mismatch: ours={ref['year']}, SS={cr_year}")
                    status = "WARN"
            else:
                issues.append("NOT FOUND in CrossRef or Semantic Scholar")
                status = "FAIL"

        if not issues:
            issues = ["All fields match"]

        results.append({
            "id": rid,
            "status": status,
            "first_author": ref["first_author"],
            "year": ref["year"],
            "journal": ref["journal"],
            "doi": doi,
            "title_sim": tsim,
            "source": source,
            "cr_title": cr_title,
            "cr_author": cr_author,
            "cr_year": cr_year,
            "issues": issues,
        })
    return results


def print_table(results):
    print("\n" + "=" * 90)
    print("REFERENCE VALIDATION REPORT")
    print("=" * 90)
    ok = sum(1 for r in results if r["status"] == "OK")
    warn = sum(1 for r in results if r["status"] == "WARN")
    fail = sum(1 for r in results if r["status"] == "FAIL")
    print(f"\nSummary: {ok} OK  |  {warn} WARN  |  {fail} FAIL  |  {len(results)} total\n")
    for r in results:
        doi_str = r["doi"][:40] if r["doi"] else "no DOI"
        line = (f"[{r['id']:2d}] {r['status']:<4s}  {r['first_author']:<20s} "
                f"{r['year']}  {r['journal']:<22s}  "
                f"sim={r['title_sim']:.0%}  {doi_str}")
        print(line)
        if r["status"] != "OK":
            for iss in r["issues"]:
                print(f"        -> {iss}")
    print()


if __name__ == "__main__":
    import sys, io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    results = validate_all()
    print_table(results)
