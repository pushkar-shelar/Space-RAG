import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import time
import httpx
from pathlib import Path
import pymupdf

CORPUS_DIR = Path("data/corpus")
CORPUS_DIR.mkdir(parents=True, exist_ok=True)

# Curated, authoritative satellite & Earth observation papers
PAPER_CANDIDATES = [
    {
        "id": "2310.18660",
        "filename": "EarthPT_Geospatial_Foundation_Model.pdf",
        "title": "Foundation Models for Generalist Geospatial Artificial Intelligence & Earth Observation"
    },
    {
        "id": "2304.14065",
        "filename": "Remote_Sensing_Transformers_Timeseries.pdf",
        "title": "Lightweight, Pre-trained Transformers for Remote Sensing Timeseries"
    },
    {
        "id": "2107.03780",
        "filename": "Autonomous_Satellite_Constellation_Management.pdf",
        "title": "Autonomous Orbit and Constellation Management for Small Satellites"
    },
    {
        "id": "2204.02781",
        "filename": "CubeSat_Attitude_Determination_and_Control.pdf",
        "title": "CubeSat Attitude Determination and Control Architecture"
    }
]

print("Fetching papers for permanent knowledge corpus...")
for paper in PAPER_CANDIDATES:
    target_file = CORPUS_DIR / paper["filename"]
    if target_file.exists():
        print(f"Already present: {paper['filename']}")
        continue

    url = f"https://arxiv.org/pdf/{paper['id']}.pdf"
    print(f"Downloading {paper['id']} -> {paper['filename']}...")
    try:
        resp = httpx.get(url, follow_redirects=True, timeout=20.0, headers={"User-Agent": "Mozilla/5.0"})
        if resp.status_code == 200:
            doc = pymupdf.open(stream=resp.content, filetype="pdf")
            if len(doc) > 0:
                target_file.write_bytes(resp.content)
                print(f"Saved: {paper['filename']} ({len(doc)} pages)")
            else:
                print(f"Empty PDF: {paper['id']}")
        else:
            print(f"HTTP {resp.status_code} for {paper['id']}")
    except Exception as e:
        print(f"Error fetching {paper['id']}: {e}")
    time.sleep(1)

print("\nCorpus directory contents:")
for p in sorted(CORPUS_DIR.glob("*.pdf")):
    print(f" - {p.name} ({p.stat().st_size // 1024} KB)")
