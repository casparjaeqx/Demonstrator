"""
SignBank Video Downloader
NGT Sign Language Demonstrator project

Uses a browser cookie file to authenticate — no login form needed.

Setup:
  1. Log into SignBank in your browser
  2. Export cookies for signbank.cls.ru.nl using a browser extension
     (Chrome: "Get cookies.txt LOCALLY", Firefox: "cookies.txt")
  3. Save the file as signbank_cookies.txt in the same folder as this script

Usage:
  python download_signbank.py

Videos are saved to:
  data/
  ├── SIGN_ONE/
  │   └── video.mp4
  ├── SIGN_TWO/
  │   └── video.mp4
  └── ...
"""

import os
import http.cookiejar
import requests
from bs4 import BeautifulSoup

# ── Config ─────────────────────────────────────────────────────────────────────
DATA_DIR      = "data"
SIGNBANK_BASE = "https://signbank.cls.ru.nl"
COOKIE_FILE   = "signbank_cookies.txt"


# ── Parse .ecv file ────────────────────────────────────────────────────────────
def parse_ecv(ecv_path):
    import xml.etree.ElementTree as ET

    with open(ecv_path, "r", encoding="utf-8") as f:
        content = f.read()
    root = ET.fromstring(content)

    url_map = {}
    for ext_ref in root.iter("EXTERNAL_REF"):
        ref_id = ext_ref.get("EXT_REF_ID")
        url    = ext_ref.get("VALUE")
        if ref_id and url:
            url_map[ref_id] = url

    signs = []
    for entry in root.iter("CV_ENTRY_ML"):
        ext_ref = entry.get("EXT_REF")
        url = url_map.get(ext_ref)
        if not url:
            continue

        label = None
        for cv_value in entry.findall("CVE_VALUE"):
            if cv_value.get("LANG_REF") == "nld":
                label = cv_value.text.strip()
                break
        if not label:
            for cv_value in entry.findall("CVE_VALUE"):
                label = cv_value.text.strip()
                break

        if label and url:
            signs.append({"label": label, "url": url})

    return sorted(signs, key=lambda s: s["label"])


# ── Interactive sign picker ────────────────────────────────────────────────────
def pick_signs(signs):
    selected = []
    print("\nSign picker — type part of a sign name to search, or 'done' to finish.\n")

    while True:
        query = input("Search sign (or 'done'): ").strip().upper()

        if query.lower() == "done":
            break
        if not query:
            continue

        matches = [s for s in signs if query in s["label"].upper()]

        if not matches:
            print("  No matches found.\n")
            continue

        print(f"\n  Found {len(matches)} match(es):")
        for i, sign in enumerate(matches[:20]):
            already = " ✓" if sign in selected else ""
            print(f"  [{i+1}] {sign['label']}{already}")
        if len(matches) > 20:
            print(f"  ... and {len(matches) - 20} more. Refine your search.")

        choice = input("\n  Enter number(s) to select, e.g. 1 or 1,3,5 (or Enter to search again): ").strip()
        if not choice:
            print()
            continue

        added   = []
        skipped = []
        invalid = []

        for part in choice.split(","):
            part = part.strip()
            try:
                idx = int(part) - 1
                if 0 <= idx < len(matches[:20]):
                    sign = matches[idx]
                    if sign not in selected:
                        selected.append(sign)
                        added.append(sign["label"])
                    else:
                        skipped.append(sign["label"])
                else:
                    invalid.append(part)
            except ValueError:
                invalid.append(part)

        if added:
            print(f"  Added: {', '.join(added)}")
        if skipped:
            print(f"  Already selected: {', '.join(skipped)}")
        if invalid:
            print(f"  Invalid number(s): {', '.join(invalid)}")
        print()

        print(f"  Currently selected ({len(selected)}): {', '.join(s['label'] for s in selected)}\n")

    return selected


# ── Load cookies from file ─────────────────────────────────────────────────────
def load_session(cookie_file):
    """
    Loads a Netscape-format cookies.txt file into a requests session.
    """
    if not os.path.exists(cookie_file):
        print(f"Error: cookie file '{cookie_file}' not found.")
        print("  Export your SignBank cookies using a browser extension and save as signbank_cookies.txt")
        return None

    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    })

    jar = http.cookiejar.MozillaCookieJar(cookie_file)
    try:
        jar.load(ignore_discard=True, ignore_expires=True)
    except Exception as e:
        print(f"Error loading cookie file: {e}")
        return None

    session.cookies.update(jar)
    return session


# ── Verify session is logged in ────────────────────────────────────────────────
def verify_session(session):
    """
    Checks that the session is actually logged in by visiting the profile page.
    """
    response = session.get(f"{SIGNBANK_BASE}/accounts/user_profile/")
    if "login" in response.url.lower():
        print("Error: cookies have expired. Please export fresh cookies from your browser.")
        return False
    print("Session verified — logged in successfully.\n")
    return True


# ── Find video URL from gloss page ────────────────────────────────────────────
def find_video_url(session, gloss_url):
    # Fix double slash in URL if present
    gloss_url = gloss_url.replace("cls.ru.nl//", "cls.ru.nl/")

    response = session.get(gloss_url)
    if response.status_code != 200:
        return None

    soup = BeautifulSoup(response.text, "html.parser")

    def is_video(src):
        path = src.split("?")[0].lower()
        return any(path.endswith(ext) for ext in [".mp4", ".webm", ".avi"])

    # Collect all three camera angles
    urls = []
    for video_id in ["videoplayer_middle", "videoplayer_left", "videoplayer_right"]:
        tag = soup.find("video", {"id": video_id})
        if tag:
            src = tag.get("src")
            if src and is_video(src):
                full = src if src.startswith("http") else SIGNBANK_BASE + src
                urls.append(full)

    # Fall back to any <video> or <source> tag if none found above
    if not urls:
        for tag in soup.find_all(["video", "source"]):
            src = tag.get("src") or tag.get("data-src")
            if src and is_video(src):
                full = src if src.startswith("http") else SIGNBANK_BASE + src
                if full not in urls:
                    urls.append(full)

    return urls


# ── Download a single video ────────────────────────────────────────────────────
def download_video(session, video_url, save_path):
    response = session.get(video_url, stream=True)
    if response.status_code != 200:
        return False

    total      = int(response.headers.get("content-length", 0))
    downloaded = 0

    with open(save_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    print(f"\r    {pct:.0f}%", end="", flush=True)
    print()
    return True


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("=" * 50)
    print("SignBank Video Downloader")
    print("=" * 50)

    # ── Load .ecv ─────────────────────────────────────────────────────────────
    ecv_path = input("\nPath to your .ecv file: ").strip().strip('"')
    if not os.path.exists(ecv_path):
        print(f"Error: file not found: {ecv_path}")
        return

    print("Parsing .ecv file...")
    signs = parse_ecv(ecv_path)
    print(f"Found {len(signs)} signs in the database.")

    # ── Pick signs ─────────────────────────────────────────────────────────────
    selected = pick_signs(signs)
    if not selected:
        print("No signs selected. Exiting.")
        return

    print(f"\nSelected {len(selected)} sign(s): {', '.join(s['label'] for s in selected)}")

    # ── Load cookies ───────────────────────────────────────────────────────────
    print(f"\nLoading cookies from '{COOKIE_FILE}'...")
    session = load_session(COOKIE_FILE)
    if not session:
        return

    if not verify_session(session):
        return

    # ── Download videos ────────────────────────────────────────────────────────
    print("Downloading videos...\n")
    success = 0
    failed  = 0

    for sign in selected:
        label     = sign["label"]
        gloss_url = sign["url"]
        sign_dir  = os.path.join(DATA_DIR, label)
        os.makedirs(sign_dir, exist_ok=True)

        print(f"[{label}]")
        print(f"  Fetching: {gloss_url}")

        video_urls = find_video_url(session, gloss_url)

        if not video_urls:
            print("  [!] Could not find video on this page. Try downloading manually.")
            failed += 1
            print()
            continue

        # Download each camera angle as a separate file
        angle_names = ["middle", "left", "right"]
        for i, video_url in enumerate(video_urls):
            angle = angle_names[i] if i < len(angle_names) else str(i)
            ext   = os.path.splitext(video_url.split("?")[0])[1] or ".mp4"
            save_path = os.path.join(sign_dir, f"video_{angle}{ext}")

            print(f"  [{angle}] {video_url}")

            if os.path.exists(save_path):
                print(f"    Already downloaded, skipping.")
                success += 1
                continue

            ok = download_video(session, video_url, save_path)
            if ok:
                print(f"    Saved → {save_path}")
                success += 1
            else:
                print(f"    [!] Download failed.")
                failed += 1
        print()

    # ── Summary ────────────────────────────────────────────────────────────────
    print("=" * 50)
    print(f"Done. {success} downloaded, {failed} failed.")
    if success > 0:
        print(f"\nVideos saved to '{DATA_DIR}/'. Run phase2_extract.py next.")
    if failed > 0:
        print(f"\nFor failed signs, download manually from SignBank")
        print(f"and place the video in the correct subfolder inside '{DATA_DIR}/'.")


if __name__ == "__main__":
    main()