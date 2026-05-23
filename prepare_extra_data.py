"""
Download and prepare extra Chinese training data:
1. THUCNews (836K news articles from HuggingFace)
2. Chinese classic novels (四大名著 from GitHub)
"""
import os
import sys

# Record new input files we create
new_inputs = []


def save_text(text, filename):
    """Save text to a file, return filename."""
    filepath = os.path.join(os.path.dirname(__file__), filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(text)
    size_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f"  Saved {filename} ({size_mb:.1f} MB)")
    return filename


def download_classic_novels():
    """Download 四大名著 from tennessine/corpus on GitHub."""
    print("\n=== Downloading classic Chinese novels ===")

    import urllib.parse
    import urllib.request

    novels_encoded = {
        "三国演义": urllib.parse.quote("三国演义", safe=""),
        "水浒传": urllib.parse.quote("水浒传", safe=""),
        "红楼梦": urllib.parse.quote("红楼梦", safe=""),
        "西游记": urllib.parse.quote("西游记", safe=""),
    }

    combined = ""
    for name, enc_name in novels_encoded.items():
        url = f"https://raw.githubusercontent.com/tennessine/corpus/master/{enc_name}.txt"
        print(f"  Downloading {name}...")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req) as f:
                text = f.read().decode("utf-8")
            combined += text + "\n\n"
            print(f"    {len(text)} chars")
        except Exception as e:
            print(f"    Failed: {e}")

    if combined:
        fname = save_text(combined, "input20.txt")
        new_inputs.append(fname)


def download_thucnews():
    """Download THUCNews from HuggingFace."""
    print("\n=== Downloading THUCNews (836K news articles) ===")

    from datasets import load_dataset

    ds = load_dataset("SirlyDreamer/THUCNews", split="train", streaming=True)

    out_path = os.path.join(os.path.dirname(__file__), "input21.txt")
    count = 0
    chars = 0

    with open(out_path, "w", encoding="utf-8") as f:
        for item in ds:
            # Combine title and text
            line = item["text"].strip()
            if item["title"]:
                line = item["title"].strip() + "\n" + line
            f.write(line + "\n\n")
            count += 1
            chars += len(line)
            if count % 100000 == 0:
                print(f"  {count} articles ({chars / 1024 / 1024:.1f} MB raw text)...")

    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"  Done: {count} articles, {chars} chars, {size_mb:.1f} MB on disk")
    new_inputs.append("input21.txt")


if __name__ == "__main__":
    download_classic_novels()
    download_thucnews()

    print(f"\n=== Done! Created {len(new_inputs)} new input files ===")
    for f in new_inputs:
        fpath = os.path.join(os.path.dirname(__file__), f)
        size = os.path.getsize(fpath) / (1024 * 1024)
        print(f"  {f} ({size:.1f} MB)")
