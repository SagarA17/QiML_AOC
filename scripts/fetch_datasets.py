"""Download the project's Zenodo datasets and verify them against Zenodo's md5 checksums.

Usage: python scripts/fetch_datasets.py [name ...]      (no names = all datasets)
Data root: $QIML_AOC_DATA, default /lus/eagle/projects/ATLAS_workflow_ALCF/sagar/QiML_AOC.
Files already present with the right checksum are skipped; downloads go to *.part and are renamed on success.
"""
import hashlib, json, os, sys, urllib.request

DATA_ROOT = os.environ.get("QIML_AOC_DATA", "/lus/eagle/projects/ATLAS_workflow_ALCF/sagar/QiML_AOC")
DATASETS = {
    # Shlomi et al., "Secondary vertex finding in jets with neural networks", arXiv:2008.02831. CC-BY-4.0.
    "sv_jets": dict(record=4044628, doi="10.5281/zenodo.4044628"),
}
API = "https://zenodo.org/api/records/{}"


def md5sum(path, chunk=1 << 22):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""): h.update(block)
    return h.hexdigest()


def fetch(name):
    ds = DATASETS[name]; dest = os.path.join(DATA_ROOT, name); os.makedirs(dest, exist_ok=True)
    with urllib.request.urlopen(API.format(ds["record"])) as r: rec = json.load(r)
    with open(os.path.join(dest, "zenodo_record.json"), "w") as f: json.dump(rec, f, indent=1)
    for fi in rec["files"]:
        path, want = os.path.join(dest, fi["key"]), fi["checksum"].split(":", 1)[1]
        if os.path.exists(path) and md5sum(path) == want:
            print(f"[{name}] {fi['key']}: ok (cached)"); continue
        print(f"[{name}] {fi['key']}: downloading {fi['size']/1e6:.0f} MB", flush=True)
        with urllib.request.urlopen(fi["links"]["self"]) as r, open(path + ".part", "wb") as f:
            for block in iter(lambda: r.read(1 << 22), b""): f.write(block)
        got = md5sum(path + ".part")
        if got != want: sys.exit(f"[{name}] {fi['key']}: md5 mismatch ({got} != {want})")
        os.replace(path + ".part", path); print(f"[{name}] {fi['key']}: ok")


if __name__ == "__main__":
    for n in sys.argv[1:] or DATASETS: fetch(n)
