"""
build_indic_dataset.py   (version 2)

Builds a small, balanced Tamil / Telugu evaluation set from two public datasets:

  REAL (bonafide): Kathbath, part of IndicSUPERB (AI4Bharat, CC0 packaging).
                   Read speech, stored as .m4a (AAC).
  FAKE (spoof):    IndicSynth (IIIT-Delhi, CC BY-NC 4.0, academic use only).
                   Generators: xtts_v2 (TTS), vits (TTS), freevc24 (voice conversion).

Why there are TWO versions of every fake file
---------------------------------------------
Kathbath real audio is AAC-compressed (.m4a); the fake audio is not. A detector
could learn "AAC artefacts = real" instead of real synthesis artefacts. So every
fake file is also passed through an AAC encode/decode (ffmpeg), using the
bitrate and sample rate measured on the real files. Evaluate both versions:
    kind = real        real speech (decoded from .m4a to 16 kHz wav)
    kind = fake_raw    fake speech as released, resampled to 16 kHz
    kind = fake_aac    the same fake clips after an AAC round trip
Real and fake are resampled to 16 kHz with the SAME tool (ffmpeg).

How the fake clips are sampled
------------------------------
IndicSynth files are ordered (long runs of one generator and one speaker), so
taking the first N rows would be badly biased. This script scans the generator
column of an evenly spread set of shards (default 24; cached in scan_<Language>.csv), finds which shards contain each generator, downloads a few
shards per generator (each is roughly 0.6 - 1.1 GB), samples rows from them and
deletes the shards again. Generators that do not exist for a language are skipped
with a warning, and --n-fake is split equally over the generators that do.

Needs:  pip install huggingface_hub pyarrow soundfile librosa pandas
        ffmpeg and ffprobe on PATH (check with: ffmpeg -version)

Usage (try --smoke first; it is tiny):
    python build_indic_dataset.py --smoke --kathbath-tar C:\\path\\valid_audio.tar
    python build_indic_dataset.py --kathbath-tar C:\\path\\valid_audio.tar

Output: indic_data/indic_files.csv  (+ the wav files under indic_data/)
"""

import os

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import argparse
import io
import shutil
import subprocess
import tarfile
import time
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf

SR = 16000
FAKE_REPO = "vdivyasharma/IndicSynth"
GENERATORS = ["xtts_v2", "vits", "freevc24"]


# ----------------------------------------------------------------- helpers
def find_col(names, *keys):
    for n in names:
        low = n.lower()
        if all(k in low for k in keys):
            return n
    return None


def normalise_model(s):
    s = str(s).lower()
    if "xtts" in s:
        return "xtts_v2"
    if "freevc" in s:
        return "freevc24"
    if "vits" in s:
        return "vits"
    return s


def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True)


def probe(path):
    """Return (bit_rate, sample_rate) of an audio file via ffprobe (None if unknown)."""
    r = run(["ffprobe", "-v", "error", "-select_streams", "a:0",
             "-show_entries", "stream=bit_rate,sample_rate", "-of", "default=nw=1", str(path)])
    info = {}
    for line in r.stdout.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            info[k.strip()] = v.strip()

    def to_int(x):
        try:
            return int(x)
        except (TypeError, ValueError):
            return None

    return to_int(info.get("bit_rate")), to_int(info.get("sample_rate"))


def to_wav16(src, dst):
    r = run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-ac", "1", "-ar", str(SR), str(dst)])
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip()[:300])


def aac_roundtrip(in_wav, out_wav, bitrate, aac_sr):
    """in_wav (any rate) -> mono AAC at aac_sr / bitrate -> 16 kHz wav."""
    tmp = str(out_wav) + ".m4a"
    r = run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(in_wav), "-ac", "1", "-ar", str(aac_sr),
             "-c:a", "aac", "-b:a", str(bitrate), tmp])
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip()[:300])
    to_wav16(tmp, out_wav)
    os.remove(tmp)


def process_fake_clip(wav_bytes, fid, raw_dir, aac_dir, tmp_dir, bitrate, aac_sr):
    """Decode one fake clip; write the raw 16 kHz version and the AAC-matched version."""
    y, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = np.clip(y, -1.0, 1.0)
    orig = Path(tmp_dir) / f"{fid}_orig.wav"
    sf.write(str(orig), y, sr, subtype="PCM_16")
    raw_path = Path(raw_dir) / f"{fid}.wav"
    aac_path = Path(aac_dir) / f"{fid}.wav"
    to_wav16(orig, raw_path)                       # same ffmpeg resampler as the real files
    aac_roundtrip(orig, aac_path, bitrate, aac_sr)  # original rate -> AAC -> 16 kHz
    duration = sf.info(str(raw_path)).duration
    os.remove(orig)
    return sr, duration, raw_path, aac_path


# ------------------------------------------------------------- real speech
def build_real(args, lang, out_dir, rng):
    lang_l = lang.lower()
    real_dir = out_dir / lang / "real"
    real_dir.mkdir(parents=True, exist_ok=True)

    cands = []  # (name_or_path, member_or_None)
    tf = None
    if args.kathbath_dir:
        for p in Path(args.kathbath_dir).rglob("*.m4a"):
            if lang_l in [x.lower() for x in p.parts]:
                cands.append((str(p), None))
    else:
        print(f"  scanning tar for {lang_l} files (this can take a minute)...")
        tf = tarfile.open(args.kathbath_tar, "r:")
        for m in tf.getmembers():
            name = "/" + m.name.replace("\\", "/").lower()
            if m.isfile() and name.endswith(".m4a") and f"/{lang_l}/" in name:
                cands.append((m.name, m))
    print(f"  {lang}: {len(cands)} real candidates found")
    if not cands:
        raise SystemExit(f"No Kathbath .m4a files found for {lang}. Check the tar / folder path.")

    spk_ids = {(Path(c[0]).stem.split("-") + ["", ""])[1] for c in cands}
    print(f"  {lang}: {len(spk_ids)} different speakers among the candidates")
    # sample with a per-speaker cap (file name: <utt>-<speaker>-<gender>.m4a)
    order = rng.permutation(len(cands))
    per_spk, chosen = {}, []
    for i in order:
        stem = Path(cands[i][0]).stem
        parts = stem.split("-")
        spk = parts[1] if len(parts) >= 3 else stem
        if per_spk.get(spk, 0) >= args.max_per_speaker:
            continue
        per_spk[spk] = per_spk.get(spk, 0) + 1
        chosen.append(i)
        if len(chosen) >= args.n_real:
            break

    if len(chosen) < args.n_real:
        print(f"  WARNING: only {len(chosen)} real clips possible with --max-per-speaker "
              f"{args.max_per_speaker} ({len(spk_ids)} speakers). Raise --max-per-speaker to get "
              f"{args.n_real}, at the cost of fewer different speakers per clip.")
    rows, bitrates, srates = [], [], []
    raw_dir = out_dir / lang / "raw_m4a"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for n, i in enumerate(chosen, 1):
        name, member = cands[i]
        stem = Path(name).stem
        parts = stem.split("-")
        spk = parts[1] if len(parts) >= 3 else ""
        gender = parts[2] if len(parts) >= 3 else ""
        if member is not None:
            member.name = Path(name).name  # flatten the path inside the tar
            tf.extract(member, path=raw_dir)
            src = raw_dir / member.name
        else:
            src = Path(name)
        dst = real_dir / f"{stem}.wav"
        try:
            to_wav16(src, dst)
        except Exception as e:
            print(f"  skip {stem}: {e}")
            continue
        br, orig_sr = probe(src)
        if br:
            bitrates.append(br)
        if orig_sr:
            srates.append(orig_sr)
        rows.append(dict(
            file_id=f"{lang[:2].lower()}_real_{stem}", filepath=str(dst), language=lang,
            kind="real", label="bonafide", generator="real", speaker=spk, gender=gender,
            orig_sr=orig_sr, duration_s=sf.info(str(dst)).duration, pair_id="",
        ))
        if n % 50 == 0:
            print(f"  real {n}/{len(chosen)}")
    if tf is not None:
        tf.close()
    shutil.rmtree(raw_dir, ignore_errors=True)
    return rows, bitrates, srates


# ------------------------------------------------------------- fake speech
def scan_shard_metadata(fileobj, shard_name, full=False):
    """Read the generator column of one parquet shard (plus gender / speaker if full=True).

    Over the network only the generator column is read (fewer requests, much faster).
    For a shard that is already on disk, full=True also reads gender and speaker.
    """
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(fileobj)
    names = pf.schema_arrow.names
    mcol = find_col(names, "generative")
    gcol = find_col(names, "gender") if full else None
    scol = find_col(names, "target", "speaker") if full else None
    cols = [c for c in (mcol, gcol, scol) if c]
    df = pf.read(columns=cols).to_pandas().reset_index(drop=True)
    return pd.DataFrame({
        "shard": shard_name,
        "row": np.arange(len(df)),
        "model": df[mcol].map(normalise_model) if mcol else "unknown",
        "gender": df[gcol] if gcol else "",
        "speaker": df[scol] if scol else "",
    })


def fetch_audio_rows(local_path, wanted_rows):
    """Return {row_index: wav_bytes} for the wanted rows of a parquet shard."""
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(local_path)
    wanted = sorted(set(wanted_rows))
    out, offset = {}, 0
    for rg in range(pf.num_row_groups):
        n = pf.metadata.row_group(rg).num_rows
        sel = [r for r in wanted if offset <= r < offset + n]
        if sel:
            col = pf.read_row_group(rg, columns=["audio"]).column("audio")
            for r in sel:
                out[r] = col[r - offset].as_py()["bytes"]
        offset += n
    return out


def scan_one_shard(fs, shard, tries=3):
    for t in range(tries):
        try:
            with fs.open(f"datasets/{FAKE_REPO}/{shard}", "rb") as f:
                return scan_shard_metadata(f, shard)
        except Exception as e:
            print(f"    {shard}: attempt {t + 1}/{tries} failed ({type(e).__name__})")
            time.sleep(3 * (t + 1))
    return None


def scan_language(args, lang, out_dir, shards):
    scan_csv = out_dir / f"scan_{lang}.csv"
    if scan_csv.exists() and not args.rescan:
        print(f"  using cached scan {scan_csv} (delete it or use --rescan to redo)")
        return pd.read_csv(scan_csv)

    from huggingface_hub import HfFileSystem
    k_scan = len(shards) if args.scan_shards <= 0 else min(args.scan_shards, len(shards))
    scan_idx = np.unique(np.linspace(0, len(shards) - 1, k_scan).astype(int))
    print(f"  scanning the generator column of {len(scan_idx)} evenly spread shards "
          f"(no audio is downloaded)...")
    fs = HfFileSystem()
    frames, failed, t0 = [], 0, time.time()
    for n, si in enumerate(scan_idx, 1):
        df = scan_one_shard(fs, shards[si])
        if df is None:
            failed += 1
        else:
            frames.append(df)
        if n % 5 == 0 or n == len(scan_idx):
            el = time.time() - t0
            print(f"  scanned {n}/{len(scan_idx)}  ({el:.0f}s, ~{el / n * (len(scan_idx) - n):.0f}s left)")
    if not frames:
        raise SystemExit("Could not read any IndicSynth shard (network problem?). Try again later.")
    if failed:
        print(f"  note: {failed} shards could not be scanned and are ignored")
    index = pd.concat(frames, ignore_index=True)
    index.to_csv(scan_csv, index=False)
    return index


def process_shard(local, sh, wants, rng, raw_dir, aac_dir, tmp_dir, bitrate, aac_sr, lang):
    """From one downloaded shard, sample the wanted number of rows per generator and build the files."""
    meta = scan_shard_metadata(local, sh, full=True)
    picks = []
    for g, n in wants.items():
        rows = meta[meta["model"] == g]
        if len(rows):
            picks.append(rows.sample(min(n, len(rows)), random_state=int(rng.integers(1_000_000))))
    if not picks:
        return []
    picks = pd.concat(picks)
    audio_bytes = fetch_audio_rows(local, picks["row"].tolist())
    shard_no = Path(sh).stem.replace("train-", "").split("-of-")[0]
    out = []
    for _, r in picks.iterrows():
        b = audio_bytes.get(int(r["row"]))
        if b is None:
            continue
        fid = f"{lang[:2].lower()}_{r['model']}_{shard_no}_{int(r['row'])}"
        try:
            sr, dur, raw_path, aac_path = process_fake_clip(b, fid, raw_dir, aac_dir, tmp_dir, bitrate, aac_sr)
        except Exception as e:
            print(f"  skipped {fid}: {e}")
            continue
        common = dict(language=lang, label="spoof", generator=r["model"], speaker=r["speaker"],
                      gender=r["gender"], orig_sr=sr, duration_s=dur, pair_id=fid)
        out.append(dict(file_id=fid + "_raw", filepath=str(raw_path), kind="fake_raw", **common))
        out.append(dict(file_id=fid + "_aac", filepath=str(aac_path), kind="fake_aac", **common))
    return out


def build_fake(args, lang, out_dir, rng, bitrate, aac_sr):
    from huggingface_hub import HfApi, hf_hub_download

    raw_dir = out_dir / lang / "fake_raw"
    aac_dir = out_dir / lang / "fake_aac"
    tmp_dir = out_dir / lang / "tmp"
    for d in (raw_dir, aac_dir, tmp_dir):
        d.mkdir(parents=True, exist_ok=True)

    files = HfApi().list_repo_files(FAKE_REPO, repo_type="dataset")
    shards = sorted(f for f in files if f.startswith(f"{lang}/") and f.endswith(".parquet"))
    print(f"  {lang}: {len(shards)} IndicSynth shards in the repository")
    if not shards:
        raise SystemExit(f"No IndicSynth shards found for {lang}.")

    index = scan_language(args, lang, out_dir, shards)
    print("  generators found:", index["model"].value_counts().to_dict())
    found = [g for g in GENERATORS if (index["model"] == g).any()]
    for g in GENERATORS:
        if g not in found:
            print(f"  WARNING: generator '{g}' does not appear in any shard for {lang}; skipping it")
    if not found:
        raise SystemExit("None of the expected generators were found.")
    per_gen = int(np.ceil(args.n_fake / len(found)))
    print(f"  using generators {found}, {per_gen} clips each")

    plan = {}  # shard -> {generator: number of clips wanted}
    for g in found:
        counts = index[index["model"] == g].groupby("shard").size()
        k = min(args.shards_per_gen, len(counts))
        chosen_shards = rng.choice(counts.index.to_numpy(), size=k, replace=False)
        base, extra = divmod(per_gen, k)
        for j, sh in enumerate(chosen_shards):
            plan.setdefault(sh, {})[g] = base + (1 if j < extra else 0)
    print(f"  need {len(plan)} shards (about 0.6 - 1.1 GB each)")

    rows_out = []
    cache_dir = out_dir / "hf_cache"
    for sh, wants in plan.items():
        print(f"  downloading {sh} for {wants} ...")
        local = hf_hub_download(FAKE_REPO, filename=sh, repo_type="dataset", cache_dir=str(cache_dir))
        rows_out += process_shard(local, sh, wants, rng, raw_dir, aac_dir, tmp_dir, bitrate, aac_sr, lang)
        shutil.rmtree(cache_dir, ignore_errors=True)  # free the disk space
    shutil.rmtree(tmp_dir, ignore_errors=True)
    return rows_out


# -------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", nargs="+", default=["Tamil", "Telugu"])
    ap.add_argument("--out", default="indic_data")
    ap.add_argument("--kathbath-tar", help="path to Kathbath valid_audio.tar (clean split)")
    ap.add_argument("--kathbath-dir", help="alternatively: folder with extracted Kathbath .m4a files")
    ap.add_argument("--n-real", type=int, default=200, help="real clips per language")
    ap.add_argument("--n-fake", type=int, default=200,
                    help="fake clips per language, split equally over the generators found")
    ap.add_argument("--max-per-speaker", type=int, default=10)
    ap.add_argument("--scan-shards", type=int, default=24, help="how many shards to scan (0 = all; slow)")
    ap.add_argument("--shards-per-gen", type=int, default=2)
    ap.add_argument("--rescan", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true", help="tiny run to check that everything works")
    args = ap.parse_args()

    if args.smoke:
        args.n_real, args.n_fake, args.scan_shards, args.shards_per_gen = 6, 6, 8, 1
        args.out = args.out + "_smoke"
    if not (args.kathbath_tar or args.kathbath_dir):
        raise SystemExit("Give --kathbath-tar or --kathbath-dir (see the header of this file).")
    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            raise SystemExit(f"{tool} not found on PATH. Install ffmpeg first (check: ffmpeg -version).")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    all_rows = []

    for lang in args.langs:
        print(f"\n===== {lang} =====")
        real_rows, bitrates, srates = build_real(args, lang, out_dir, rng)
        bitrate = int(np.median(bitrates)) if bitrates else 128000
        aac_sr = int(np.median(srates)) if srates else 44100
        print(f"  AAC settings for the fake copies: {bitrate} bit/s at {aac_sr} Hz "
              f"({'measured from the real files' if bitrates else 'fallback defaults'})")
        fake_rows = build_fake(args, lang, out_dir, rng, bitrate, aac_sr)
        all_rows += real_rows + fake_rows
        pd.DataFrame(all_rows).to_csv(out_dir / "indic_files.csv", index=False)

    df = pd.DataFrame(all_rows)
    print("\n================ SUMMARY ================")
    print(df.groupby(["language", "kind", "generator"]).size().to_string())
    print("\nMedian duration (s) by kind:")
    print(df.groupby("kind")["duration_s"].median().round(2).to_string())
    print("\nOriginal sample rates by kind (fake_aac mirrors fake_raw):")
    print(df.groupby(["kind", "orig_sr"]).size().to_string())
    print(f"\nSaved: {out_dir / 'indic_files.csv'}")
    print("Check the duration table: very different lengths between real and fake can be a shortcut.")


if __name__ == "__main__":
    main()
