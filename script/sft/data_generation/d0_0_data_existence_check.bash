#!/usr/bin/env bash

set -euo pipefail

# Check and filter entries in a JSON manifest to only those whose
# referenced video files exist AND are decodable (via decord).
# The output overwrites the same JSON by default.

# Ensure environment has decord if available
if [[ -z "${CONDA_PREFIX:-}" || "${CONDA_DEFAULT_ENV:-}" != "video-colbert" ]]; then
  if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate video-colbert || true
  fi
fi

JSON_PATH="${JSON_PATH:-/home/seohyun/vid_understanding/video_retrieval/data/anet_sampled.json}"
VIDEO_BASE="${VIDEO_BASE:-/hub_data2/dohwan/data/retrieval/activitynet/videos}"
OUTPUT_JSON="${OUTPUT_JSON:-${JSON_PATH}}"

# Print missing/unreadable filenames
LIST_MISSING="${LIST_MISSING:-false}"

# Decode validation controls
VALIDATE_DECODE="${VALIDATE_DECODE:-true}"
TEST_FRAMES="${TEST_FRAMES:-3}"

# Parallelism
WORKERS="${WORKERS:-0}"  # 0 = auto(cpu_count)

echo "JSON_PATH       : ${JSON_PATH}"
echo "VIDEO_BASE      : ${VIDEO_BASE}"
echo "OUTPUT_JSON     : ${OUTPUT_JSON}"
echo "VALIDATE_DECODE : ${VALIDATE_DECODE} (frames=${TEST_FRAMES})"
echo "WORKERS         : ${WORKERS} (0=auto)"

python3 - "$JSON_PATH" "$VIDEO_BASE" "$OUTPUT_JSON" << 'PY'
import json, os, sys
from concurrent.futures import ThreadPoolExecutor

json_path = sys.argv[1]
video_base = sys.argv[2]
output_json = sys.argv[3]

with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

if not isinstance(data, list):
    print(f"❌ Expected list at {json_path}", file=sys.stderr)
    sys.exit(1)

validate_decode = os.environ.get('VALIDATE_DECODE', 'true').lower() == 'true'
test_frames = int(os.environ.get('TEST_FRAMES', '3'))
list_missing = os.environ.get('LIST_MISSING', 'false').lower() == 'true'
workers_env = int(os.environ.get('WORKERS', '0') or 0)
workers = workers_env if workers_env > 0 else (os.cpu_count() or 4)

try:
    from decord import VideoReader
    import numpy as np  # noqa: F401
    HAVE_DECORD = True
except Exception as e:
    HAVE_DECORD = False
    decord_err = e

try:
    import cv2  # type: ignore
    HAVE_CV2 = True
except Exception:
    HAVE_CV2 = False

seen = set()
unique_entries = []
for entry in data:
    video = entry.get('video')
    if not isinstance(video, str):
        continue
    v = video.strip()
    if not v:
        continue
    if v in seen:
        continue
    seen.add(v)
    unique_entries.append(entry)

kept = []
missing = []
unreadable = []
rewritten = []  # (old -> new) when extension fallback works

def can_decode(path: str) -> bool:
    if not validate_decode:
        return True
    if not HAVE_DECORD:
        # If decord is unavailable, try OpenCV fallback; otherwise skip validation
        if HAVE_CV2:
            try:
                cap = cv2.VideoCapture(path)
                ok = cap.isOpened()
                if not ok:
                    return False
                # attempt to read one frame
                ok, _ = cap.read()
                cap.release()
                return bool(ok)
            except Exception as e:
                print(f"[fail] cv2 decode {path}: {e}", file=sys.stderr)
                return False
        print("[warn] decord not available; skipping decode validation", file=sys.stderr)
        return True
    try:
        vr = VideoReader(path)
        n = len(vr)
        if n <= 0:
            return False
        # Probe a few frames across the video
        k = max(1, min(test_frames, n))
        import numpy as _np
        idxs = [int(round(i)) for i in _np.linspace(0, n - 1, k)]
        try:
            _ = vr.get_batch(idxs)
        except Exception:
            for i in idxs:
                _ = vr[i]
        return True
    except Exception as e:
        print(f"[fail] decode {path}: {e}", file=sys.stderr)
        return False

def _alt_candidates(v: str):
    v_lower = v.lower()
    cands = []
    # Prefer mp4 <-> mkv swap
    if v_lower.endswith('.mp4'):
        cands.append(v[:-4] + '.mkv')
    elif v_lower.endswith('.mkv'):
        cands.append(v[:-4] + '.mp4')
    return cands

def validate_entry(entry):
    video = entry.get('video')
    v = (video or '').strip()
    local_path = os.path.join(video_base, v)

    exists = os.path.exists(local_path)
    if exists and can_decode(local_path):
        return ('ok', v, entry)

    # Try extension fallback (mp4 <-> mkv)
    for alt in _alt_candidates(v):
        alt_path = os.path.join(video_base, alt)
        if os.path.exists(alt_path) and can_decode(alt_path):
            new_entry = dict(entry)
            new_entry['video'] = alt
            return ('ok_rewrite', v + ' -> ' + alt, new_entry)

    # If original exists but cannot decode and no valid fallback
    if exists:
        return ('unreadable', v, entry)
    # Otherwise fully missing
    return ('missing', v, entry)

with ThreadPoolExecutor(max_workers=max(1, int(workers))) as ex:
    for status, v, entry in ex.map(validate_entry, unique_entries, chunksize=1):
        if status == 'ok':
            kept.append(entry)
        elif status == 'ok_rewrite':
            kept.append(entry)
            rewritten.append(v)
        elif status == 'missing':
            missing.append(v)
        else:
            unreadable.append(v)

os.makedirs(os.path.dirname(output_json), exist_ok=True)
with open(output_json, 'w', encoding='utf-8') as f:
    json.dump(kept, f, ensure_ascii=False, indent=2)

msg = (
    f"Total: {len(data)} | Unique: {len(unique_entries)} | Exists+Decodable: {len(kept)} | Missing: {len(missing)} | Unreadable: {len(unreadable)} | Rewritten: {len(rewritten)}"
)
print(msg, file=sys.stderr)

if list_missing:
    if missing:
        print("\n# Missing files:")
        for m in missing:
            print(m)
    if unreadable:
        print("\n# Unreadable files:")
        for m in unreadable:
            print(m)
    if rewritten:
        print("\n# Rewritten (mp4<->mkv):")
        for m in rewritten:
            print(m)
PY

echo "✅ Wrote filtered JSON to ${OUTPUT_JSON}"
