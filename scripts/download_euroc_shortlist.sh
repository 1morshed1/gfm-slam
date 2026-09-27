#!/usr/bin/env bash
# Download EuRoC shortlist via HuggingFace GlowBond mirror (ETHZ often unreachable).
# Extracts only the sequences we need from room archives.
set -eo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA="$ROOT/data/euroc"
HF_DIR="$DATA/_hf"
EXT_LINK="$ROOT/ext/MASt3R-SLAM/datasets/euroc"
mkdir -p "$DATA" "$HF_DIR" "$EXT_LINK"

CONDA_ROOT="${CONDA_ROOT:-/office/dev_workspace/morshed/miniconda3}"
set +u
eval "$("${CONDA_ROOT}/bin/conda" shell.bash hook)"
conda activate edgeslam
set -u

need_seq() {
  local seq="$1"
  [ -d "$DATA/$seq/mav0/cam0/data" ] && return 1
  return 0
}

link_seq() {
  local seq="$1"
  ln -sfn "$DATA/$seq" "$EXT_LINK/$seq"
}

# already have V1_01
link_seq V1_01_easy

# Decide which room zips we need
NEED_V1=0 NEED_V2=0 NEED_MH=0
need_seq V1_02_medium && NEED_V1=1
need_seq V2_01_easy && NEED_V2=1
need_seq MH_01_easy && NEED_MH=1
need_seq MH_02_easy && NEED_MH=1

cd "$HF_DIR"
if [ "$NEED_V1" = 1 ] && [ ! -f vicon_room1.zip ]; then
  echo "HF download vicon_room1.zip (~6GB)"
  hf download GlowBond/EuRoC_MAV_Dataset vicon_room1.zip --repo-type dataset --local-dir .
fi
if [ "$NEED_V2" = 1 ] && [ ! -f vicon_room2.zip ]; then
  echo "HF download vicon_room2.zip (~6GB)"
  hf download GlowBond/EuRoC_MAV_Dataset vicon_room2.zip --repo-type dataset --local-dir .
fi
if [ "$NEED_MH" = 1 ] && [ ! -f machine_hall.zip ]; then
  echo "HF download machine_hall.zip (~12GB)"
  hf download GlowBond/EuRoC_MAV_Dataset machine_hall.zip --repo-type dataset --local-dir .
fi

extract_one() {
  local zip="$1" seq="$2" room_prefix="$3"
  if [ -d "$DATA/$seq/mav0/cam0/data" ]; then
    echo "SKIP extract $seq"
    link_seq "$seq"
    return 0
  fi
  echo "Extract nested $seq from $zip"
  python3 - <<PY
import zipfile, shutil
from pathlib import Path
zip_path = Path("$zip")
seq = "$seq"
room = "$room_prefix"
DATA = Path("$DATA")
inner_name = f"{room}/{seq}/{seq}.zip"
dest = DATA / seq
tmp = DATA / f"_tmp_{seq}"
if tmp.exists():
    shutil.rmtree(tmp)
tmp.mkdir()
with zipfile.ZipFile(zip_path) as outer:
    data = outer.read(inner_name)
inner = DATA / f"_{seq}.zip"
inner.write_bytes(data)
with zipfile.ZipFile(inner) as zf:
    zf.extractall(tmp)
inner.unlink(missing_ok=True)
mavs = [p for p in tmp.rglob("mav0") if p.is_dir()]
assert mavs, f"no mav0 for {seq}"
dest.mkdir(parents=True, exist_ok=True)
if (dest / "mav0").exists():
    shutil.rmtree(dest / "mav0")
shutil.move(str(mavs[0]), str(dest / "mav0"))
shutil.rmtree(tmp, ignore_errors=True)
assert (dest / "mav0/cam0/data").is_dir()
print(f"OK {seq}")
PY
  link_seq "$seq"
}

if [ -f "$HF_DIR/vicon_room1.zip" ]; then
  extract_one "$HF_DIR/vicon_room1.zip" V1_02_medium vicon_room1
  extract_one "$HF_DIR/vicon_room1.zip" V1_03_difficult vicon_room1
fi
if [ -f "$HF_DIR/vicon_room2.zip" ]; then
  extract_one "$HF_DIR/vicon_room2.zip" V2_01_easy vicon_room2
  extract_one "$HF_DIR/vicon_room2.zip" V2_02_medium vicon_room2
  extract_one "$HF_DIR/vicon_room2.zip" V2_03_difficult vicon_room2
fi
if [ -f "$HF_DIR/machine_hall.zip" ]; then
  extract_one "$HF_DIR/machine_hall.zip" MH_01_easy machine_hall
  extract_one "$HF_DIR/machine_hall.zip" MH_02_easy machine_hall
  extract_one "$HF_DIR/machine_hall.zip" MH_03_medium machine_hall
  extract_one "$HF_DIR/machine_hall.zip" MH_04_difficult machine_hall
  extract_one "$HF_DIR/machine_hall.zip" MH_05_difficult machine_hall
fi

echo "EUROC_DOWNLOAD_DONE"
ls -la "$EXT_LINK"
