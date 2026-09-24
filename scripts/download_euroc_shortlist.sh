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
  local zip="$1" seq="$2"
  if [ -d "$DATA/$seq/mav0/cam0/data" ]; then
    echo "SKIP extract $seq"
    link_seq "$seq"
    return 0
  fi
  echo "Extract $seq from $zip"
  mkdir -p "$DATA/_tmp_$seq"
  # archives nest as <seq>/mav0/...
  unzip -qo "$zip" "${seq}/*" -d "$DATA/_tmp_$seq" || \
    unzip -qo "$zip" "*/${seq}/*" -d "$DATA/_tmp_$seq"
  if [ -d "$DATA/_tmp_$seq/$seq/mav0" ]; then
    mv "$DATA/_tmp_$seq/$seq" "$DATA/$seq"
  elif [ -d "$DATA/_tmp_$seq/mav0" ]; then
    mkdir -p "$DATA/$seq"
    mv "$DATA/_tmp_$seq/mav0" "$DATA/$seq/"
  else
    # find mav0
    found=$(find "$DATA/_tmp_$seq" -type d -name mav0 | head -1)
    [ -n "$found" ] || { echo "FATAL: mav0 not found for $seq"; exit 1; }
    mkdir -p "$DATA/$seq"
    mv "$found" "$DATA/$seq/"
  fi
  rm -rf "$DATA/_tmp_$seq"
  test -d "$DATA/$seq/mav0/cam0/data" || { echo "FATAL: bad extract $seq"; exit 1; }
  link_seq "$seq"
  echo "OK $seq"
}

[ -f "$HF_DIR/vicon_room1.zip" ] && extract_one "$HF_DIR/vicon_room1.zip" V1_02_medium
[ -f "$HF_DIR/vicon_room2.zip" ] && extract_one "$HF_DIR/vicon_room2.zip" V2_01_easy
if [ -f "$HF_DIR/machine_hall.zip" ]; then
  extract_one "$HF_DIR/machine_hall.zip" MH_01_easy
  extract_one "$HF_DIR/machine_hall.zip" MH_02_easy
fi

echo "EUROC_DOWNLOAD_DONE"
ls -la "$EXT_LINK"
