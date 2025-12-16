#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="/hub_data4/seohyun/ecva"
WORKERS="${WORKERS:-64}"
mkdir -p "$OUT_DIR"

# URLS=(
# 'https://cdn-lfs-cn-1.modelscope.cn/prod/lfs-objects/af/59/20f9b5cf257ab5666167a401141294506ad9d1276fbb9286f6ebd11302c6?filename=Video_group_5.zip&namespace=gouchenyi&repository=ECVA&revision=master&tag=dataset&auth_key=1765761129-cf7289b0432047e9af032c63cb2c16a7-0-3e153313335e4ab2321e9cc40a3a9d33'
# 'https://cdn-lfs-cn-1.modelscope.cn/prod/lfs-objects/f1/81/33a4f1a73be2f4a3a1dec2f904b6fd34054c7d984f77b3a8ac87b32191a7?filename=Video_group_4.zip&namespace=gouchenyi&repository=ECVA&revision=master&tag=dataset&auth_key=1765761445-42947c2f1e824975aca96c355d826824-0-5547949aba56108fba00ed9b57107148'
# 'https://cdn-lfs-cn-1.modelscope.cn/prod/lfs-objects/63/44/4d2c330062f19b9ef63ea4309eeeeed54c1c529dec3f5f282d531c1c9e8c?filename=Video_group_3.zip&namespace=gouchenyi&repository=ECVA&revision=master&tag=dataset&auth_key=1765761470-c69e5c60c1f14a2a914ecca6291f5b25-0-88b638810432f171b5f1601b57c8cabc'
# 'https://cdn-lfs-cn-1.modelscope.cn/prod/lfs-objects/ca/8d/1e650a9a2f709875a692c4a9df88de02cbf182fa228df5f619e8d2ee8402?filename=Video_group_2.zip&namespace=gouchenyi&repository=ECVA&revision=master&tag=dataset&auth_key=1765761498-0a5bb7b7ecbd498faf4e092a46bb70f5-0-7ed47fef86182bf60a5d0d27882623fe'
# 'https://cdn-lfs-cn-1.modelscope.cn/prod/lfs-objects/df/b5/a2893ff648b6a864bf33f3cb38e2fa9d01b0d42a9fa137ba9aac25ef13f5?filename=Video_group_1.zip&namespace=gouchenyi&repository=ECVA&revision=master&tag=dataset&auth_key=1765761561-062c6371dd2941389da6c1156f6a5458-0-b79dcc8b16cef5068bf195796a88b5db'
# 'https://cdn-lfs-cn-1.modelscope.cn/prod/lfs-objects/80/58/ce15dff93213ba4f5abd10fd80debb0eb9ddd5030b15ee049f8deb357e3a?filename=Video_group_0.zip&namespace=gouchenyi&repository=ECVA&revision=master&tag=dataset&auth_key=1765761572-391e361c03be4780a07ec880f369416b-0-08d51ed295ee5d0f0544263bde3c9668'
# )

URLS=(
'https://cdn-lfs-cn-1.modelscope.cn/prod/lfs-objects/63/44/4d2c330062f19b9ef63ea4309eeeeed54c1c529dec3f5f282d531c1c9e8c?filename=Video_group_3.zip&namespace=gouchenyi&repository=ECVA&revision=master&tag=dataset&auth_key=1765761470-c69e5c60c1f14a2a914ecca6291f5b25-0-88b638810432f171b5f1601b57c8cabc'
'https://cdn-lfs-cn-1.modelscope.cn/prod/lfs-objects/80/58/ce15dff93213ba4f5abd10fd80debb0eb9ddd5030b15ee049f8deb357e3a?filename=Video_group_0.zip&namespace=gouchenyi&repository=ECVA&revision=master&tag=dataset&auth_key=1765761572-391e361c03be4780a07ec880f369416b-0-08d51ed295ee5d0f0544263bde3c9668'
)

printf '%s\n' "${URLS[@]}" | xargs -n 1 -P "$WORKERS" -I {} bash -c '
  set -euo pipefail
  url="$1"
  out_dir="$2"

  fname="${url##*filename=}"
  fname="${fname%%&*}"

  part="$out_dir/${fname}.part"
  out="$out_dir/$fname"

  echo "==> [$$] downloading $fname"
  curl -L -C - --fail --retry 5 --retry-delay 2 \
    -H "Referer: https://www.modelscope.cn/" \
    -o "$part" \
    "$url"
  mv -f "$part" "$out"
' _ {} "$OUT_DIR"

echo "Done:"
ls -lh "$OUT_DIR"/Video_group_*.zip