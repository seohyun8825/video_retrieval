#!/usr/bin/env bash
set -euo pipefail

cd /hub_data4/seohyun/ucf

URL='https://uc4d4f324b782f47c08bf4ac3366.dl.dropboxusercontent.com/zip_download_get/CafKl4w4Fs0Va8K0xqc0RUhMQ8rgzmp_ugV9PdMiuk-vlSYh-NBHov9-vxEe7F0ALy2-QJXVNUnqi3vAezFMy3Z3S1nY1NkJcL6mLF6rVPSttA?_download_id=967549854280791819796257634152825041448212928678956024049082895725&_log_download_success=1&_notify_domain=www.dropbox.com&dl=1'

OUT='dropbox_folder.zip'
rm -f "$OUT"


curl -L --fail \
  --retry 5 --retry-delay 2 \
  "$URL" \
  -H 'accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7' \
  -H 'accept-language: ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7,ru;q=0.6' \
  -H 'priority: u=0, i' \
  -H 'referer: https://www.dropbox.com/' \
  -H 'sec-ch-ua: "Google Chrome";v="143", "Chromium";v="143", "Not A(Brand";v="24"' \
  -H 'sec-ch-ua-mobile: ?0' \
  -H 'sec-ch-ua-platform: "macOS"' \
  -H 'sec-fetch-dest: iframe' \
  -H 'sec-fetch-mode: navigate' \
  -H 'sec-fetch-site: cross-site' \
  -H 'sec-fetch-storage-access: active' \
  -H 'sec-fetch-user: ?1' \
  -H 'upgrade-insecure-requests: 1' \
  -H 'user-agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36' \
  -o "$OUT"

echo "[+] Downloaded: $(ls -lh "$OUT")"
file "$OUT" || true
xxd -l 4 "$OUT" || true

# zip 확인: PK로 시작해야 정상 zip
if ! xxd -l 2 "$OUT" | grep -qi "50 4b"; then
  echo "[!] Not a ZIP (likely HTML/error). Showing first 200 bytes:"
  head -c 200 "$OUT" | cat
  echo
  echo "[!] HTTP headers:"
  curl -I -L "$URL" | head -n 40
  exit 1
fi

echo "[+] ZIP looks valid. Testing..."
unzip -t "$OUT" | head

mkdir -p dropbox_folder
unzip -q "$OUT" -d dropbox_folder
echo "extracted to: /hub_data4/seohyun/ucf/dropbox_folder"
