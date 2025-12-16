#!/usr/bin/env python3
import os
import string
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

OUT_DIR = "/hub_data4/seohyun/hanwha_eval_videos"
os.makedirs(OUT_DIR, exist_ok=True)

# 병렬 worker 수
MAX_WORKERS = 8

# ---- 공통 헤더 기본값 ----
COMMON_HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
    "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7,ru;q=0.6",
    "Cache-Control": "max-age=0",
    "Connection": "keep-alive",
    "Content-Type": "application/x-www-form-urlencoded",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "same-origin",
    "Sec-Fetch-User": "?1",
    "Upgrade-Insecure-Requests": "1",
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36",
}

def get_config_for_part(part: str):
    """
    part: 'aa', 'ab', ... 'bh'
    각 구간별로 url / header / data 를 세팅해주는 함수
    """
    fname = f"eval_videos.tar.gz{part}"

    # 1) aa ~ ag  (맨 처음 bash 스크립트)
    if "aa" <= part <= "ag":
        base_url = "https://ep.circle.hanwha.com/neo/bigmail/modules/mail/downloadMassFileWithCert.mvc"
        id_ = "c0ffbabd-bf20-4ab0-bb2c-2ee43780a977"
        cert = "2912H4"
        cookie = (
            "JSESSIONID_NCP_EP=SToVRoX1a4CUo4sfb8omD4YvlR0FJWItfolXBk5s5UdHVS9Ltk8nfabp1IOSpwGa.Y3BfRVBET01BSU4vYmlnbWFpbDAx; "
            "TS013fc109=0123a3fb00ea552e18e22fb321a2dda6f78d07e745e303b5686e67ce1acea101082f73373f200431a90b619282a5d319b0144648ab; "
            "WMONID=guUF9cxRu5r; "
            "TS01ec8595=0123a3fb00ea552e18e22fb321a2dda6f78d07e745e303b5686e67ce1acea101082f73373f200431a90b619282a5d319b0144648ab"
        )
        headers = {
            **COMMON_HEADERS,
            "Origin": "https://ep.circle.hanwha.com",
            "Referer": f"https://ep.circle.hanwha.com/neo/bigmail/modules/mail/downloadMassFile.mvc?id={id_}&name={fname}",
            "Cookie": cookie,
        }
        data = {
            "dlmType": "file",
            "id": id_,
            "mailId": "",
            "name": fname,
            "name2": "",
            "size": "",
            "certNum": cert,
        }
        return base_url, headers, data

    # 2) ah ~ an  (두 번째 curl 스니펫)
    if "ah" <= part <= "an":
        base_url = "https://ep.circle.hanwha.com/neo/bigmail/modules/mail/downloadMassFileWithCert.mvc"
        id_ = "9b45d044-50bf-498f-91ad-36b24c6cb75e"
        cert = "0169XN"
        cookie = (
            "JSESSIONID_NCP_EP=SToVRoX1a4CUo4sfb8omD4YvlR0FJWItfolXBk5s5UdHVS9Ltk8nfabp1IOSpwGa.Y3BfRVBET01BSU4vYmlnbWFpbDAx; "
            "TS013fc109=0123a3fb00ea552e18e22fb321a2dda6f78d07e745e303b5686e67ce1acea101082f73373f200431a90b619282a5d319b0144648ab; "
            "WMONID=guUF9cxRu5r; "
            "TS01ec8595=0123a3fb0045779a036bb31d84181fc7d06fd155188ea07f597dafdff64bfcab18b8ec86a2e7c6cf398f8e243fd4edae268646cabc"
        )
        headers = {
            **COMMON_HEADERS,
            "Origin": "https://ep.circle.hanwha.com",
            "Referer": f"https://ep.circle.hanwha.com/neo/bigmail/modules/mail/downloadMassFile.mvc?id={id_}&name={fname}",
            "sec-ch-ua": '"Chromium";v="142", "Google Chrome";v="142", "Not_A Brand";v="99"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"macOS"',
            "Cookie": cookie,
        }
        data = {
            "dlmType": "file",
            "id": id_,
            "mailId": "",
            "name": fname,
            "name2": "",
            "size": "",
            "certNum": cert,
        }
        return base_url, headers, data

    # 3) ao ~ au  ← 여기 수정: 이제 zip 말고 file API + 새 cookie/id/cert 사용
    if "ao" <= part <= "au":
        base_url = "https://ep.circle.hanwha.com/neo/bigmail/modules/mail/downloadMassFileWithCert.mvc"
        id_ = "761e201f-2570-4087-990f-7ccfb7f6352d"
        cert = "1A3O9Y"
        cookie = (
            "JSESSIONID_NCP_EP=PNkJA5BAXs0GKjlVBzcQCoI9rGhLNh2qfoJ7n81eFHP1bwZttMSmi7tLDNn2L76R.Y3BfRVBET01BSU4vYmlnbWFpbDAx; "
            "TS013fc109=0123a3fb0045779a036bb31d84181fc7d06fd155188ea07f597dafdff64bfcab18b8ec86a2e7c6cf398f8e243fd4edae268646cabc; "
            "WMONID=guUF9cxRu5r; "
            "TS01ec8595=0123a3fb008cf2e8ec0b8114f97ab92dcd9bd5abd5a869e7fbee8396c08a916c14969f5bdc1b6261d14a2fee93f9d0ceb735804acd"
        )
        headers = {
            **COMMON_HEADERS,
            "Origin": "https://ep.circle.hanwha.com",
            "Referer": f"https://ep.circle.hanwha.com/neo/bigmail/modules/mail/downloadMassFile.mvc?id={id_}&name={fname}",
            "sec-ch-ua": '"Chromium";v="142", "Google Chrome";v="142", "Not_A Brand";v="99"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"macOS"',
            "Cookie": cookie,
        }
        data = {
            "dlmType": "file",
            "id": id_,
            "mailId": "",
            "name": fname,
            "name2": "",
            "size": "",
            "certNum": cert,
        }
        return base_url, headers, data

    # 4) av ~ bb (네 번째 curl 스니펫)
    if "av" <= part <= "bb":
        base_url = "https://ep.circle.hanwha.com/neo/bigmail/modules/mail/downloadMassFileWithCert.mvc"
        id_ = "7f7bec76-14b6-40fc-a66e-04f185016139"
        cert = "VB0FLM"
        cookie = (
            "JSESSIONID_NCP_EP=PNkJA5BAXs0GKjlVBzcQCoI9rGhLNh2qfoJ7n81eFHP1bwZttMSmi7tLDNn2L76R.Y3BfRVBET01BSU4vYmlnbWFpbDAx; "
            "TS013fc109=0123a3fb0045779a036bb31d84181fc7d06fd155188ea07f597dafdff64bfcab18b8ec86a2e7c6cf398f8e243fd4edae268646cabc; "
            "WMONID=guUF9cxRu5r; "
            "TS01ec8595=0123a3fb0045779a036bb31d84181fc7d06fd155188ea07f597dafdff64bfcab18b8ec86a2e7c6cf398f8e243fd4edae268646cabc"
        )
        headers = {
            **COMMON_HEADERS,
            "Origin": "https://ep.circle.hanwha.com",
            "Referer": f"https://ep.circle.hanwha.com/neo/bigmail/modules/mail/downloadMassFile.mvc?id={id_}&name={fname}",
            "sec-ch-ua": '"Chromium";v="142", "Google Chrome";v="142", "Not_A Brand";v="99"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"macOS"',
            "Cookie": cookie,
        }
        data = {
            "dlmType": "file",
            "id": id_,
            "mailId": "",
            "name": fname,
            "name2": "",
            "size": "",
            "certNum": cert,
        }
        return base_url, headers, data

    # 5) bc ~ bh (다섯 번째 curl 스니펫)
    if "bc" <= part <= "bh":
        base_url = "https://ep.circle.hanwha.com/neo/bigmail/modules/mail/downloadMassFileWithCert.mvc"
        id_ = "0e6c97cf-c4d4-4437-99c1-5f1f2db9b016"
        cert = "6QKX42"
        cookie = (
            "JSESSIONID_NCP_EP=PNkJA5BAXs0GKjlVBzcQCoI9rGhLNh2qfoJ7n81eFHP1bwZttMSmi7tLDNn2L76R.Y3BfRVBET01BSU4vYmlnbWFpbDAx; "
            "TS013fc109=0123a3fb0045779a036bb31d84181fc7d06fd155188ea07f597dafdff64bfcab18b8ec86a2e7c6cf398f8e243fd4edae268646cabc; "
            "WMONID=guUF9cxRu5r; "
            "TS01ec8595=0123a3fb0045779a036bb31d84181fc7d06fd155188ea07f597dafdff64bfcab18b8ec86a2e7c6cf398f8e243fd4edae268646cabc"
        )
        headers = {
            **COMMON_HEADERS,
            "Origin": "https://ep.circle.hanwha.com",
            "Referer": f"https://ep.circle.hanwha.com/neo/bigmail/modules/mail/downloadMassFile.mvc?id={id_}&name={fname}",
            "sec-ch-ua": '"Chromium";v="142", "Google Chrome";v="142", "Not_A Brand";v="99"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"macOS"',
            "Cookie": cookie,
        }
        data = {
            "dlmType": "file",
            "id": id_,
            "mailId": "",
            "name": fname,
            "name2": "",
            "size": "",
            "certNum": cert,
        }
        return base_url, headers, data

    raise ValueError(f"part={part} 에 대한 설정이 없음")


def download_one_part(part: str, retry: int = 3, min_size_bytes: int = 1024) -> None:
    fname = f"eval_videos.tar.gz{part}"
    out_path = os.path.join(OUT_DIR, fname)

    base_url, headers, data = get_config_for_part(part)

    for attempt in range(1, retry + 1):
        try:
            print(f"[{part}] ▶ downloading (try {attempt}) ...")
            resp = requests.post(base_url, headers=headers, data=data, timeout=60)
            resp.raise_for_status()

            content = resp.content
            size = len(content)

            if size < min_size_bytes:
                print(f"[{part}] ⚠ size={size} bytes (너무 작음, HTML 에러일 수도 있음)")
            else:
                print(f"[{part}] ✅ downloaded size={size / (1024 * 1024):.2f} MB")

            with open(out_path, "wb") as f:
                f.write(content)

            return
        except Exception as e:
            print(f"[{part}] ❌ error on attempt {attempt}: {e}")
            if attempt < retry:
                time.sleep(2)
            else:
                print(f"[{part}] 🚨 failed after {retry} attempts")
                return


def main():
    parts = []
    for first in ["a", "b"]:
        for second in string.ascii_lowercase:
            part = f"{first}{second}"
            if part > "bh":
                break
            parts.append(part)

    print("총 파트 수:", len(parts), parts[0], "→", parts[-1])

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_part = {
            executor.submit(download_one_part, part): part for part in parts
        }

        for future in as_completed(future_to_part):
            part = future_to_part[future]
            try:
                future.result()
            except Exception as e:
                print(f"[{part}] 🔥 unexpected exception in future: {e}")

    print("🎉 all download tasks finished (성공/실패 여부는 위 로그 참고)")


if __name__ == "__main__":
    main()
