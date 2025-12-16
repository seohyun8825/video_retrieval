from huggingface_hub import HfApi

NEW_REPO_ID = "happy8825/valid_ecva_fixed"
REPO_TYPE = "dataset"
LOCAL_JSONL_PATH = "/home/seohyun/vid_understanding/video_retrieval/data/abnormal_sft.jsonl"  # gt 포함된 최신 파일

api = HfApi()

# 1️⃣ 레포 생성 (이미 있으면 그냥 통과)
api.create_repo(
    repo_id=NEW_REPO_ID,
    repo_type=REPO_TYPE,
    exist_ok=True,
)

# 2️⃣ data.jsonl 업로드
api.upload_file(
    path_or_fileobj=LOCAL_JSONL_PATH,
    path_in_repo="data.jsonl",
    repo_id=NEW_REPO_ID,
    repo_type=REPO_TYPE,
    commit_message="Initial upload: data.jsonl with gt field",
)

print(f"✅ Uploaded to https://huggingface.co/datasets/{NEW_REPO_ID}")