"""
YouTube 댓글 수집기 (닉네임/시간 제외, 내용만 저장)

동작:
1) yt-dlp로 댓글 메타를 JSON으로 수집(--write-comments)
2) JSON의 'comments' 항목에서 본문(text)과 대댓글(replies[].text)만 추출
3) comments/{video_id}.comments.txt 로 저장

필요: yt-dlp 설치
"""

import json
import re
import subprocess
import sys
from pathlib import Path


def extract_video_id(url_or_id: str) -> str:
    """URL 또는 11자리 영상 ID에서 videoId 추출"""
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", url_or_id):
        return url_or_id
    from urllib.parse import urlparse, parse_qs
    p = urlparse(url_or_id)
    if p.netloc:
        q = parse_qs(p.query)
        return q.get("v", [""])[0]
    return ""


def run_yt_dlp_comments(url: str, work_dir: Path, use_cookies: bool = True) -> Path | None:
    """yt-dlp로 댓글 JSON(.info.json)을 생성하고 그 경로를 반환"""
    work_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "yt-dlp",
        "--skip-download",
        "--write-comments",
        "-o",
        "%(id)s.%(ext)s",
        url,
    ]

    cookies = Path.cwd() / "cookies.txt"
    if use_cookies and cookies.exists():
        cmd.extend(["--cookies", str(cookies)])

    # 조용한 출력
    result = subprocess.run(cmd, cwd=work_dir, capture_output=True)
    if result.returncode != 0:
        print("오류: yt-dlp 실행 실패")
        if result.stderr:
            try:
                print(result.stderr.decode("utf-8", errors="ignore"))
            except Exception:
                pass
        return None

    # 생성된 info.json 탐색
    candidates = sorted(work_dir.glob("*.info.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def extract_comment_texts(info_json_path: Path) -> list[str]:
    """yt-dlp info.json에서 댓글과 대댓글의 본문만 추출하여 리스트로 반환"""
    try:
        with open(info_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print("오류: JSON 로드 실패:", e)
        return []

    out: list[str] = []
    comments = data.get("comments") or []
    for c in comments:
        text = (c.get("text") or "").strip()
        if text:
            out.append(text)
        # 대댓글
        replies = c.get("replies") or []
        for r in replies:
            r_text = (r.get("text") or "").strip()
            if r_text:
                out.append(r_text)
    return out


def save_comments(video_id: str, texts: list[str], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{video_id}.comments.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        for line in texts:
            # 닉/시간 제거는 yt-dlp JSON에서 본문만 가져오므로 추가 처리 불필요
            # 안전을 위해 줄바꿈/캐리지 리턴은 공백으로 정규화
            f.write(line.replace("\r", " ").replace("\n", " ") + "\n")
    return out_path


def main():
    print("=== YouTube 댓글 수집기 ===")
    url = input("유튜브 URL 또는 video ID 입력: ").strip()

    video_id = extract_video_id(url)
    if not video_id:
        print("오류: 유효한 video ID를 추출할 수 없습니다.")
        sys.exit(1)

    base = Path.cwd()
    tmp_dir = base / "_comments_tmp"
    out_dir = base / "comments"

    info_json = run_yt_dlp_comments(f"https://www.youtube.com/watch?v={video_id}", tmp_dir)
    if not info_json:
        print("오류: 댓글 정보를 가져오지 못했습니다.")
        sys.exit(2)

    texts = extract_comment_texts(info_json)
    if not texts:
        print("경고: 추출된 댓글이 없습니다.")
    out_path = save_comments(video_id, texts, out_dir)
    print(f"[완료] 댓글 저장: {out_path}")


if __name__ == "__main__":
    main()


