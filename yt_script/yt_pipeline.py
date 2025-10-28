"""
YouTube → 오디오 → Whisper 전사 → 키워드 추출 + 요약
완전한 파이프라인을 하나의 파일로 구성
"""

import re
import json
import subprocess
import sys
from pathlib import Path

try:
    import whisper
except ImportError:
    print("Whisper가 설치되지 않았습니다: pip install openai-whisper")
    sys.exit(1)

# 키워드 추출 로직 제거됨

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

def download_audio(video_id: str, audio_dir: Path):
    """yt-dlp로 오디오 다운로드 (audio_dir에 저장)"""
    print("\n[1단계] 오디오 다운로드 중...")

    audio_dir.mkdir(parents=True, exist_ok=True)

    # audio_dir을 작업 폴더로 설정하여 파일이 해당 폴더에 저장되도록 함
    result = subprocess.run([
        "yt-dlp",
        "-f", "ba",  # best audio
        "-o", "%(id)s.%(ext)s",
        "--quiet",
        "--no-warnings",
        f"https://www.youtube.com/watch?v={video_id}"
    ], capture_output=True, cwd=audio_dir)

    if result.returncode != 0:
        print("오류: 오디오 다운로드 실패")
        sys.exit(1)

    # 오디오 파일 찾기
    for ext in [".m4a", ".webm", ".opus", ".mp3"]:
        audio_path = audio_dir / f"{video_id}{ext}"
        if audio_path.exists():
            print(f"[OK] 오디오 다운로드: {audio_path}")
            return audio_path

    print("오류: 오디오 파일을 찾을 수 없습니다")
    sys.exit(1)

def whisper_transcribe(audio_path: Path, scripts_dir: Path, model_name: str = "small"):
    """Whisper로 한국어 전사 (scripts_dir에 저장)"""
    print(f"\n[2단계] Whisper 전사 (모델: {model_name})...")
    
    try:
        print("모델 로딩 중...")
        model = whisper.load_model(model_name)
        
        print("전사 중...")
        result = model.transcribe(
            str(audio_path), 
            language="ko", 
            verbose=False,
            fp16=False
        )
        
        scripts_dir.mkdir(parents=True, exist_ok=True)
        video_id = audio_path.stem
        txt_path = scripts_dir / f"{video_id}.txt"
        text = result.get("text", "").strip()
        
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)
        
        print(f"[OK] Whisper 전사 저장: {txt_path.name}")
        return txt_path, text
    
    except FileNotFoundError as e:
        print("\n오류: ffmpeg를 찾을 수 없습니다.")
        print("해결: conda install -c conda-forge ffmpeg")
        sys.exit(1)
    except Exception as e:
        print(f"\n오류: Whisper 전사 실패 - {e}")
        sys.exit(1)

# 키워드 추출 단계는 제거되었습니다.

def extract_summary(text: str, max_sentences: int = 5):
    """핵심 문장 추출 (일반 리스트)"""
    print(f"\n[3단계] 요약 추출 중...")

    sentences = re.split(r'[.!?。\n]+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 30]

    # 길이 기준 상위 문장 선택
    sentences.sort(key=len, reverse=True)
    summary = sentences[:max_sentences]

    print(f"[OK] {max_sentences}개 핵심 문장 추출 완료")
    return summary

def build_structured_summary(text: str):
    """지침 템플릿(방송 보도 요약)용 문장 분류"""
    # 후보 문장 풀
    sentences = re.split(r'[.!?。\n]+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

    # 분류 키워드
    kw_overview = ["사건", "확인", "발생", "출입", "발표", "단독", "보도", "체류", "입장", "들어갔", "방문"]
    kw_sayings = ["밝혔", "주장", "반박", "입장", "말했", "설명", "해명", "지적"]
    kw_context = ["배경", "맥락", "경위", "정황", "집중", "이전", "과거", "관련", "연관"]
    kw_result = ["결과", "영향", "조치", "계획", "예정", "전망", "추가", "후속", "조사", "확인"]

    def pick(cands, kws):
        for s in cands:
            if any(k in s for k in kws):
                return s
        return cands[0] if cands else ""

    # 길이 기준 상위 12문장 내에서 선정
    cands = sorted(sentences, key=len, reverse=True)[:12]

    structured = {
        "overview": pick(cands, kw_overview),
        "sayings": pick(cands, kw_sayings),
        "context": pick(cands, kw_context),
        "result": pick(cands, kw_result),
    }

    # 중복 제거
    seen = set()
    for k, v in structured.items():
        if v in seen:
            # 대체 후보로 길이 순 리스트에서 다음 문장 선택
            alt = next((s for s in cands if s not in seen), "")
            structured[k] = alt
            seen.add(alt)
        else:
            seen.add(v)

    return structured

def build_editorial_summary(text: str):
    """방송사 보도체 요약 포맷 생성 (문단 + 불릿 5개)"""
    s = build_structured_summary(text)

    # 문단 요약 구성: 개요 → 발언/입장 → 배경 → 결과/계획 순
    parts = [s.get("overview", ""), s.get("sayings", ""), s.get("context", ""), s.get("result", "")]
    paragraph = " ".join(p for p in parts if p)
    paragraph = paragraph.strip()

    # 길이 조정: 350~500자 목표
    if len(paragraph) > 500:
        paragraph = paragraph[:500].rstrip() + ""
    elif len(paragraph) < 350:
        # 부족하면 원문에서 긴 문장 보충
        extra = [t for t in re.split(r'[.!?。\n]+', text) if len(t.strip()) > 40]
        for e in extra:
            if e not in paragraph:
                paragraph = (paragraph + " " + e.strip()).strip()
                if len(paragraph) >= 350:
                    break
        if len(paragraph) > 500:
            paragraph = paragraph[:500].rstrip()

    bullets = {
        "summary": s.get("overview", ""),
        "saying": s.get("sayings", ""),
        "party": s.get("context", ""),
        "allegation": s.get("result", ""),
        "next": s.get("result", ""),
    }

    return paragraph, bullets

def tokenize_ko_en(text: str) -> list[str]:
    ko = re.findall(r"[가-힣]{2,}", text)
    en = re.findall(r"[A-Za-z]{3,}", text)
    num = re.findall(r"\d{2,}", text)
    return ko + [w.lower() for w in en] + num

def relevance_score(comment: str, query_terms: set[str]) -> float:
    terms = set(tokenize_ko_en(comment))
    if not terms or not query_terms:
        return 0.0
    inter = terms & query_terms
    # 가중치: 교집합 크기 + 길이 보정(아주 짧은 댓글 가중치 낮음)
    return float(len(inter)) + min(len(comment) / 500.0, 0.5)

def main():
    print("=" * 60)
    print("YouTube → 오디오 → Whisper → 키워드 + 요약")
    print("=" * 60)
    
    url = input("\n유튜브 URL 또는 video ID 입력: ").strip()
    
    video_id = extract_video_id(url)
    if not video_id:
        print("오류: 유효한 video ID를 추출할 수 없습니다.")
        sys.exit(1)
    
    print(f"\n영상 ID: {video_id}")
    
    # 출력 폴더 구성
    base_dir = Path.cwd()
    audio_dir = base_dir / "audio"
    scripts_dir = base_dir / "scripts"
    summaries_dir = base_dir / "summaries"
    comments_dir = base_dir / "comments"

    # 1. 오디오 다운로드
    audio_path = download_audio(video_id, audio_dir)
    
    # 2. Whisper 전사
    txt_path, text = whisper_transcribe(audio_path, scripts_dir)
    
    # 3. 요약 추출
    summary = extract_summary(text, max_sentences=5)
    
    # 5. 결과 출력 및 저장
    print("\n" + "=" * 60)
    print("결과 요약")
    print("=" * 60)

    paragraph, bullets = build_editorial_summary(text)

    print("\n------------------------------------")
    print("[요약문]")
    print(paragraph)
    print("\n[핵심 포인트]")
    print(f"- (사건 요약) {bullets.get('summary','')}")
    print(f"- (핵심 인물 발언) {bullets.get('saying','')}")
    print(f"- (정당/기관 입장) {bullets.get('party','')}")
    print(f"- (의혹 또는 반박 내용) {bullets.get('allegation','')}")
    print(f"- (향후 대응/전망) {bullets.get('next','')}")
    print("------------------------------------")
    
    # 결과 저장 (summaries 폴더)
    summaries_dir.mkdir(parents=True, exist_ok=True)
    output_file = summaries_dir / f"{video_id}.summary.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("------------------------------------\n")
        f.write("[요약문]\n")
        f.write(paragraph + "\n\n")
        f.write("[핵심 포인트]\n")
        f.write(f"- (사건 요약) {bullets.get('summary','')}\n")
        f.write(f"- (핵심 인물 발언) {bullets.get('saying','')}\n")
        f.write(f"- (정당/기관 입장) {bullets.get('party','')}\n")
        f.write(f"- (의혹 또는 반박 내용) {bullets.get('allegation','')}\n")
        f.write(f"- (향후 대응/전망) {bullets.get('next','')}\n")
        f.write("------------------------------------\n")
    
    print(f"\n[완료] 결과 저장: {output_file.name}")
    print(f"[참고] 오디오 파일 '{audio_path.name}'이 남아있습니다.")

    # 4. 댓글 수집 및 저장
    print("\n[부가] 댓글 수집 중...")
    comments_dir.mkdir(parents=True, exist_ok=True)

    # yt-dlp로 댓글 JSON 생성
    def run_yt_dlp_comments(url: str, work_dir: Path, use_cookies: bool = True) -> Path | None:
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
        res = subprocess.run(cmd, cwd=work_dir, capture_output=True)
        if res.returncode != 0:
            return None
        cands = sorted(work_dir.glob("*.info.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        return cands[0] if cands else None

    tmp_dir = base_dir / "_comments_tmp"
    info_json = run_yt_dlp_comments(f"https://www.youtube.com/watch?v={video_id}", tmp_dir)
    if info_json:
        try:
            with open(info_json, "r", encoding="utf-8") as f:
                data = json.load(f)
            raw_texts: list[str] = []
            for c in (data.get("comments") or []):
                t = (c.get("text") or "").strip()
                if t:
                    raw_texts.append(t)
                for r in (c.get("replies") or []):
                    rt = (r.get("text") or "").strip()
                    if rt:
                        raw_texts.append(rt)

            # 요약문 기반 관련도 정렬
            query_str = paragraph + " " + " ".join(bullets.values())
            query_terms = set(tokenize_ko_en(query_str))
            scored = sorted(
                ((txt, relevance_score(txt, query_terms)) for txt in raw_texts),
                key=lambda x: x[1], reverse=True
            )
            texts = [t for t, _ in scored]

            out_comments = comments_dir / f"{video_id}.comments.txt"
            with open(out_comments, "w", encoding="utf-8") as f:
                for line in texts:
                    f.write(line.replace("\r", " ").replace("\n", " ") + "\n")
            print(f"[완료] 댓글 저장(관련도 정렬): {out_comments}")
        except Exception:
            print("[경고] 댓글 저장에 실패했습니다.")
    else:
        print("[경고] 댓글 정보를 가져오지 못했습니다.") 

if __name__ == "__main__":
    main()

