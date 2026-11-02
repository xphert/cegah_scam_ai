import base64
import json
import mimetypes
import os
import sys
import time
import requests
from typing import Any, Dict, Optional

# Library UI terminal
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

console = Console()

# ============================================================
# CONFIG — CEGAHSCAM AI V3
# ============================================================
API_URL = "https://openrouter.ai/api/v1/chat/completions"
BEARER_TOKEN = os.getenv("OPENROUTER_API_KEY", "").strip()

# Endpoint API Tools (FastAPI)
TOOLS_API_BASE = "http://127.0.0.1:8000/api/v1"

# Konfigurasi Model
MODEL = "nvidia/nemotron-3.5-super-vl:free"
# Kontrol Respon AI
TEMPERATURE = 0.1            # Tingkat kreativitas output (0.1 = sangat deterministik/konsisten, ideal untuk format JSON)
MAX_TOKENS = 6000            # Batas maksimum jumlah token/kata yang boleh dihasilkan AI dalam satu jawaban
TOP_P = 1.0                  # Nucleus sampling (1.0 = AI mempertimbangkan seluruh opsi kata yang relevan)

# Reasoning Config (Konfigurasi Proses Berpikir AI)
REASONING_ENABLED = True     # Mengaktifkan fitur "extended thinking/reasoning" agar AI menganalisis secara mendalam
REASONING_EFFORT = "high"    # Tingkat kedalaman berpikir AI ("low", "medium", atau "high" untuk analisis paling teliti)
REASONING_EXCLUDE = True     # Menyembunyikan teks proses berpikir internal agar output bersih dan hanya berisi JSON

# Network, Retry, & Payload Controls
STREAM_OUTPUT = True         # Mengirimkan teks respon secara bertahap (real-time/streaming) saat AI mengetik
MAX_RETRIES = 3              # Batas maksimal percobaan ulang (retry) jika request API ke OpenRouter gagal/timeout
RETRY_BACKOFF_SECONDS = 1.5  # Jeda waktu (dalam detik) sebelum sistem mencoba melakukan request ulang
MAX_IMAGE_BYTES = 15 * 1024 * 1024  # Batas maksimum ukuran file gambar yang diizinkan untuk diunggah (15 MB)

KEYWORDS_FILE = "scam_keywords.json"

# ============================================================
# LOAD KNOWLEDGE BASE (JSON)
# ============================================================
def load_scam_keywords() -> list:
    if os.path.exists(KEYWORDS_FILE):
        try:
            with open(KEYWORDS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            console.print(f"[bold red]Gagal membaca {KEYWORDS_FILE}: {e}[/bold red]")
            return []
    return []

# ============================================================
# PROMPTS
# ============================================================
# Prompt Analisis Normal dengan Output Bilingual JSON persis versi V2
COMBINED_REGULAR_PROMPT = """
You are the final fraud/scam analysis engine specialized in Indonesia.
You will receive an image (if provided) and user text. Treat them as ONE combined multimodal context.

INPUT CONTEXT:
1. User Text: {user_text}
2. Checked Phone Number Info (from API): {phone_info}
3. Scam Keywords Knowledge Base: {kb_keywords}
4. Official Institutions DB (White List): {db_institutions}

TASK:
Analyze all evidence. 
1. Determine if scam indicators exist based on the image, text, phone info, and KB.
2. Cross-check any claimed organization against the Official Institutions DB. If not in DB or unverified, set related_institution values to "Unknown" or "N/A".
3. Extract scammer contact details from the input/phone info into "scammer_information".
4. Assign a Risk Score (0-100).
5. Provide prevention steps and indicators.

Return ONLY valid JSON with exactly this structure (no markdown fences, no text outside JSON):
{{
  "en": {{
    "title": "Scam title in English",
    "type": "Scam type",
    "score": 0,
    "description": "Description in English",
    "indicators": ["Indicator 1", "Indicator 2"],
    "prevention": "Step-by-step prevention in English; use \\n for new lines",
    "related_institution": {{
      "name": "Name or Unknown",
      "contact": "Verified official contact or N/A",
      "website": "Verified official website or N/A",
      "social": []
    }},
    "scammer_information": [
      {{
        "contact_name": "Name or Unknown",
        "contact_number": "Number or Unknown",
        "status": "scam, suspicious, unknown, or normal",
        "scam_score": "high, medium, low, or unknown"
      }}
    ]
  }},
  "id": {{
    "title": "Judul penipuan dalam bahasa Indonesia",
    "type": "Jenis scam",
    "score": 0,
    "description": "Deskripsi dalam bahasa Indonesia",
    "indicators": ["Indikator 1", "Indikator 2"],
    "prevention": "Langkah pencegahan dalam bahasa Indonesia; gunakan \\n untuk baris baru",
    "related_institution": {{
      "name": "Nama atau Unknown",
      "contact": "Kontak resmi terverifikasi atau N/A",
      "website": "Situs resmi terverifikasi atau N/A",
      "social": []
    }},
    "scammer_information": [
      {{
        "contact_name": "Nama atau Unknown",
        "contact_number": "Nomor atau Unknown",
        "status": "scam, suspicious, unknown, atau normal",
        "scam_score": "high, medium, low, atau unknown",
        "other": "info lain jika ada"
      }}
    ]
  }}
}}
"""

# Prompt Analisis Darurat
EMERGENCY_PROMPT = """
You are the Emergency Fraud Mitigation AI.
The user has experienced a HIGH RISK scam (Score: {score}).

EMERGENCY INPUTS:
- Regular Analysis Result: {regular_result}
- User's Emergency Data: {emergency_data}

TASK:
Based on the specific emergency data and the regular analysis result, dynamically generate a highly relevant and detailed emergency action plan (SOP Mitigasi) tailored to this EXACT situation. Do NOT use generic advice if specific steps can be deduced from the context.
Generate this detailed emergency action plan and a draft police/platform report.

Return ONLY valid JSON with exactly this structure (no markdown fences):
{{
  "emergency_steps": ["Langkah mitigasi spesifik 1", "Langkah mitigasi spesifik 2"],
  "draft_laporan_markdown": "Isi draf laporan kronologis dalam format Markdown untuk dilapor ke polisi/platform. Gunakan placeholder [..] jika ada data yang kurang.",
  "kontak_darurat_rekomendasi": "Instansi yang harus dihubungi segera berdasarkan konteks"
}}
"""

# ============================================================
# CORE AI REQUEST
# ============================================================
def image_to_data_url(image_path: str) -> str:
    mime_type, _ = mimetypes.guess_type(image_path)
    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"

def cloud_ai_request(messages, stream_output=STREAM_OUTPUT):
    if not BEARER_TOKEN:
        raise RuntimeError("OPENROUTER_API_KEY belum diatur.")

    headers = {
        "Authorization": f"Bearer {BEARER_TOKEN}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "top_p": TOP_P,
        "stream": stream_output,
    }

    if REASONING_ENABLED:
        payload["reasoning"] = {"effort": REASONING_EFFORT, "exclude": REASONING_EXCLUDE}

    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(API_URL, headers=headers, json=payload, stream=stream_output, timeout=300)
            if response.status_code >= 400:
                if response.status_code in (408, 429, 500, 502, 503, 504):
                    time.sleep(RETRY_BACKOFF_SECONDS * (attempt + 1))
                    continue
                raise RuntimeError(f"OpenRouter HTTP {response.status_code}: {response.text[:500]}")

            if not stream_output:
                content = response.json().get("choices", [])[0].get("message", {}).get("content", "")
                return content

            collected_content = []
            for line in response.iter_lines(decode_unicode=True):
                if line and line.startswith("data:"):
                    raw_data = line[5:].strip()
                    if raw_data == "[DONE]": break
                    try:
                        chunk = json.loads(raw_data)
                        token = chunk.get("choices", [])[0].get("delta", {}).get("content", "")
                        if token:
                            collected_content.append(token)
                            print(token, end="", flush=True)
                    except: continue
            
            print()
            return "".join(collected_content)

        except requests.RequestException as exc:
            last_error = exc
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_BACKOFF_SECONDS * (attempt + 1))
                continue

    raise RuntimeError(f"Gagal menghubungi OpenRouter: {last_error}")

def parse_json_output(content: str) -> Dict[str, Any]:
    content = content.strip()
    if content.startswith("```json"): content = content[7:]
    if content.startswith("```"): content = content[3:]
    if content.endswith("```"): content = content[:-3]
    return json.loads(content.strip())

# ============================================================
# API FETCHERS
# ============================================================
def fetch_phone_info(phone_number: str) -> dict:
    try:
        res = requests.get(f"{TOOLS_API_BASE}/verify/{phone_number}", timeout=10)
        return res.json() if res.status_code == 200 else {"error": "Gagal verifikasi nomor"}
    except: return {"error": "API Tools tidak merespons"}

def fetch_institutions() -> list:
    try:
        res = requests.get(f"{TOOLS_API_BASE}/institutions", timeout=10)
        return res.json().get("data", []) if res.status_code == 200 else []
    except: return []

# ============================================================
# MAIN ANALYSIS
# ============================================================
def analyze(user_text: str, image_path: Optional[str], phone_number: str, emergency_data: dict):
    table = Table(title="CegahScam V3 Configuration", show_header=False)
    table.add_row("LLM Model", MODEL)
    table.add_row("Reasoning", f"Enabled ({REASONING_EFFORT})")
    table.add_row("Tools API", TOOLS_API_BASE)
    table.add_row("Keywords DB", KEYWORDS_FILE)
    console.print(table)

    scam_keywords = load_scam_keywords()
    phone_data = {}
    db_inst = []

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        if phone_number:
            task1 = progress.add_task(f"[cyan]Verifikasi nomor {phone_number} via API...", total=None)
            phone_data = fetch_phone_info(phone_number)
            progress.update(task1, completed=True)
        
        task2 = progress.add_task("[green]Mengambil White List DB Instansi via API...", total=None)
        db_inst = fetch_institutions()
        progress.update(task2, completed=True)

    # 1. TAHAP ANALISIS REGULER
    console.print("\n[bold yellow]>>> Memulai Analisis Multimodal (Regular Flow)[/bold yellow]")
    
    prompt = COMBINED_REGULAR_PROMPT.format(
        user_text=user_text,
        phone_info=json.dumps(phone_data),
        kb_keywords=json.dumps(scam_keywords),
        db_institutions=json.dumps(db_inst[:15])
    )

    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    if image_path:
        messages[0]["content"].append({"type": "image_url", "image_url": {"url": image_to_data_url(image_path)}})

    raw_regular = cloud_ai_request(messages, stream_output=STREAM_OUTPUT)
    reg_result = parse_json_output(raw_regular)
    
    console.print("\n[bold green]Hasil Analisis Regular:[/bold green]")
    console.print(json.dumps(reg_result, indent=2, ensure_ascii=False))

    # 2. TAHAP ANALISIS DARURAT (AUTO-TRIGGER)
    # Membaca skor risiko dari objek id atau en
    score = reg_result.get("id", {}).get("score", reg_result.get("en", {}).get("score", 0))
    is_emergency_input = any(emergency_data.values())

    if score >= 80 or is_emergency_input:
        console.print(f"\n[bold red]!!! PERINGATAN DARURAT (Skor: {score}) !!![/bold red]")
        console.print("[bold red]Menjalankan Emergency Flow (AI Dynamic SOP Mitigation)...[/bold red]\n")
        
        emg_prompt = EMERGENCY_PROMPT.format(
            score=score,
            regular_result=json.dumps(reg_result, ensure_ascii=False),
            emergency_data=json.dumps(emergency_data, ensure_ascii=False)
        )
        
        emg_messages = [{"role": "user", "content": emg_prompt}]
        raw_emg = cloud_ai_request(emg_messages, stream_output=STREAM_OUTPUT)
        emg_result = parse_json_output(raw_emg)

        console.print("\n[bold green]Hasil Analisis Darurat (Emergency):[/bold green]")
        console.print(json.dumps(emg_result, indent=2, ensure_ascii=False))
        
        with open("draft_laporan_darurat.md", "w", encoding="utf-8") as f:
            f.write(emg_result.get("draft_laporan_markdown", ""))
        console.print("\n[cyan]Draf laporan telah disimpan sebagai 'draft_laporan_darurat.md'[/cyan]")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="CegahScam V3 AI Engine")
    parser.add_argument("--text", default="", help="Teks pesan")
    parser.add_argument("--image", help="Path gambar")
    parser.add_argument("--phone", default="", help="Nomor telepon pelaku")
    
    parser.add_argument("--bank", default="", help="Bank/E-Wallet korban")
    parser.add_argument("--kerugian", default="", help="Nominal kerugian")
    parser.add_argument("--data-bocor", default="", help="Jenis data bocor")
    
    args = parser.parse_args()
    
    emergency_data = {
        "bank_ewallet": args.bank,
        "nominal_kerugian": args.kerugian,
        "jenis_data_bocor": args.data_bocor
    }
    
    try:
        analyze(args.text, args.image, args.phone, emergency_data)
    except Exception as e:
        console.print(f"\n[bold red]ERROR:[/bold red] {e}")