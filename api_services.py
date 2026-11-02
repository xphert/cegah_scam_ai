import csv
import logging
import traceback
from io import StringIO
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
import requests

# Import langsung fungsi dari modul phone_number_verification
from phone_number_verification import (
    GtcError,
    api_search,
    get_cred,
    load_store,
    normalize_phone,
)

# Logging Debug untuk pantau error di terminal FastAPI
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("CegahScamAPI")

app = FastAPI(title="CegahScam Tools API v3")

SHEET_URL = "https://docs.google.com/spreadsheets/d/17dDzhzAefyeTp2Gb9Wq311-prqcus9GU/export?format=csv&gid=1746131459"


# Handler Global 500 untuk menampilkan detail traceback di terminal dan respons JSON
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    tb_str = traceback.format_exc()
    logger.error(f"Internal Server Error pada {request.url}:\n{tb_str}")
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "error_type": type(exc).__name__,
            "message": str(exc),
            "traceback": tb_str.splitlines()[-5:]
        }
    )


@app.get("/api/v1/verify/{phone_number}")
def verify_phone(phone_number: str, source: str = "profile"):
    """Mengeksekusi api_search() langsung dengan default source 'profile' dan mengembalikan JSON mentah."""
    logger.info(f"==> Verifikasi nomor: {phone_number} (source: {source})")
    try:
        # Ambil kredensial aktif dari file store
        store = load_store()
        name, cred = get_cred(store, None)
        phone = normalize_phone(phone_number)

        logger.info(f"Menggunakan akun GetContact: '{name}' | Target: '{phone}'")

        # Panggil fungsi api_search secara langsung
        body = api_search(cred, phone, source)

        logger.info(f"Verifikasi sukses untuk {phone}")
        return body

    except GtcError as e:
        logger.error(f"GtcError pada {phone_number}: {e}")
        raise HTTPException(
            status_code=400,
            detail={
                "status": "gtc_error",
                "message": str(e),
                "hint": "Cek akun aktif GetContact via CLI 'gtc cred list', captcha, atau sisa kuota."
            }
        )
    except Exception as e:
        logger.error(f"Error 500:\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail={
                "status": "server_error",
                "error_type": type(e).__name__,
                "message": str(e),
                "traceback": traceback.format_exc().splitlines()[-4:]
            }
        )


@app.get("/api/v1/institutions")
def get_institutions():
    """Endpoint parsing DB Instansi Resmi dari Google Sheets."""
    try:
        res = requests.get(SHEET_URL, timeout=15)
        res.raise_for_status()

        csv_data = StringIO(res.text)
        reader = csv.DictReader(csv_data)

        institutions = []
        for row in reader:
            if row.get("name"):
                institutions.append({
                    "name": row.get("name", "").strip(),
                    "call": row.get("call", "").strip(),
                    "email": row.get("email", "").strip(),
                    "instagram": row.get("instagram", "").strip(),
                    "twitter": row.get("twitter", "").strip(),
                })
        return {"status": "success", "count": len(institutions), "data": institutions}
    except Exception as e:
        logger.error(f"Gagal memuat DB Instansi: {e}")
        raise HTTPException(status_code=500, detail=f"Gagal mengambil DB Instansi: {str(e)}")
