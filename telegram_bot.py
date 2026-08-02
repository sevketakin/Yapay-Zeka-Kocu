"""
Koçum - Telegram Bot Sürümü
================================================================

Chainlit sürümüyle aynı beyni (Gemini + video/eski sohbet arşivi)
kullanır, ama Telegram üzerinden çalışır. Basit, kendi kurduğumuz
bir veritabanı tablosuyla (Chainlit'in karmaşık şemasına gerek
kalmadan) her kullanıcının kendi sohbet geçmişini ayrı ayrı tutar.

Birden fazla kişi (örn. sen ve kız arkadaşın) aynı botu kullanabilir,
Telegram otomatik olarak kim yazdığını ayırt eder, herkesin kendi
geçmişi ayrı kalır. Ortak olan tek şey video/bilgi arşivi.

Gereksinim:
    pip install python-telegram-bot chromadb google-genai openpyxl psycopg2-binary requests

Ortam değişkenleri (Railway'de ayarlanacak):
    TELEGRAM_BOT_TOKEN, GEMINI_API_ANAHTARI, DATABASE_URL, VERI_KLASORU
"""

import os
import re
import json
import uuid
import zipfile
import glob
import base64
from io import BytesIO
from datetime import datetime, timedelta

from google import genai
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
import psycopg2
import requests

from telegram import Update, InputFile, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, MessageHandler, CommandHandler, CallbackQueryHandler,
    ContextTypes, filters,
)

# ============== AYARLAR ==============
VERI_KLASORU = os.environ.get("VERI_KLASORU", ".")
DB_KLASORU = os.path.join(VERI_KLASORU, "veritabani")
GEMINI_API_ANAHTARI = os.environ.get("GEMINI_API_ANAHTARI", "")
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
DATABASE_URL = os.environ.get("DATABASE_URL", "")
KOLEKSIYON_ADI = "video_transkriptleri"
GEMINI_MODEL_LISTESI = [
    "gemini-flash-latest",
    "gemini-3.5-flash",
    "gemini-3.1-flash-lite",
    "gemini-3-flash-preview",
]
KAC_PARCA_GETIRILSIN = 25
UYGULAMA_ADI = "Koçum"

STRAVA_CLIENT_ID = os.environ.get("STRAVA_CLIENT_ID", "")
STRAVA_CLIENT_SECRET = os.environ.get("STRAVA_CLIENT_SECRET", "")

# Buraya, "yumuşak ton" ile konuşulmasını istediğin kullanıcıların
# Telegram ID'lerini ekleyebilirsin (örn. kız arkadaşının ID'si).
# ID'yi öğrenmek için: bota /id yazması yeterli, sana ID'sini gösterir.
YUMUSAK_TON_KULLANICILARI = set()

GITHUB_ZIP_URL = (
    "https://github.com/sevketakin/Yapay-Zeka-Kocu/releases/download/v1/veritabani.zip"
)

AYLAR_TR = ["Ocak", "Şubat", "Mart", "Nisan", "Mayıs", "Haziran",
            "Temmuz", "Ağustos", "Eylül", "Ekim", "Kasım", "Aralık"]
GUNLER_TR = ["Pazartesi", "Salı", "Çarşamba", "Perşembe", "Cuma", "Cumartesi", "Pazar"]
GUN_INDEX = {"Pazartesi": 0, "Salı": 1, "Çarşamba": 2, "Perşembe": 3,
             "Cuma": 4, "Cumartesi": 5, "Pazar": 6}


# ============== VERİTABANI ZIP'İNİ OTOMATİK İNDİRME + AÇMA ==============
def _zip_saglam_mi(yol):
    try:
        with zipfile.ZipFile(yol, 'r') as z:
            return z.testzip() is None
    except Exception:
        return False


def _veritabanini_hazirla():
    _tamamlandi_isareti = os.path.join(VERI_KLASORU, "_veritabani_tamamlandi.txt")
    if os.path.exists(_tamamlandi_isareti):
        return

    if os.path.exists(DB_KLASORU):
        import shutil as _shutil
        print("Yarım kalmış eski veritabanı klasörü bulundu, temizleniyor...")
        _shutil.rmtree(DB_KLASORU, ignore_errors=True)

    os.makedirs(VERI_KLASORU, exist_ok=True)
    _tek_zip = os.path.join(VERI_KLASORU, "veritabani.zip")

    if os.path.exists(_tek_zip) and not _zip_saglam_mi(_tek_zip):
        os.remove(_tek_zip)

    if not os.path.exists(_tek_zip):
        try:
            print("Veritabanı GitHub Release'ten indiriliyor... (bu birkaç dakika sürebilir)")
            with requests.get(GITHUB_ZIP_URL, stream=True, allow_redirects=True, timeout=120) as yanit:
                yanit.raise_for_status()
                with open(_tek_zip, "wb") as f:
                    for parca in yanit.iter_content(chunk_size=8 * 1024 * 1024):
                        f.write(parca)
            print("İndirme tamamlandı.")
        except Exception as e:
            print(f"GitHub'dan indirme başarısız: {e}")
            if os.path.exists(_tek_zip):
                os.remove(_tek_zip)
            return

    if not _zip_saglam_mi(_tek_zip):
        print("HATA: İndirilen zip bozuk.")
        return

    print(f"'{_tek_zip}' açılıyor... (bu biraz sürebilir)")
    with zipfile.ZipFile(_tek_zip, 'r') as z:
        z.extractall(VERI_KLASORU)
    os.remove(_tek_zip)

    with open(_tamamlandi_isareti, "w") as f:
        f.write("ok")
    print("Veritabanı hazır.")


# ============== GEMINI EMBEDDING (arama için) ==============
class GeminiEmbeddingFonksiyonu(EmbeddingFunction):
    def __init__(self, api_anahtari):
        self.client = genai.Client(api_key=api_anahtari)

    def __call__(self, input: Documents) -> Embeddings:
        return [self._tek_embedding_al(metin) for metin in input]

    def _tek_embedding_al(self, metin, max_deneme=3):
        import time
        for deneme in range(1, max_deneme + 1):
            try:
                sonuc = self.client.models.embed_content(
                    model="gemini-embedding-001", contents=metin,
                )
                return sonuc.embeddings[0].values
            except Exception as e:
                hata_metni = str(e)
                if "429" in hata_metni or "RESOURCE_EXHAUSTED" in hata_metni or "UNAVAILABLE" in hata_metni:
                    time.sleep(5)
                    continue
                raise
        raise RuntimeError(f"Embedding alınamadı: {metin[:50]}...")


_client_gemini = None
_koleksiyon = None


def istemcileri_al():
    global _client_gemini, _koleksiyon
    if _client_gemini is None:
        _client_gemini = genai.Client(api_key=GEMINI_API_ANAHTARI)
    if _koleksiyon is None:
        embed_fonksiyonu = GeminiEmbeddingFonksiyonu(GEMINI_API_ANAHTARI)
        db = chromadb.PersistentClient(path=DB_KLASORU)
        _koleksiyon = db.get_collection(name=KOLEKSIYON_ADI, embedding_function=embed_fonksiyonu)
    return _client_gemini, _koleksiyon


def baglami_hazirla(bulunan_parcalar):
    metinler = bulunan_parcalar['documents'][0]
    metadatalar = bulunan_parcalar['metadatas'][0]
    baglam_parcalari = []
    kaynaklar = []
    for metin, meta in zip(metinler, metadatalar):
        video_id = meta.get('video_id', 'bilinmiyor')
        kaynak_turu = meta.get('kaynak', '')
        onizleme = metin[:250].strip() + ("..." if len(metin) > 250 else "")
        if kaynak_turu == "eski_sohbet":
            baglam_parcalari.append(
                f"[GERÇEK KİŞİSEL GEÇMİŞ — {video_id} — bu kullanıcının SENİNLE "
                f"gerçekten daha önce konuştuğu bir şey, genel bir video örneği DEĞİL]\n{metin}"
            )
            kaynaklar.append({"link": None, "baslik": video_id, "onizleme": onizleme})
        else:
            link = f"https://youtube.com/watch?v={video_id}"
            baglam_parcalari.append(
                f"[Genel video içeriği (BAŞKA insanların/koçların anlattığı örnekler, "
                f"kullanıcının kendi kişisel geçmişi DEĞİL): {link}]\n{metin}"
            )
            kaynaklar.append({"link": link, "baslik": None, "onizleme": onizleme})
    return "\n\n---\n\n".join(baglam_parcalari), kaynaklar


# ============== BASİT SOHBET GEÇMİŞİ (kendi tablomuz) ==============
def _basit_semayi_hazirla():
    if not DATABASE_URL:
        return
    try:
        baglanti = psycopg2.connect(DATABASE_URL)
        baglanti.autocommit = True
        imlec = baglanti.cursor()
        imlec.execute("""
            CREATE TABLE IF NOT EXISTS tg_mesajlar (
                id SERIAL PRIMARY KEY,
                kullanici_id BIGINT NOT NULL,
                rol TEXT NOT NULL,
                icerik TEXT NOT NULL,
                zaman TIMESTAMP DEFAULT NOW()
            );
            CREATE TABLE IF NOT EXISTS tg_ayarlar (
                kullanici_id BIGINT PRIMARY KEY,
                yumusak_ton BOOLEAN DEFAULT FALSE
            );
            CREATE TABLE IF NOT EXISTS strava_baglantilar (
                kullanici_id BIGINT PRIMARY KEY,
                refresh_token TEXT NOT NULL,
                athlete_id BIGINT,
                son_gorulen_aktivite_id BIGINT DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS kullanici_profil (
                kullanici_id BIGINT PRIMARY KEY,
                profil TEXT DEFAULT ''
            );
            ALTER TABLE tg_ayarlar ADD COLUMN IF NOT EXISTS sabah_mesaji BOOLEAN DEFAULT FALSE;
        """)
        imlec.close()
        baglanti.close()
        print("Telegram bot tabloları hazır.")
    except Exception as e:
        print(f"Tablo hazırlanırken hata: {e}")


def _gecmis_tarih_etiketi(zaman):
    """Bir geçmiş mesajın, BUGÜNE göre kaç gün önce olduğunu Türkçe
    etiketler — böylece model 'dün konuştuğumuz şey' ile 'bugün
    konuştuğumuz şey'i birbirinden ayırt edebilir."""
    try:
        from zoneinfo import ZoneInfo
        if zaman.tzinfo is None:
            zaman_tr = zaman.replace(tzinfo=ZoneInfo("UTC")).astimezone(ZoneInfo("Europe/Istanbul"))
        else:
            zaman_tr = zaman.astimezone(ZoneInfo("Europe/Istanbul"))
    except Exception:
        zaman_tr = zaman

    bugun = _turkiye_simdi().date()
    fark = (bugun - zaman_tr.date()).days
    saat_str = zaman_tr.strftime("%H:%M")

    if fark <= 0:
        return f"BUGÜN, saat {saat_str}"
    elif fark == 1:
        return f"DÜN, saat {saat_str}"
    else:
        return f"{fark} gün önce ({zaman_tr.day} {AYLAR_TR[zaman_tr.month - 1]}), saat {saat_str}"


def gecmisi_oku(kullanici_id, limit=40):
    if not DATABASE_URL:
        return []
    try:
        baglanti = psycopg2.connect(DATABASE_URL)
        imlec = baglanti.cursor()
        imlec.execute(
            "SELECT rol, icerik, zaman FROM ("
            "  SELECT rol, icerik, zaman FROM tg_mesajlar "
            "  WHERE kullanici_id = %s ORDER BY zaman DESC LIMIT %s"
            ") alt ORDER BY zaman ASC",
            (kullanici_id, limit),
        )
        satirlar = imlec.fetchall()
        imlec.close()
        baglanti.close()
        sonuc = []
        for rol, icerik, zaman in satirlar:
            etiket = _gecmis_tarih_etiketi(zaman)
            icerik_etiketli = f"[{etiket}] {icerik}"
            sonuc.append({"role": rol, "parts": [{"text": icerik_etiketli}]})
        return sonuc
    except Exception as e:
        print(f"Geçmiş okunurken hata: {e}")
        return []


def mesaji_kaydet(kullanici_id, rol, icerik):
    if not DATABASE_URL:
        return
    try:
        baglanti = psycopg2.connect(DATABASE_URL)
        baglanti.autocommit = True
        imlec = baglanti.cursor()
        imlec.execute(
            "INSERT INTO tg_mesajlar (kullanici_id, rol, icerik) VALUES (%s, %s, %s)",
            (kullanici_id, rol, icerik),
        )
        imlec.close()
        baglanti.close()
    except Exception as e:
        print(f"Mesaj kaydedilirken hata: {e}")


def yumusak_ton_mu(kullanici_id):
    if kullanici_id in YUMUSAK_TON_KULLANICILARI:
        return True
    if not DATABASE_URL:
        return False
    try:
        baglanti = psycopg2.connect(DATABASE_URL)
        imlec = baglanti.cursor()
        imlec.execute("SELECT yumusak_ton FROM tg_ayarlar WHERE kullanici_id = %s", (kullanici_id,))
        satir = imlec.fetchone()
        imlec.close()
        baglanti.close()
        return bool(satir[0]) if satir else False
    except Exception:
        return False


def yumusak_ton_ayarla(kullanici_id, deger):
    if not DATABASE_URL:
        return
    try:
        baglanti = psycopg2.connect(DATABASE_URL)
        baglanti.autocommit = True
        imlec = baglanti.cursor()
        imlec.execute(
            "INSERT INTO tg_ayarlar (kullanici_id, yumusak_ton) VALUES (%s, %s) "
            "ON CONFLICT (kullanici_id) DO UPDATE SET yumusak_ton = %s",
            (kullanici_id, deger, deger),
        )
        imlec.close()
        baglanti.close()
    except Exception as e:
        print(f"Ayar kaydedilirken hata: {e}")


# ============== STRAVA ENTEGRASYONU ==============
def strava_baglantisini_kaydet(kullanici_id, refresh_token, athlete_id=None):
    if not DATABASE_URL:
        return
    baglanti = psycopg2.connect(DATABASE_URL)
    baglanti.autocommit = True
    imlec = baglanti.cursor()
    imlec.execute(
        "INSERT INTO strava_baglantilar (kullanici_id, refresh_token, athlete_id) "
        "VALUES (%s, %s, %s) "
        "ON CONFLICT (kullanici_id) DO UPDATE SET refresh_token = %s, athlete_id = %s",
        (kullanici_id, refresh_token, athlete_id, refresh_token, athlete_id),
    )
    imlec.close()
    baglanti.close()


def strava_baglantisini_getir(kullanici_id):
    if not DATABASE_URL:
        return None
    baglanti = psycopg2.connect(DATABASE_URL)
    imlec = baglanti.cursor()
    imlec.execute(
        "SELECT refresh_token, athlete_id, son_gorulen_aktivite_id "
        "FROM strava_baglantilar WHERE kullanici_id = %s",
        (kullanici_id,),
    )
    satir = imlec.fetchone()
    imlec.close()
    baglanti.close()
    if not satir:
        return None
    return {"refresh_token": satir[0], "athlete_id": satir[1], "son_gorulen": satir[2]}


def strava_son_gorulen_guncelle(kullanici_id, aktivite_id):
    if not DATABASE_URL:
        return
    baglanti = psycopg2.connect(DATABASE_URL)
    baglanti.autocommit = True
    imlec = baglanti.cursor()
    imlec.execute(
        "UPDATE strava_baglantilar SET son_gorulen_aktivite_id = %s WHERE kullanici_id = %s",
        (aktivite_id, kullanici_id),
    )
    imlec.close()
    baglanti.close()


def strava_erisim_tokeni_al(refresh_token):
    """refresh_token'dan (kalıcı) taze bir access_token (6 saatlik) üretir."""
    yanit = requests.post(
        "https://www.strava.com/oauth/token",
        data={
            "client_id": STRAVA_CLIENT_ID,
            "client_secret": STRAVA_CLIENT_SECRET,
            "refresh_token": refresh_token,
            "grant_type": "refresh_token",
        },
        timeout=15,
    )
    yanit.raise_for_status()
    return yanit.json()["access_token"]


def strava_son_aktiviteleri_getir(access_token, kac_tane=5):
    yanit = requests.get(
        "https://www.strava.com/api/v3/athlete/activities",
        headers={"Authorization": f"Bearer {access_token}"},
        params={"per_page": kac_tane},
        timeout=15,
    )
    yanit.raise_for_status()
    return yanit.json()


def strava_aktiviteyi_metne_cevir(aktivite):
    isim = aktivite.get("name", "Aktivite")
    tur = aktivite.get("type", "")
    mesafe_km = (aktivite.get("distance", 0) or 0) / 1000
    sure_dk = (aktivite.get("moving_time", 0) or 0) / 60
    ort_nabiz = aktivite.get("average_heartrate")
    yukseklik = aktivite.get("total_elevation_gain")
    tarih = aktivite.get("start_date_local", "")

    satirlar = [f"Aktivite: {isim} ({tur})", f"Tarih: {tarih}",
                f"Mesafe: {mesafe_km:.2f} km", f"Süre: {sure_dk:.0f} dakika"]
    if ort_nabiz:
        satirlar.append(f"Ortalama nabız: {ort_nabiz:.0f}")
    if yukseklik:
        satirlar.append(f"Toplam tırmanış: {yukseklik:.0f} m")
    return "\n".join(satirlar)


def _pace_hesapla(mesafe_km, sure_dk):
    if not mesafe_km or mesafe_km <= 0:
        return None
    pace_dk_km = sure_dk / mesafe_km
    dk = int(pace_dk_km)
    sn = int(round((pace_dk_km - dk) * 60))
    return f"{dk}:{sn:02d} dk/km"


def strava_kosu_pace_ozeti(access_token, kac_tane=8):
    """Son koşularının pace (tempo) verilerini özetler — antrenman
    önerilerinde 'gerçek pace'ine göre' konuşabilmek için kullanılır."""
    aktiviteler = strava_son_aktiviteleri_getir(access_token, kac_tane=30)
    kosular = [a for a in aktiviteler if a.get("type") in ("Run", "TrailRun", "VirtualRun")][:kac_tane]
    if not kosular:
        return ""

    satirlar = ["Son koşularımın pace (tempo) verileri:"]
    tum_pace_degerleri = []
    for a in kosular:
        mesafe_km = (a.get("distance", 0) or 0) / 1000
        sure_dk = (a.get("moving_time", 0) or 0) / 60
        pace = _pace_hesapla(mesafe_km, sure_dk)
        ort_nabiz = a.get("average_heartrate")
        tarih = (a.get("start_date_local", "") or "")[:10]
        if pace:
            satir = f"- {tarih}: {mesafe_km:.1f} km, pace {pace}"
            if ort_nabiz:
                satir += f", ort. nabız {ort_nabiz:.0f}"
            satirlar.append(satir)
            tum_pace_degerleri.append(sure_dk / mesafe_km if mesafe_km else None)

    gecerli_paceler = [p for p in tum_pace_degerleri if p]
    if gecerli_paceler:
        ort_pace = sum(gecerli_paceler) / len(gecerli_paceler)
        en_iyi_pace = min(gecerli_paceler)
        dk_ort, sn_ort = int(ort_pace), int(round((ort_pace - int(ort_pace)) * 60))
        dk_iyi, sn_iyi = int(en_iyi_pace), int(round((en_iyi_pace - int(en_iyi_pace)) * 60))
        satirlar.append(f"\nOrtalama pace: {dk_ort}:{sn_ort:02d} dk/km")
        satirlar.append(f"En iyi (en hızlı) pace: {dk_iyi}:{sn_iyi:02d} dk/km")

    return "\n".join(satirlar)


_KOSU_ANAHTAR_KELIMELER = [
    "pace", "tempo", "interval", "koşu hız", "km/saat", "dakika/km",
    "dk/km", "hangi hızla", "ne hızla", "koşu antrenman",
]


def _kosu_sorusu_mu(soru):
    soru_kucuk = soru.lower()
    return any(k in soru_kucuk for k in _KOSU_ANAHTAR_KELIMELER)


# ============== KALICI KULLANICI PROFİLİ ==============
def profili_oku(kullanici_id):
    if not DATABASE_URL:
        return ""
    try:
        baglanti = psycopg2.connect(DATABASE_URL)
        imlec = baglanti.cursor()
        imlec.execute("SELECT profil FROM kullanici_profil WHERE kullanici_id = %s", (kullanici_id,))
        satir = imlec.fetchone()
        imlec.close()
        baglanti.close()
        return satir[0] if satir else ""
    except Exception:
        return ""


def profili_yaz(kullanici_id, yeni_profil):
    if not DATABASE_URL:
        return
    baglanti = psycopg2.connect(DATABASE_URL)
    baglanti.autocommit = True
    imlec = baglanti.cursor()
    imlec.execute(
        "INSERT INTO kullanici_profil (kullanici_id, profil) VALUES (%s, %s) "
        "ON CONFLICT (kullanici_id) DO UPDATE SET profil = %s",
        (kullanici_id, yeni_profil, yeni_profil),
    )
    imlec.close()
    baglanti.close()


def profili_otomatik_guncelle(client_gemini, kullanici_id, soru, cevap):
    """Her konuşmadan sonra, kalıcı olarak hatırlanması gereken YENİ bir
    bilgi (hedef, kilo, sakatlık, tercih vb.) geçip geçmediğini kontrol
    eder, varsa profile ekler. Ucuz/hızlı bir model kullanır."""
    mevcut_profil = profili_oku(kullanici_id)
    talimat = (
        "Aşağıda bir kullanıcı ile antrenörü arasındaki SON mesaj çifti var. "
        "Kullanıcının MEVCUT PROFİLİ de altta. Eğer bu son mesajlarda, "
        "profile eklenmeye değer YENİ ve KALICI bir kişisel bilgi geçiyorsa "
        "(hedef, kilo, boy, yaş, sakatlık/kısıtlama, tercih, hedef yarış "
        "tarihi vb. — GEÇİCİ/günlük şeyler değil) profili güncelleyip TAM "
        "HALİNİ döndür. Yeni bir şey yoksa, SADECE 'DEĞİŞİKLİK_YOK' yaz. "
        "Profil kısa, madde madde, Türkçe olmalı, 15 satırı geçmemeli.\n\n"
        f"MEVCUT PROFİL:\n{mevcut_profil or '(henüz boş)'}\n\n"
        f"SON MESAJLAR:\nKullanıcı: {soru}\nAntrenör: {cevap}"
    )
    try:
        yanit = client_gemini.models.generate_content(
            model="gemini-flash-latest", contents=talimat,
        )
        sonuc = (yanit.text or "").strip()
        if sonuc and "DEĞİŞİKLİK_YOK" not in sonuc.upper():
            profili_yaz(kullanici_id, sonuc)
    except Exception as e:
        print(f"Profil güncellenirken hata: {e}")


# ============== TARİH/SAAT ==============
def _turkiye_simdi():
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo("Europe/Istanbul"))
    except Exception:
        return datetime.now()


def bugunun_tarihi():
    simdi = _turkiye_simdi()
    return f"{simdi.day} {AYLAR_TR[simdi.month - 1]} {simdi.year}, {GUNLER_TR[simdi.weekday()]}"


def su_anki_saat():
    simdi = _turkiye_simdi()
    saat = simdi.hour
    if 5 <= saat < 12:
        vakit = "sabah"
    elif 12 <= saat < 17:
        vakit = "öğleden sonra"
    elif 17 <= saat < 21:
        vakit = "akşam"
    else:
        vakit = "gece"
    return simdi.strftime("%H:%M"), vakit


# ============== SES YAZIYA ÇEVİRME ==============
def ses_yaziya_cevir(client_gemini, ses_bytes, mime_tipi="audio/ogg"):
    ses_b64 = base64.b64encode(ses_bytes).decode("utf-8")
    denenecek_tipler = [mime_tipi] + [
        t for t in ["audio/ogg", "audio/wav", "audio/mp3", "audio/mp4", "audio/webm"]
        if t != mime_tipi
    ]
    for tip in denenecek_tipler:
        for model_adi in GEMINI_MODEL_LISTESI:
            try:
                yanit = client_gemini.models.generate_content(
                    model=model_adi,
                    contents=[
                        {"role": "user", "parts": [
                            {"text": "Bu ses kaydını sadece yazıya çevir, başka hiçbir şey ekleme."},
                            {"inline_data": {"mime_type": tip, "data": ses_b64}},
                        ]}
                    ],
                )
                if yanit.text and yanit.text.strip():
                    return yanit.text.strip()
            except Exception:
                continue
    return None


# ============== ANA CEVAP ÜRETME ==============
def cevap_uret(client_gemini, soru, baglam, gecmis, gorsel_b64=None, gorsel_mime=None,
                yumusak=False, profil=""):
    bugun = bugunun_tarihi()
    saat, vakit = su_anki_saat()

    ton_talimati = (
        "TON: Benimle samimi, sıcak, motive edici ve gerçek bir antrenör "
        "gibi konuş — resmi/mesafeli bir asistan gibi değil."
        if not yumusak else
        "TON: Yumuşak, nazik, destekleyici ve sabırlı bir dille konuş — "
        "baskıcı/sert bir koç gibi değil, anlayışlı bir rehber gibi."
    )

    profil_blogu = (
        f"📋 KULLANICI HAKKINDA KALICI BİLGİLER (bunlar her zaman doğrudur, "
        f"konuşma ne kadar eski olursa olsun unutma):\n{profil}\n\n"
        if profil else ""
    )

    sistem_mesaji = (
        "🚨 EN ÖNEMLİ KURAL — EN BAŞTA OKU VE HER ZAMAN UYGULA:\n"
        "Kullanıcı sana 'geçen hafta/gün ne yaptık', 'dün ne konuşmuştuk', "
        "'hatırlıyor musun', kaldırdığı ağırlıklar, koştuğu mesafeler gibi "
        "KENDİ KİŞİSEL geçmişiyle ilgili bir şey sorduğunda: SADECE ve "
        "SADECE aşağıda sana verilen gerçek konuşma geçmişinde (ÖNCEKİ "
        "MESAJLAR), '📋 KULLANICI HAKKINDA KALICI BİLGİLER' bölümünde ya da "
        "'[GERÇEK KİŞİSEL GEÇMİŞ ...]' etiketli notlarda GERÇEKTEN yazan "
        "bilgiyi kullan. '[Genel video içeriği ...]' etiketli notlar BAŞKA "
        "insanların kendi hikayeleri — bunları asla kullanıcının kendi "
        "hikayesiymiş gibi anlatma. Bilmiyorsan dürüstçe söyle, asla "
        "kişisel veri uydurma.\n\n"
        "🚨 TARİH ALGISI KURALI: Aşağıdaki ÖNCEKİ MESAJLAR'ın her birinin "
        "başında [BUGÜN, saat ...] / [DÜN, saat ...] / [X gün önce (...), "
        "saat ...] şeklinde bir zaman etiketi var. Bu etiketleri MUTLAKA "
        "dikkate al: 'DÜN' ya da 'X gün önce' etiketli bir mesajda geçen "
        "bir olayı (örn. 'bugün doğum günüm', 'bugün dinleniyorum' gibi) "
        "ASLA şu anki 'bugün'müş gibi ele alma — o, o GÜNÜN 'bugünü' idi, "
        "senin şu anki BUGÜN'ün değil. Sadece gerçekten [BUGÜN] etiketli "
        "mesajlardaki bilgi, gerçek zamanlı bugünü yansıtır.\n\n"
        "🚨 SPESİFİK PROGRAM/İÇERİK UYDURMA KURALI: Kullanıcı belirli bir "
        "kanalın/kişinin/programın (örn. 'Asla Durma'nın 8 haftalık "
        "programı') TAM İÇERİĞİNİ, hafta hafta/gün gün yapısını sorduğunda: "
        "Bu detayları SADECE sana verilen notlarda GERÇEKTEN yazıyorsa "
        "kullan. Notlarda yoksa, kendi genel spor bilgini kullanarak "
        "'muhtemelen böyle olabilir, tipik bir yapı şöyledir' diye GENEL "
        "bir çerçeve sunabilirsin AMA bunu o programın 'kesin, doğrulanmış "
        "içeriğiymiş' gibi sunma — belirsizliği açıkça belirt (örn. 'tam "
        "dakikaları elimde yok ama genel mantık şöyle işler' gibi). Kesin "
        "biliyormuş gibi uydurma sayılar/haftalar vermek yasak.\n\n"
        "📌 FARKLI KAYNAK/KOÇ FARKINDALIĞI: Notlar arasında farklı "
        "video_id'lerden (yani farklı kanallardan/koçlardan) gelen bilgiler "
        "birbirleriyle ÇELİŞİYORSA, bunu görmezden gelip birini seçme — "
        "kısaca 'bazı kaynaklar şöyle diyor, bazıları böyle' diye açık "
        "şekilde belirt, sonra kendi önerini sun.\n\n"
        "🚨 PACE/TEMPO/HIZ ÖNERİLERİNDE GERÇEK VERİYE DAYAN — GENEL VARSAYIM "
        "YAPMA: Eğer sana '[GERÇEK STRAVA VERİSİ ...]' etiketli, kullanıcının "
        "GERÇEK koşu pace verileri verilmişse, önerini KESİNLİKLE bu gerçek "
        "sayılara dayandır (örn. 'ortalama pace'in 6:30 dk/km, bugünkü "
        "interval için bunun biraz altında, 6:00 dk/km hedefleyelim' gibi). "
        "ASLA kilo/boy/genel fiziksel özelliklerden yola çıkarak "
        "('100 kg birisin, muhtemelen bu hızda koşarsın' gibi) GENEL VE "
        "SOYUT bir pace tahmini uydurma — bu, elindeki gerçek veriyi göz "
        "ardı edip tembel bir varsayımla konuşmak demektir, YASAK. Gerçek "
        "veri yoksa (Strava bağlı değilse ya da hiç koşu verisi yoksa), "
        "bunu açıkça söyle ve kullanıcıdan mevcut pace'ini/hissini sor — "
        "yine de kilo bazlı genel bir tahmin uydurma.\n\n"
        f"{profil_blogu}"
        f"Bugünün tarihi: {bugun}. Şu anki saat: {saat} ({vakit}). Bu bilgiyi "
        f"antrenman/beslenme önerilerinde dikkate al.\n\n"
        "Sen benim kişisel hybrid antrenörümsün. Amacın, beni hybrid "
        "sistemde (koşu, bisiklet, yüzme, kuvvet vb.) en iyi versiyonuma "
        "ulaştırmak için elinden gelen tüm yardımı sağlamak. Spor bilimi, "
        "antrenman periyotlaması, beslenme, toparlanma konularında "
        "deneyimli bir antrenör gibisin.\n\n"
        "Sana verilen notlar (varsa) senin kendi bilgi birikimin gibi "
        "davran, dışarıdan kaynak gibi sunma. Alakalıysa öncelikle bunlara "
        "dayan, alakasızsa kendi genel uzmanlığından cevap ver.\n\n"
        f"{ton_talimati}"
    )

    contents = list(gecmis)
    kullanici_mesaji = f"(Alakalı notlar varsa aşağıda, yoksa yok say):\n\n{baglam}\n\nSORU: {soru}"
    parts = [{"text": kullanici_mesaji}]
    if gorsel_b64:
        parts.append({"inline_data": {"mime_type": gorsel_mime, "data": gorsel_b64}})
    contents.append({"role": "user", "parts": parts})

    for model_adi in GEMINI_MODEL_LISTESI:
        try:
            yanit = client_gemini.models.generate_content(
                model=model_adi, contents=contents,
                config={"system_instruction": sistem_mesaji},
            )
            if yanit.text and yanit.text.strip():
                return yanit.text
        except Exception:
            continue
    return "Şu an cevap üretemedim, lütfen tekrar dener misin?"


# ============== PROGRAM -> TAKVİM/EXCEL (kısaltılmış, arayuz.py'dekiyle aynı mantık) ==============
def programdan_json_cikar(client_gemini, program_metni):
    talimat = (
        "Aşağıdaki metinde bir antrenman/beslenme programı var. SADECE "
        "JSON döndür, başka hiçbir şey yazma:\n\n"
        '[{"gun": "Pazartesi", "baslik": "...", "baslangic_saat": "18:00", '
        '"bitis_saat": "19:00", "aciklama": "..."}]\n\n'
        "gun değeri: Pazartesi, Salı, Çarşamba, Perşembe, Cuma, Cumartesi, Pazar. "
        "Program yoksa boş liste [] döndür.\n\n"
        f"METİN:\n{program_metni}"
    )
    for model_adi in GEMINI_MODEL_LISTESI:
        try:
            yanit = client_gemini.models.generate_content(model=model_adi, contents=talimat)
            metin = re.sub(r"^```json\s*|\s*```$", "", yanit.text.strip(), flags=re.MULTILINE).strip("`").strip()
            return json.loads(metin)
        except Exception:
            continue
    return None


def sonraki_gun_tarihi(hedef_gun_index):
    bugun = datetime.now().date()
    fark = (hedef_gun_index - bugun.weekday()) % 7
    return bugun + timedelta(days=fark)


def ics_plan_olustur(etkinlikler):
    satirlar = ["BEGIN:VCALENDAR", "VERSION:2.0", "PRODID:-//Kocum//TR//", "CALSCALE:GREGORIAN"]
    for e in etkinlikler:
        gun = e.get("gun", "")
        if gun not in GUN_INDEX:
            continue
        try:
            bs_saat, bs_dk = map(int, e.get("baslangic_saat", "18:00").split(":"))
            bt_saat, bt_dk = map(int, e.get("bitis_saat", "19:00").split(":"))
        except Exception:
            continue
        tarih = sonraki_gun_tarihi(GUN_INDEX[gun])
        satirlar.append("BEGIN:VEVENT")
        satirlar.append(f"UID:{uuid.uuid4()}")
        satirlar.append(f"DTSTAMP:{datetime.now().strftime('%Y%m%dT%H%M%SZ')}")
        satirlar.append(f"DTSTART;TZID=Europe/Istanbul:{tarih.strftime('%Y%m%d')}T{bs_saat:02d}{bs_dk:02d}00")
        satirlar.append(f"DTEND;TZID=Europe/Istanbul:{tarih.strftime('%Y%m%d')}T{bt_saat:02d}{bt_dk:02d}00")
        satirlar.append(f"SUMMARY:[{UYGULAMA_ADI}] {e.get('baslik', 'Etkinlik')}")
        if e.get("aciklama"):
            satirlar.append(f"DESCRIPTION:{e['aciklama']}")
        satirlar.append("END:VEVENT")
    satirlar.append("END:VCALENDAR")
    return "\r\n".join(satirlar)


def programdan_excel_json_cikar(client_gemini, program_metni):
    talimat = (
        "Aşağıdaki metinde bir antrenman/beslenme programı var. SADECE "
        "JSON döndür:\n\n"
        '[{"gun": "Pazartesi", "kategori": "Antrenman", "baslik": "...", '
        '"detay": "...", "sure_kalori": "60 dk"}]\n\n'
        f"METİN:\n{program_metni}"
    )
    for model_adi in GEMINI_MODEL_LISTESI:
        try:
            yanit = client_gemini.models.generate_content(model=model_adi, contents=talimat)
            metin = re.sub(r"^```json\s*|\s*```$", "", yanit.text.strip(), flags=re.MULTILINE).strip("`").strip()
            return json.loads(metin)
        except Exception:
            continue
    return None


def excel_plan_olustur(satirlar):
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side

    wb = Workbook()
    ws = wb.active
    ws.title = "Program"
    basliklar = ["Gün", "Kategori", "Başlık", "Detay", "Süre/Kalori"]
    b_font = Font(name="Arial", bold=True, color="FFFFFF", size=11)
    b_fill = PatternFill(start_color="1E293B", end_color="1E293B", fill_type="solid")
    h_font = Font(name="Arial", size=10)
    kenar = Border(*(Side(style="thin", color="D1D5DB"),) * 4)

    for i, m in enumerate(basliklar, start=1):
        h = ws.cell(row=1, column=i, value=m)
        h.font, h.fill, h.border = b_font, b_fill, kenar
        h.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)

    for r, e in enumerate(satirlar, start=2):
        degerler = [e.get("gun", ""), e.get("kategori", ""), e.get("baslik", ""),
                    e.get("detay", ""), e.get("sure_kalori", "")]
        for c, d in enumerate(degerler, start=1):
            h = ws.cell(row=r, column=c, value=d)
            h.font, h.border = h_font, kenar
            h.alignment = Alignment(vertical="top", wrap_text=True)

    for i, g in enumerate([12, 12, 22, 45, 14], start=1):
        ws.column_dimensions[ws.cell(row=1, column=i).column_letter].width = g
    ws.freeze_panes = "A2"

    tampon = BytesIO()
    wb.save(tampon)
    return tampon.getvalue()


# ============== TELEGRAM HANDLER'LARI ==============
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        f"Merhaba! Ben {UYGULAMA_ADI}, senin kişisel hybrid antrenörünüm. 💪\n\n"
        f"Yazarak, sesli mesajla ya da fotoğraf göndererek soru sorabilirsin. "
        f"Eski sohbet (.json) ya da video transkripti (.md) dosyalarını da "
        f"gönderip arşive ekleyebilirsin.\n\n"
        f"📋 Profil:\n"
        f"/profil_goster — hakkında bildiklerim\n"
        f"/profil_ekle <bilgi> — kalıcı bir bilgi ekle\n"
        f"/profil_sil — profili sıfırla\n\n"
        f"🏃 Strava:\n"
        f"/strava_baglan <token> — hesabını bağla\n"
        f"/son_antrenman — son aktiviteni yorumlat\n"
        f"/strava_ozet [gün] — trend özeti (varsayılan 7 gün)\n\n"
        f"☀️ Sabah mesajı:\n"
        f"/sabah_ac — her sabah 07:00'de otomatik mesaj\n"
        f"/sabah_kapat — kapat\n\n"
        f"🔍 Arşiv araçları:\n"
        f"/video_var_mi <id> — bir video arşivde var mı kontrol et\n"
        f"/zorla_video <id> — bir videoyu garanti kullan\n\n"
        f"⚙️ Genel:\n"
        f"/id — Telegram ID'ni gösterir\n"
        f"/yumusak_ac, /yumusak_kapat — ton ayarı\n"
        f"/temizle — sohbet geçmişini sıfırla\n"
        f"/web_sohbetlerini_getir <email> — eski web sohbetlerini taşı"
    )


async def id_goster(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(f"Telegram ID'n: {update.effective_user.id}")


async def yumusak_ac(update: Update, context: ContextTypes.DEFAULT_TYPE):
    yumusak_ton_ayarla(update.effective_user.id, True)
    await update.message.reply_text("Tamamdır, bundan sonra daha yumuşak bir tonla konuşacağım. 🌿")


async def yumusak_kapat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    yumusak_ton_ayarla(update.effective_user.id, False)
    await update.message.reply_text("Tamamdır, normal (motive edici) tona döndüm. 💪")


async def temizle(update: Update, context: ContextTypes.DEFAULT_TYPE):
    kullanici_id = update.effective_user.id
    if DATABASE_URL:
        try:
            baglanti = psycopg2.connect(DATABASE_URL)
            baglanti.autocommit = True
            imlec = baglanti.cursor()
            imlec.execute("DELETE FROM tg_mesajlar WHERE kullanici_id = %s", (kullanici_id,))
            imlec.close()
            baglanti.close()
        except Exception:
            pass
    await update.message.reply_text("Sohbet geçmişin temizlendi, sıfırdan başlıyoruz.")


async def profil_goster(update: Update, context: ContextTypes.DEFAULT_TYPE):
    kullanici_id = update.effective_user.id
    profil = profili_oku(kullanici_id)
    if not profil:
        await update.message.reply_text(
            "Henüz kalıcı bir profil bilgin yok. Sohbet ettikçe otomatik "
            "oluşacak, ya da /profil_ekle ile elle ekleyebilirsin."
        )
        return
    await update.message.reply_text(f"📋 Senin hakkında bildiklerim:\n\n{profil}")


async def profil_ekle(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text(
            "Kullanım: /profil_ekle <bilgi>\nÖrn: /profil_ekle Sol dizimde eski bir sakatlık var, ağır squat yapamıyorum"
        )
        return
    kullanici_id = update.effective_user.id
    yeni_bilgi = " ".join(context.args)
    mevcut = profili_oku(kullanici_id)
    guncel = (mevcut + "\n- " + yeni_bilgi).strip() if mevcut else "- " + yeni_bilgi
    profili_yaz(kullanici_id, guncel)
    await update.message.reply_text("✅ Profiline eklendi, bundan sonra bunu hep hatırlayacağım.")


async def profil_sil(update: Update, context: ContextTypes.DEFAULT_TYPE):
    kullanici_id = update.effective_user.id
    profili_yaz(kullanici_id, "")
    await update.message.reply_text("Profilin sıfırlandı.")


async def sabah_ac(update: Update, context: ContextTypes.DEFAULT_TYPE):
    kullanici_id = update.effective_user.id
    if DATABASE_URL:
        baglanti = psycopg2.connect(DATABASE_URL)
        baglanti.autocommit = True
        imlec = baglanti.cursor()
        imlec.execute(
            "INSERT INTO tg_ayarlar (kullanici_id, sabah_mesaji) VALUES (%s, TRUE) "
            "ON CONFLICT (kullanici_id) DO UPDATE SET sabah_mesaji = TRUE",
            (kullanici_id,),
        )
        imlec.close()
        baglanti.close()
    await update.message.reply_text("☀️ Tamamdır, her sabah 07:00'de sana otomatik bir mesaj göndereceğim.")


async def sabah_kapat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    kullanici_id = update.effective_user.id
    if DATABASE_URL:
        baglanti = psycopg2.connect(DATABASE_URL)
        baglanti.autocommit = True
        imlec = baglanti.cursor()
        imlec.execute(
            "INSERT INTO tg_ayarlar (kullanici_id, sabah_mesaji) VALUES (%s, FALSE) "
            "ON CONFLICT (kullanici_id) DO UPDATE SET sabah_mesaji = FALSE",
            (kullanici_id,),
        )
        imlec.close()
        baglanti.close()
    await update.message.reply_text("Sabah mesajları kapatıldı.")


async def sabah_mesaji_isi(context: ContextTypes.DEFAULT_TYPE):
    """Her sabah 07:00'de (Türkiye saati), sabah mesajını açmış tüm
    kullanıcılara KENDİ profillerine/verilerine göre kişisel bir
    günaydın mesajı gönderir."""
    if not DATABASE_URL:
        return
    try:
        baglanti = psycopg2.connect(DATABASE_URL)
        imlec = baglanti.cursor()
        imlec.execute("SELECT kullanici_id FROM tg_ayarlar WHERE sabah_mesaji = TRUE")
        kullanicilar = [r[0] for r in imlec.fetchall()]
        imlec.close()
        baglanti.close()
    except Exception:
        return

    client_gemini, koleksiyon = istemcileri_al()

    for kullanici_id in kullanicilar:
        try:
            yumusak = yumusak_ton_mu(kullanici_id)
            profil = profili_oku(kullanici_id)
            gecmis = gecmisi_oku(kullanici_id, limit=15)

            strava_ozeti = ""
            baglanti_bilgisi = strava_baglantisini_getir(kullanici_id)
            if baglanti_bilgisi:
                try:
                    access_token = strava_erisim_tokeni_al(baglanti_bilgisi["refresh_token"])
                    son_aktiviteler = strava_son_aktiviteleri_getir(access_token, kac_tane=1)
                    if son_aktiviteler:
                        strava_ozeti = f"\n\nEn son aktivitem:\n{strava_aktiviteyi_metne_cevir(son_aktiviteler[0])}"
                except Exception:
                    pass

            soru = f"Günaydın koç! Bugün için bana kısa bir motivasyon ve gün planı önerir misin?{strava_ozeti}"
            bulunan = koleksiyon.query(query_texts=[soru], n_results=KAC_PARCA_GETIRILSIN)
            baglam, _ = baglami_hazirla(bulunan) if bulunan['documents'][0] else ("", [])
            cevap = cevap_uret(client_gemini, soru, baglam, gecmis, yumusak=yumusak, profil=profil)

            mesaji_kaydet(kullanici_id, "user", soru)
            mesaji_kaydet(kullanici_id, "model", cevap)
            await context.bot.send_message(chat_id=kullanici_id, text=f"☀️ Günaydın!\n\n{cevap}")
        except Exception as e:
            print(f"Sabah mesajı hatası (kullanıcı {kullanici_id}): {e}")


async def web_sohbetlerini_getir(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Koçum'un web (Chainlit) sürümündeki sohbetleri, aynı PostgreSQL
    veritabanından okuyup Telegram'ın kendi basit tablosuna kopyalar.
    Kullanım: /web_sohbetlerini_getir seninmailin@gmail.com"""
    if not context.args:
        await update.message.reply_text(
            "Kullanım: /web_sohbetlerini_getir email@adresin.com\n"
            "(Koçum web'e giriş yaptığın e-posta neyse onu yaz)"
        )
        return

    web_email = context.args[0]
    kullanici_id = update.effective_user.id

    if not DATABASE_URL:
        await update.message.reply_text("Veritabanı bağlantısı yok, işlem yapılamadı.")
        return

    try:
        baglanti = psycopg2.connect(DATABASE_URL)
        imlec = baglanti.cursor()

        imlec.execute('SELECT id FROM users WHERE identifier = %s', (web_email,))
        satir = imlec.fetchone()
        if not satir:
            await update.message.reply_text(
                f"'{web_email}' ile web'de giriş yapılmış bir hesap bulamadım."
            )
            imlec.close()
            baglanti.close()
            return
        web_user_id = satir[0]

        imlec.execute(
            'SELECT id, name FROM threads WHERE "userId" = %s ORDER BY "createdAt" ASC',
            (web_user_id,),
        )
        threadler = imlec.fetchall()

        toplam_mesaj = 0
        for thread_id, thread_adi in threadler:
            imlec.execute(
                'SELECT type, output, input, "createdAt" FROM steps '
                'WHERE "threadId" = %s AND type IN (\'user_message\', \'assistant_message\') '
                'ORDER BY "createdAt" ASC',
                (thread_id,),
            )
            adimlar = imlec.fetchall()
            for tur, cikti, girdi, _zaman in adimlar:
                icerik = cikti or girdi or ""
                if not icerik.strip():
                    continue
                rol = "user" if tur == "user_message" else "model"
                mesaji_kaydet(kullanici_id, rol, icerik)
                toplam_mesaj += 1

        imlec.close()
        baglanti.close()

        await update.message.reply_text(
            f"✅ Web'deki {len(threadler)} sohbetten toplam {toplam_mesaj} mesaj "
            f"buraya (gerçek konuşma geçmişine) aktarıldı. Artık onları da hatırlıyorum."
        )
    except Exception as e:
        await update.message.reply_text(f"Aktarma sırasında hata oluştu: {e}")


async def strava_baglan(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Kullanım: /strava_baglan <refresh_token>
    Her kullanıcı KENDİ Strava hesabını, kendi refresh_token'ıyla bağlar —
    böylece hesaplar birbirine karışmaz, herkesin verisi kendine özel kalır."""
    if not context.args:
        await update.message.reply_text(
            "Kullanım: /strava_baglan <refresh_token>\n\n"
            "Kendi Strava hesabını bağlamak için:\n"
            "1) strava.com/settings/api adresinden bir uygulama oluştur\n"
            "2) Bana Client ID ve Secret'ı gönder, sana yetkilendirme linki hazırlayayım\n"
            "3) O linkten aldığın kodu bana ilet, refresh_token üreteyim\n"
            "4) O refresh_token'ı bu komutla kaydet"
        )
        return

    refresh_token = context.args[0]
    kullanici_id = update.effective_user.id

    try:
        access_token = strava_erisim_tokeni_al(refresh_token)
        aktiviteler = strava_son_aktiviteleri_getir(access_token, kac_tane=1)
        athlete_id = None
        strava_baglantisini_kaydet(kullanici_id, refresh_token, athlete_id)
        if aktiviteler:
            strava_son_gorulen_guncelle(kullanici_id, aktiviteler[0]["id"])
        await update.message.reply_text(
            "✅ Strava hesabın başarıyla bağlandı! Artık antrenmanlarını görebiliyorum. "
            "Yeni bir aktivite bitirdiğinde sana otomatik haber vereceğim."
        )
    except Exception as e:
        await update.message.reply_text(f"Bağlantı başarısız: {e}")


async def video_var_mi(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Kullanım: /video_var_mi <video_id_veya_link>
    Bir videonun gerçekten arşivde (ChromaDB'de) olup olmadığını,
    varsa kaç parçaya bölündüğünü ve içeriğinden bir önizlemeyi gösterir."""
    if not context.args:
        await update.message.reply_text("Kullanım: /video_var_mi <video_id_veya_link>")
        return

    girdi = context.args[0]
    video_id = girdi
    eslesme = re.search(r"watch\?v=([\w-]+)", girdi)
    if eslesme:
        video_id = eslesme.group(1)

    try:
        _, koleksiyon = istemcileri_al()
        sonuc = koleksiyon.get(where={"video_id": video_id})
        if not sonuc.get("ids"):
            await update.message.reply_text(f"❌ '{video_id}' arşivde bulunamadı.")
            return
        onizleme = sonuc["documents"][0][:300] if sonuc.get("documents") else ""
        await update.message.reply_text(
            f"✅ '{video_id}' arşivde var! ({len(sonuc['ids'])} parça)\n\n"
            f"İçerik önizlemesi:\n{onizleme}..."
        )
    except Exception as e:
        await update.message.reply_text(f"Kontrol sırasında hata: {e}")


async def son_antrenman(update: Update, context: ContextTypes.DEFAULT_TYPE):
    kullanici_id = update.effective_user.id
    baglanti_bilgisi = strava_baglantisini_getir(kullanici_id)
    if not baglanti_bilgisi:
        await update.message.reply_text(
            "Strava hesabın bağlı değil. Bağlamak için /strava_baglan yaz."
        )
        return

    try:
        access_token = strava_erisim_tokeni_al(baglanti_bilgisi["refresh_token"])
        aktiviteler = strava_son_aktiviteleri_getir(access_token, kac_tane=1)
        if not aktiviteler:
            await update.message.reply_text("Henüz hiç aktivite bulamadım.")
            return

        aktivite_metni = strava_aktiviteyi_metne_cevir(aktiviteler[0])
        client_gemini, koleksiyon = istemcileri_al()
        yumusak = yumusak_ton_mu(kullanici_id)

        soru = f"Az önce bitirdiğim antrenmanı yorumlar mısın?\n\n{aktivite_metni}"
        bulunan = koleksiyon.query(query_texts=[soru], n_results=KAC_PARCA_GETIRILSIN)
        baglam, _ = baglami_hazirla(bulunan) if bulunan['documents'][0] else ("", [])
        gecmis = gecmisi_oku(kullanici_id)
        cevap = cevap_uret(client_gemini, soru, baglam, gecmis, yumusak=yumusak)

        mesaji_kaydet(kullanici_id, "user", soru)
        mesaji_kaydet(kullanici_id, "model", cevap)
        await update.message.reply_text(cevap)
    except Exception as e:
        await update.message.reply_text(f"Antrenman getirilirken hata: {e}")


async def strava_ozet(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Kullanım: /strava_ozet [gun_sayisi] — varsayılan son 7 gün.
    Toplam mesafe, süre, aktivite sayısı gibi TREND bilgisini özetler."""
    kullanici_id = update.effective_user.id
    baglanti_bilgisi = strava_baglantisini_getir(kullanici_id)
    if not baglanti_bilgisi:
        await update.message.reply_text("Strava hesabın bağlı değil. Bağlamak için /strava_baglan yaz.")
        return

    gun_sayisi = 7
    if context.args:
        try:
            gun_sayisi = int(context.args[0])
        except ValueError:
            pass

    try:
        access_token = strava_erisim_tokeni_al(baglanti_bilgisi["refresh_token"])
        aktiviteler = strava_son_aktiviteleri_getir(access_token, kac_tane=100)

        esik_tarih = datetime.now() - timedelta(days=gun_sayisi)
        secilenler = []
        for a in aktiviteler:
            try:
                a_tarih = datetime.fromisoformat(a["start_date_local"].replace("Z", ""))
                if a_tarih >= esik_tarih:
                    secilenler.append(a)
            except Exception:
                continue

        if not secilenler:
            await update.message.reply_text(f"Son {gun_sayisi} günde hiç aktivite bulamadım.")
            return

        toplam_mesafe = sum((a.get("distance", 0) or 0) for a in secilenler) / 1000
        toplam_sure = sum((a.get("moving_time", 0) or 0) for a in secilenler) / 60
        toplam_tirmanis = sum((a.get("total_elevation_gain", 0) or 0) for a in secilenler)
        tur_sayisi = {}
        for a in secilenler:
            tur_sayisi[a.get("type", "?")] = tur_sayisi.get(a.get("type", "?"), 0) + 1

        ozet_metni = (
            f"Son {gun_sayisi} gün özeti:\n"
            f"- Toplam aktivite: {len(secilenler)}\n"
            f"- Toplam mesafe: {toplam_mesafe:.1f} km\n"
            f"- Toplam süre: {toplam_sure:.0f} dakika\n"
            f"- Toplam tırmanış: {toplam_tirmanis:.0f} m\n"
            f"- Türlere göre: {tur_sayisi}"
        )

        client_gemini, koleksiyon = istemcileri_al()
        yumusak = yumusak_ton_mu(kullanici_id)
        soru = f"Son {gun_sayisi} günlük antrenman verilerimi değerlendirir misin, trend olarak nasılım?\n\n{ozet_metni}"
        bulunan = koleksiyon.query(query_texts=[soru], n_results=KAC_PARCA_GETIRILSIN)
        baglam, _ = baglami_hazirla(bulunan) if bulunan['documents'][0] else ("", [])
        gecmis = gecmisi_oku(kullanici_id)
        profil = profili_oku(kullanici_id)
        cevap = cevap_uret(client_gemini, soru, baglam, gecmis, yumusak=yumusak, profil=profil)

        mesaji_kaydet(kullanici_id, "user", soru)
        mesaji_kaydet(kullanici_id, "model", cevap)
        await update.message.reply_text(cevap)
    except Exception as e:
        await update.message.reply_text(f"Özet oluşturulurken hata: {e}")


async def strava_kontrol_isi(context: ContextTypes.DEFAULT_TYPE):
    """Periyodik olarak (JobQueue ile) her bağlı kullanıcının yeni bir
    Strava aktivitesi olup olmadığını kontrol eder, varsa OTOMATİK
    olarak SADECE O KULLANICIYA yorum gönderir — başkasına gitmez."""
    if not DATABASE_URL:
        return
    try:
        baglanti = psycopg2.connect(DATABASE_URL)
        imlec = baglanti.cursor()
        imlec.execute("SELECT kullanici_id, refresh_token, son_gorulen_aktivite_id FROM strava_baglantilar")
        tum_baglantilar = imlec.fetchall()
        imlec.close()
        baglanti.close()
    except Exception:
        return

    client_gemini, koleksiyon = istemcileri_al()

    for kullanici_id, refresh_token, son_gorulen in tum_baglantilar:
        try:
            access_token = strava_erisim_tokeni_al(refresh_token)
            aktiviteler = strava_son_aktiviteleri_getir(access_token, kac_tane=3)
            for aktivite in reversed(aktiviteler):
                if aktivite["id"] <= (son_gorulen or 0):
                    continue

                aktivite_metni = strava_aktiviteyi_metne_cevir(aktivite)
                yumusak = yumusak_ton_mu(kullanici_id)
                soru = f"Az önce şu antrenmanı bitirdim, yorumlar mısın?\n\n{aktivite_metni}"
                bulunan = koleksiyon.query(query_texts=[soru], n_results=KAC_PARCA_GETIRILSIN)
                baglam, _ = baglami_hazirla(bulunan) if bulunan['documents'][0] else ("", [])
                gecmis = gecmisi_oku(kullanici_id)
                cevap = cevap_uret(client_gemini, soru, baglam, gecmis, yumusak=yumusak)

                mesaji_kaydet(kullanici_id, "user", soru)
                mesaji_kaydet(kullanici_id, "model", cevap)

                await context.bot.send_message(chat_id=kullanici_id, text=f"🏃 Yeni antrenman algılandı!\n\n{cevap}")
                strava_son_gorulen_guncelle(kullanici_id, aktivite["id"])
        except Exception as e:
            print(f"Strava kontrol hatası (kullanıcı {kullanici_id}): {e}")


def _hybrid_arama(koleksiyon, soru, kac_tane):
    """Normal anlamsal aramaya ek olarak, sorudaki net ifadeleri
    (örn. '5. hafta', '3. gün') KELİME OLARAK da arar ve sonuçları
    birleştirir. Bu, 'Zone 2' ya da 'X. hafta' gibi çok net ama
    anlamsal aramanın bazen kaçırdığı ifadeleri yakalamayı sağlar."""
    semantik = koleksiyon.query(query_texts=[soru], n_results=kac_tane)
    dokumanlar = list(semantik['documents'][0]) if semantik['documents'][0] else []
    metadatalar = list(semantik['metadatas'][0]) if semantik['metadatas'][0] else []
    gorulen_idler = set(semantik['ids'][0]) if semantik.get('ids') and semantik['ids'][0] else set()

    anahtar_ifadeler = re.findall(r"\d+\s*\.\s*(?:hafta|gün|hafta\w*|gün\w*)", soru, flags=re.IGNORECASE)
    for ifade in anahtar_ifadeler[:2]:
        try:
            anahtar_sonuc = koleksiyon.get(
                where_document={"$contains": ifade.strip()}, limit=5,
            )
            for i, doc_id in enumerate(anahtar_sonuc.get("ids", [])):
                if doc_id not in gorulen_idler:
                    gorulen_idler.add(doc_id)
                    dokumanlar.append(anahtar_sonuc["documents"][i])
                    metadatalar.append(anahtar_sonuc["metadatas"][i])
        except Exception:
            continue

    if not dokumanlar:
        return {"documents": [[]], "metadatas": [[]]}
    return {"documents": [dokumanlar], "metadatas": [metadatalar]}


async def _soruyu_isle(update, context, soru, gorsel_b64=None, gorsel_mime=None):
    client_gemini, koleksiyon = istemcileri_al()
    kullanici_id = update.effective_user.id
    yumusak = yumusak_ton_mu(kullanici_id)

    bulunan = _hybrid_arama(koleksiyon, soru, KAC_PARCA_GETIRILSIN)
    baglam, kaynaklar = "", []
    if bulunan['documents'][0]:
        baglam, kaynaklar = baglami_hazirla(bulunan)

    zorla_video_id = context.user_data.get("zorla_video")
    if zorla_video_id:
        try:
            zorla_sonuc = koleksiyon.get(where={"video_id": zorla_video_id})
            if zorla_sonuc and zorla_sonuc.get("documents"):
                zorla_metin = "\n\n".join(zorla_sonuc["documents"])
                baglam = (
                    f"[ÖNEMLİ — kullanıcı bu videoyu özellikle belirtti, "
                    f"MUTLAKA dikkate al: {zorla_video_id}]\n{zorla_metin}\n\n---\n\n"
                ) + baglam
        except Exception:
            pass

    # Soru pace/tempo/interval ile ilgiliyse, Strava bağlıysa gerçek pace
    # verilerini OTOMATİK olarak çekip bağlama ekle — kullanıcının ayrıca
    # /strava_ozet çalıştırmasına gerek kalmadan.
    if _kosu_sorusu_mu(soru):
        baglanti_bilgisi = strava_baglantisini_getir(kullanici_id)
        if baglanti_bilgisi:
            try:
                erisim_tokeni = strava_erisim_tokeni_al(baglanti_bilgisi["refresh_token"])
                pace_ozeti = strava_kosu_pace_ozeti(erisim_tokeni)
                if pace_ozeti:
                    baglam = (
                        f"[GERÇEK STRAVA VERİSİ — kullanıcının son koşularının gerçek pace "
                        f"değerleri, öneri verirken buna dayan]\n{pace_ozeti}\n\n---\n\n"
                    ) + baglam
            except Exception:
                pass

    gecmis = gecmisi_oku(kullanici_id)
    profil = profili_oku(kullanici_id)
    cevap = cevap_uret(client_gemini, soru, baglam, gecmis, gorsel_b64, gorsel_mime, yumusak, profil)

    mesaji_kaydet(kullanici_id, "user", soru)
    mesaji_kaydet(kullanici_id, "model", cevap)
    profili_otomatik_guncelle(client_gemini, kullanici_id, soru, cevap)

    context.chat_data["son_cevap"] = cevap

    dugmeler = InlineKeyboardMarkup([
        [InlineKeyboardButton("📅 Takvime Hazırla", callback_data="takvim"),
         InlineKeyboardButton("📊 Excel Yap", callback_data="excel")],
    ])
    await update.message.reply_text(cevap, reply_markup=dugmeler)


async def zorla_video_ayarla(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Kullanım: /zorla_video <video_id_veya_link> — bu videoyu bundan
    sonraki TÜM sorularında garanti olarak dikkate alır (arama şansına
    bırakmaz). Kapatmak için: /zorla_video kapat"""
    if not context.args:
        mevcut = context.user_data.get("zorla_video")
        await update.message.reply_text(
            f"Şu an aktif: {mevcut}" if mevcut else
            "Kullanım: /zorla_video <video_id_veya_link>\nKapatmak için: /zorla_video kapat"
        )
        return

    girdi = context.args[0]
    if girdi.lower() == "kapat":
        context.user_data.pop("zorla_video", None)
        await update.message.reply_text("Zorla video kapatıldı, normal aramaya döndük.")
        return

    video_id = girdi
    eslesme = re.search(r"watch\?v=([\w-]+)", girdi)
    if eslesme:
        video_id = eslesme.group(1)

    context.user_data["zorla_video"] = video_id
    await update.message.reply_text(
        f"Tamamdır, '{video_id}' videosunu bundan sonraki sorularında garanti "
        f"olarak dikkate alacağım. Kapatmak için: /zorla_video kapat"
    )


async def mesaj_geldi(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await _soruyu_isle(update, context, update.message.text)


async def ses_geldi(update: Update, context: ContextTypes.DEFAULT_TYPE):
    client_gemini, _ = istemcileri_al()
    dosya = await context.bot.get_file(update.message.voice.file_id)
    ses_bytes = bytes(await dosya.download_as_bytearray())
    yazi = ses_yaziya_cevir(client_gemini, ses_bytes, "audio/ogg")
    if not yazi:
        await update.message.reply_text("Ses anlaşılamadı, tekrar dener misin?")
        return
    await update.message.reply_text(f"🎤 Anladığım: \"{yazi}\"")
    await _soruyu_isle(update, context, yazi)


async def foto_geldi(update: Update, context: ContextTypes.DEFAULT_TYPE):
    dosya = await context.bot.get_file(update.message.photo[-1].file_id)
    foto_bytes = bytes(await dosya.download_as_bytearray())
    gorsel_b64 = base64.b64encode(foto_bytes).decode("utf-8")
    soru = update.message.caption or "Bu fotoğrafa bakıp yorumlar mısın?"
    await _soruyu_isle(update, context, soru, gorsel_b64, "image/jpeg")


async def buton_tiklandi(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    client_gemini, _ = istemcileri_al()
    son_cevap = context.chat_data.get("son_cevap", "")
    if not son_cevap:
        await query.message.reply_text("Önce bir program oluşturmam lazım.")
        return

    if query.data == "takvim":
        etkinlikler = programdan_json_cikar(client_gemini, son_cevap)
        if not etkinlikler:
            await query.message.reply_text("Bu cevapta takvime çevrilecek bir program bulamadım.")
            return
        ics_veri = ics_plan_olustur(etkinlikler)
        await query.message.reply_document(
            document=InputFile(BytesIO(ics_veri.encode("utf-8")), filename="program.ics"),
            caption="📅 Takvim dosyan hazır, Google Takvim'e aktarabilirsin."
        )
    elif query.data == "excel":
        satirlar = programdan_excel_json_cikar(client_gemini, son_cevap)
        if not satirlar:
            await query.message.reply_text("Bu cevapta Excel'e çevrilecek bir program bulamadım.")
            return
        excel_veri = excel_plan_olustur(satirlar)
        await query.message.reply_document(
            document=InputFile(BytesIO(excel_veri), filename="program.xlsx"),
            caption="📊 Excel dosyan hazır."
        )


async def _md_dosyasini_isle(update: Update, context: ContextTypes.DEFAULT_TYPE, belge):
    """Bir video transkripti (.md) dosyasını doğrudan canlı ChromaDB
    arşivine ekler — zip yükleme/redeploy derdi olmadan, anında."""
    dosya = await context.bot.get_file(belge.file_id)
    icerik_bytes = bytes(await dosya.download_as_bytearray())
    try:
        metin = icerik_bytes.decode("utf-8")
    except Exception as e:
        await update.message.reply_text(f"Dosya okunamadı: {e}")
        return

    if len(metin.strip()) < 50:
        await update.message.reply_text("Dosya çok kısa/boş görünüyor, atlandı.")
        return

    # "# Video Linki: https://youtube.com/watch?v=XXXX" satırından video_id çıkar
    video_id = belge.file_name.replace(".md", "")
    eslesme = re.search(r"watch\?v=([\w-]+)", metin)
    if eslesme:
        video_id = eslesme.group(1)

    try:
        _, koleksiyon = istemcileri_al()

        boyut, ortusme = 800, 150
        parcalar, baslangic = [], 0
        while baslangic < len(metin):
            parcalar.append(metin[baslangic:baslangic + boyut])
            baslangic += (boyut - ortusme)

        # Bu video daha önce eklenmişse eski kayıtlarını temizle (tekrar etmesin)
        try:
            eski = koleksiyon.get(where={"video_id": video_id}, include=[])
            if eski.get("ids"):
                koleksiyon.delete(ids=eski["ids"])
        except Exception:
            pass

        ids = [f"{video_id}_parca_{j}" for j in range(len(parcalar))]
        metadatalar = [{"video_id": video_id, "kaynak": belge.file_name} for _ in parcalar]
        koleksiyon.add(documents=parcalar, ids=ids, metadatas=metadatalar)

        await update.message.reply_text(
            f"✅ '{video_id}' videosu canlı arşive eklendi ({len(parcalar)} parça). "
            f"Artık bu içerikten sorular sorabilirsin."
        )
    except Exception as e:
        await update.message.reply_text(f"Arşive eklenirken hata oluştu: {e}")


async def belge_geldi(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """.json (eski sohbet) ya da .md (video transkripti) dosyası
    gönderildiğinde: doğrudan canlı arşive (ChromaDB) ekler."""
    belge = update.message.document

    if belge.file_name.endswith(".md"):
        await _md_dosyasini_isle(update, context, belge)
        return

    if not belge.file_name.endswith(".json"):
        await update.message.reply_text(
            "Şu an .json (eski sohbet) ya da .md (video transkripti) dosyası kabul ediyorum."
        )
        return

    dosya = await context.bot.get_file(belge.file_id)
    icerik_bytes = bytes(await dosya.download_as_bytearray())

    try:
        veri = json.loads(icerik_bytes.decode("utf-8"))
    except Exception as e:
        await update.message.reply_text(f"Dosya okunamadı: {e}")
        return

    mesajlar = veri.get("mesajlar", [])
    if not mesajlar:
        await update.message.reply_text("Bu dosyada mesaj bulunamadı, atlandı.")
        return

    baslik = veri.get("baslik", belge.file_name)
    kullanici_id = update.effective_user.id

    # 1) Arşive (ChromaDB) ekle — bilgi olarak her zaman aranabilir olsun
    try:
        _, koleksiyon = istemcileri_al()
        satirlar = [f"# Eski Sohbet: {baslik}\n"]
        for m in mesajlar:
            rol = "Kullanıcı" if m.get("role") == "user" else "Koçum"
            icerik = m.get("content", "")
            if icerik:
                satirlar.append(f"{rol}: {icerik}")
        metin = "\n\n".join(satirlar)

        parca_sayisi = 0
        if len(metin.strip()) >= 50:
            boyut, ortusme = 800, 150
            parcalar, baslangic = [], 0
            while baslangic < len(metin):
                parcalar.append(metin[baslangic:baslangic + boyut])
                baslangic += (boyut - ortusme)
            etiket = f"eski_sohbet_{uuid.uuid4().hex[:8]}"
            ids = [f"{etiket}_parca_{j}" for j in range(len(parcalar))]
            metadatalar = [{"video_id": f"Eski sohbet: {baslik}", "kaynak": "eski_sohbet"}
                            for _ in parcalar]
            koleksiyon.add(documents=parcalar, ids=ids, metadatas=metadatalar)
            parca_sayisi = len(parcalar)
    except Exception as e:
        parca_sayisi = f"HATA: {e}"

    # 2) GERÇEK konuşma geçmişine ekle — böylece bota "geçen sohbette
    #    ne demiştik" dediğinde arama yapmadan doğrudan hatırlar
    eklenen_mesaj = 0
    for m in mesajlar:
        icerik = m.get("content", "")
        rol = m.get("role")
        if not icerik:
            continue
        mesaji_kaydet(kullanici_id, "user" if rol == "user" else "model", icerik)
        eklenen_mesaj += 1

    await update.message.reply_text(
        f"📥 '{baslik}' eklendi:\n"
        f"- Arşive: {parca_sayisi} parça\n"
        f"- Gerçek sohbet geçmişine: {eklenen_mesaj} mesaj — artık doğrudan hatırlıyorum, "
        f"arama yapmama bile gerek yok."
    )


def main():
    if not TELEGRAM_BOT_TOKEN:
        print("HATA: TELEGRAM_BOT_TOKEN ayarlanmamış.")
        return
    _veritabanini_hazirla()
    _basit_semayi_hazirla()
    istemcileri_al()

    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("id", id_goster))
    app.add_handler(CommandHandler("yumusak_ac", yumusak_ac))
    app.add_handler(CommandHandler("yumusak_kapat", yumusak_kapat))
    app.add_handler(CommandHandler("temizle", temizle))
    app.add_handler(CommandHandler("web_sohbetlerini_getir", web_sohbetlerini_getir))
    app.add_handler(CommandHandler("strava_baglan", strava_baglan))
    app.add_handler(CommandHandler("son_antrenman", son_antrenman))
    app.add_handler(CommandHandler("video_var_mi", video_var_mi))
    app.add_handler(CommandHandler("zorla_video", zorla_video_ayarla))
    app.add_handler(CommandHandler("profil_goster", profil_goster))
    app.add_handler(CommandHandler("profil_ekle", profil_ekle))
    app.add_handler(CommandHandler("profil_sil", profil_sil))
    app.add_handler(CommandHandler("strava_ozet", strava_ozet))
    app.add_handler(CommandHandler("sabah_ac", sabah_ac))
    app.add_handler(CommandHandler("sabah_kapat", sabah_kapat))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, mesaj_geldi))
    app.add_handler(MessageHandler(filters.VOICE, ses_geldi))
    app.add_handler(MessageHandler(filters.PHOTO, foto_geldi))
    app.add_handler(MessageHandler(filters.Document.ALL, belge_geldi))
    app.add_handler(CallbackQueryHandler(buton_tiklandi))

    # Her 15 dakikada bir, tüm bağlı kullanıcıların yeni Strava aktivitesi
    # olup olmadığını kontrol eder — her kullanıcıya SADECE KENDİ verisi gider.
    if app.job_queue:
        app.job_queue.run_repeating(strava_kontrol_isi, interval=900, first=30)

        # Her sabah 07:00'de (Türkiye saati), /sabah_ac demiş kullanıcılara
        # otomatik, kişisel bir günaydın mesajı gönderir.
        try:
            from zoneinfo import ZoneInfo
            from datetime import time as _time
            app.job_queue.run_daily(
                sabah_mesaji_isi, time=_time(7, 0, tzinfo=ZoneInfo("Europe/Istanbul")),
            )
        except Exception as e:
            print(f"Sabah mesajı zamanlanamadı: {e}")

    print(f"{UYGULAMA_ADI} Telegram botu başlıyor...")
    app.run_polling()


if __name__ == "__main__":
    main()
