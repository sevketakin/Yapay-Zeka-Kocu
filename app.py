"""
Koçum - Chainlit Tabanlı Kişisel Antrenör Asistanı
================================================================

Streamlit'ten Chainlit'e geçiş. Chainlit, LLM sohbet uygulamaları için
özel tasarlanmış bir framework olduğu için, ChatGPT'ye çok benzer,
kutudan çıktığı gibi cilalı bir arayüz, çoklu sohbet geçmişi ve daha
stabil dosya/görsel yükleme sağlıyor.

KURULUM:
    pip install chainlit sqlalchemy aiosqlite chromadb google-genai openpyxl

BİR KEZ YAPILACAK - Kimlik doğrulama anahtarı oluştur:
    chainlit create-secret
    (çıkan satırı .env dosyasına CHAINLIT_AUTH_SECRET=... şeklinde kaydet)

ÇALIŞTIRMA:
    chainlit run app.py -w

Bu komut tarayıcında otomatik olarak bir sekme açacak (localhost:8000).
"""

import os
import re
import json
import uuid
import base64
import zipfile
from datetime import datetime, timedelta
from io import BytesIO

import chainlit as cl
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from google import genai

# ============== AYARLAR ==============
# VERI_KLASORU: Railway'de kalıcı Volume'un bağlandığı yer (örn. /data).
# Yerelde test ederken normal klasörleri kullanır (Volume yoksa).
VERI_KLASORU = os.environ.get("VERI_KLASORU", ".")
DB_KLASORU = os.path.join(VERI_KLASORU, "veritabani")
SOHBET_DB_YOLU = os.path.join(VERI_KLASORU, "koc_data.db")

KOLEKSIYON_ADI = "video_transkriptleri"
GEMINI_API_ANAHTARI = os.environ.get("GEMINI_API_ANAHTARI", "BURAYA_API_ANAHTARINI_YAPISTIR")
GEMINI_MODEL_LISTESI = [
    "gemini-flash-latest",
    "gemini-3.5-flash",
    "gemini-3.1-flash-lite",
    "gemini-3-flash-preview",
]
KAC_PARCA_GETIRILSIN = 15
UYGULAMA_ADI = "Koçum"
# =======================================

# ============== VERİTABANI ZIP'İNİ OTOMATİK İNDİRME + BİRLEŞTİRME + AÇMA ==============
# Öncelik sırası:
# 1) '/data/veritabani' zaten varsa hiçbir şey yapma
# 2) Yoksa, önce mevcut zip'in SAĞLAM olup olmadığını kontrol et — bozuksa sil
# 3) GitHub Release'ten indir (GITHUB_ZIP_URL tanımlıysa)
# 4) Ya da yerelde yüklenmiş parçaları (veritabani.zip.001, ...) birleştir
# 5) Zip'i aç, açma da bozuksa tekrar sil ve baştan indir (bir kere daha dene)
GITHUB_ZIP_URL = (
    "https://github.com/sevketakin/Yapay-Zeka-Kocu/releases/download/"
    "v1/veritabani.zip"
)


def _zip_saglam_mi(yol):
    try:
        with zipfile.ZipFile(yol, 'r') as z:
            return z.testzip() is None
    except Exception:
        return False


def _veritabanini_hazirla():
    import glob as _glob
    import requests

    _tek_zip = os.path.join(VERI_KLASORU, "veritabani.zip")
    _birlesik_zip = os.path.join(VERI_KLASORU, "_veritabani_birlesik.zip")

    # Var olan ama BOZUK dosyaları temizle (önceki başarısız denemelerden kalmış olabilir)
    for _yol in [_tek_zip, _birlesik_zip]:
        if os.path.exists(_yol) and not _zip_saglam_mi(_yol):
            print(f"'{_yol}' bozuk görünüyor, siliniyor...")
            os.remove(_yol)

    if GITHUB_ZIP_URL and not os.path.exists(_tek_zip):
        try:
            print("Veritabanı GitHub Release'ten indiriliyor... (bu birkaç dakika sürebilir)")
            with requests.get(GITHUB_ZIP_URL, stream=True, allow_redirects=True, timeout=120) as yanit:
                yanit.raise_for_status()
                toplam = int(yanit.headers.get("content-length", 0))
                indirilen = 0
                with open(_tek_zip, "wb") as f:
                    for parca in yanit.iter_content(chunk_size=8 * 1024 * 1024):
                        f.write(parca)
                        indirilen += len(parca)
                        if toplam:
                            print(f"  {indirilen / (1024*1024):.0f}MB / {toplam / (1024*1024):.0f}MB")
            print("İndirme tamamlandı.")
        except Exception as e:
            print(f"GitHub'dan indirme başarısız: {e}")

    if not os.path.exists(_tek_zip) or not _zip_saglam_mi(_tek_zip):
        _parcalar = sorted(_glob.glob(os.path.join(VERI_KLASORU, "veritabani.zip.*")))
        if _parcalar and not os.path.exists(_birlesik_zip):
            print(f"{len(_parcalar)} parça bulundu, birleştiriliyor...")
            with open(_birlesik_zip, "wb") as cikti:
                for parca in _parcalar:
                    with open(parca, "rb") as p:
                        cikti.write(p.read())
            print("Parçalar birleştirildi.")

    _acilacak_zip = _tek_zip if _zip_saglam_mi(_tek_zip) else (
        _birlesik_zip if os.path.exists(_birlesik_zip) and _zip_saglam_mi(_birlesik_zip) else None
    )

    if not _acilacak_zip:
        print("HATA: Sağlam bir veritabanı zip'i bulunamadı/indirilemedi.")
        return

    print(f"'{_acilacak_zip}' açılıyor... (bu biraz sürebilir)")
    with zipfile.ZipFile(_acilacak_zip, 'r') as z:
        z.extractall(VERI_KLASORU)
    print("Veritabanı hazır.")


if not os.path.exists(DB_KLASORU):
    _veritabanini_hazirla()

AYLAR_TR = ["Ocak", "Şubat", "Mart", "Nisan", "Mayıs", "Haziran",
            "Temmuz", "Ağustos", "Eylül", "Ekim", "Kasım", "Aralık"]
GUNLER_TR = ["Pazartesi", "Salı", "Çarşamba", "Perşembe", "Cuma", "Cumartesi", "Pazar"]
GUN_INDEX = {"Pazartesi": 0, "Salı": 1, "Çarşamba": 2, "Perşembe": 3,
             "Cuma": 4, "Cumartesi": 5, "Pazar": 6}


class GeminiEmbeddingFonksiyonu(EmbeddingFunction):
    """Veritabanı 'gemini-embedding-001' ile indexlendiği için, arama
    yaparken de AYNI modeli kullanmamız gerekiyor — aksi halde
    embedding'ler uyumsuz olur ve arama anlamsızlaşır."""

    def __init__(self, api_anahtari):
        self.client = genai.Client(api_key=api_anahtari)

    def __call__(self, input: Documents) -> Embeddings:
        return [self._tek_embedding_al(metin) for metin in input]

    def _tek_embedding_al(self, metin, max_deneme=3):
        import time
        for deneme in range(1, max_deneme + 1):
            try:
                sonuc = self.client.models.embed_content(
                    model="gemini-embedding-001",
                    contents=metin,
                )
                return sonuc.embeddings[0].values
            except Exception as e:
                hata_metni = str(e)
                if "429" in hata_metni or "RESOURCE_EXHAUSTED" in hata_metni or "UNAVAILABLE" in hata_metni:
                    time.sleep(5)
                    continue
                raise
        raise RuntimeError(f"Embedding alınamadı: {metin[:50]}...")


def bugunun_tarihi():
    simdi = datetime.now()
    return f"{simdi.day} {AYLAR_TR[simdi.month - 1]} {simdi.year}, {GUNLER_TR[simdi.weekday()]}"


def su_anki_saat():
    simdi = datetime.now()
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


# ============== TEKİL İSTEMCİLER (bir kere oluşturulur) ==============
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
        link = f"https://youtube.com/watch?v={video_id}"
        baglam_parcalari.append(f"[Kaynak video: {link}]\n{metin}")
        onizleme = metin[:250].strip() + ("..." if len(metin) > 250 else "")
        kaynaklar.append({"link": link, "onizleme": onizleme})
    return "\n\n---\n\n".join(baglam_parcalari), kaynaklar


def ses_yaziya_cevir(client_gemini, ses_bytes, ses_mime_tipi="audio/wav"):
    ses_b64 = base64.b64encode(ses_bytes).decode("utf-8")
    for model_adi in GEMINI_MODEL_LISTESI:
        try:
            yanit = client_gemini.models.generate_content(
                model=model_adi,
                contents=[
                    {"role": "user", "parts": [
                        {"text": "Bu ses kaydını sadece yazıya çevir, başka hiçbir şey ekleme, sadece söylenen metni döndür."},
                        {"inline_data": {"mime_type": ses_mime_tipi, "data": ses_b64}},
                    ]}
                ],
            )
            return yanit.text.strip()
        except Exception:
            continue
    return None


def cevap_uret(client_gemini, soru, baglam, gecmis, gorsel_b64=None, gorsel_mime=None):
    bugun = bugunun_tarihi()
    saat, vakit = su_anki_saat()
    sistem_mesaji = (
        f"Bugünün tarihi: {bugun}. Şu anki saat: {saat} ({vakit}). "
        f"Sorulara göre (örneğin bu haftaki antrenman planı, kaç gün kaldı, "
        f"bugün ne yesem, şu an antrenman yapmalı mıyım gibi) bu tarih ve "
        f"saat bilgisini mutlaka dikkate al ve cevabını buna göre uyarla. "
        f"Örneğin: gece geç saatte ağır bir antrenman veya kafeinli bir "
        f"şey önerme, sabah/akşam selamlaşmanı zaman dilimine göre yap, "
        f"'bu akşam' veya 'yarın sabah' gibi ifadeleri gerçek zamana göre "
        f"doğru kullan.\n\n"
        "Sen benim kişisel hybrid antrenörümsün. Amacın, beni hybrid "
        "sistemde (koşu, bisiklet, yüzme, kuvvet vb. dallarda) en iyi "
        "versiyonuma ulaştırmak için elinden gelen tüm yardımı, bilgiyi "
        "ve desteği sağlamak. Spor bilimi, antrenman periyotlaması, "
        "beslenme, toparlanma ve hybrid atletizm konularında yıllarını "
        "vermiş, sahada bizzat çalışmış, deneyimli bir antrenör/sporcu "
        "gibisin.\n\n"
        "Sana ayrıca, senin yıllar içinde izlediğin, öğrendiğin ve "
        "tecrübe edindiğin antrenman videolarından/içeriklerden bazı "
        "hatırlatma notları verilecek. BUNLAR SENİN KENDİ BİLGİ "
        "BİRİKİMİN — dışarıdan bir kaynak, döküman veya alıntı DEĞİL. "
        "Bu bilgiyi asla 'bu videoda deniyor ki', 'transkriptte geçiyor "
        "ki', 'kaynağa göre' gibi ifadelerle aktarma ve asla dışarıdan "
        "bir referans gibi sunma. Bunun yerine, sanki yıllar içinde "
        "bizzat öğrenip özümsediğin, kendi tecrüben ve bilgin haline "
        "gelmiş bir şeymiş gibi, doğal ve kendinden emin bir şekilde "
        "anlat — tıpkı gerçek bir antrenörün, okuduğu/izlediği onlarca "
        "kaynaktan süzdüğü bilgiyi kendi sözleriyle, kendi tecrübesi "
        "gibi aktarması gibi.\n\n"
        "ÖNCELİK SIRASI (ÇOK ÖNEMLİ): Bu notlar soruyla alakalıysa, "
        "cevabını ÖNCELİKLE ve AĞIRLIKLI OLARAK bu notlara dayandır — "
        "onlar senin birincil, en somut ve en güncel bilgi kaynağın gibi "
        "davran. Kendi genel bilgin sadece bu notları TAMAMLAMAK, "
        "boşlukları doldurmak veya bağlam eklemek için ikincil bir "
        "destek olarak kullanılır. Yani alakalı not varken genel "
        "bilgini öne çıkarıp notu arka planda bırakma — tam tersi "
        "olsun. Notlar arasında çelişki varsa ya da soru notlarla hiç "
        "alakalı değilse, o zaman kendi genel uzmanlığından tam ve "
        "kaliteli bir cevap ver.\n\n"
        "TON: Benimle samimi, sıcak, motive edici ve gerçek bir antrenör "
        "gibi konuş — resmi/mesafeli bir asistan gibi değil. Beni "
        "tanıyan, hedeflerime yatırım yapmış biri gibi davran. Önceki "
        "mesajları dikkate alarak sohbetin bağlamını koru."
    )

    contents = list(gecmis)
    kullanici_mesaji = f"""(Senin kendi bilgi birikimin/hafızandan gelen, bu soruyla \
alakalı olabilecek notlar — alakasızsa yok say, alakalıysa kendi tecrübenmiş gibi kullan):

{baglam}

SORU: {soru}"""

    parts = [{"text": kullanici_mesaji}]
    if gorsel_b64:
        parts.append({"inline_data": {"mime_type": gorsel_mime, "data": gorsel_b64}})
    contents.append({"role": "user", "parts": parts})

    son_hata = None
    for model_adi in GEMINI_MODEL_LISTESI:
        for deneme in range(1, 3):
            try:
                yanit = client_gemini.models.generate_content(
                    model=model_adi,
                    contents=contents,
                    config={"system_instruction": sistem_mesaji},
                )
                metin = yanit.text
                if metin and metin.strip():
                    return metin
                son_hata = "Model boş cevap döndürdü"
                break
            except Exception as e:
                hata_metni = str(e)
                son_hata = hata_metni
                if "503" in hata_metni or "UNAVAILABLE" in hata_metni:
                    continue
                elif "404" in hata_metni or "NOT_FOUND" in hata_metni:
                    break
                else:
                    continue

    return (f"Şu an cevap üretemedim, lütfen tekrar dener misin? "
            f"(Teknik detay: {son_hata})")


# ============== PROGRAM -> TAKVİM (.ics) DÖNÜŞÜMÜ ==============
def programdan_json_cikar(client_gemini, program_metni):
    talimat = (
        "Aşağıdaki metinde bir antrenman/beslenme/aktivite programı var. "
        "Bunu SADECE aşağıdaki JSON formatında, başka hiçbir açıklama/metin "
        "eklemeden döndür:\n\n"
        '[{"gun": "Pazartesi", "baslik": "Bacak Antrenmanı", '
        '"baslangic_saat": "18:00", "bitis_saat": "19:00", '
        '"aciklama": "kısa açıklama"}, ...]\n\n'
        "Kurallar:\n"
        "- gun değeri sadece şunlardan biri olmalı: Pazartesi, Salı, Çarşamba, "
        "Perşembe, Cuma, Cumartesi, Pazar\n"
        "- Saat belirtilmemişse mantıklı bir saat öner\n"
        "- Metinde gerçekten gün/aktivite içeren bir program yoksa boş liste "
        "[] döndür\n\n"
        f"METİN:\n{program_metni}"
    )
    for model_adi in GEMINI_MODEL_LISTESI:
        try:
            yanit = client_gemini.models.generate_content(model=model_adi, contents=talimat)
            metin = yanit.text.strip()
            metin = re.sub(r"^```json\s*|\s*```$", "", metin.strip(), flags=re.MULTILINE)
            metin = metin.strip("`").strip()
            return json.loads(metin)
        except Exception:
            continue
    return None


def sonraki_gun_tarihi(hedef_gun_index):
    bugun = datetime.now().date()
    fark = (hedef_gun_index - bugun.weekday()) % 7
    return bugun + timedelta(days=fark)


def ics_plan_olustur(etkinlikler, baslik_oneki=""):
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
        uid = str(uuid.uuid4())
        baslik = e.get("baslik", "Etkinlik")
        if baslik_oneki:
            baslik = f"[{baslik_oneki}] {baslik}"
        satirlar.append("BEGIN:VEVENT")
        satirlar.append(f"UID:{uid}")
        satirlar.append(f"DTSTAMP:{datetime.now().strftime('%Y%m%dT%H%M%SZ')}")
        satirlar.append(f"DTSTART;TZID=Europe/Istanbul:{tarih.strftime('%Y%m%d')}T{bs_saat:02d}{bs_dk:02d}00")
        satirlar.append(f"DTEND;TZID=Europe/Istanbul:{tarih.strftime('%Y%m%d')}T{bt_saat:02d}{bt_dk:02d}00")
        satirlar.append(f"SUMMARY:{baslik}")
        if e.get("aciklama"):
            satirlar.append(f"DESCRIPTION:{e['aciklama']}")
        satirlar.append("END:VEVENT")
    satirlar.append("END:VCALENDAR")
    return "\r\n".join(satirlar)


# ============== PROGRAM -> EXCEL (.xlsx) DÖNÜŞÜMÜ ==============
def programdan_excel_json_cikar(client_gemini, program_metni):
    talimat = (
        "Aşağıdaki metinde bir antrenman/beslenme/aktivite programı var. "
        "Bunu SADECE aşağıdaki JSON formatında, başka hiçbir açıklama/metin "
        "eklemeden döndür:\n\n"
        '[{"gun": "Pazartesi", "kategori": "Antrenman", '
        '"baslik": "Bacak Antrenmanı", '
        '"detay": "Squat 4x8, Leg Press 3x12, ...", '
        '"sure_kalori": "60 dk"}, ...]\n\n'
        "Kurallar:\n"
        "- kategori değeri 'Antrenman', 'Beslenme' ya da 'Diğer' olmalı\n"
        "- detay alanına set/tekrar sayılarını, öğün içeriğini vb. eksiksiz yaz\n"
        "- Metinde gerçekten bir program yoksa boş liste [] döndür\n\n"
        f"METİN:\n{program_metni}"
    )
    for model_adi in GEMINI_MODEL_LISTESI:
        try:
            yanit = client_gemini.models.generate_content(model=model_adi, contents=talimat)
            metin = yanit.text.strip()
            metin = re.sub(r"^```json\s*|\s*```$", "", metin.strip(), flags=re.MULTILINE)
            metin = metin.strip("`").strip()
            return json.loads(metin)
        except Exception:
            continue
    return None


def excel_plan_olustur(satirlar, baslik="Program"):
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side

    wb = Workbook()
    ws = wb.active
    ws.title = baslik[:31] if baslik else "Program"

    basliklar = ["Gün", "Kategori", "Başlık", "Detay", "Süre/Kalori"]
    baslik_yazi_tipi = Font(name="Arial", bold=True, color="FFFFFF", size=11)
    baslik_dolgu = PatternFill(start_color="1E293B", end_color="1E293B", fill_type="solid")
    hucre_yazi_tipi = Font(name="Arial", size=10)
    kenarlik = Border(*(Side(style="thin", color="D1D5DB"),) * 4)

    for sutun, metin in enumerate(basliklar, start=1):
        hucre = ws.cell(row=1, column=sutun, value=metin)
        hucre.font = baslik_yazi_tipi
        hucre.fill = baslik_dolgu
        hucre.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        hucre.border = kenarlik

    for satir_no, e in enumerate(satirlar, start=2):
        degerler = [e.get("gun", ""), e.get("kategori", ""), e.get("baslik", ""),
                    e.get("detay", ""), e.get("sure_kalori", "")]
        for sutun, deger in enumerate(degerler, start=1):
            hucre = ws.cell(row=satir_no, column=sutun, value=deger)
            hucre.font = hucre_yazi_tipi
            hucre.alignment = Alignment(vertical="top", wrap_text=True)
            hucre.border = kenarlik

    genislikler = [12, 12, 22, 45, 14]
    for i, genislik in enumerate(genislikler, start=1):
        ws.column_dimensions[ws.cell(row=1, column=i).column_letter].width = genislik
    ws.freeze_panes = "A2"

    arabellek = BytesIO()
    wb.save(arabellek)
    return arabellek.getvalue()


# ============== KİMLİK DOĞRULAMA (basit, tek kullanıcı - local kullanım) ==============
@cl.password_auth_callback
def auth_callback(username: str, password: str):
    # Sadece kendi bilgisayarında/local kullanım için - herhangi bir
    # kullanıcı adıyla giriş kabul edilir, gerçek bir güvenlik katmanı değil.
    return cl.User(identifier=username or "ben")


@cl.data_layer
def get_data_layer():
    return SQLAlchemyDataLayer(conninfo=f"sqlite+aiosqlite:///{SOHBET_DB_YOLU}")


# ============== CHAINLIT OLAYLARI ==============
@cl.on_chat_start
async def basla():
    istemcileri_al()
    cl.user_session.set("gecmis", [])
    await cl.Message(
        content=f"Merhaba! Ben **{UYGULAMA_ADI}**, senin kişisel hybrid antrenörünüm. "
                f"Bugün nasıl yardımcı olabilirim?\n\n"
                f"💡 İpucu: Bir fotoğraf ya da ses kaydı eklemek için mesaj kutusundaki "
                f"ataç ikonunu kullanabilirsin."
    ).send()


@cl.on_message
async def mesaj_geldi(message: cl.Message):
    client_gemini, koleksiyon = istemcileri_al()
    soru = message.content or ""

    gorsel_b64, gorsel_mime = None, None
    for el in message.elements:
        mime = getattr(el, "mime", "") or ""
        yol = getattr(el, "path", None)
        if not yol:
            continue
        if mime.startswith("image"):
            with open(yol, "rb") as f:
                gorsel_b64 = base64.b64encode(f.read()).decode("utf-8")
            gorsel_mime = mime
        elif mime.startswith("audio"):
            with open(yol, "rb") as f:
                ses_bytes = f.read()
            async with cl.Step(name="Ses yazıya çevriliyor", type="tool"):
                yazi = await cl.make_async(ses_yaziya_cevir)(client_gemini, ses_bytes, mime)
            if yazi:
                soru = (soru + "\n" + yazi).strip() if soru else yazi

    if not soru:
        await cl.Message(content="Bir metin, fotoğraf ya da ses kaydı gönderir misin?").send()
        return

    async with cl.Step(name="Arşiv taranıyor", type="tool"):
        bulunan = await cl.make_async(koleksiyon.query)(
            query_texts=[soru], n_results=KAC_PARCA_GETIRILSIN
        )
        baglam, kaynaklar = "", []
        if bulunan['documents'][0]:
            baglam, kaynaklar = baglami_hazirla(bulunan)

    gecmis = cl.user_session.get("gecmis", [])
    cevap = await cl.make_async(cevap_uret)(client_gemini, soru, baglam, gecmis, gorsel_b64, gorsel_mime)

    gecmis = gecmis + [
        {"role": "user", "parts": [{"text": soru}]},
        {"role": "model", "parts": [{"text": cevap}]},
    ]
    cl.user_session.set("gecmis", gecmis)

    actions = [
        cl.Action(name="takvim_yap", payload={"metin": cevap}, label="📅 Takvime Hazırla"),
        cl.Action(name="excel_yap", payload={"metin": cevap}, label="📊 Excel Yap"),
    ]

    elementler = []
    if kaynaklar:
        kaynak_metni = "\n".join(f"- {k['link']}" for k in kaynaklar[:10])
        elementler.append(cl.Text(name="Kullanılan kaynaklar", content=kaynak_metni, display="side"))

    await cl.Message(content=cevap, actions=actions, elements=elementler).send()


@cl.action_callback("takvim_yap")
async def takvim_yap(action: cl.Action):
    metin = action.payload.get("metin", "")
    client_gemini, _ = istemcileri_al()
    async with cl.Step(name="Program takvime çevriliyor", type="tool"):
        etkinlikler = await cl.make_async(programdan_json_cikar)(client_gemini, metin)
    if not etkinlikler:
        await cl.Message(content="Bu mesajda takvime çevrilecek bir program bulunamadı.").send()
        return
    ics_veri = ics_plan_olustur(etkinlikler, baslik_oneki=UYGULAMA_ADI)
    dosya = cl.File(name="program.ics", content=ics_veri.encode("utf-8"))
    await cl.Message(content="📅 Takvim dosyan hazır, indirip Google Takvim'e aktarabilirsin:",
                      elements=[dosya]).send()


@cl.action_callback("excel_yap")
async def excel_yap(action: cl.Action):
    metin = action.payload.get("metin", "")
    client_gemini, _ = istemcileri_al()
    async with cl.Step(name="Program Excel'e çevriliyor", type="tool"):
        satirlar = await cl.make_async(programdan_excel_json_cikar)(client_gemini, metin)
    if not satirlar:
        await cl.Message(content="Bu mesajda Excel'e çevrilecek bir program bulunamadı.").send()
        return
    excel_veri = excel_plan_olustur(satirlar, baslik=UYGULAMA_ADI)
    dosya = cl.File(name="program.xlsx", content=excel_veri)
    await cl.Message(content="📊 Excel dosyan hazır:", elements=[dosya]).send()
