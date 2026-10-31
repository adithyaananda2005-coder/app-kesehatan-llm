import time

import pandas as pd
import streamlit as st
from groq import Groq

from classifier import NAMA_FITUR, classify, load_config

st.set_page_config(page_title="Klasifikasi Massa Jaringan Payudara (LLM)", page_icon="🤖")

# ---------- 1. Muat konfigurasi hasil evaluasi & siapkan client ----------
cfg = load_config("config.json")
FITUR = list(NAMA_FITUR)
BATAS_SESI = 25          # maksimal panggilan API per pengunjung (hemat kuota)
MAKS_BARIS_CSV = 10

# rentang nilai pada data latih (untuk nilai awal dan peringatan)
RENTANG = {
    "radius": (6.981, 28.11), "tekstur": (9.71, 39.28), "keliling": (43.79, 188.5),
    "luas": (143.5, 2501.0), "kehalusan": (0.053, 0.163), "kekompakan": (0.019, 0.345),
    "cekungan": (0.0, 0.427), "titik_cekung": (0.0, 0.201),
}

try:
    API_KEY = st.secrets["GROQ_API_KEY"]
except Exception:
    st.error("GROQ_API_KEY belum diatur. Isi di .streamlit/secrets.toml (lokal) "
             "atau menu Secrets (Streamlit Community Cloud).")
    st.stop()


@st.cache_resource
def get_client():
    return Groq(api_key=API_KEY)


@st.cache_resource
def get_simpanan():
    return {}            # hasil valid disimpan; data yang sama tidak memanggil API lagi


client = get_client()
if "panggilan" not in st.session_state:
    st.session_state["panggilan"] = 0


def prediksi(nilai):
    """Kembalikan label. Hasil 'invalid' tidak disimpan agar bisa dicoba lagi."""
    kunci = tuple(round(float(nilai[f]), 6) for f in FITUR)
    simpanan = get_simpanan()
    if kunci in simpanan:
        return simpanan[kunci]
    if st.session_state["panggilan"] >= BATAS_SESI:
        return "batas"
    st.session_state["panggilan"] += 1
    hasil = classify(client, cfg, {f: float(nilai[f]) for f in FITUR})
    if hasil != "invalid":
        simpanan[kunci] = hasil
    return hasil


def tampilkan(hasil):
    if hasil == "Jinak":
        st.success(f"Prediksi: **{hasil}**")
    elif hasil == "Ganas":
        st.error(f"Prediksi: **{hasil}**")
    elif hasil == "batas":
        st.warning("Batas penggunaan per sesi tercapai. Muat ulang halaman untuk memulai lagi.")
    else:
        st.warning("Layanan LLM sedang tidak merespons (kemungkinan kuota habis). Coba lagi nanti.")


# ---------- 2. Sidebar: informasi konfigurasi yang dikunci ----------
ev = cfg["evaluasi"]
with st.sidebar:
    st.header("Tentang Model")
    st.write(f"LLM: `{cfg['model']}` (Groq)")
    st.write(f"Few-shot: {len(cfg['few_shot'])} contoh")
    st.metric("F1 macro LLM (data uji)", f"{ev['f1_macro']:.3f}")
    if "f1_library_pembanding" in ev:
        st.metric("F1 macro model library (pembanding)", f"{ev['f1_library_pembanding']:.3f}")
    st.caption(f"Dievaluasi pada {ev['n']} data uji yang sama dengan model library, "
               f"tanggal {ev['tanggal']}")
    st.caption(f"Panggilan API sesi ini: {st.session_state['panggilan']} / {BATAS_SESI}")
    with st.expander("Lihat system prompt"):
        st.code(cfg["system_prompt"], language=None)

st.title("🤖 Klasifikasi Massa Jaringan Payudara dengan LLM")
st.write("Masukkan ukuran inti sel untuk memprediksi apakah massa **Jinak** atau **Ganas**.")
st.info("Aplikasi ini hanya untuk pembelajaran (tugas praktikum) dan "
        "**bukan alat diagnosis medis**. Hasil tidak boleh dipakai untuk keputusan kesehatan.")

tab1, tab2 = st.tabs(["Prediksi Satu Data", "Prediksi dari File CSV"])

# ---------- 3. Prediksi satu data ----------
with tab1:
    with st.form("form_prediksi"):
        nilai = {}
        kolom = st.columns(2)
        for i, f in enumerate(FITUR):
            lo, hi = RENTANG[f]
            with kolom[i % 2]:
                nilai[f] = st.number_input(
                    NAMA_FITUR[f], min_value=0.0, value=float(round((lo + hi) / 2, 4)),
                    step=float(round((hi - lo) / 100, 5)) or 0.001, format="%.5f",
                    help=f"Rentang pada data latih: {lo:g} sampai {hi:g}")
        kirim = st.form_submit_button("Prediksi")

    if kirim:
        luar = [c for c in FITUR if not RENTANG[c][0] <= nilai[c] <= RENTANG[c][1]]
        if luar:
            st.warning(f"Nilai {', '.join(luar)} di luar rentang data latih; "
                       "hasil prediksi bisa kurang akurat.")
        with st.spinner("Menghubungi Groq..."):
            hasil = prediksi(nilai)
        tampilkan(hasil)

# ---------- 4. Prediksi banyak data dari CSV ----------
with tab2:
    st.write("Unggah CSV dengan kolom: " + ", ".join(f"`{c}`" for c in FITUR)
             + f" (maksimal {MAKS_BARIS_CSV} baris agar hemat kuota).")
    with open("contoh_data_pasien.csv", "rb") as f_contoh:
        st.download_button("Unduh sampel data (contoh_data_pasien.csv)",
                           f_contoh.read(), file_name="contoh_data_pasien.csv",
                           mime="text/csv")
    file = st.file_uploader("Pilih file CSV", type="csv")
    if file is not None:
        df = pd.read_csv(file)
        kurang = [c for c in FITUR if c not in df.columns]
        if kurang:
            st.error(f"Kolom berikut tidak ditemukan: {', '.join(kurang)}")
        else:
            df = df.head(MAKS_BARIS_CSV)
            if st.button("Prediksi semua"):
                bar = st.progress(0.0)
                hasil = []
                for i, (_, b) in enumerate(df.iterrows(), start=1):
                    hasil.append(prediksi({f: b[f] for f in FITUR}))
                    bar.progress(i / len(df))
                    time.sleep(3)               # jaga batas request per menit
                df["prediksi_diagnosis"] = hasil
                st.dataframe(df)
                st.write(df["prediksi_diagnosis"].value_counts())
                st.download_button("Unduh hasil prediksi", df.to_csv(index=False),
                                   file_name="hasil_prediksi_llm.csv", mime="text/csv")