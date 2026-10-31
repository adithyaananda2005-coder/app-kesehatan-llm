"""Klasifikasi diagnosis (Jinak/Ganas) menggunakan LLM (Groq API).
Dipakai saat EVALUASI dan saat DEPLOY di Streamlit."""
import json
import re
import time
import groq

NAMA_FITUR = {
    "radius": "Radius inti sel",
    "tekstur": "Tekstur (simpangan baku nilai abu-abu)",
    "keliling": "Keliling inti sel",
    "luas": "Luas inti sel",
    "kehalusan": "Kehalusan kontur (variasi panjang radius lokal)",
    "kekompakan": "Kekompakan (keliling kuadrat / luas - 1)",
    "cekungan": "Tingkat cekungan kontur",
    "titik_cekung": "Jumlah titik cekung pada kontur",
}

def load_config(path="config.json"):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def format_data(data):
    return "; ".join(f"{nama}: {data[kolom]}" for kolom, nama in NAMA_FITUR.items())

def build_messages(cfg, data):
    msgs = [{"role": "system", "content": cfg["system_prompt"]}]
    for ex in cfg.get("few_shot", []):
        msgs.append({"role": "user", "content": format_data(ex["data"])})
        msgs.append({"role": "assistant", "content": ex["label"]})
    msgs.append({"role": "user", "content": format_data(data)})
    return msgs

def parse_label(output, labels):
    jawab = re.sub(r"[^a-z]", "", (output or "").lower())
    peta = {label.lower(): label for label in labels}
    return peta.get(jawab, "invalid")

def classify(client, cfg, data, max_retry=3):
    for percobaan in range(max_retry):
        try:
            kwargs = dict(
                model=cfg["model"],
                temperature=cfg["temperature"],
                max_tokens=cfg["max_tokens"],
                messages=build_messages(cfg, data),
            )
            if cfg.get("reasoning_effort"):
                kwargs["reasoning_effort"] = cfg["reasoning_effort"]
            resp = client.chat.completions.create(**kwargs)
            return parse_label(resp.choices[0].message.content, cfg["labels"])
        except groq.RateLimitError:
            time.sleep(15 * (percobaan + 1))
    return "invalid"