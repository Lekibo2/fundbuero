import json
import sqlite3
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image


# --------------------------------------------------
# Einstellungen
# --------------------------------------------------

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "model" / "kleidung_model.h5"
LABELS_PATH = BASE_DIR / "labels" / "classes.json"
DATABASE_PATH = BASE_DIR / "data" / "fundbuero.db"

# Diese Größe muss zur Eingabegröße deines Modells passen.
# Häufig sind es 224 x 224 oder 128 x 128.
IMAGE_SIZE = (224, 224)


# --------------------------------------------------
# Streamlit-Seiteneinstellungen
# --------------------------------------------------

st.set_page_config(
    page_title="Fundbüro Kleidung",
    page_icon="👕",
    layout="wide"
)


# --------------------------------------------------
# Modell und Kategorien laden
# --------------------------------------------------

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error(f"Modell nicht gefunden: {MODEL_PATH}")
        st.stop()

    return tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )



@st.cache_data
def load_labels():
    if not LABELS_PATH.exists():
        st.error(f"Kategorien nicht gefunden: {LABELS_PATH}")
        st.stop()

    with open(LABELS_PATH, "r", encoding="utf-8") as file:
        labels = json.load(file)

    # Unterstützt sowohl eine Liste als auch ein Dictionary
    if isinstance(labels, list):
        return labels

    if isinstance(labels, dict):
        labels = dict(sorted(labels.items(), key=lambda item: int(item[0])))
        return list(labels.values())

    st.error("Ungültiges Format in classes.json.")
    st.stop()


model = load_model()
labels = load_labels()


# --------------------------------------------------
# Datenbank
# --------------------------------------------------

def get_connection():
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(DATABASE_PATH)


def initialize_database():
    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS entries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            category TEXT NOT NULL,
            confidence REAL NOT NULL,
            upload_date TEXT NOT NULL,
            image BLOB NOT NULL
        )
        """
    )

    connection.commit()
    connection.close()


def save_entry(filename, category, confidence, upload_date, image_bytes):
    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        """
        INSERT INTO entries
        (filename, category, confidence, upload_date, image)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            filename,
            category,
            confidence,
            upload_date,
            image_bytes
        )
    )

    connection.commit()
    connection.close()


def get_all_entries():
    connection = get_connection()

    dataframe = pd.read_sql_query(
        """
        SELECT
            id,
            filename,
            category,
            confidence,
            upload_date,
            image
        FROM entries
        ORDER BY id DESC
        """,
        connection
    )

    connection.close()
    return dataframe


initialize_database()


# --------------------------------------------------
# Bilderkennung
# --------------------------------------------------

def prepare_image(image):
    """
    Bereitet das Bild für das Modell vor.

    Die Normalisierung /255.0 passt für viele Modelle.
    Falls dein Modell anders trainiert wurde, musst du diese Stelle anpassen.
    """

    image = image.convert("RGB")
    image = image.resize(IMAGE_SIZE)

    image_array = np.array(image).astype("float32")
    image_array = image_array / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    return image_array


def predict_category(image):
    prepared_image = prepare_image(image)

    prediction = model.predict(prepared_image, verbose=0)
    prediction = np.asarray(prediction)

    # Für normale Klassifikation:
    # prediction[0] enthält die Wahrscheinlichkeiten je Kategorie.
    probabilities = prediction[0]

    predicted_index = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_index])

    if predicted_index >= len(labels):
        category = f"Unbekannte Kategorie ({predicted_index})"
    else:
        category = labels[predicted_index]

    return category, confidence


# --------------------------------------------------
# Darstellung einzelner Einträge
# --------------------------------------------------

def display_entry(entry):
    image_bytes = entry["image"]

    image_column, info_column = st.columns([1, 2])

    with image_column:
        st.image(
            image_bytes,
            use_container_width=True
        )

    with info_column:
        st.subheader(entry["category"])
        st.write(f"**Dateiname:** {entry['filename']}")
        st.write(f"**Hochgeladen am:** {entry['upload_date']}")
        st.write(
            f"**Erkennungswahrscheinlichkeit:** "
            f"{entry['confidence'] * 100:.2f} %"
        )


# --------------------------------------------------
# Seiten-Navigation
# --------------------------------------------------

st.sidebar.title("Fundbüro")

page = st.sidebar.radio(
    "Navigation",
    [
        "Startseite",
        "Kleidungsstück hochladen",
        "Alle Einträge"
    ]
)


# --------------------------------------------------
# Startseite
# --------------------------------------------------

if page == "Startseite":
    st.title("👕 Fundbüro für Kleidungsstücke")

    st.write(
        "Lade ein Foto eines gefundenen Kleidungsstücks hoch. "
        "Das Modell erkennt automatisch die passende Kategorie."
    )

    dataframe = get_all_entries()

    # Suchfeld auf der Startseite
    search_text = st.text_input(
        "🔎 Einträge durchsuchen",
        placeholder="Zum Beispiel: Jacke, Schuhe oder 2026-09-21"
    )

    if dataframe.empty:
        st.info("Bisher wurden noch keine Kleidungsstücke hochgeladen.")

        if st.button("Ersten Eintrag hochladen"):
            st.switch_page("app.py")

    else:
        filtered_dataframe = dataframe.copy()

        if search_text:
            search_text = search_text.lower()

            filtered_dataframe = filtered_dataframe[
                filtered_dataframe["category"]
                .str.lower()
                .str.contains(search_text, na=False)
                |
                filtered_dataframe["filename"]
                .str.lower()
                .str.contains(search_text, na=False)
                |
                filtered_dataframe["upload_date"]
                .str.lower()
                .str.contains(search_text, na=False)
            ]

        st.subheader("Neueste Einträge")

        newest_entries = filtered_dataframe.head(6)

        if newest_entries.empty:
            st.warning("Keine passenden Einträge gefunden.")
        else:
            for _, entry in newest_entries.iterrows():
                display_entry(entry)
                st.divider()

        if st.button("Alle Einträge anzeigen"):
            st.switch_page("app.py")


# --------------------------------------------------
# Upload-Seite
# --------------------------------------------------

elif page == "Kleidungsstück hochladen":
    st.title("📤 Kleidungsstück hochladen")

    uploaded_file = st.file_uploader(
        "Foto auswählen",
        type=["jpg", "jpeg", "png", "webp"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        st.subheader("Hochgeladenes Bild")
        st.image(image, caption=uploaded_file.name, width=400)

        if st.button("Bild analysieren", type="primary"):
            with st.spinner("Das Kleidungsstück wird erkannt..."):
                category, confidence = predict_category(image)

            st.success(f"Erkannte Kategorie: {category}")

            st.write(
                f"Erkennungswahrscheinlichkeit: "
                f"{confidence * 100:.2f} %"
            )

            # Datum und Uhrzeit des Uploads
            upload_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            image_bytes = uploaded_file.getvalue()

            save_entry(
                filename=uploaded_file.name,
                category=category,
                confidence=confidence,
                upload_date=upload_date,
                image_bytes=image_bytes
            )

            st.success("Der Eintrag wurde erfolgreich gespeichert.")

            st.cache_data.clear()


# --------------------------------------------------
# Alle Einträge
# --------------------------------------------------

elif page == "Alle Einträge":
    st.title("📋 Alle Fundbüro-Einträge")

    dataframe = get_all_entries()

    if dataframe.empty:
        st.info("Es sind noch keine Einträge vorhanden.")
    else:
        category_filter = st.selectbox(
            "Nach Kategorie filtern",
            ["Alle"] + sorted(dataframe["category"].unique().tolist())
        )

        filtered_dataframe = dataframe.copy()

        if category_filter != "Alle":
            filtered_dataframe = filtered_dataframe[
                filtered_dataframe["category"] == category_filter
            ]

        st.write(
            f"{len(filtered_dataframe)} Einträge gefunden."
        )

        for _, entry in filtered_dataframe.iterrows():
            display_entry(entry)
            st.divider()
