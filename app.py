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
    page_title="Fundbüro",
    page_icon="👕",
    layout="wide"
)

st.markdown(
    """
    <style>
        /* Seitenbereich */
        .block-container {
            padding-top: 1rem;
            padding-left: 3rem;
            padding-right: 3rem;
            max-width: 1400px;
        }

        /* Obere Leiste */
        .topbar {
            background-color: #1f2937;
            padding: 22px 30px;
            border-radius: 14px;
            margin-bottom: 25px;
            text-align: center;
        }

        .topbar h1 {
            color: white;
            margin: 0;
            font-size: 32px;
        }

        /* Zentrierte Buttons */
        .center-button {
            display: flex;
            justify-content: center;
            margin: 18px 0 28px 0;
        }

        /* Karten für Einträge */
        .entry-card {
            background-color: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 12px;
            min-height: 300px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
        }

        .entry-title {
            font-size: 18px;
            font-weight: 600;
            margin-top: 8px;
            color: #1f2937;
        }

        .entry-info {
            font-size: 13px;
            color: #64748b;
            margin-top: 5px;
        }

        /* Upload-Bereich */
        .upload-area {
            margin-top: 30px;
            padding: 25px;
            text-align: center;
            background-color: #003366;
            border-radius: 14px;
        }

        /* Sidebar ausblenden */
        [data-testid="stSidebar"] {
            display: none;
        }

        /* Buttons etwas größer */
        .stButton > button {
            border-radius: 8px;
            font-weight: 600;
        }
    </style>
    """,
    unsafe_allow_html=True
)

if "page" not in st.session_state:
    st.session_state.page = "home"


def go_to_page(page_name):
    st.session_state.page = page_name


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
# Startseite
# --------------------------------------------------

if st.session_state.page == "home":
    # Obere Leiste
    st.markdown(
        """
        <div class="topbar">
            <h1>👕 Fundbüro</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Suchleiste
    search_text = st.text_input(
        "Suche",
        placeholder="Nach Kategorie, Dateiname oder Datum suchen...",
        label_visibility="collapsed"
    )

    # Alle-Einträge-Button
    st.markdown('<div class="center-button">', unsafe_allow_html=True)

    all_entries_button = st.button(
        "Alle Einträge anzeigen",
        use_container_width=False
    )

    if all_entries_button:
        go_to_page("all")

    st.markdown("</div>", unsafe_allow_html=True)

    dataframe = get_all_entries()

    if search_text:
        search_text_lower = search_text.lower()

        dataframe = dataframe[
            dataframe["category"]
            .str.lower()
            .str.contains(search_text_lower, na=False)
            |
            dataframe["filename"]
            .str.lower()
            .str.contains(search_text_lower, na=False)
            |
            dataframe["upload_date"]
            .str.lower()
            .str.contains(search_text_lower, na=False)
        ]

    st.subheader("Letzte Einträge")

    if dataframe.empty:
        st.info("Bisher wurden noch keine Einträge gespeichert.")
    else:
        latest_entries = dataframe.head(4)

        # Vier horizontale Spalten
        columns = st.columns(4)

        for column, (_, entry) in zip(columns, latest_entries.iterrows()):
            with column:
                st.markdown(
                    '<div class="entry-card">',
                    unsafe_allow_html=True
                )

                st.image(
                    entry["image"],
                    use_container_width=True
                )

                st.markdown(
                    f'<div class="entry-title">{entry["category"]}</div>',
                    unsafe_allow_html=True
                )

                st.markdown(
                    f'<div class="entry-info">'
                    f'{entry["upload_date"]}'
                    f'</div>',
                    unsafe_allow_html=True
                )

                st.markdown(
                    f'<div class="entry-info">'
                    f'Erkennung: {entry["confidence"] * 100:.1f} %'
                    f'</div>',
                    unsafe_allow_html=True
                )

                st.markdown("</div>", unsafe_allow_html=True)

    # Upload-Bereich
    st.markdown(
        """
        <div class="upload-area">
            <h3>Neuen Fund hinzufügen</h3>
            <p>Fotografiere oder lade ein Kleidungsstück beziehungsweise einen Gegenstand hoch.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    upload_button = st.button(
        "📤 Gegenstand hochladen",
        use_container_width=True
    )

    if upload_button:
        go_to_page("upload")



# --------------------------------------------------
# Upload-Seite
# --------------------------------------------------

elif st.session_state.page == "upload":
    st.title("📤 Gegenstand hochladen")

    if st.button("← Zurück zur Startseite"):
        go_to_page("home")

    uploaded_file = st.file_uploader(
        "Foto auswählen",
        type=["jpg", "jpeg", "png", "webp"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        st.image(
            image,
            caption=uploaded_file.name,
            width=400
        )

        if st.button("Bild analysieren", type="primary"):
            with st.spinner("Der Gegenstand wird erkannt..."):
                category, confidence = predict_category(image)

            st.success(f"Erkannte Kategorie: {category}")

            st.write(
                f"Erkennungswahrscheinlichkeit: "
                f"{confidence * 100:.2f} %"
            )

            upload_date = datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )

            image_bytes = uploaded_file.getvalue()

            save_entry(
                filename=uploaded_file.name,
                category=category,
                confidence=confidence,
                upload_date=upload_date,
                image_bytes=image_bytes
            )

            st.cache_data.clear()

            st.success("Der Eintrag wurde erfolgreich gespeichert.")

            if st.button("Zur Startseite"):
                go_to_page("home")



# --------------------------------------------------
# Alle Einträge
# --------------------------------------------------

elif st.session_state.page == "all":
    st.title("📋 Alle Einträge")

    if st.button("← Zurück zur Startseite"):
        go_to_page("home")

    dataframe = get_all_entries()

    if dataframe.empty:
        st.info("Es sind noch keine Einträge vorhanden.")
    else:
        category_filter = st.selectbox(
            "Nach Kategorie filtern",
            ["Alle"] + sorted(
                dataframe["category"].unique().tolist()
            )
        )

        if category_filter != "Alle":
            dataframe = dataframe[
                dataframe["category"] == category_filter
            ]

        st.write(f"{len(dataframe)} Einträge gefunden.")

        for _, entry in dataframe.iterrows():
            left_column, right_column = st.columns([1, 3])

            with left_column:
                st.image(
                    entry["image"],
                    use_container_width=True
                )

            with right_column:
                st.subheader(entry["category"])
                st.write(f"Dateiname: {entry['filename']}")
                st.write(f"Upload-Datum: {entry['upload_date']}")
                st.write(
                    "Erkennungswahrscheinlichkeit: "
                    f"{entry['confidence'] * 100:.2f} %"
                )

            st.divider()

