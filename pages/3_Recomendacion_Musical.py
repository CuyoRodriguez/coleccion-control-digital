import base64
from concurrent.futures import ThreadPoolExecutor
import difflib
import json
import mimetypes
from pathlib import Path
import re
from urllib.parse import urlencode
import unicodedata

import pandas as pd
import requests
import streamlit as st


st.set_page_config(page_title="Recomendación Musical", layout="wide")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ITUNES_CACHE_PATH = PROJECT_ROOT / "data" / "itunes_cache.json"
CATALOG_COLUMNS = ["Track name", "Artist name", "Album", "Genre_Group"]
TOP_RECOMMENDATIONS_LIMIT = 10
TOP_RECOMMENDATIONS = {
    "Clásica": [
        ("Schwanengesang, D. 957: IV. Ständchen", "Anne-Sophie Mutter & Daniil Trifonov"),
        ("Gymnopédie No. 1", "Erik Satie"),
        ("Clair de Lune", "Claude Debussy"),
        ("Nocturne in E-Flat Major, Op. 9 No. 2", "Frédéric Chopin"),
        ("Moonlight Sonata: I. Adagio sostenuto", "Ludwig van Beethoven"),
        ("Canon in D", "Johann Pachelbel"),
        ("Air on the G String", "Johann Sebastian Bach"),
        ("The Four Seasons: Winter (L'Inverno)", "Antonio Vivaldi"),
        ("Liebestraum No. 3 in A-Flat Major", "Franz Liszt"),
        ("Cello Suite No. 1 in G Major: Prélude", "Johann Sebastian Bach"),
        ("Prelude in C Major, BWV 846", "Johann Sebastian Bach"),
        ("Swan Lake, Op. 20: Scene", "Pyotr Ilyich Tchaikovsky"),
        ("Adagio for Strings, Op. 11", "Samuel Barber"),
        ("Waltz in C-Sharp Minor, Op. 64 No. 2", "Frédéric Chopin"),
        ("Arabesque No. 1", "Claude Debussy"),
    ],
    "Alternativa": [
        ("No Surprises (Remastered)", "Radiohead"),
        ("Fascination Street (Remastered)", "The Cure"),
        ("Jigsaw Falling Into Place", "Radiohead"),
        ("Elephant Woman", "Blonde Redhead"),
        ("Nude", "Radiohead"),
        ("Neon Pill", "Cage the Elephant"),
        ("Misery Is a Butterfly", "Blonde Redhead"),
        ("How Are You True", "Cage the Elephant"),
        ("Dine N'Dash", "The Strokes"),
        ("The Fruits of Conquest", "The Strokes"),
        ("I Belong to You", "Muse"),
        ("Champagne", "Declan McKenna"),
        ("Float Into The Sky", "Cage the Elephant"),
        ("The Key to Life on Earth", "Declan McKenna"),
        ("7 Horses", "The Voidz"),
    ],
    "Indie": [
        ("88", "Inner Wave"),
        ("Ccbw", "Inner Wave"),
        ("Diamond Eyes", "Inner Wave"),
        ("Locket", "Crumb"),
        ("Nina", "Crumb"),
        ("Part III", "Crumb"),
        ("Busted and Blue", "Gorillaz"),
        ("Homage", "Mild High Club"),
        ("Hold My Place in Heaven", "Saint Motel"),
        ("Wait & See", "Saint Motel"),
        ("Fine Wine", "Saint Motel"),
        ("Tony Stark", "The Symposium"),
        ("Paraphernalia", "Temples"),
        ("Cicada", "Temples"),
        ("Afterlife", "Temples"),
    ],
    "Experimental": [
        ("Dream Sweet in Sea Major", "Miracle Musical"),
        ("Introduction to the Snow", "Miracle Musical"),
        ("White Ball", "Miracle Musical"),
        ("Euthanasia", "Will Wood"),
        ("Suburbia Overture", "Will Wood"),
        ("The Song with Five Names", "Will Wood"),
        ("Rule #34", "Fish in a Birdcage"),
        ("Rule #5 - James Picard", "Fish in a Birdcage"),
        ("Rule #4 - Fish in a Birdcage", "Fish in a Birdcage"),
        ("Fate of the Stars", "Tally Hall"),
        ("Choke", "I DONT KNOW HOW BUT THEY FOUND ME"),
        ("楽園都市", "Masayoshi Oishi"),
        ("Alien Days", "MGMT"),
        ("Planets", "Everything Everything"),
        ("Cough Cough", "Everything Everything"),
    ],
    "Rock": [
        ("A Little Piece of Heaven", "Avenged Sevenfold"),
        ("Afterlife", "Avenged Sevenfold"),
        ("The Stage", "Avenged Sevenfold"),
        ("God Damn", "Avenged Sevenfold"),
        ("Acid Rain", "Avenged Sevenfold"),
        ("Deutschland", "Rammstein"),
        ("Sonne", "Rammstein"),
        ("Du Hast", "Rammstein"),
        ("2112: Overture", "Rush"),
        ("Tom Sawyer", "Rush"),
        ("A Passage To Bangkok", "Rush"),
        ("Sadie's a Sadist", "The Buttertones"),
        ("November", "Psychedelic Porn Crumpets"),
        ("Corazon", "Mystic Braves"),
        ("Like All Before You", "The Voidz"),
    ],
    "Pop": [
        ("Instant Crush", "Daft Punk & Julian Casablancas"),
        ("Instantané", "Paradis"),
        ("Matahari", "L'Impératrice"),
        ("Erreur 404", "L'Impératrice"),
        ("Sonate Pacifique", "L'Impératrice"),
        ("Hiéroglyphes", "Pépite"),
        ("Parfait Tirage", "Paradis"),
        ("Within", "Daft Punk"),
        ("The Game of Love", "Daft Punk"),
        ("Lisztomania", "Phoenix"),
        ("Chamber Of Reflection", "Mac DeMarco"),
        ("Just the Two of Us", "Grover Washington, Jr."),
        ("Ruby", "Foster the People"),
        ("Beleza Pula", "Masayoshi Takanaka"),
        ("Bird In The Snare", "Israel's Arcade"),
    ],
}


@st.cache_data(show_spinner=False)
def load_itunes_cache(
    cache_path: str,
    modified_time_ns: int,
) -> dict[tuple[str, str], dict[str, str]]:
    path = Path(cache_path)
    if not path.is_file():
        return {}

    try:
        records = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"No se pudo leer el caché de iTunes: {error}") from error

    if not isinstance(records, list):
        raise ValueError("El caché de iTunes debe contener una lista de canciones.")

    cache: dict[tuple[str, str], dict[str, str]] = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("El caché de iTunes contiene un registro inválido.")
        title = record.get("title")
        artist = record.get("artist")
        if not isinstance(title, str) or not isinstance(artist, str):
            raise ValueError("Un registro del caché no incluye título y artista válidos.")
        cache[(title, artist)] = {
            key: value
            for key in ("artwork_url", "preview_url")
            if isinstance((value := record.get(key)), str) and value
        }

    return cache


@st.cache_data(ttl=86400, show_spinner=False)
def get_itunes_preview(
    track: str,
    artist: str,
    cache_modified_time_ns: int,
) -> dict[str, str] | None:
    cached = load_itunes_cache(
        str(ITUNES_CACHE_PATH),
        cache_modified_time_ns,
    ).get((track, artist), {})
    if cached.get("artwork_url") and cached.get("preview_url"):
        return {
            "artwork_url": cached["artwork_url"],
            "audio_url": cached["preview_url"],
        }

    def normalized(value: str) -> str:
        return " ".join(
            re.findall(
                r"[^\W_]+",
                normalize_text(value).replace("’", "'"),
                flags=re.UNICODE,
            )
        )

    target_track = normalized(track)
    matches = []
    lookup_error = None
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36"
        )
    }
    for term in (f"{track} {artist}", track):
        params = {
            "term": term,
            "media": "music",
            "entity": "song",
            "limit": 25,
        }
        try:
            response = requests.get(
                "https://itunes.apple.com/search",
                params=params,
                headers=headers,
                timeout=5,
            )
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError) as error:
            lookup_error = error
            continue

        if not isinstance(payload, dict):
            raise ValueError("iTunes devolvió una respuesta de búsqueda inválida.")
        results = payload.get("results", [])
        if not isinstance(results, list):
            raise ValueError("iTunes devolvió una respuesta de búsqueda inválida.")

        for result in results:
            if not isinstance(result, dict):
                continue
            result_track = result.get("trackName")
            if not isinstance(result_track, str):
                continue
            track_similarity = difflib.SequenceMatcher(
                None,
                target_track,
                normalized(result_track),
            ).ratio()
            if track_similarity < 0.78:
                continue
            result_artist = result.get("artistName", "")
            artist_similarity = difflib.SequenceMatcher(
                None,
                normalized(artist),
                normalized(str(result_artist)),
            ).ratio()
            matches.append((track_similarity, artist_similarity, result))

        if matches:
            break

    if not matches:
        if lookup_error is not None and not cached:
            raise RuntimeError(f"No se pudo consultar iTunes: {lookup_error}") from lookup_error
        if cached:
            return {
                "artwork_url": cached.get("artwork_url", ""),
                "audio_url": cached.get("preview_url", ""),
            }
        return None

    result = max(matches, key=lambda match: (match[0], match[1]))[2]
    artwork_url = result.get("artworkUrl100")
    if isinstance(artwork_url, str):
        artwork_url = re.sub(r"\d+x\d+bb(?=\.)", "1000x1000bb", artwork_url)
    else:
        artwork_url = ""
    preview_url = result.get("previewUrl")
    return {
        "artwork_url": cached.get("artwork_url", artwork_url),
        "audio_url": cached.get(
            "preview_url",
            preview_url if isinstance(preview_url, str) else "",
        ),
    }


def render_genre_recommendations(
    genre_name: str,
    tracks: list[tuple[str, str]],
) -> None:
    st.markdown(
        f'<div class="music-card"><h3>Top 15: {genre_name}</h3>'
        "<p>Recomendaciones en orden de frecuencia de escucha.</p></div>",
        unsafe_allow_html=True,
    )

    def load_preview(
        item: tuple[str, str],
    ) -> dict[str, str] | None | Exception:
        try:
            return get_itunes_preview(*item, cache_modified_time_ns)
        except (RuntimeError, ValueError) as error:
            return error

    try:
        cache_modified_time_ns = ITUNES_CACHE_PATH.stat().st_mtime_ns
    except FileNotFoundError:
        cache_modified_time_ns = 0

    with st.spinner("Buscando portadas y muestras de audio..."):
        with ThreadPoolExecutor(max_workers=3) as executor:
            previews = list(executor.map(load_preview, tracks))

    for (track, artist), preview in zip(tracks, previews):
        cover_column, details_column, audio_column, youtube_column = st.columns(
            [1, 4, 4, 2],
            vertical_alignment="center",
        )
        with cover_column:
            if isinstance(preview, dict) and preview["artwork_url"]:
                st.image(preview["artwork_url"], width=90)
            else:
                st.markdown('<div class="genre-icon">🎵</div>', unsafe_allow_html=True)
        with details_column:
            st.markdown(f"**{track}**  \n{artist}")
        with audio_column:
            if isinstance(preview, dict) and preview.get("audio_url"):
                st.audio(preview["audio_url"], format="audio/mp4")
            elif isinstance(preview, Exception):
                st.caption(str(preview))
            else:
                st.caption("Muestra de audio no disponible en iTunes.")
        with youtube_column:
            youtube_url = "https://www.youtube.com/results?" + urlencode(
                {"search_query": f"{track} {artist}"}
            )
            st.link_button("▶ YouTube", youtube_url, width="stretch")



@st.cache_data
def load_music_library(
    csv_path: str,
    modified_time_ns: int,
) -> pd.DataFrame:
    header = pd.read_csv(csv_path, encoding="utf-8-sig", nrows=0)
    missing_columns = set(CATALOG_COLUMNS).difference(header.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Faltan columnas requeridas en el catálogo: {missing}.")
    library = pd.read_csv(
        csv_path,
        encoding="utf-8-sig",
        usecols=CATALOG_COLUMNS,
    )
    library["Artist name"] = library["Artist name"].map(canonicalize_artist)
    library.drop_duplicates(
        subset=["Track name", "Artist name"],
        inplace=True,
    )
    return library[CATALOG_COLUMNS].reset_index(drop=True)


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    normalized = unicodedata.normalize("NFKD", str(value))
    return "".join(character for character in normalized if not unicodedata.combining(character)).casefold()


def canonicalize_artist(artist_name: object) -> object:
    if pd.isna(artist_name):
        return artist_name

    normalized = normalize_text(artist_name).replace("’", "'")
    if normalized in {"will wood", "will wood and the tapeworms"}:
        return "Will Wood"
    if normalized in {"jack stauber", "jack stauber's micropop"}:
        return "Jack Stauber"
    return artist_name


def render_top_recommendations(title: str, tracks: pd.DataFrame) -> None:
    recommendations = (
        tracks.sort_values(
            ["Artist name", "Track name"],
            key=lambda column: column.fillna("").astype(str).str.casefold(),
        )
        .head(TOP_RECOMMENDATIONS_LIMIT)
        .reset_index(drop=True)
    )
    st.markdown(
        f'<div class="music-card"><h3>Top {len(recommendations)}: {title}</h3>'
        "<p>Selección del catálogo, ordenada alfabéticamente porque no incluye datos de popularidad.</p></div>",
        unsafe_allow_html=True,
    )
    with st.container(border=True):
        st.dataframe(
            recommendations[CATALOG_COLUMNS],
            hide_index=True,
            use_container_width=True,
        )


def select_music_genre(genre: str) -> None:
    st.session_state.music_genre_filter = genre
    st.session_state.selected_genre = genre


def select_music_artist(artist: str) -> None:
    st.session_state.selected_music_artist = artist


generos_config = [
    {
        "nombre": "Clásica",
        "clave": "Clásica",
        "img": "assets/genres/clasica.png",
        "grupo": "🎻 Instrumental, BSO & Ambient",
    },
    {
        "nombre": "Alternativa",
        "clave": "Alternativa",
        "img": "assets/genres/alternativa.png",
        "grupo": "🎸 Música Alternativa",
    },
    {
        "nombre": "Indie & Synth",
        "clave": "Indie",
        "img": "assets/genres/indie.png",
        "grupo": "✨ Indie, Synth & Psicodelia",
    },
    {
        "nombre": "Experimental",
        "clave": "Experimental",
        "img": "assets/genres/experimental.png",
        "grupo": "🎭 Experimental & Avant-Garde",
    },
    {
        "nombre": "Rock & Metal",
        "clave": "Rock",
        "img": "assets/genres/rock.png",
        "grupo": "🔥 Rock & Heavy Metal",
    },
    {
        "nombre": "Pop y otros",
        "clave": "Pop",
        "img": "assets/genres/pop.png",
        "grupo": "🎧 Pop, Variados & Otros",
    },
]
genre_image_css = []
for index, genre_config in enumerate(generos_config):
    image_path = PROJECT_ROOT / genre_config["img"]
    image_mime = mimetypes.guess_type(image_path.name)[0] or "image/png"
    image_data = base64.b64encode(image_path.read_bytes()).decode("ascii")
    image_selector = (
        f'[class*="st-key-music_filter_{index}"] button::before, '
        f'[class*="st-key-genre_card_{index}"] button::before'
    )
    genre_image_css.append(
        f'{image_selector} {{ background-image: url("data:{image_mime};base64,{image_data}"); }}'
    )

css_path = Path(__file__).resolve().parent.parent / "assets" / "style.css"
st.markdown(
    f"<style>{css_path.read_text(encoding='utf-8')}</style>",
    unsafe_allow_html=True,
)

st.markdown(
    """
    <style>
        section[data-testid="stSidebar"],
        section[data-testid="stSidebar"] > div {
            background-color: #FA2D48;
        }

        section[data-testid="stSidebar"] * {
            color: #FFFFFF;
        }

        section[data-testid="stSidebar"] a {
            color: #FFFFFF;
        }

        [class*="st-key-music_filter_"] button,
        [class*="st-key-genre_card_"] button,
        [class*="st-key-btn_pagina_central"] button {
            display: flex !important;
            align-items: center !important;
            justify-content: flex-start !important;
            gap: 12px !important;
            min-height: 53px !important;
            padding: 4px 16px 4px 4px !important;
            border: 2px solid #000000 !important;
            border-radius: 40px !important;
            background: #FFFFFF !important;
            color: #000000 !important;
            font-size: 16px !important;
            font-weight: 700 !important;
            text-align: left !important;
            transition: box-shadow 160ms ease, transform 160ms ease !important;
        }

        [class*="st-key-music_filter_"] button::before,
        [class*="st-key-genre_card_"] button::before {
            content: "";
            display: block;
            flex: 0 0 45px;
            width: 45px;
            height: 45px;
            border-radius: 50%;
            border: 1px solid #000000;
            background-position: center;
            background-size: cover;
        }

        [class*="st-key-music_filter_"] button [data-testid="stMarkdownContainer"],
        [class*="st-key-genre_card_"] button [data-testid="stMarkdownContainer"] {
            flex: 1;
            text-align: center;
        }

        [class*="st-key-music_filter_"] button p,
        [class*="st-key-genre_card_"] button p,
        [class*="st-key-btn_pagina_central"] button p {
            margin: 0 !important;
            color: #000000 !important;
            font-size: 16px !important;
            font-weight: 700 !important;
        }

        [class*="st-key-music_filter_"] button:hover,
        [class*="st-key-genre_card_"] button:hover,
        [class*="st-key-btn_pagina_central"] button:hover {
            border-color: #000000 !important;
            background: #FFFFFF !important;
            color: #000000 !important;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.16) !important;
            transform: translateY(-2px);
        }

        [class*="st-key-music_filter_"] button:hover p,
        [class*="st-key-genre_card_"] button:hover p,
        [class*="st-key-btn_pagina_central"] button:hover p {
            color: #000000 !important;
        }

        [data-testid="stBaseButton-primary"] {
            background-color: #FA2D48;
            border-color: #FA2D48;
            color: #FFFFFF;
        }

        [data-testid="stBaseButton-primary"]:hover {
            background-color: #D91F3A;
            border-color: #D91F3A;
            color: #FFFFFF;
        }

        [data-testid="stTabs"] [role="tab"][aria-selected="true"] {
            color: #FA2D48;
            border-bottom: 2px solid #FA2D48;
        }

        [data-testid="stTabs"] [role="tab"]:hover {
            color: #D91F3A;
        }

        [data-testid="stVerticalBlockBorderWrapper"] {
            border: 1px solid #222222 !important;
            border-radius: 18px !important;
            background-color: #FFFFFF;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.08);
        }

        .music-page-header,
        .music-page-banner {
            border: 1px solid #222222;
            border-radius: 18px;
            background: #FFFFFF;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.08);
            margin: 0.5rem 0 1rem;
            padding: 1.15rem 1.5rem;
            text-align: center;
        }

        .music-page-header h1 {
            color: #FA2D48;
            font-size: clamp(1.8rem, 4vw, 2.6rem);
            margin: 0;
        }

        .music-page-banner p {
            color: #292929;
            font-size: 1.05rem;
            margin: 0;
        }

        .genre-icon {
            align-items: center;
            background: #FFF0F2;
            border: 1px solid rgba(250, 45, 72, 0.2);
            border-radius: 50%;
            display: flex;
            font-size: 1.5rem;
            height: 3rem;
            justify-content: center;
            margin: 0.4rem auto;
            width: 3rem;
        }

        .music-card {
            background: #FFFFFF;
            border: 1px solid #222222;
            border-left: 5px solid #FA2D48;
            border-radius: 18px;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.08);
            margin: 0.5rem 0 1rem;
            padding: 1.1rem 1.35rem;
        }

        .music-card h3 {
            color: #FA2D48;
            margin: 0 0 0.4rem;
        }

        .music-card p {
            color: #292929;
            margin: 0;
        }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    f"<style>{''.join(genre_image_css)}</style>",
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="music-page-header"><h1>Recomendacion musical</h1></div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="music-page-banner"><p>Recomendaciones musicales por parte del creador de la pagina</p></div>',
    unsafe_allow_html=True,
)

valid_genres = {genre["clave"] for genre in generos_config}
if st.session_state.get("selected_genre") not in valid_genres:
    st.session_state.selected_genre = None
if st.session_state.get("music_genre_filter") not in valid_genres:
    st.session_state.music_genre_filter = None

with st.sidebar:
    st.header("Mi Colección")
    st.caption("Explora por género")
    for index, genre in enumerate(generos_config):
        st.button(
            genre["nombre"],
            key=f"music_filter_{index}",
            use_container_width=True,
            on_click=select_music_genre,
            args=(genre["clave"],),
        )
    st.markdown("---")
    if st.button(
        "🏠 Página Central",
        key="btn_pagina_central",
        use_container_width=True,
    ):
        st.switch_page("app.py")

catalog_candidates = [
    PROJECT_ROOT / "data" / "curated_music.csv",
    PROJECT_ROOT / "curated_music.csv",
]
csv_path = next(
    (candidate for candidate in catalog_candidates if candidate.is_file()),
    catalog_candidates[0],
)
music_catalog: pd.DataFrame | None = None

try:
    modified_time_ns = csv_path.stat().st_mtime_ns
    music_catalog = load_music_library(
        str(csv_path),
        modified_time_ns,
    )
except FileNotFoundError:
    st.error(
        "No se encontró **curated_music.csv**. Colócalo en "
        f"`{catalog_candidates[0].parent}` o en `{PROJECT_ROOT}`."
    )
except (OSError, UnicodeDecodeError, pd.errors.ParserError, ValueError) as error:
    st.error(f"No se pudo cargar el catálogo musical: {error}")

recommendation_tab, artist_tab = st.tabs(["🎧 Géneros", "🎤 Artistas"])

with recommendation_tab:
    st.subheader("Explora por género")
    for row_start in range(0, len(generos_config), 3):
        genre_columns = st.columns(3)
        row_genres = generos_config[row_start : row_start + 3]
        for offset, (column, genre) in enumerate(zip(genre_columns, row_genres)):
            index = row_start + offset
            with column:
                st.button(
                    genre["nombre"],
                    key=f"genre_card_{index}",
                    use_container_width=True,
                    on_click=select_music_genre,
                    args=(genre["clave"],),
                )

    selected_genre = st.session_state.music_genre_filter
    if selected_genre is None:
        st.info("Seleccione un genero para ver las recomendaciones")
    else:
        selected_config = next(
            genre for genre in generos_config
            if genre["clave"] == selected_genre
        )
        recommended_tracks = TOP_RECOMMENDATIONS.get(selected_genre)
        if recommended_tracks is None:
            st.error(f"No hay una curaduría Top 15 configurada para {selected_genre}.")
        else:
            render_genre_recommendations(
                selected_config["nombre"],
                recommended_tracks,
            )

with artist_tab:
    if music_catalog is None:
        st.markdown(
            """
            <div class="music-card">
              <h3>Datos de artistas</h3>
              <p>El resumen estará disponible cuando el catálogo curado esté en data/curated_music.csv o en la raíz del proyecto.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        artistas_excluidos = [
            "The Living Tombstone",
            "Hitoshizuku & Yama",
            "ALI",
            "Creepy Nuts",
            "平野義久",
            "Various Artists",
            "Varios Artistas",
            "C418",
            "Bruno Wen-li",
            "Arctic Monkeys",
            "Artic Monkeys",
            "Safari Riot",
            "Epic Mountain",
            "Lemon Demon",
            "Lorien Testard",
            "Lonen Testard",
        ]
        artistas_excluidos_normalizados = {
            normalize_text(artist) for artist in artistas_excluidos
        }
        filtered_library = music_catalog[
            ~music_catalog["Artist name"].map(
                lambda artist: normalize_text(artist)
                in artistas_excluidos_normalizados
            )
        ]
        artist_counts = (
            filtered_library["Artist name"]
            .dropna()
            .value_counts()
            .rename_axis("Artist name")
            .reset_index(name="Track count")
        )
        frequent_artists = artist_counts[artist_counts["Track count"] >= 15]
        st.subheader("Artistas con 15 o más canciones")

        if frequent_artists.empty:
            st.info("No hay artistas con 15 o más canciones en la biblioteca.")
        else:
            artists = frequent_artists["Artist name"].tolist()
            for row_start in range(0, len(artists), 3):
                artist_columns = st.columns(3)
                for offset, (column, artist) in enumerate(
                    zip(artist_columns, artists[row_start : row_start + 3])
                ):
                    index = row_start + offset
                    with column:
                        with st.container(border=True):
                            icon_column, label_column = st.columns([0.3, 0.7])
                            with icon_column:
                                st.markdown(
                                    '<div class="genre-icon">🎤</div>',
                                    unsafe_allow_html=True,
                                )
                            with label_column:
                                st.button(
                                    artist,
                                    key=f"artist_card_{index}",
                                    use_container_width=True,
                                    on_click=select_music_artist,
                                    args=(artist,),
                                )

        selected_artist = st.session_state.get("selected_music_artist")
        eligible_artists = set(frequent_artists["Artist name"])
        if selected_artist not in eligible_artists:
            selected_artist = None
            st.session_state.selected_music_artist = None
        if selected_artist is None:
            st.info("Selecciona una tarjeta de artista para ver sus canciones.")
        else:
            artist_tracks = filtered_library[
                filtered_library["Artist name"] == selected_artist
            ]
            render_top_recommendations(selected_artist, artist_tracks)
