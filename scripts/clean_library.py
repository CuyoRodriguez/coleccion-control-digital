from pathlib import Path
import unicodedata

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE_PATH = PROJECT_ROOT / "My Apple Music Library.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "curated_music.csv"

ARTISTS_EXCLUDED = {
    "the living tombstone",
    "hitoshizuku & yama",
    "ali",
    "creepy nuts",
    "平野義久",
    "various artists",
    "varios artistas",
    "c418",
    "bruno wen-li",
    "arctic monkeys",
    "artic monkeys",
    "safari riot",
    "epic mountain",
    "lemon demon",
    "lorien testard",
    "lonen testard",
}
EXCLUDED_ARTIST_TERMS = (
    "vocaloid",
    "j-music",
    "j music",
    "japanese music",
    "j-pop",
    "jpop",
)
OUTPUT_COLUMNS = [
    "Track name",
    "Artist name",
    "Album",
    "Genre_Group",
    "Favorite",
]


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    normalized = unicodedata.normalize("NFKD", str(value))
    return "".join(
        character
        for character in normalized
        if not unicodedata.combining(character)
    ).casefold()


NORMALIZED_ARTISTS_EXCLUDED = {
    normalize_text(artist) for artist in ARTISTS_EXCLUDED
}


def canonicalize_artist(artist_name: object) -> object:
    if pd.isna(artist_name):
        return artist_name

    normalized = normalize_text(artist_name).replace("’", "'")
    if normalized in {"will wood", "will wood and the tapeworms"}:
        return "Will Wood"
    if normalized in {"jack stauber", "jack stauber's micropop"}:
        return "Jack Stauber"
    return artist_name


def classify_genre_group(artist_name: object, playlist_name: object) -> str:
    artist = normalize_text(artist_name)
    playlist = normalize_text(playlist_name)

    if any(
        name in artist
        for name in (
            "the strokes",
            "cage the elephant",
            "radiohead",
            "foster the people",
            "arctic monkeys",
            "interpol",
            "declan mckenna",
        )
    ) or any(term in playlist for term in ("alternative", "rock alternativo")):
        return "🎸 Música Alternativa"

    if any(
        name in artist
        for name in (
            "gorillaz",
            "inner wave",
            "everything everything",
            "saint motel",
            "l'imperatrice",
            "peach pit",
            "daft punk",
            "temples",
            "mgmt",
            "metronomy",
            "crumb",
            "tame impala",
            "the buttertones",
        )
    ):
        return "✨ Indie, Synth & Psicodelia"

    if any(
        name in artist
        for name in (
            "will wood",
            "fish in a birdcage",
            "jack stauber",
            "tally hall",
            "idkhow",
            "lemon demon",
        )
    ):
        return "🎭 Experimental & Avant-Garde"

    if any(
        name in artist
        for name in ("c418", "epic mountain", "lorien testard", "waveshaper")
    ) or any(
        term in playlist
        for term in (
            "orchestra",
            "orchestral",
            "orquesta",
            "piano",
            "classical",
            "clasica",
            "minecraft",
            "kurzgesagt",
            "fromsoftware",
            "from software",
        )
    ):
        return "🎻 Instrumental, BSO & Ambient"

    if any(
        name in artist
        for name in (
            "avenged sevenfold",
            "rammstein",
            "muse",
            "rush",
            "slipknot",
            "system of a down",
        )
    ) or any(term in playlist for term in ("metal", "a7x", "guitar hero")):
        return "🔥 Rock & Heavy Metal"

    return "🎧 Pop, Variados & Otros"


def is_excluded_artist(artist_name: object) -> bool:
    normalized_artist = normalize_text(artist_name)
    has_japanese_script = any(
        "\u3040" <= character <= "\u30ff"
        or "\u3400" <= character <= "\u9fff"
        for character in str(artist_name)
    )
    return (
        normalized_artist in NORMALIZED_ARTISTS_EXCLUDED
        or has_japanese_script
        or any(term in normalized_artist for term in EXCLUDED_ARTIST_TERMS)
    )


def curate_library(library: pd.DataFrame) -> pd.DataFrame:
    required_columns = {"Track name", "Artist name", "Album", "Playlist name"}
    missing_columns = required_columns.difference(library.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Faltan columnas requeridas en el CSV: {missing}.")

    curated = library.copy()
    curated["Artist name"] = curated["Artist name"].map(canonicalize_artist)
    if "Favorite" not in curated.columns:
        curated["Favorite"] = False

    curated.dropna(subset=["Track name", "Artist name"], inplace=True)
    curated.drop_duplicates(
        subset=["Track name", "Artist name"],
        inplace=True,
    )
    curated = curated.loc[~curated["Artist name"].map(is_excluded_artist)].copy()
    curated["Genre_Group"] = curated.apply(
        lambda track: classify_genre_group(
            track["Artist name"],
            track["Playlist name"],
        ),
        axis=1,
    )
    return curated[OUTPUT_COLUMNS].reset_index(drop=True)


def main() -> None:
    if not SOURCE_PATH.is_file():
        raise FileNotFoundError(
            f"No se encontró el archivo de entrada: {SOURCE_PATH}"
        )

    source = pd.read_csv(SOURCE_PATH, encoding="utf-8-sig")
    curated = curate_library(source)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    curated.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print(
        f"Biblioteca curada guardada en {OUTPUT_PATH} "
        f"({len(curated):,} pistas; columnas: {', '.join(OUTPUT_COLUMNS)})."
    )


if __name__ == "__main__":
    main()
