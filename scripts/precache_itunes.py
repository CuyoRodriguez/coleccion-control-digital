import ast
import difflib
import json
from pathlib import Path
import re
import time
import unicodedata
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PAGE_PATH = PROJECT_ROOT / "pages" / "3_Recomendacion_Musical.py"
OUTPUT_PATH = PROJECT_ROOT / "data" / "itunes_cache.json"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/129.0.0.0 Safari/537.36"
)


def load_recommendations() -> dict[str, list[tuple[str, str]]]:
    syntax_tree = ast.parse(PAGE_PATH.read_text(encoding="utf-8"))
    for node in syntax_tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name)
            and target.id == "TOP_RECOMMENDATIONS"
            for target in node.targets
        ):
            recommendations = ast.literal_eval(node.value)
            if not isinstance(recommendations, dict):
                break
            return recommendations
    raise ValueError(f"No se encontró TOP_RECOMMENDATIONS en {PAGE_PATH}.")


def normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    unaccented = "".join(
        character for character in normalized
        if not unicodedata.combining(character)
    ).casefold()
    return " ".join(
        re.findall(r"[^\W_]+", unaccented, flags=re.UNICODE)
    )


def search_itunes(
    title: str,
    artist: str,
    request_count: int,
) -> tuple[dict[str, object] | None, int, str | None]:
    for term in (f"{title} {artist}", title):
        query = urlencode(
            {
                "term": term,
                "media": "music",
                "entity": "song",
                "limit": 50,
                "country": "US",
            }
        )
        request = Request(
            f"https://itunes.apple.com/search?{query}",
            headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
        )
        if request_count:
            time.sleep(0.1)
        request_count += 1

        payload = None
        lookup_error = None
        for attempt in range(3):
            try:
                with urlopen(request, timeout=20) as response:
                    payload = json.load(response)
                break
            except HTTPError as error:
                if error.code not in {403, 429, 500, 502, 503, 504} or attempt == 2:
                    lookup_error = str(error)
                    break
                time.sleep(1 + attempt * 2)
            except (URLError, TimeoutError, json.JSONDecodeError) as error:
                lookup_error = str(error)
                break

        if lookup_error:
            continue
        if not isinstance(payload, dict):
            return None, request_count, "iTunes devolvió una respuesta inválida."

        results = payload.get("results", [])
        if not isinstance(results, list):
            return None, request_count, "iTunes devolvió una respuesta inválida."

        normalized_title = normalize_text(title)
        normalized_artist = normalize_text(artist)
        candidates = []
        for result in results:
            if not isinstance(result, dict):
                continue
            result_title = result.get("trackName")
            result_artist = result.get("artistName")
            if not isinstance(result_title, str) or not isinstance(result_artist, str):
                continue

            title_similarity = difflib.SequenceMatcher(
                None,
                normalized_title,
                normalize_text(result_title),
            ).ratio()
            artist_similarity = difflib.SequenceMatcher(
                None,
                normalized_artist,
                normalize_text(result_artist),
            ).ratio()
            if title_similarity >= 0.68:
                candidates.append(
                    (title_similarity, artist_similarity, result)
                )

        if candidates:
            return max(
                candidates,
                key=lambda candidate: (
                    candidate[0] * 0.75 + candidate[1] * 0.25,
                    candidate[0],
                ),
            )[2], request_count, None

    return None, request_count, lookup_error


def save_cache(records: list[dict[str, object]]) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = OUTPUT_PATH.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(OUTPUT_PATH)


def main() -> int:
    recommendations = load_recommendations()
    total = sum(len(tracks) for tracks in recommendations.values())
    existing_records = (
        json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
        if OUTPUT_PATH.is_file()
        else []
    )
    if not isinstance(existing_records, list):
        raise ValueError(f"El caché existente no es una lista JSON: {OUTPUT_PATH}.")
    existing_by_key = {
        (record.get("genre"), record.get("title"), record.get("artist")): record
        for record in existing_records
        if isinstance(record, dict)
    }
    records: list[dict[str, object]] = []
    request_count = 0

    for genre, tracks in recommendations.items():
        for title, artist in tracks:
            key = (genre, title, artist)
            cached_record = existing_by_key.get(key)
            if (
                cached_record
                and cached_record.get("artwork_url")
                and cached_record.get("preview_url")
            ):
                records.append(cached_record)
                print(f"[{len(records)}/{total}] {genre}: {title} — {artist}: Cache")
                continue

            result, request_count, lookup_error = search_itunes(
                title,
                artist,
                request_count,
            )
            record: dict[str, object] = {
                "genre": genre,
                "title": title,
                "artist": artist,
                "artwork_url": None,
                "preview_url": None,
            }
            if result is not None:
                artwork_url = result.get("artworkUrl100")
                if isinstance(artwork_url, str):
                    artwork_url = re.sub(
                        r"\d+x\d+bb(?=\.)",
                        "1000x1000bb",
                        artwork_url,
                    )
                record["artwork_url"] = artwork_url
                preview_url = result.get("previewUrl")
                record["preview_url"] = (
                    preview_url if isinstance(preview_url, str) else None
                )
                if not record["artwork_url"] or not record["preview_url"]:
                    record["lookup_error"] = (
                        "iTunes no devolvió portada HD o muestra de audio."
                    )
            elif lookup_error:
                record["lookup_error"] = lookup_error
            else:
                record["lookup_error"] = "No se encontró una coincidencia en iTunes."

            records.append(record)
            save_cache(records)
            status = (
                "OK"
                if record["artwork_url"] and record["preview_url"]
                else str(record.get("lookup_error", "Faltan datos de iTunes."))
            )
            print(
                f"[{len(records)}/{total}] {genre}: {title} — {artist}: {status}"
            )

    save_cache(records)
    complete_records = sum(
        bool(record["artwork_url"] and record["preview_url"])
        for record in records
    )
    print(
        f"Guardadas {len(records)} canciones; {complete_records} con portada "
        f"y audio. Archivo: {OUTPUT_PATH}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
