"""Retrieve and cache Materials Project summary records for bulk structure discovery."""

from __future__ import annotations

from datetime import datetime, timezone
from fractions import Fraction
import hashlib
from http.client import IncompleteRead
import json
import os
from pathlib import Path
import re
import ssl
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import HTTPRedirectHandler, Request, build_opener


SUMMARY_ENDPOINT = "https://api.materialsproject.org/materials/summary/"
SUMMARY_FIELDS = (
    "material_id", "formula_pretty", "structure", "symmetry", "energy_above_hull",
    "band_gap", "is_metal", "is_magnetic", "deprecated", "last_updated",
)
MATERIAL_ID = re.compile(r"(?:mp|mvc)-(?:[0-9]+|[a-z]+)")
FORMULA_TOKEN = re.compile(r"([A-Z][a-z]?|\))([0-9]+(?:\.[0-9]+)?|\.[0-9]+)?|(\()")
SUBSCRIPTS = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
ELEMENTS = frozenset((
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge "
    "As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm "
    "Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th "
    "Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og"
).split())
PAGE_SIZE = 50
RETRY_STATUSES = {429, 500, 502, 503, 504}


class _NoRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def material_id_key(material_id: str) -> tuple[str, int]:
    """Match Materials Project decimal IDs and their documented base-26 AlphaIDs."""
    material_id = material_id.strip().lower()
    if not MATERIAL_ID.fullmatch(material_id):
        raise ValueError(f"Invalid Materials Project ID: {material_id!r}")
    prefix, identifier = material_id.split("-", 1)
    if identifier.isdigit():
        return prefix, int(identifier)
    number = 0
    for letter in identifier:
        number = 26 * number + ord(letter) - ord("a")
    return prefix, number


def _composition(formula: str) -> dict[str, Fraction]:
    """Return atom fractions, independent of element order and formula-unit size."""
    stack: list[dict[str, Fraction]] = [{}]
    offset = 0
    while offset < len(formula):
        match = FORMULA_TOKEN.match(formula, offset)
        if match is None:
            raise ValueError(f"Invalid chemical formula: {formula!r}")
        name, number, opening = match.groups()
        offset = match.end()
        if opening:
            stack.append({})
            continue
        multiplier = Fraction(number or "1")
        if multiplier <= 0:
            raise ValueError(f"Non-positive stoichiometry in formula: {formula!r}")
        if name == ")":
            if len(stack) == 1 or not stack[-1]:
                raise ValueError(f"Invalid parentheses in chemical formula: {formula!r}")
            atoms = stack.pop()
        else:
            if name not in ELEMENTS:
                raise ValueError(f"Unknown element {name!r} in chemical formula: {formula!r}")
            atoms = {name: Fraction(1)}
        for element, count in atoms.items():
            stack[-1][element] = stack[-1].get(element, Fraction(0)) + count * multiplier
    if len(stack) != 1 or not stack[0]:
        raise ValueError(f"Invalid chemical formula: {formula!r}")
    total = sum(stack[0].values())
    return {element: count / total for element, count in stack[0].items()}


def _formula_text(text: str) -> str:
    return "".join(text.translate(SUBSCRIPTS).split())


def _cache_path(directory: Path, query: str, by_id: bool) -> Path:
    kind = "id" if by_id else "formula"
    readable = re.sub(r"[^A-Za-z0-9-]", "_", query)[:30]
    digest = hashlib.sha256(query.encode("ascii")).hexdigest()[:12]
    return directory / f"{kind}-{readable}-{digest}.json"


def _record(raw: dict, seed_id: str | None = None) -> dict:
    if not isinstance(raw, dict):
        raise ValueError("Materials Project result contains a non-object record")
    material_id = raw.get("material_id")
    formula = raw.get("formula_pretty")
    if not isinstance(material_id, str) or not MATERIAL_ID.fullmatch(material_id):
        raise ValueError("Materials Project result is missing a valid material_id")
    if not isinstance(formula, str) or not formula:
        raise ValueError(f"Materials Project result is missing formula_pretty for {material_id}")
    result = {name: raw[name] for name in SUMMARY_FIELDS if name in raw}
    result["requested_id"] = material_id == seed_id
    return result


def _exact_formula(records: list[dict], formula: str) -> list[dict]:
    composition = _composition(formula)
    return [
        record for record in records
        if _composition(_formula_text(record["formula_pretty"])) == composition
    ]


def _seed(records: list[dict], query: str, allow_alias: bool) -> dict:
    key = material_id_key(query)
    for record in records:
        if material_id_key(record["material_id"]) == key:
            return record
    if allow_alias and len(records) == 1:
        return records[0]
    if not records or not allow_alias:
        raise ValueError(f"No Materials Project summary found for ID {query}")
    raise ValueError(f"Materials Project ID {query} resolved to multiple records without an exact match")


def _read_local(path: Path, query: str, by_id: bool, *, allow_fixture: bool) -> list[dict]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(payload, dict) and payload.get("schema") == 1:
            if payload["query"] != query or payload["by_id"] is not by_id:
                raise ValueError("cached query does not match")
            resolved = payload["resolved_material_id"]
            records = [_record(record, resolved) for record in payload["records"]]
            if sum(record["requested_id"] for record in records) != int(by_id):
                raise ValueError("cache has an inconsistent resolved material ID")
        elif allow_fixture:
            raw_records = payload["data"] if isinstance(payload, dict) else payload
            if not isinstance(raw_records, list):
                raise ValueError("fixture needs a list of summary records")
            records = [_record(record) for record in raw_records]
            if by_id:
                seed = _seed(records, query, allow_alias=False)
                records = [
                    _record(record, seed["material_id"])
                    for record in _exact_formula(records, _formula_text(seed["formula_pretty"]))
                ]
            else:
                records = _exact_formula(records, query)
        else:
            raise ValueError("unsupported cache schema")
        if len({record["material_id"] for record in records}) != len(records):
            raise ValueError("duplicate material IDs")
        return sorted(records, key=lambda record: record["material_id"])
    except (OSError, TypeError, ValueError, KeyError) as error:
        raise ValueError(f"Cannot read Materials Project records from {path}: {error}") from error


def _retry_delay(attempt: int, retry_after: str | None) -> float:
    if retry_after and retry_after.isascii() and retry_after.isdigit() and len(retry_after) <= 4:
        return min(15, int(retry_after))
    return 2 ** attempt


def _request_json(opener, url: str, api_key: str) -> dict:
    request = Request(url, headers={
        "X-API-KEY": api_key,
        "Accept": "application/json",
        "User-Agent": "ferroelectricity-dft/1",
    })
    for attempt in range(3):
        try:
            with opener.open(request, timeout=20) as response:
                payload = json.load(response)
            if not isinstance(payload, dict):
                raise RuntimeError("Materials Project returned an invalid JSON response")
            return payload
        except HTTPError as error:
            status = error.code
            retry_after = error.headers.get("Retry-After") if error.headers else None
            error.close()
            if status in (401, 403):
                raise RuntimeError(f"Materials Project rejected the request (HTTP {status}); check MP_API_KEY and access") from None
            if 300 <= status < 400:
                raise RuntimeError("Materials Project returned an unexpected redirect; request was stopped") from None
            if status not in RETRY_STATUSES or attempt == 2:
                raise RuntimeError(f"Materials Project summary request failed (HTTP {status}); check the query or retry later") from None
            time.sleep(_retry_delay(attempt, retry_after))
        except (URLError, OSError, IncompleteRead) as error:
            if isinstance(getattr(error, "reason", error), ssl.SSLCertVerificationError):
                raise RuntimeError("Cannot verify the Materials Project HTTPS certificate; check local CA settings") from None
            if attempt == 2:
                raise RuntimeError("Cannot connect to the Materials Project API; check network access or use a cached response") from None
            time.sleep(_retry_delay(attempt, None))
        except (json.JSONDecodeError, UnicodeError):
            raise RuntimeError("Materials Project returned invalid JSON; retry later") from None
    raise AssertionError("request attempts exhausted")


def _search(opener, api_key: str, parameter: str, query: str, endpoints: list[dict]) -> list[dict]:
    by_material_id: dict[str, dict] = {}
    expected_total = None
    while True:
        url = SUMMARY_ENDPOINT + "?" + urlencode({
            parameter: query,
            "_fields": ",".join(SUMMARY_FIELDS),
            "_sort_fields": "material_id",
            "_limit": PAGE_SIZE,
            "_skip": len(by_material_id),
        })
        payload = _request_json(opener, url, api_key)
        page = payload.get("data")
        metadata = payload.get("meta", {})
        if not isinstance(page, list) or not isinstance(metadata, dict):
            raise RuntimeError("Materials Project summary response has no valid data/meta envelope")
        total = metadata.get("total_doc")
        if total is not None and (type(total) is not int or total < 0):
            raise RuntimeError("Materials Project summary response has invalid pagination metadata")
        if expected_total is not None and total != expected_total:
            raise RuntimeError("Materials Project query total changed during pagination; retry the query")
        expected_total = total
        endpoints.append({
            "url": url, "api_version": metadata.get("api_version"),
            "server_time": metadata.get("time_stamp"), "total_doc": total,
        })
        for item in page:
            record = _record(item)
            material_id = record["material_id"]
            if material_id in by_material_id:
                raise RuntimeError("Materials Project returned duplicate IDs during pagination; retry the query")
            by_material_id[material_id] = record
        if total is not None:
            if len(by_material_id) == total:
                return list(by_material_id.values())
            if len(by_material_id) > total or not page:
                raise RuntimeError("Materials Project pagination ended with an inconsistent result count")
        elif len(page) < PAGE_SIZE:
            return list(by_material_id.values())


def _save_cache(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    output = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False)
    staged = Path(output.name)
    try:
        with output:
            json.dump(payload, output, ensure_ascii=False, allow_nan=False)
            output.write("\n")
        staged.replace(path)
    finally:
        staged.unlink(missing_ok=True)


def fetch_records(query: str, *, by_id: bool, cache_dir: Path, refresh: bool = False) -> list[dict]:
    """Return exact-composition summary records, sorted by material ID.

    ID queries mark the resolved record with requested_id=True and include its
    same-composition polymorphs. A directory caches each complete query until
    explicitly refreshed. Pass a summary-response JSON file as cache_dir for
    a fully offline query, limited to the records in that fixture.
    """
    query = query.strip().lower() if by_id else _formula_text(query)
    if by_id:
        material_id_key(query)
    else:
        _composition(query)

    if cache_dir.is_file():
        if refresh:
            raise ValueError("An offline fixture cannot be refreshed; pass a cache directory for a live query")
        return _read_local(cache_dir, query, by_id, allow_fixture=True)
    cache = _cache_path(cache_dir, query, by_id)
    if cache.exists() and not refresh:
        return _read_local(cache, query, by_id, allow_fixture=False)

    api_key = os.environ.get("MP_API_KEY", "").strip()
    if not api_key or any(not 33 <= ord(character) <= 126 for character in api_key):
        raise RuntimeError("Set MP_API_KEY in this process environment or supply a cached Materials Project response")
    try:
        opener = build_opener(_NoRedirects())
    except ssl.SSLError:
        raise RuntimeError("Cannot initialize HTTPS for Materials Project; check local Python SSL settings or use a cache") from None
    endpoints: list[dict] = []
    seed_id = None
    if by_id:
        seed = _seed(_search(opener, api_key, "material_ids", query, endpoints), query, allow_alias=True)
        formula = _formula_text(seed["formula_pretty"])
        seed_id = seed["material_id"]
    else:
        formula = query
    records = _exact_formula(_search(opener, api_key, "formula", formula, endpoints), formula)
    by_material_id = {record["material_id"]: record for record in records}
    if seed_id:
        by_material_id[seed_id] = seed
    records = [_record(by_material_id[material_id], seed_id) for material_id in sorted(by_material_id)]
    _save_cache(cache, {
        "schema": 1,
        "source": "Materials Project",
        "query": query,
        "by_id": by_id,
        "resolved_material_id": seed_id,
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "endpoints": endpoints,
        "records": records,
    })
    return records
