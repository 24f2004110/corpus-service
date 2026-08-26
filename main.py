import json
import re
import hashlib
import unicodedata
from datetime import datetime, timezone
from typing import Any

import google_crc32c
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


app = FastAPI()


# --------------------------------------------------
# CONSTANTS
# --------------------------------------------------

SAFE_INT_MAX = 9007199254740991

OBJECT_CODES = {
    "URI_INVALID",
    "GENERATION_INVALID",
    "GENERATION_MISMATCH",
    "CRC32C_INVALID",
    "CRC32C_MISMATCH",
    "SCHEMA_INVALID",
    "JSONL_INVALID",
}

ROW_CODES = {
    "DUPLICATE",
    "POLICY_INVALID",
    "OUT_OF_WINDOW",
    "TRAIN_CONTAMINATION",
}


# Exact structure: gs://bucket/object
URI_RE = re.compile(r"^gs://[^/\s]+/.+$")

# Decimal strings
GENERATION_RE = re.compile(r"^[0-9]+$")

# Exactly 8 lowercase hex digits
CRC_RE = re.compile(r"^[0-9a-f]{8}$")

# YYYY-MM-DDTHH:mm:ss[.sss](Z|±HH:mm)
TIME_RE = re.compile(
    r"^"
    r"\d{4}-\d{2}-\d{2}"
    r"T"
    r"\d{2}:\d{2}:\d{2}"
    r"(?:\.\d{1,3})?"
    r"(?:Z|[+-]\d{2}:\d{2})"
    r"$"
)


# --------------------------------------------------
# HELPERS
# --------------------------------------------------

def utf8_key(value: str) -> bytes:
    return value.encode("utf-8")


def compact_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False
    )


def valid_timestamp(value: Any) -> bool:
    if not isinstance(value, str):
        return False

    if not TIME_RE.fullmatch(value):
        return False

    try:
        if value.endswith("Z"):
            base = value[:-1] + "+00:00"
        else:
            base = value

        dt = datetime.fromisoformat(base)

        # Offset constraints
        offset = dt.utcoffset()
        if offset is None:
            return False

        total_seconds = abs(int(offset.total_seconds()))
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60

        if hours > 14:
            return False

        if hours == 14 and minutes != 0:
            return False

        return True

    except Exception:
        return False


def parse_timestamp(value: str) -> datetime:
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"

    return datetime.fromisoformat(value)


def normalize_event_time(value: str) -> str:
    dt = parse_timestamp(value)

    utc = dt.astimezone(timezone.utc)

    milliseconds = utc.microsecond // 1000

    return (
        utc.strftime("%Y-%m-%dT%H:%M:%S")
        + f".{milliseconds:03d}Z"
    )


def canonicalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value)
    value = value.lower()

    # Unicode whitespace -> one ASCII space
    value = " ".join(value.split())

    return value


def calculate_crc32c(content: str) -> str:
    checksum = google_crc32c.value(content.encode("utf-8"))
    return f"{checksum:08x}"


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def bucket_for_entity(entity: str) -> int:
    digest = hashlib.sha256(entity.encode("utf-8")).digest()

    return digest[0] % 10


def split_for_entity(entity: str) -> str:
    bucket = bucket_for_entity(entity)

    if bucket <= 5:
        return "train"

    if bucket <= 7:
        return "validation"

    return "test"


def word_set(text: str) -> set[str]:
    # lowercase Unicode letters/numbers sequences
    words = []
    current = []

    for ch in text:
        category = unicodedata.category(ch)

        if category.startswith("L") or category.startswith("N"):
            current.append(ch)
        else:
            if current:
                words.append("".join(current))
                current = []

    if current:
        words.append("".join(current))

    return set(words)


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0

    union = a | b

    if not union:
        return 1.0

    return len(a & b) / len(union)


def sorted_codes(codes):
    return sorted(set(codes), key=lambda x: x.encode("utf-8"))


def valid_policy(policy: Any):
    if not isinstance(policy, dict):
        return None

    min_time = policy.get("minTime")
    max_time = policy.get("maxTime")
    threshold = policy.get("contaminationThreshold")

    if not valid_timestamp(min_time):
        return None

    if not valid_timestamp(max_time):
        return None

    if (
        not isinstance(threshold, (int, float))
        or isinstance(threshold, bool)
    ):
        return None

    try:
        if threshold != threshold:
            return None

        if threshold in (float("inf"), float("-inf")):
            return None
    except Exception:
        return None

    if threshold < 0 or threshold > 1:
        return None

    min_dt = parse_timestamp(min_time).astimezone(timezone.utc)
    max_dt = parse_timestamp(max_time).astimezone(timezone.utc)

    if min_dt > max_dt:
        return None

    return {
        "min": min_dt,
        "max": max_dt,
        "threshold": float(threshold)
    }


def invalid_input():
    return JSONResponse(
        status_code=400,
        content={"error": "INVALID_INPUT"}
    )


# --------------------------------------------------
# MAIN ENDPOINT
# --------------------------------------------------

@app.post("/build-corpus")
async def build_corpus(request: Request):

    try:
        body = await request.json()
    except Exception:
        return invalid_input()

    if not isinstance(body, dict):
        return invalid_input()

    if "policy" not in body:
        return invalid_input()

    if not isinstance(body.get("objects"), list):
        return invalid_input()

    policy = valid_policy(body.get("policy"))

    objects = body["objects"]

    rejected_objects = []
    rejected_rows = []
    lineage = []

    candidate_rows = []

    # --------------------------------------------------
    # PROCESS OBJECTS
    # --------------------------------------------------

    for obj in objects:

        codes = []

        if not isinstance(obj, dict):
            rejected_objects.append({
                "uri": None,
                "reasonCodes": ["URI_INVALID"]
            })
            continue

        uri = obj.get("uri")
        generation = obj.get("generation")
        fetched_generation = obj.get("fetchedGeneration")
        crc32c = obj.get("crc32c")
        schema_id = obj.get("schemaId")
        content = obj.get("content")

        output_uri = uri if isinstance(uri, str) else None

        # URI
        if not isinstance(uri, str) or not URI_RE.fullmatch(uri):
            codes.append("URI_INVALID")

        # Generations
        generation_valid = (
            isinstance(generation, str)
            and GENERATION_RE.fullmatch(generation)
        )

        fetched_generation_valid = (
            isinstance(fetched_generation, str)
            and GENERATION_RE.fullmatch(fetched_generation)
        )

        if not generation_valid or not fetched_generation_valid:
            codes.append("GENERATION_INVALID")

        if generation != fetched_generation:
            codes.append("GENERATION_MISMATCH")

        # CRC
        crc_valid = (
            isinstance(crc32c, str)
            and CRC_RE.fullmatch(crc32c)
        )

        if not crc_valid:
            codes.append("CRC32C_INVALID")

        if (
            isinstance(content, str)
            and crc_valid
        ):
            actual_crc = calculate_crc32c(content)

            if actual_crc != crc32c:
                codes.append("CRC32C_MISMATCH")

        # Schema
        if schema_id != "training-v1":
            codes.append("SCHEMA_INVALID")

        if not isinstance(content, str):
            codes.append("SCHEMA_INVALID")

        parsed_rows = []

        if isinstance(content, str):

            non_blank_lines = [
                line
                for line in content.splitlines()
                if line.strip() != ""
            ]

            if len(non_blank_lines) == 0:
                codes.append("SCHEMA_INVALID")
            else:
                for line in non_blank_lines:
                    try:
                        parsed = json.loads(line)
                        parsed_rows.append(parsed)
                    except Exception:
                        codes.append("JSONL_INVALID")

        # Validate parsed row schema
        if not codes:

            for row in parsed_rows:

                expected = {
                    "id",
                    "entity",
                    "eventTime",
                    "revision",
                    "text"
                }

                if not isinstance(row, dict):
                    codes.append("SCHEMA_INVALID")
                    break

                if set(row.keys()) != expected:
                    codes.append("SCHEMA_INVALID")
                    break

                if (
                    not isinstance(row["id"], str)
                    or not isinstance(row["entity"], str)
                    or not isinstance(row["eventTime"], str)
                    or not isinstance(row["text"], str)
                ):
                    codes.append("SCHEMA_INVALID")
                    break

                revision = row["revision"]

                if (
                    isinstance(revision, bool)
                    or not isinstance(revision, int)
                    or revision < 0
                    or revision > SAFE_INT_MAX
                ):
                    codes.append("SCHEMA_INVALID")
                    break

                if not valid_timestamp(row["eventTime"]):
                    codes.append("SCHEMA_INVALID")
                    break

        if codes:

            rejected_objects.append({
                "uri": output_uri,
                "reasonCodes": sorted_codes(codes)
            })

            continue

        # Valid object -> lineage
        lineage.append({
            "uri": uri,
            "generation": generation,
            "crc32c": crc32c,
            "schemaId": schema_id
        })

        # Canonicalize rows
        for row in parsed_rows:

            normalized = {
                "id": row["id"],
                "entity": canonicalize_text(row["entity"]),
                "eventTime": normalize_event_time(row["eventTime"]),
                "revision": row["revision"],
                "text": canonicalize_text(row["text"])
            }

            normalized["_uri"] = uri

            candidate_rows.append(normalized)

    # --------------------------------------------------
    # DEDUPLICATION
    # --------------------------------------------------

    groups = {}

    for row in candidate_rows:

        key = (
            row["entity"],
            row["eventTime"],
            row["text"]
        )

        groups.setdefault(key, []).append(row)

    retained = []

    for _, rows in groups.items():

        rows_sorted = sorted(
            rows,
            key=lambda r: (
                -r["revision"],
                utf8_key(r["id"])
            )
        )

        winner = rows_sorted[0]

        retained.append(winner)

        for loser in rows_sorted[1:]:

            rejected_rows.append({
                "id": loser["id"],
                "reasonCodes": ["DUPLICATE"]
            })

    # --------------------------------------------------
    # POLICY
    # --------------------------------------------------

    policy_rows = []

    if policy is None:

        for row in retained:

            rejected_rows.append({
                "id": row["id"],
                "reasonCodes": ["POLICY_INVALID"]
            })

    else:

        for row in retained:

            event_dt = parse_timestamp(
                row["eventTime"]
            ).astimezone(timezone.utc)

            if (
                event_dt < policy["min"]
                or event_dt > policy["max"]
            ):

                rejected_rows.append({
                    "id": row["id"],
                    "reasonCodes": ["OUT_OF_WINDOW"]
                })

            else:
                policy_rows.append(row)

    # --------------------------------------------------
    # SPLIT
    # --------------------------------------------------

    splits = {
        "train": [],
        "validation": [],
        "test": []
    }

    for row in policy_rows:

        split = split_for_entity(row["entity"])

        splits[split].append(row)

    # --------------------------------------------------
    # TRAIN CONTAMINATION
    # --------------------------------------------------

    if policy is not None:

        train_sets = [
            word_set(row["text"])
            for row in splits["train"]
        ]

        for split_name in ["validation", "test"]:

            clean_rows = []

            for row in splits[split_name]:

                candidate_words = word_set(row["text"])

                contaminated = False

                for train_words in train_sets:

                    similarity = jaccard(
                        candidate_words,
                        train_words
                    )

                    if similarity >= policy["threshold"]:
                        contaminated = True
                        break

                if contaminated:

                    rejected_rows.append({
                        "id": row["id"],
                        "reasonCodes": [
                            "TRAIN_CONTAMINATION"
                        ]
                    })

                else:
                    clean_rows.append(row)

            splits[split_name] = clean_rows

    # --------------------------------------------------
    # REMOVE INTERNAL KEYS
    # --------------------------------------------------

    for split_name in splits:

        clean = []

        for row in splits[split_name]:

            clean.append({
                "id": row["id"],
                "entity": row["entity"],
                "eventTime": row["eventTime"],
                "revision": row["revision"],
                "text": row["text"]
            })

        splits[split_name] = clean

    # --------------------------------------------------
    # SORT SPLITS
    # --------------------------------------------------

    for split_name in splits:

        splits[split_name].sort(
            key=lambda row: (
                utf8_key(row["id"]),
                compact_json(row).encode("utf-8")
            )
        )

    # --------------------------------------------------
    # DIGESTS
    # --------------------------------------------------

    digests = {}

    for split_name, rows in splits.items():

        serialized = b""

        for row in rows:

            line = compact_json(row).encode("utf-8")
            serialized += line + b"\n"

        digests[split_name] = sha256_hex(serialized)

    # --------------------------------------------------
    # SORT REJECTED OBJECTS
    # --------------------------------------------------

    rejected_objects.sort(
        key=lambda item: (
            utf8_key(item["uri"])
            if isinstance(item["uri"], str)
            else b"",
            compact_json(item).encode("utf-8")
        )
    )

    # --------------------------------------------------
    # SORT REJECTED ROWS
    # --------------------------------------------------

    rejected_rows.sort(
        key=lambda item: (
            utf8_key(item["id"]),
            compact_json(item).encode("utf-8")
        )
    )

    # --------------------------------------------------
    # SORT LINEAGE
    # --------------------------------------------------

    lineage.sort(
        key=lambda item: (
            utf8_key(item["uri"]),
            compact_json(item).encode("utf-8")
        )
    )

    return {
        "splits": splits,
        "rejectedObjects": rejected_objects,
        "rejectedRows": rejected_rows,
        "digests": {
            "train": digests["train"],
            "validation": digests["validation"],
            "test": digests["test"]
        },
        "lineage": lineage
    }