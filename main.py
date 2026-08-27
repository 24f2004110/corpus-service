import json
import re
import hashlib
import math
from datetime import datetime, timezone

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


app = FastAPI()

SAFE_INT_MAX = 9007199254740991

TIMESTAMP_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}"
    r"T\d{2}:\d{2}:\d{2}"
    r"(?:\.\d{1,3})?"
    r"(?:Z|[+-]\d{2}:\d{2})$"
)

DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")

# Stateful storage
RUNS = {}


def utf8_key(value):
    return value.encode("utf-8")


def compact_json(value, sort_keys=False):
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
        sort_keys=sort_keys
    )


def is_safe_int(value):
    return (
        type(value) is int
        and 0 <= value <= SAFE_INT_MAX
    )


def is_finite_number(value):
    return (
        type(value) in (int, float)
        and math.isfinite(value)
    )


def is_finite_01(value):
    return (
        is_finite_number(value)
        and 0 <= value <= 1
    )


def sorted_codes(codes):
    return sorted(set(codes), key=utf8_key)


def valid_run_id(value):
    return (
        isinstance(value, str)
        and 1 <= len(value) <= 128
    )


def parse_timestamp(value):
    if not isinstance(value, str):
        raise ValueError("Invalid timestamp")

    if not TIMESTAMP_RE.fullmatch(value):
        raise ValueError("Invalid timestamp")

    try:
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"

        dt = datetime.fromisoformat(value)

        offset = dt.utcoffset()

        if offset is None:
            raise ValueError

        seconds = abs(int(offset.total_seconds()))
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60

        if hours > 14:
            raise ValueError

        if hours == 14 and minutes != 0:
            raise ValueError

        return dt.astimezone(timezone.utc)

    except Exception:
        raise ValueError("Invalid timestamp")


def normalize_timestamp(value):
    dt = parse_timestamp(value)

    return (
        dt.strftime("%Y-%m-%dT%H:%M:%S.")
        + f"{dt.microsecond // 1000:03d}Z"
    )


def invalid_http():
    return JSONResponse(
        status_code=400,
        content={"error": "INVALID_INPUT"}
    )


def selection_response(
    run_id,
    selected_trial_id,
    train_ids,
    eval_ids,
    feature_names,
    dataset_digest,
    reason_codes
):
    return {
        "runId": run_id,
        "selectedTrialId": selected_trial_id,
        "trainRowIds": train_ids,
        "evalRowIds": eval_ids,
        "featureNames": feature_names,
        "datasetDigest": dataset_digest,
        "reasonCodes": sorted_codes(reason_codes)
    }


# ============================================================
# SELECT PHASE
# ============================================================

def process_select(body):

    run_id = body.get("runId")

    if not valid_run_id(run_id):
        return (
            None,
            selection_response(
                run_id if isinstance(run_id, str) else None,
                None,
                [],
                [],
                [],
                None,
                ["INVALID_INPUT"]
            )
        )

    forbidden = body.get("forbiddenFeatures")
    num_trials_limit = body.get("numTrialsLimit")
    rows = body.get("rows")
    trials = body.get("trials")

    # ---------- Basic validation ----------

    malformed = False

    if (
        not isinstance(forbidden, list)
        or any(not isinstance(x, str) for x in forbidden)
    ):
        malformed = True

    if (
        type(num_trials_limit) is not int
        or num_trials_limit <= 0
    ):
        malformed = True

    if not isinstance(rows, list) or len(rows) == 0:
        malformed = True

    if not isinstance(trials, list):
        malformed = True

    if malformed:
        return (
            run_id,
            selection_response(
                run_id,
                None,
                [],
                [],
                [],
                None,
                ["INVALID_INPUT"]
            )
        )

    reason_codes = []

    if len(trials) > num_trials_limit:
        reason_codes.append("TRIAL_LIMIT_EXCEEDED")

    # ---------- Validate rows ----------

    expected_row_keys = {
        "id",
        "entity",
        "eventTime",
        "predictionTime",
        "version",
        "split",
        "features"
    }

    seen_row_ids = set()
    groups = {}

    invalid_rows = False

    for row in rows:

        if (
            not isinstance(row, dict)
            or set(row.keys()) != expected_row_keys
        ):
            invalid_rows = True
            break

        row_id = row["id"]

        if (
            not isinstance(row_id, str)
            or not isinstance(row["entity"], str)
            or not is_safe_int(row["version"])
            or row["split"] not in ("TRAIN", "EVAL")
            or not isinstance(row["features"], dict)
        ):
            invalid_rows = True
            break

        if row_id in seen_row_ids:
            invalid_rows = True
            break

        seen_row_ids.add(row_id)

        try:
            normalized_event_time = normalize_timestamp(
                row["eventTime"]
            )

            prediction_dt = parse_timestamp(
                row["predictionTime"]
            )

        except ValueError:
            invalid_rows = True
            break

        # Validate all features
        for feature_name, feature in row["features"].items():

            if not isinstance(feature_name, str):

                invalid_rows = True
                break

            if (
                not isinstance(feature, dict)
                or set(feature.keys()) != {
                    "value",
                    "availableAt"
                }
            ):
                invalid_rows = True
                break

            if not isinstance(feature["value"], str):
                invalid_rows = True
                break

            try:
                parse_timestamp(
                    feature["availableAt"]
                )

            except ValueError:
                invalid_rows = True
                break

        if invalid_rows:
            break

        dedup_key = (
            row["entity"],
            normalized_event_time
        )

        internal_row = dict(row)

        internal_row["_eventTimeUTC"] = normalized_event_time
        internal_row["_predictionDT"] = prediction_dt

        groups.setdefault(
            dedup_key,
            []
        ).append(internal_row)

    if invalid_rows:

        return (
            run_id,
            selection_response(
                run_id,
                None,
                [],
                [],
                [],
                None,
                ["INVALID_INPUT"]
            )
        )

    # ---------- Deduplicate ----------

    retained_rows = []

    for duplicate_group in groups.values():

        winner = sorted(
            duplicate_group,
            key=lambda row: (
                -row["version"],
                utf8_key(row["id"])
            )
        )[0]

        retained_rows.append(winner)

    # ---------- Validate trials ----------

    seen_trial_ids = set()
    successful_trials = []

    invalid_trials = False

    for trial in trials:

        if not isinstance(trial, dict):

            invalid_trials = True
            break

        if set(trial.keys()) != {
            "trialId",
            "status",
            "evalMetric"
        }:

            invalid_trials = True
            break

        trial_id = trial["trialId"]

        if not is_safe_int(trial_id):

            invalid_trials = True
            break

        if trial_id in seen_trial_ids:

            invalid_trials = True
            break

        seen_trial_ids.add(trial_id)

        if trial["status"] not in (
            "SUCCEEDED",
            "FAILED"
        ):

            invalid_trials = True
            break

        metric = trial["evalMetric"]

        if not is_finite_number(metric):

            invalid_trials = True
            break

        if trial["status"] == "SUCCEEDED":

            successful_trials.append(trial)

    if invalid_trials:

        return (
            run_id,
            selection_response(
                run_id,
                None,
                [],
                [],
                [],
                None,
                ["INVALID_INPUT"]
            )
        )

    # Contract failure

    if reason_codes:

        return (
            run_id,
            selection_response(
                run_id,
                None,
                [],
                [],
                [],
                None,
                reason_codes
            )
        )

    if not successful_trials:

        return (
            run_id,
            selection_response(
                run_id,
                None,
                [],
                [],
                [],
                None,
                ["NO_SUCCESSFUL_TRIAL"]
            )
        )

    # ---------- Feature eligibility ----------

    forbidden_set = set(forbidden)

    feature_names = set(
        retained_rows[0]["features"].keys()
    )

    for row in retained_rows[1:]:

        feature_names &= set(
            row["features"].keys()
        )

    eligible_features = []

    for feature_name in feature_names:

        if feature_name in forbidden_set:
            continue

        eligible = True

        for row in retained_rows:

            available_at = parse_timestamp(
                row["features"][feature_name]["availableAt"]
            )

            if available_at > row["_predictionDT"]:

                eligible = False
                break

        if eligible:
            eligible_features.append(feature_name)

    eligible_features.sort(key=utf8_key)

    # ---------- Split IDs ----------

    train_row_ids = sorted(
        [
            row["id"]
            for row in retained_rows
            if row["split"] == "TRAIN"
        ],
        key=utf8_key
    )

    eval_row_ids = sorted(
        [
            row["id"]
            for row in retained_rows
            if row["split"] == "EVAL"
        ],
        key=utf8_key
    )

    # ---------- Dataset digest ----------

    digest_payload = {
        "trainRowIds": train_row_ids,
        "evalRowIds": eval_row_ids,
        "featureNames": eligible_features
    }

    digest_bytes = compact_json(
        digest_payload
    ).encode("utf-8")

    dataset_digest = hashlib.sha256(
        digest_bytes
    ).hexdigest()

    # ---------- Select best trial ----------

    winner = min(
        successful_trials,
        key=lambda trial: (
            -trial["evalMetric"],
            trial["trialId"]
        )
    )

    response = selection_response(
        run_id,
        winner["trialId"],
        train_row_ids,
        eval_row_ids,
        eligible_features,
        dataset_digest,
        []
    )

    return run_id, response


# ============================================================
# EVALUATE PHASE
# ============================================================

def process_evaluate(body):

    run_id = body.get("runId")
    selected_trial_id = body.get("selectedTrialId")
    dataset_digest = body.get("datasetDigest")

    metric_floor = body.get("metricFloor")
    required_slices = body.get("requiredSlices")
    rows = body.get("rows")
    bytes_processed = body.get("bytesProcessed")
    max_bytes = body.get("maxBytes")

    reason_codes = []

    invalid_input = False

    # ---------- Input validation ----------

    if not valid_run_id(run_id):
        invalid_input = True

    if not is_safe_int(selected_trial_id):
        invalid_input = True

    if (
        not isinstance(dataset_digest, str)
        or not DIGEST_RE.fullmatch(dataset_digest)
    ):
        invalid_input = True

    if not is_finite_01(metric_floor):
        invalid_input = True

    if not isinstance(required_slices, dict):
        invalid_input = True

    else:

        for slice_name, slice_floor in required_slices.items():

            if (
                not isinstance(slice_name, str)
                or slice_name == ""
                or not is_finite_01(slice_floor)
            ):
                invalid_input = True
                break

    if not isinstance(rows, list):
        invalid_input = True

    if not is_safe_int(bytes_processed):
        invalid_input = True

    if not is_safe_int(max_bytes):
        invalid_input = True

    if invalid_input:
        reason_codes.append("INVALID_INPUT")

    # ---------- Lineage ----------

    stored_run = (
        RUNS.get(run_id)
        if valid_run_id(run_id)
        else None
    )

    lineage_valid = False

    if stored_run is not None:

        stored_response = stored_run["response"]

        if (
            stored_response["selectedTrialId"] is not None
            and stored_response["selectedTrialId"]
            == selected_trial_id
            and stored_response["datasetDigest"]
            == dataset_digest
        ):
            lineage_valid = True

    if not lineage_valid:
        reason_codes.append("INVALID_LINEAGE")

    # ---------- Validate test rows ----------

    invalid_test_row = False
    parsed_rows = []

    if isinstance(rows, list):

        for row in rows:

            if (
                not isinstance(row, dict)
                or set(row.keys()) != {
                    "label",
                    "prediction",
                    "slice"
                }
            ):

                invalid_test_row = True
                break

            if (
                type(row["label"]) is not int
                or row["label"] not in (0, 1)
            ):

                invalid_test_row = True
                break

            if (
                type(row["prediction"]) is not int
                or row["prediction"] not in (0, 1)
            ):

                invalid_test_row = True
                break

            if (
                not isinstance(row["slice"], str)
                or row["slice"] == ""
            ):

                invalid_test_row = True
                break

            parsed_rows.append(row)

    else:
        invalid_test_row = True

    if invalid_test_row:
        reason_codes.append("INVALID_TEST_ROW")

    # ---------- Metric and slices ----------

    test_metric = None
    critical_slice_pass = True

    can_compute_metrics = (
        isinstance(rows, list)
        and len(rows) > 0
        and not invalid_test_row
        and not invalid_input
    )

    if can_compute_metrics:

        correct = sum(
            row["label"] == row["prediction"]
            for row in parsed_rows
        )

        test_metric = round(
            correct / len(parsed_rows),
            12
        )

        if test_metric < metric_floor:

            reason_codes.append(
                "AGGREGATE_FLOOR"
            )

        for slice_name, slice_floor in required_slices.items():

            slice_rows = [
                row
                for row in parsed_rows
                if row["slice"] == slice_name
            ]

            if not slice_rows:

                reason_codes.append(
                    f"MISSING_SLICE:{slice_name}"
                )

                critical_slice_pass = False

                continue

            slice_correct = sum(
                row["label"] == row["prediction"]
                for row in slice_rows
            )

            slice_accuracy = round(
                slice_correct / len(slice_rows),
                12
            )

            if slice_accuracy < slice_floor:

                reason_codes.append(
                    f"SLICE_FLOOR:{slice_name}"
                )

                critical_slice_pass = False

    else:

        test_metric = None

    # criticalSlicePass conditions

    if invalid_input:
        critical_slice_pass = False

    if not lineage_valid:
        critical_slice_pass = False

    if invalid_test_row:
        critical_slice_pass = False

    # ---------- Byte limit ----------

    if (
        is_safe_int(bytes_processed)
        and is_safe_int(max_bytes)
        and bytes_processed > max_bytes
    ):

        reason_codes.append(
            "BYTE_LIMIT"
        )

    reason_codes = sorted_codes(
        reason_codes
    )

    decision = (
        "admit"
        if len(reason_codes) == 0
        else "reject"
    )

    return {
        "runId": (
            run_id
            if isinstance(run_id, str)
            else None
        ),
        "selectedTrialId": (
            selected_trial_id
            if is_safe_int(selected_trial_id)
            else None
        ),
        "datasetDigest": (
            dataset_digest
            if isinstance(dataset_digest, str)
            else None
        ),
        "testMetric": test_metric,
        "criticalSlicePass": critical_slice_pass,
        "decision": decision,
        "bytesProcessed": (
            bytes_processed
            if is_safe_int(bytes_processed)
            else None
        ),
        "reasonCodes": reason_codes
    }


# ============================================================
# API
# ============================================================

@app.post("/bqml")
async def bqml(request: Request):

    try:
        body = await request.json()

    except Exception:
        return invalid_http()

    if not isinstance(body, dict):
        return invalid_http()

    phase = body.get("phase")

    if phase not in ("select", "evaluate"):
        return invalid_http()

    # ========================================================
    # SELECT
    # ========================================================

    if phase == "select":

        run_id, response = process_select(body)

        if run_id is None:
            return response

        # Canonical fingerprint of selection input
        try:

            fingerprint = compact_json(
                body,
                sort_keys=True
            )

        except Exception:

            response = selection_response(
                run_id,
                None,
                [],
                [],
                [],
                None,
                ["INVALID_INPUT"]
            )

            return response

        # Existing runId
        if run_id in RUNS:

            if RUNS[run_id]["input"] == fingerprint:

                # Identical replay
                return RUNS[run_id]["response"]

            # Different input
            return JSONResponse(
                status_code=409,
                content={
                    "error": "RUN_ID_CONFLICT"
                }
            )

        # Persist complete response
        RUNS[run_id] = {
            "input": fingerprint,
            "response": response
        }

        return response

    # ========================================================
    # EVALUATE
    # ========================================================

    return process_evaluate(body)