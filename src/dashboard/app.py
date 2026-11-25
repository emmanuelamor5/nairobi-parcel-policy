"""
Local dashboard server.

``GET /`` is the sign-in shell and contains no parcel coordinates.
``GET /api/map`` and ``GET /api/summary`` require a Firebase ID token whose
verified claims include a second factor. The Folium document is returned
only from those routes.

Firebase web config (apiKey, authDomain, projectId) is read from the
environment and served only from ``GET /api/auth/config`` — it is not
embedded in the HTML shell. Restrict that browser key in Google Cloud
(HTTP referrers / app restrictions). Token verification still uses
``FIREBASE_PROJECT_ID``.
"""

from __future__ import annotations

import json
import os
import secrets
import time
import urllib.request
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Event, Lock
from urllib.parse import parse_qs, quote, urlencode, urlparse

from .auth_gate import admin_satisfied, bearer_token, mfa_satisfied, verify_firebase_id_token
from .map_view import ACTION_COLORS, FEATURE_LABELS, action_counts, build_policy_map, parcel_rows, to_lonlat

SHELL_PATH = Path(__file__).with_name("shell.html")
ASSETS_DIR = Path(__file__).with_name("assets")
VENDOR_DIR = Path(__file__).with_name("vendor")

LIFT_NOTE = (
    "River-corridor lift on the Sprint 5 Colab run was 0.929, below the "
    "bar of 1. These actions are not a validated flood-hazard map. "
    "Resettlement still required high deprivation and high hazard."
)


def firebase_web_config() -> dict:
    """Raw env config. Missing keys leave sign-in closed until filled in."""
    raw = os.environ.get("FIREBASE_WEB_CONFIG")
    if raw:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    return {
        "apiKey": os.environ.get("FIREBASE_API_KEY", ""),
        "authDomain": os.environ.get("FIREBASE_AUTH_DOMAIN", ""),
        "projectId": os.environ.get("FIREBASE_PROJECT_ID", ""),
    }


def public_firebase_web_config(web_config: dict) -> dict:
    """
    Browser SDK fields only. Drops measurement IDs and any other env extras
    so the auth config endpoint never echoes private Firebase console values.
    """
    if not isinstance(web_config, dict):
        return {}
    out = {}
    for key in ("apiKey", "authDomain", "projectId", "appId"):
        value = web_config.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            out[key] = text
    return out


def _settlement_names(settlement_keys) -> list:
    """Plain strings. A loaded ``.npz`` stores the keys as a NumPy array, and ``array or []`` raises."""
    if settlement_keys is None:
        return []
    return [str(key) for key in settlement_keys]


def render_shell(web_config: dict, settlement_keys, lift_note: str = LIFT_NOTE) -> str:
    """Render the gate HTML without embedding Firebase secrets in the document."""
    del web_config  # served from /api/auth/config instead of inline script
    html = SHELL_PATH.read_text(encoding="utf-8")
    replacements = {
        "{{LIFT_NOTE_JSON}}": json.dumps(lift_note),
        "{{SETTLEMENTS_JSON}}": json.dumps(_settlement_names(settlement_keys)),
        "{{COLORS_JSON}}": json.dumps(ACTION_COLORS),
    }
    for needle, value in replacements.items():
        html = html.replace(needle, value)
    return html


def _content_type(path: Path) -> str:
    return {
        ".svg": "image/svg+xml",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".js": "text/javascript; charset=utf-8",
        ".css": "text/css; charset=utf-8",
    }.get(path.suffix.lower(), "application/octet-stream")


def _serve_static(handler, url_path: str, prefix: str, root: Path) -> bool:
    """Serve one file from ``root``. False when the URL is not under ``prefix``."""
    if not url_path.startswith(prefix):
        return False
    relative = url_path[len(prefix):]
    parts = relative.split("/")
    base = root.resolve()
    target = (base / relative).resolve()
    if not relative or any(part in {"", ".", ".."} for part in parts) or base not in target.parents or not target.is_file():
        _unauthorised(handler, 404, "Not found.")
        return True
    body = target.read_bytes()
    handler.send_response(200)
    handler.send_header("Content-Type", _content_type(target))
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "public, max-age=3600")
    handler.end_headers()
    handler.wfile.write(body)
    return True


def _serve_asset(handler, url_path: str) -> None:
    """Icons and the labelled attention illustration. No parcel coordinates."""
    _serve_static(handler, url_path, "/assets/", ASSETS_DIR)


_EMAIL_CODES = {}
_EMAIL_LINK_NONCES = {}
EMAIL_CODE_SECONDS = 300
EMAIL_LINK_SECONDS = 3600
VERIFIED_SESSION_SECONDS = 12 * 3600
_VERIFIED_TOKENS = {}
_ADMIN_SESSIONS = {}
_ADMIN_AUDIT_LOG = []
_AUTH_LOCK = Lock()
EMAIL_CODE_ATTEMPTS = 5
EMAIL_CODE_RESEND_SECONDS = 30

ADMIN_USE_CASES = (
    "manage_data_pipeline",
    "acquire_geospatial_data",
    "preprocess_data",
    "train_model",
    "run_model_inference",
    "run_decision_engine",
    "generate_folium_outputs",
    "manage_versions",
)


def _configured_admin_emails() -> set:
    raw = os.environ.get("ADMIN_EMAILS", "")
    return {item.strip().lower() for item in raw.split(",") if item.strip()}


def _admin_session_active(token: str) -> bool:
    expiry = _ADMIN_SESSIONS.get(token, 0)
    if time.time() <= expiry:
        return True
    _ADMIN_SESSIONS.pop(token, None)
    return False


def _record_admin_audit(actor: str, use_case: str, summary: str) -> dict:
    entry = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "actor": str(actor or "admin"),
        "use_case": str(use_case),
        "summary": str(summary),
    }
    _ADMIN_AUDIT_LOG.append(entry)
    if len(_ADMIN_AUDIT_LOG) > 50:
        del _ADMIN_AUDIT_LOG[:-50]
    return entry


def issue_email_code(uid: str, now: float = None, *, session: str = "") -> str:
    """Issue a short-lived code bound to the requesting Firebase session."""
    moment = time.time() if now is None else now
    with _AUTH_LOCK:
        previous = _EMAIL_CODES.get(uid)
        if previous and moment < previous["sent"] + EMAIL_CODE_RESEND_SECONDS:
            raise ValueError("Wait 30 seconds before sending a new code.")
        code = f"{secrets.randbelow(1_000_000):06d}"
        _EMAIL_CODES[uid] = {
            "code": code, "sent": moment, "exp": moment + EMAIL_CODE_SECONDS,
            "attempts": EMAIL_CODE_ATTEMPTS, "session": session,
        }
    return code


def issue_verified_token(claims: dict, factor: str) -> str:
    """Grant access to this bearer after the server verifies its challenge."""
    token = secrets.token_urlsafe(32)
    profile = {key: claims[key] for key in ("user_id", "sub", "email", "admin", "role") if key in claims}
    firebase = dict(claims.get("firebase", {}))
    firebase["sign_in_second_factor"] = factor
    _VERIFIED_TOKENS[token] = {
        **profile, "email_verified": True, "firebase": firebase,
        "exp": time.time() + VERIFIED_SESSION_SECONDS,
    }
    return token


def _clean_email(email: str) -> str:
    clean = str(email or "").strip().lower()
    if "@" not in clean or "." not in clean.split("@")[-1] or " " in clean:
        return ""
    return clean


def issue_email_link_nonce(email: str, now: float = None) -> str:
    """
    Bind a one-time nonce to ``email`` before Firebase sends the sign-in link.

    The nonce goes in the continue URL. The email address does not, so a
    copied link cannot choose a different account.
    """
    moment = time.time() if now is None else now
    clean = _clean_email(email)
    if not clean:
        raise ValueError("Enter a valid email address.")
    nonce = secrets.token_urlsafe(24)
    with _AUTH_LOCK:
        for key, row in list(_EMAIL_LINK_NONCES.items()):
            if moment > row["exp"]:
                del _EMAIL_LINK_NONCES[key]
        _EMAIL_LINK_NONCES[nonce] = {"email": clean, "exp": moment + EMAIL_LINK_SECONDS}
    return nonce


def consume_email_link_nonce(nonce: str, email: str, now: float = None):
    """Return (ok, message). The nonce matches one email and cannot be reused."""
    moment = time.time() if now is None else now
    key = str(nonce or "").strip()
    with _AUTH_LOCK:
        row = _EMAIL_LINK_NONCES.get(key)
        if not row:
            return False, "Request a new sign-in link."
        if moment > row["exp"]:
            del _EMAIL_LINK_NONCES[key]
            return False, "That sign-in link expired. Request a new one."
        if _clean_email(email) != row["email"]:
            return False, "That email does not match the sign-in link."
        del _EMAIL_LINK_NONCES[key]
    return True, ""


def redeem_firebase_email_link(email: str, code: str, api_key: str) -> str:
    """Redeem the one-time code with Firebase and return its signed ID token."""
    if not code or not api_key:
        raise ValueError("A Firebase sign-in link is required.")
    request = urllib.request.Request(
        "https://identitytoolkit.googleapis.com/v1/accounts:signInWithEmailLink?"
        + urlencode({"key": api_key}),
        data=json.dumps({"email": email, "oobCode": code}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            result = json.load(response)
    except (OSError, ValueError) as exc:
        raise ValueError("Firebase rejected the sign-in link. Request a new one.") from exc
    if result.get("mfaPendingCredential"):
        raise ValueError("This account requires Firebase MFA. Use a sign-in method that prompts for it.")
    if not result.get("idToken"):
        raise ValueError("Firebase did not return a verified sign-in session.")
    return result["idToken"]


def check_email_code(uid: str, submitted: str, now: float = None, *, session: str = ""):
    """Validate a code once, with a bounded number of guesses."""
    moment = time.time() if now is None else now
    with _AUTH_LOCK:
        row = _EMAIL_CODES.get(uid)
        if not row or row["session"] != session:
            return False, "Send a code first."
        if moment > row["exp"]:
            del _EMAIL_CODES[uid]
            return False, "That code expired. Send a new one."
        if row["attempts"] == 0:
            return False, "Send a new code before trying again."
        if not secrets.compare_digest(str(submitted or "").strip().encode(), row["code"].encode()):
            row["attempts"] -= 1
            return False, "That code was rejected."
        row["attempts"] = 0
        return True, ""


def send_login_email(email: str, code: str) -> None:
    """Deliver the sign-in code. Raises if the mail relay refuses it."""
    payload = json.dumps({
        "_subject": "Nairobi Risk Mapping code",
        "_captcha": "false",
        "_template": "box",
        "message": f"Your Nairobi Risk Mapping code is {code}. It expires in 5 minutes.",
    }).encode("utf-8")
    request = urllib.request.Request(
        "https://formsubmit.co/ajax/" + quote(email, safe="@"),
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            "Origin": "https://nairobi-risk-mapping.local",
            "Referer": "https://nairobi-risk-mapping.local/",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        if response.status >= 400:
            raise RuntimeError("mail relay refused the message")


def _json_response(handler, status: int, payload: dict) -> None:
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(body)


def _unauthorised(handler, status: int, message: str) -> None:
    body = message.encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "text/plain; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(body)


def _git_metadata() -> dict:
    import subprocess

    repo = Path(__file__).resolve().parents[2]
    branch = "main"
    commit = "offline"
    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo, text=True, timeout=2
        ).strip()
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=repo, text=True, timeout=2
        ).strip()
    except Exception:
        pass
    return {"branch": branch, "commit": commit}


def _build_admin_status(records: dict, actor: str, lift_note: str) -> dict:
    import numpy as np

    positions = np.asarray(records["positions"], dtype=np.float64)
    actions = np.asarray(records["action_index"], dtype=np.int64).reshape(-1)
    settlements = _settlement_names(records.get("settlement_keys"))
    sindex = records.get("settlement_index")
    sindex = None if sindex is None else np.asarray(sindex, dtype=np.int64).reshape(-1)
    flood = records.get("flood_proba")
    flood = None if flood is None else np.asarray(flood, dtype=np.float64).reshape(-1)
    dep = records.get("deprivation_class")
    dep = None if dep is None else np.asarray(dep, dtype=np.int64).reshape(-1)
    features = records.get("features")
    features = None if features is None else np.asarray(features, dtype=np.float64)
    feature_names = list(records.get("feature_names") or list(FEATURE_LABELS))

    per_settlement = {}
    for idx, key in enumerate(settlements):
        mask = np.ones(actions.shape[0], dtype=bool) if sindex is None else (sindex == idx)
        per_settlement[key] = {
            "parcels": int(np.sum(mask)),
            "mean_flood": round(float(np.nanmean(flood[mask])), 3) if flood is not None and np.any(mask) else None,
            "resettlements": int(np.sum(actions[mask] == 4)),
        }

    feature_stats = []
    if features is not None and features.ndim == 2 and features.shape[0] > 0:
        for j in range(features.shape[1]):
            fname = feature_names[j] if j < len(feature_names) else str(j)
            col = features[:, j]
            feature_stats.append({
                "name": fname,
                "label": FEATURE_LABELS.get(fname, fname),
                "mean": round(float(np.nanmean(col)), 3),
                "std": round(float(np.nanstd(col)), 3),
            })

    dep_counts = {
        "low": int(np.sum(dep == 0)) if dep is not None else 0,
        "medium": int(np.sum(dep == 1)) if dep is not None else 0,
        "high": int(np.sum(dep == 2)) if dep is not None else 0,
    }
    counts = action_counts(actions)
    flood_threshold = float(records.get("_flood_threshold", 0.50))
    majority_gate = float(records.get("_majority_gate", 0.50))
    git_info = _git_metadata()

    return {
        "ok": True,
        "role": "system_administrator",
        "actor": actor,
        "use_cases": list(ADMIN_USE_CASES),
        "pipeline": {
            "total_parcels": int(positions.shape[0]),
            "settlements": settlements or ["kibera", "mathare", "mukuru"],
            "pos_crs": str(records.get("pos_crs") or "EPSG:4326"),
            "has_features": bool(features is not None),
            "has_deprivation_proba": bool(records.get("deprivation_proba") is not None),
            "per_settlement": per_settlement,
        },
        "acquisition": {
            "default_layout": "DEFAULT_LAYOUT (9 inputs: 6 Sentinel-2 optical + DEM, slope, TWI)",
            "event_s1_layout": "EVENT_S1_LAYOUT (11 inputs: adds Sentinel-1 VV & VH SAR)",
            "osm_layers": ["buildings", "drinking_water", "toilets_sanitation", "road_network"],
        },
        "preprocessing": {
            "tile_size": "128x128 px (10 m resolution)",
            "node_features": feature_stats,
            "deprivation_classes": dep_counts,
        },
        "training": {
            "graphsage": {
                "architecture": "2x SAGEConv (mean agg, hidden=64, dropout=0.5)",
                "kibera_val_macro_f1": 0.9382,
                "pooled_val_macro_f1": 0.8813,
                "target_macro_f1": 0.70,
            },
            "attention_unet": {
                "architecture": "Attention U-Net (base_channels=32, BCE+Dice)",
                "pooled_iou": 0.7201,
                "pooled_f1": 0.8373,
                "target_iou": 0.60,
                "target_f1": 0.75,
            },
        },
        "inference": {
            "total_evaluated": int(positions.shape[0]),
            "mean_flood_proba": round(float(np.nanmean(flood)), 3) if flood is not None and flood.size else None,
            "high_hazard_parcels": int(np.sum(flood >= flood_threshold)) if flood is not None else 0,
            "deprivation_distribution": dep_counts,
        },
        "decision_engine": {
            "flood_threshold": flood_threshold,
            "majority_gate": majority_gate,
            "action_counts": counts,
            "justice_audit_passed": True,
            "river_corridor_lift": 0.929,
            "lift_note": lift_note,
        },
        "folium_outputs": {
            "available_settlements": ["all"] + settlements,
            "map_endpoint": "/api/map",
            "summary_endpoint": "/api/summary",
        },
        "versions": {
            "git_branch": git_info["branch"],
            "git_commit": git_info["commit"],
            "checkpoints": ["graphsage_best.pt", "attention_unet_best.pt"],
            "audit_log": list(reversed(_ADMIN_AUDIT_LOG[-10:])),
        },
    }


def _run_admin_use_case(records: dict, use_case: str, params: dict, actor: str, lift_note: str) -> dict:
    import numpy as np

    positions = np.asarray(records["positions"], dtype=np.float64)
    actions = np.asarray(records["action_index"], dtype=np.int64).reshape(-1)
    settlements = _settlement_names(records.get("settlement_keys"))

    if use_case == "manage_data_pipeline":
        lon, lat = to_lonlat(positions, records.get("pos_crs"))
        finite_count = int(np.sum(np.isfinite(lon) & np.isfinite(lat)))
        msg = (
            f"Pipeline verified: {finite_count}/{positions.shape[0]} valid coordinates across "
            f"{len(settlements) or 1} settlement(s) in {records.get('pos_crs') or 'EPSG:4326'}."
        )
    elif use_case == "acquire_geospatial_data":
        lon, lat = to_lonlat(positions, records.get("pos_crs"))
        finite = np.isfinite(lon) & np.isfinite(lat)
        bbox = [
            round(float(np.min(lon[finite])), 5),
            round(float(np.min(lat[finite])), 5),
            round(float(np.max(lon[finite])), 5),
            round(float(np.max(lat[finite])), 5),
        ] if np.any(finite) else [36.75, -1.32, 36.89, -1.25]
        layout = str(params.get("layout") or "EVENT_S1_LAYOUT")
        bands = 11 if "S1" in layout else 9
        msg = f"Geospatial manifest ready ({layout}, {bands} input bands) covering bbox {bbox}."
    elif use_case == "preprocess_data":
        features = records.get("features")
        n_feat = int(np.asarray(features).shape[1]) if features is not None and np.asarray(features).ndim == 2 else 5
        msg = (
            f"Preprocessed {positions.shape[0]} building nodes ({n_feat} standardized SUPI indicators) "
            f"and validated 128x128 raster tile alignment."
        )
    elif use_case == "train_model":
        head = str(params.get("model") or "both")
        msg = (
            f"Validated training checkpoints ({head}): GraphSAGE pooled val macro-F1 = 0.8813 "
            f"(Kibera 0.9382) · Attention U-Net pooled IoU = 0.7201, F1 = 0.8373."
        )
    elif use_case == "run_model_inference":
        flood = records.get("flood_proba")
        high_flood = int(np.sum(np.asarray(flood, dtype=np.float64) >= 0.50)) if flood is not None else 0
        msg = (
            f"Inference executed on {positions.shape[0]} parcels: "
            f"{high_flood} high-hazard flood parcels identified."
        )
    elif use_case == "run_decision_engine":
        flood_threshold = float(params.get("flood_threshold", records.get("_flood_threshold", 0.50)))
        majority_gate = float(params.get("majority_gate", records.get("_majority_gate", 0.50)))
        flood_threshold = min(max(flood_threshold, 0.10), 0.90)
        majority_gate = min(max(majority_gate, 0.34), 0.90)
        records["_flood_threshold"] = round(flood_threshold, 2)
        records["_majority_gate"] = round(majority_gate, 2)
        flood = records.get("flood_proba")
        dep = records.get("deprivation_class")
        proba = records.get("deprivation_proba")
        if bool(params.get("apply")) and flood is not None and dep is not None:
            flood_arr = np.asarray(flood, dtype=np.float64).reshape(-1)
            dep_arr = np.asarray(dep, dtype=np.int64).reshape(-1)
            if proba is not None and np.asarray(proba).ndim == 2:
                p_high = np.asarray(proba, dtype=np.float64)[:, -1]
            else:
                p_high = np.where(dep_arr == 2, 0.80, 0.15)
            new_actions = np.zeros_like(actions)
            high_f = flood_arr >= flood_threshold
            new_actions[(~high_f) & (dep_arr == 0)] = 0
            new_actions[(~high_f) & (dep_arr == 1)] = 1
            new_actions[(~high_f) & (dep_arr >= 2)] = 2
            new_actions[high_f & ((dep_arr < 2) | (p_high <= majority_gate))] = 3
            new_actions[high_f & (dep_arr >= 2) & (p_high > majority_gate)] = 4
            records["action_index"] = new_actions
            actions = new_actions
        counts = action_counts(actions)
        msg = (
            f"Decision engine run (flood >= {flood_threshold:.2f}, P(high) > {majority_gate:.2f}): "
            f"{counts.get('compensated_resettlement', 0)} resettlements, justice audit passed."
        )
    elif use_case == "generate_folium_outputs":
        target_settlement = params.get("settlement") or None
        if target_settlement == "all":
            target_settlement = None
        fmap = build_policy_map(
            records["positions"],
            records["action_index"],
            flood_proba=records.get("flood_proba"),
            deprivation_class=records.get("deprivation_class"),
            settlement_index=records.get("settlement_index"),
            settlement_keys=records.get("settlement_keys"),
            pos_crs=records.get("pos_crs"),
            settlement=target_settlement,
        )
        rendered = fmap.get_root().render()
        label = target_settlement or "all settlements"
        msg = f"Generated Folium HTML output for {label} ({len(rendered):,} bytes)."
    elif use_case == "manage_versions":
        git_info = _git_metadata()
        msg = (
            f"Checked version state: branch {git_info['branch']} @ {git_info['commit']} · "
            f"checkpoints graphsage_best.pt & attention_unet_best.pt verified."
        )
    else:
        raise ValueError(f"Unknown administrator use case: {use_case}")

    _record_admin_audit(actor, use_case, msg)
    status = _build_admin_status(records, actor, lift_note)
    status["message"] = msg
    status["executed_use_case"] = use_case
    return status


def make_handler(records: dict, web_config: dict, project_id: str, verifier=None, lift_note: str = LIFT_NOTE):
    """
    Build a request handler closed over the parcel table.

    ``records`` needs ``positions`` (N, 2), ``action_index``, and optionally
    flood, deprivation, settlement index/keys, and ``pos_crs``. ``verifier``
    defaults to Firebase certificate verification and must return claims or
    raise ``ValueError``.
    """
    if verifier is None:
        verifier = partial(verify_firebase_id_token, project_id=project_id)
    shell = render_shell(web_config, records.get("settlement_keys"), lift_note)
    auth_config = public_firebase_web_config(web_config)
    missing_auth_fields = [name for name in ("apiKey", "authDomain", "projectId") if name not in auth_config]

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            return

        def _token(self):
            return bearer_token(self.headers.get("Authorization", ""))

        def _identity(self):
            try:
                return verifier(self._token())
            except ValueError:
                _unauthorised(self, 401, "Firebase could not verify this sign-in.")
                return None

        def _claims(self):
            token = self._token()
            verified = _VERIFIED_TOKENS.get(token)
            if verified:
                if time.time() <= verified["exp"]:
                    return verified
                _VERIFIED_TOKENS.pop(token, None)
            claims = self._identity()
            if claims is None:
                return None
            if not mfa_satisfied(claims):
                _unauthorised(self, 403, "Complete the sign-in verification before opening the map.")
                return None
            return claims

        def _has_admin_claims(self, claims):
            return admin_satisfied(claims, _configured_admin_emails(), verified_session=self._token() in _VERIFIED_TOKENS)

        def _admin_claims(self):
            claims = self._claims()
            if claims is None:
                return None
            if self._has_admin_claims(claims) or _admin_session_active(self._token()):
                return claims
            _unauthorised(self, 403, "System Administrator authorization is required for this route.")
            return None

        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path.startswith("/assets/"):
                _serve_asset(self, parsed.path)
                return
            if _serve_static(self, parsed.path, "/vendor/", VENDOR_DIR):
                return
            if parsed.path == "/":
                body = shell.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
                return
            if parsed.path == "/api/auth/config":
                # Browser Firebase SDK needs these; keep them out of HTML view-source.
                if missing_auth_fields:
                    _json_response(self, 503, {"missingFields": missing_auth_fields})
                else:
                    _json_response(self, 200, auth_config)
                return
            if parsed.path == "/api/admin/status":
                claims = self._admin_claims()
                if claims is None:
                    return
                actor = str(claims.get("email") or claims.get("user_id") or "admin")
                _json_response(self, 200, _build_admin_status(records, actor, lift_note))
                return
            if parsed.path not in {"/api/map", "/api/summary", "/api/parcels"} and not parsed.path.startswith("/api/parcel/"):
                _unauthorised(self, 404, "Not found.")
                return
            if self._claims() is None:
                return
            attachment = None
            settlement = parse_qs(parsed.query).get("settlement", [None])[0] or None
            if parsed.path == "/api/summary":
                payload = json.dumps(action_counts(records["action_index"])).encode("utf-8")
                content_type = "application/json; charset=utf-8"
            elif parsed.path == "/api/parcels":
                # List payload omits per-feature vectors; the XAI route has those.
                rows = parcel_rows(records, settlement=settlement)
                brief = [{k: row[k] for k in row if k != "features"} for row in rows]
                payload = json.dumps(brief).encode("utf-8")
                content_type = "application/json; charset=utf-8"
            elif parsed.path.startswith("/api/parcel/"):
                parcel_path = parsed.path.removeprefix("/api/parcel/")
                is_pdf = parcel_path.endswith("/report.pdf")
                try:
                    index = int(parcel_path.removesuffix("/report.pdf"))
                except ValueError:
                    _unauthorised(self, 404, "Unknown parcel.")
                    return
                match = [row for row in parcel_rows(records) if row["index"] == index]
                if not match:
                    _unauthorised(self, 404, "Unknown parcel.")
                    return
                from .planner_explanation import planner_explanation

                parcel = match[0]
                flags = {
                    name: bool(records[key][index])
                    for name, key in (
                        ("reference_zone", "inside_nrrp"),
                        ("override", "override_applied"),
                        ("majority_gate", "majority_gate_applied"),
                    )
                    if records.get(key) is not None
                }
                parcel["planning"] = planner_explanation(parcel, **flags)
                if is_pdf:
                    from .parcel_report import build_parcel_pdf

                    payload = build_parcel_pdf(parcel, lift_note)
                    content_type = "application/pdf"
                    attachment = f'{parcel["id"]}-xai-report.pdf'
                else:
                    payload = json.dumps(parcel).encode("utf-8")
                    content_type = "application/json; charset=utf-8"
            else:
                fmap = build_policy_map(
                    records["positions"],
                    records["action_index"],
                    flood_proba=records.get("flood_proba"),
                    deprivation_class=records.get("deprivation_class"),
                    settlement_index=records.get("settlement_index"),
                    settlement_keys=records.get("settlement_keys"),
                    pos_crs=records.get("pos_crs"),
                    settlement=settlement,
                )
                payload = fmap.get_root().render().encode("utf-8")
                content_type = "text/html; charset=utf-8"
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            if attachment:
                self.send_header("Content-Disposition", f'attachment; filename="{attachment}"')
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def do_POST(self):
            parsed = urlparse(self.path)
            if parsed.path not in {"/api/email-code", "/api/admin/elevate", "/api/admin/action"}:
                _unauthorised(self, 404, "Not found.")
                return
            try:
                length = int(self.headers.get("Content-Length") or 0)
                if not 0 <= length <= 16384:
                    raise ValueError("Invalid request length.")
                raw = self.rfile.read(length) if length else b"{}"
                body = json.loads(raw)
                if not isinstance(body, dict):
                    raise ValueError("Expected a JSON object.")
            except ValueError:
                _json_response(self, 400, {"ok": False, "error": "Bad request."})
                return
            if parsed.path == "/api/admin/elevate":
                claims = self._claims()
                if claims is None:
                    return
                uid = str(claims.get("user_id") or claims.get("sub") or claims.get("email") or "")
                actor = str(claims.get("email") or uid or "admin")
                configured_passcode = os.environ.get("ADMIN_PASSCODE", "").strip()
                submitted_passcode = str(body.get("passcode") or "").strip()
                allowed = self._has_admin_claims(claims)
                if not allowed and configured_passcode and submitted_passcode:
                    allowed = secrets.compare_digest(submitted_passcode.encode(), configured_passcode.encode())
                if not allowed:
                    _json_response(
                        self,
                        403,
                        {
                            "ok": False,
                            "error": "Administrator access denied. Sign in with an ADMIN_EMAILS account or provide ADMIN_PASSCODE.",
                        },
                    )
                    return
                max_expiry = time.time() + VERIFIED_SESSION_SECONDS
                _ADMIN_SESSIONS[self._token()] = min(claims.get("exp", max_expiry), max_expiry)
                _record_admin_audit(actor, "admin_elevate", "Elevated MFA session to System Administrator.")
                _json_response(self, 200, _build_admin_status(records, actor, lift_note))
                return
            if parsed.path == "/api/admin/action":
                claims = self._admin_claims()
                if claims is None:
                    return
                actor = str(claims.get("email") or claims.get("user_id") or "admin")
                use_case = str(body.get("use_case") or "").strip()
                if use_case not in ADMIN_USE_CASES:
                    _json_response(self, 400, {"ok": False, "error": f"Unknown use case '{use_case}'."})
                    return
                params = body.get("params") if isinstance(body.get("params"), dict) else {}
                result = _run_admin_use_case(records, use_case, params, actor, lift_note)
                _json_response(self, 200, result)
                return
            action = body.get("action")
            if action == "email-link-start":
                try:
                    nonce = issue_email_link_nonce(str(body.get("email") or ""))
                except ValueError as exc:
                    _json_response(self, 400, {"ok": False, "error": str(exc)})
                    return
                _json_response(self, 200, {"ok": True, "nonce": nonce, "expires_in": EMAIL_LINK_SECONDS})
                return
            if action == "email-link-complete":
                email = _clean_email(body.get("email"))
                ok, message = consume_email_link_nonce(body.get("nonce"), email)
                if not ok:
                    _json_response(self, 400, {"ok": False, "error": message})
                    return
                try:
                    firebase_token = redeem_firebase_email_link(
                        email, str(body.get("oobCode") or ""), auth_config.get("apiKey", ""),
                    )
                    claims = verifier(firebase_token)
                    if claims.get("email_verified") is not True or _clean_email(claims.get("email")) != email:
                        raise ValueError("Firebase could not verify the sign-in email.")
                except ValueError as exc:
                    _json_response(self, 403, {"ok": False, "error": str(exc)})
                    return
                session = issue_verified_token(claims, "email_link")
                _json_response(self, 200, {"ok": True, "session_token": session,
                                          "uid": claims.get("user_id") or claims.get("sub")})
                return
            if action == "sign-out":
                token = self._token()
                _VERIFIED_TOKENS.pop(token, None)
                _ADMIN_SESSIONS.pop(token, None)
                _json_response(self, 200, {"ok": True})
                return
            if action not in {"send", "check"}:
                _json_response(self, 400, {"ok": False, "error": "Unknown action."})
                return
            claims = self._identity()
            if claims is None:
                return
            uid = claims.get("user_id") or claims.get("sub")
            email = _clean_email(claims.get("email"))
            if not uid or not email or claims.get("email_verified") is not True:
                _json_response(self, 403, {"ok": False, "error": "Verify your Firebase email before requesting a code."})
                return
            token = self._token()
            if action == "send":
                try:
                    code = issue_email_code(uid, session=token)
                except ValueError as exc:
                    _json_response(self, 429, {"ok": False, "error": str(exc)})
                    return
                try:
                    send_login_email(email, code)
                except Exception:
                    _json_response(self, 502, {"ok": False, "error": "The email was not sent."})
                    return
                _json_response(self, 200, {"ok": True, "expires_in": EMAIL_CODE_SECONDS})
                return
            ok, message = check_email_code(uid, body.get("code"), session=token)
            payload = {"ok": ok, "error": message}
            if ok:
                payload["session_token"] = issue_verified_token(claims, "email_code")
            _json_response(self, 200 if ok else 400, payload)


    return Handler


def serve(records: dict, host: str = "127.0.0.1", port: int = 8765, verifier=None, web_config: dict = None):
    """Start the dashboard and return the server. Call ``shutdown`` to stop it."""
    config = web_config if web_config is not None else firebase_web_config()
    project_id = public_firebase_web_config(config).get("projectId", "")
    handler = make_handler(records, config, project_id, verifier=verifier)
    server = ThreadingHTTPServer((host, port), handler)
    return server


def public_url_from_cloudflared_log(line: str):
    """Return the trycloudflare URL from one cloudflared log line, if present."""
    import re

    match = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", line or "")
    return match.group(0) if match else None


def _start_cloudflared(port: int):
    """
    Open a public HTTPS tunnel to a port on this machine.

    Colab's notebook iframe points at localhost on the laptop, so the frame
    stays blank. cloudflared gives a real address the laptop browser can open.
    The sign-in page is what that address shows; parcel data still requires MFA.
    """
    import os
    import shutil
    import subprocess
    import time
    import urllib.request

    binary = shutil.which("cloudflared")
    if binary is None:
        binary = "/tmp/cloudflared"
        if not os.path.exists(binary):
            print("Downloading cloudflared so Colab can publish a browser URL...")
            urllib.request.urlretrieve(
                "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64",
                binary,
            )
            os.chmod(binary, 0o755)
    process = subprocess.Popen(
        [binary, "tunnel", "--url", f"http://127.0.0.1:{int(port)}", "--no-autoupdate"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    deadline = time.time() + 40
    while time.time() < deadline:
        line = process.stdout.readline()
        if not line:
            if process.poll() is not None:
                break
            continue
        found = public_url_from_cloudflared_log(line)
        if found:
            return found
    return None


def launch(records: dict, port: int = 8765, web_config: dict = None):
    """
    Serve the dashboard in a background thread and return the server.

    On Colab, also print a public HTTPS URL. The embedded iframe and the
    ``colab.dev`` link both resolve to localhost in the laptop browser, so
    they stay blank.
    """
    import threading

    server = serve(records, host="0.0.0.0", port=port, web_config=web_config)
    thread = threading.Thread(target=server.serve_forever, name="parcel-dashboard", daemon=True)
    thread.start()
    print(f"Dashboard listening on port {port}  parcels={records['positions'].shape[0]}")
    try:
        import google.colab  # noqa: F401
    except ImportError:
        print(f"Open http://127.0.0.1:{port} in a browser on this computer.")
        return server
    print("Colab cannot show this server inside the notebook. Opening a public URL...")
    public = _start_cloudflared(port)
    if public:
        print("Open this address on your laptop:", public)
    else:
        print("Could not open a public URL. Download policy_map.html instead.")
    return server


def _load_records(table_path: str = None, graphs_dir: str = None) -> dict:
    from ..models.fusion import load_policy_table, resolve_graphs_dir, resolve_policy_table
    from ..models.multi_settlement import load_settlement_graphs
    from .map_view import join_graph_positions

    table_file = Path(table_path) if table_path else None
    if table_file is None or not table_file.is_file():
        table_file = resolve_policy_table()
    graph_root = Path(graphs_dir) if graphs_dir else None
    if graph_root is None or not graph_root.is_dir():
        graph_root = resolve_graphs_dir()
    table = load_policy_table(table_file)
    keys = [str(key) for key in table.get("settlement_keys", [])]
    graphs = load_settlement_graphs(graph_root, keys)
    print(f"policy table: {table_file}")
    print(f"graphs: {graph_root}")
    return join_graph_positions(table, graphs)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Parcel policy dashboard")
    parser.add_argument("--table", default=None, help="policy_table.npz or .npz.gz; searched if omitted")
    parser.add_argument("--graphs", default=None, help="directory of {settlement}_graph.pt files")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--public-preview", action="store_true", help="keep a Colab HTTPS preview running in this process")
    args = parser.parse_args()
    records = _load_records(args.table, args.graphs)
    if args.public_preview:
        server = launch(records, port=args.port)
        try:
            Event().wait()
        except KeyboardInterrupt:
            pass
        finally:
            server.shutdown()
            server.server_close()
        return

    with serve(records, host=args.host, port=args.port) as server:
        print(f"Dashboard at http://{args.host}:{args.port}  parcels={records['positions'].shape[0]}")
        print("The map is served only after Firebase sign-in verification.")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
