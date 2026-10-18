"""Sign-in: salted PBKDF2 password hashes and random bearer tokens (standard library only).

Only a SHA-256 of each token is stored, so a database leak doesn't hand out live sessions.
Forgotten passwords: a 6-digit code is emailed (SMTP_* in backend/.env) or, with no SMTP
set up, printed in the backend console so a local run still works.
"""
import hashlib
import hmac
import os
import secrets
import smtplib
from datetime import timedelta
from email.message import EmailMessage

from fastapi import Depends, Header, HTTPException

from buddy.store import now, one_as_dict, open_conn, session_expiry

ROLES = ("student", "supervisor", "examiner")
STAFF_ROLES = ("supervisor", "examiner")
ITERATIONS = 200_000
RESET_MINUTES = 15
RESET_ATTEMPTS = 5


def hash_password(password):
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, ITERATIONS)
    return f"pbkdf2${salt.hex()}${digest.hex()}"


def check_password(password, stored):
    try:
        _, salt, digest = stored.split("$")
    except ValueError:
        return False
    test = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), ITERATIONS)
    return hmac.compare_digest(test.hex(), digest)


def _token_hash(token):
    return hashlib.sha256(token.encode()).hexdigest()


def start_session(conn, user_id):
    token = secrets.token_urlsafe(32)
    conn.execute("INSERT INTO sessions VALUES (?, ?, ?)", (_token_hash(token), user_id, session_expiry()))
    conn.commit()
    return token


def end_session(conn, token):
    conn.execute("DELETE FROM sessions WHERE token_hash = ?", (_token_hash(token),))
    conn.commit()


def end_other_sessions(conn, user_id, keep_token=None):
    """After a password change: sign the user out everywhere else (keep_token = this device)."""
    if keep_token:
        conn.execute("DELETE FROM sessions WHERE user_id = ? AND token_hash <> ?", (user_id, _token_hash(keep_token)))
    else:
        conn.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))
    conn.commit()


# ---------- Forgotten password ----------

def start_reset(conn, user_id):
    """New 6-digit code for this user (replaces any older one). Returns the code."""
    code = f"{secrets.randbelow(1_000_000):06d}"
    conn.execute("DELETE FROM password_resets WHERE user_id = ?", (user_id,))
    conn.execute("INSERT INTO password_resets VALUES (?, ?, ?, 0)",
                 (user_id, _token_hash(code), now(timedelta(minutes=RESET_MINUTES))))
    conn.commit()
    return code


def check_reset(conn, user_id, code):
    """True if the code is right and still valid. Wrong guesses count; after RESET_ATTEMPTS the code dies."""
    row = one_as_dict(conn.execute("SELECT * FROM password_resets WHERE user_id = ?", (user_id,)))
    if not row or row["expires_at"] <= now() or row["attempts"] >= RESET_ATTEMPTS:
        return False
    if hmac.compare_digest(_token_hash((code or "").strip()), row["code_hash"]):
        conn.execute("DELETE FROM password_resets WHERE user_id = ?", (user_id,))
        conn.commit()
        return True
    conn.execute("UPDATE password_resets SET attempts = attempts + 1 WHERE user_id = ?", (user_id,))
    conn.commit()
    return False


def send_reset_code(email, name, code):
    """Email the code. Without SMTP_HOST it is printed in the backend console instead (local runs)."""
    host = os.getenv("SMTP_HOST", "")
    if not host:
        print(f"[buddy] password reset code for {email}: {code}  (set SMTP_HOST in .env to email it)")
        return
    msg = EmailMessage()
    msg["Subject"] = "Your Buddy.ds password reset code"
    msg["From"] = os.getenv("SMTP_FROM") or os.getenv("SMTP_USER", "")
    msg["To"] = email
    msg.set_content(f"Hi {name},\n\nYour code to reset your Buddy.ds password is {code}.\n"
                    f"It works for {RESET_MINUTES} minutes. If you didn't ask for this, you can ignore this email.\n")
    with smtplib.SMTP(host, int(os.getenv("SMTP_PORT", "587")), timeout=15) as smtp:
        smtp.starttls()
        if os.getenv("SMTP_USER"):
            smtp.login(os.getenv("SMTP_USER"), os.getenv("SMTP_PASSWORD", ""))
        smtp.send_message(msg)


def staff_code_ok(code):
    """Examiner accounts need the STAFF_CODE from backend/.env (they can read all feedback)."""
    expected = os.getenv("STAFF_CODE", "")
    return bool(expected) and hmac.compare_digest(code or "", expected)


def public_user(u):
    return {"user_id": u["user_id"], "email": u["email"], "name": u["name"], "role": u["role"]}


# ---------- FastAPI dependencies ----------

def get_conn():
    conn = open_conn()
    try:
        yield conn
    finally:
        conn.close()


def _bearer(authorization):
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(401, "Please sign in.")
    return authorization.split(" ", 1)[1].strip()


def current_user(conn=Depends(get_conn), authorization: str | None = Header(None)):
    token = _bearer(authorization)
    user = one_as_dict(conn.execute(
        "SELECT u.* FROM sessions s JOIN users u ON u.user_id = s.user_id "
        "WHERE s.token_hash = ? AND s.expires_at > ?", (_token_hash(token), now())))
    if not user:
        raise HTTPException(401, "Your session has expired. Please sign in again.")
    return user


def current_token(authorization: str | None = Header(None)):
    return _bearer(authorization)
