import hashlib
import re

from component2.config import ANON_SALT

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE_RE = re.compile(r"\+?\d[\d \-]{8,}\d")


def make_code(text):
    digest = hashlib.sha256((ANON_SALT + text).encode()).hexdigest()
    return "M-" + digest[:6]


class MemberResolver:
    """Gives every person in ONE team a code like M-3f9a2c.
    If two commits share a name OR an email, they get the same code."""

    def __init__(self, team_id):
        self.team_id = team_id
        self.by_name = {}
        self.by_email = {}
        self.real_names = set()

    def get_id(self, name, email=""):
        name = (name or "").strip().lower()
        email = (email or "").strip().lower()
        member_id = self.by_name.get(name) or self.by_email.get(email)
        if member_id is None:
            member_id = make_code(self.team_id + "|" + (email or name))
        if name:
            self.by_name[name] = member_id
            self.real_names.add(name)
        if email:
            self.by_email[email] = member_id
        return member_id

    def scrub(self, text):
        text = EMAIL_RE.sub("[EMAIL]", text)
        text = PHONE_RE.sub("[PHONE]", text)
        for name in self.real_names:
            if len(name) >= 3:
                text = re.sub(r"\b" + re.escape(name) + r"\b", self.by_name[name], text, flags=re.I)
        return text