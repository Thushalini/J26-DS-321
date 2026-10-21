"""Anonymising (NFR07): member codes and scrubbing names, emails and phone numbers from text."""
import json
import re

from component2 import anonymise
from component2.anonymise import MemberResolver, make_code


def test_codes_are_short_stable_and_not_the_name():
    code = make_code("team|thusha@uni.lk")
    assert re.fullmatch(r"M-[0-9a-f]{6}", code)
    assert code == make_code("team|thusha@uni.lk") and code != make_code("team|other@uni.lk")


def test_same_email_or_name_is_the_same_person():
    r = MemberResolver("team_x")
    first = r.get_id("Thusha", "thusha@uni.lk")
    assert r.get_id("Thushalini P", "THUSHA@uni.lk") == first        # same email, other name
    assert r.get_id("thusha", "123+t@users.noreply.github.com") == first   # same name, other email
    assert r.get_id("Kamal", "kamal@uni.lk") != first


def test_same_person_gets_a_different_code_in_another_team():
    assert MemberResolver("team_x").get_id("Thusha", "t@uni.lk") != MemberResolver("team_y").get_id("Thusha", "t@uni.lk")


def test_scrub_removes_names_emails_and_phone_numbers():
    r = MemberResolver("team_x")
    code = r.get_id("Thusha", "thusha@uni.lk")
    r.get_id("Al", "al@uni.lk")
    text = r.scrub("Thusha fixed it, mail thusha@uni.lk or call +94 77 123 4567. Al agreed.")
    assert "Thusha" not in text and code in text
    assert "[EMAIL]" in text and "[PHONE]" in text and "@" not in text
    assert "Al agreed" in text                                      # names under 3 letters are left alone


def test_alias_file_joins_identities(tmp_path, monkeypatch):
    monkeypatch.setattr(anonymise, "ALIAS_DIR", tmp_path)
    (tmp_path / "team_x.json").write_text(json.dumps(
        {"people": [["thusha@uni.lk", "99+tp@users.noreply.github.com", "thusha-p"]]}), encoding="utf-8")
    r = MemberResolver("team_x")
    assert r.get_id("thusha-p") == r.get_id("", "99+tp@users.noreply.github.com") == r.get_id("", "thusha@uni.lk")
