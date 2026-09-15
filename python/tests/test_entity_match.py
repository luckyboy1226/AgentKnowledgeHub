"""Strict entity identity tests; no provider, network, or graph is used."""

from __future__ import annotations

from services.entity_match import entity_match_key, entity_names_match


def test_atlas_whitespace_variants_match():
    assert entity_names_match("Atlas事件服务", "Atlas 事件服务")


def test_unicode_form_case_and_whitespace_are_normalized_deterministically():
    assert entity_match_key(" Ａｔｌａｓ　事件服务 ") == entity_match_key("atlas事件服务")
    assert entity_match_key("Atlas\t事件服务") == "atlas事件服务"


def test_business_suffixes_are_not_removed_or_fuzzy_matched():
    assert not entity_names_match("Atlas服务", "Atlas事件服务")
    assert not entity_names_match("北极星", "北极星检索平台")
    assert not entity_names_match("天枢", "天枢知识平台")


def test_empty_values_never_match():
    assert not entity_names_match("", " ")
