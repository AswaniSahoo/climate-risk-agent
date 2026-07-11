"""Evaluation harness: the gold eval set + retrieval/answer metrics.

Authored BEFORE retrieval exists so the benchmark can't be gamed, then frozen +
content-hashed so every future "recall improved" is measured against the same
questions.
"""
