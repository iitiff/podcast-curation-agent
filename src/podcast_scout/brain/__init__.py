"""Durable brain: markdown entity pages, thesis evidence, and falsifier watch."""
from .falsifier import (
    FalsifierHit,
    QuestionHit,
    SignalInput,
    check_falsifiers,
    check_questions,
)
from .schema import (
    BrainPage,
    Company,
    ConfidenceEntry,
    Pattern,
    Question,
    Source,
    Thesis,
    append_to_section,
    join_frontmatter,
    slugify,
    split_frontmatter,
)
from .store import BrainStore, source_id_for
from .writer import BrainWriteResult, write_to_brain

__all__ = [
    "BrainPage",
    "BrainStore",
    "BrainWriteResult",
    "Company",
    "ConfidenceEntry",
    "FalsifierHit",
    "Pattern",
    "Question",
    "QuestionHit",
    "SignalInput",
    "Source",
    "Thesis",
    "append_to_section",
    "check_falsifiers",
    "check_questions",
    "join_frontmatter",
    "slugify",
    "source_id_for",
    "split_frontmatter",
    "write_to_brain",
]
