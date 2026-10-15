from datetime import date, datetime
from enum import Enum
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.logging import get_logger
from app.models.document import DocumentStatus
from app.retrieval.base import RetrievedChunk
from app.schemas.ask import ContradictionReport

logger = get_logger(__name__)

STOP_WORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of",
    "with", "by", "from", "as", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "can", "could", "should", "would",
    "will", "shall", "may", "might", "must", "that", "which", "this", "these",
    "those", "it", "its", "they", "their", "them", "we", "our", "you", "your",
    "based", "provided", "context", "documentation", "document", "stated", "states",
    "according", "policy", "guidelines", "rules", "standard", "manual", "section"
}

POSITIVE_INDICATORS = {
    "permitted", "allowed", "authorized", "mandatory", "required", "approved",
    "eligible", "entitled", "must", "shall", "granted", "applicable", "unlimited"
}

NEGATIVE_INDICATORS = {
    "prohibited", "forbidden", "disallowed", "banned", "restricted", "prohibit",
    "not permitted", "not allowed", "ineligible", "never", "cannot", "no",
    "excluded", "disapproved", "cancelled", "void"
}


class ContradictionStatus(str, Enum):
    """Overall concordance outcome across retrieved evidence chunks."""

    AGREE = "AGREE"
    PARTIALLY_CONFLICT = "PARTIALLY_CONFLICT"
    CONFLICT = "CONFLICT"


class ConflictResolutionAction(str, Enum):
    """Action taken to resolve conflicting evidence."""

    NO_CONFLICT = "NO_CONFLICT"
    RESOLVED_BY_ACTIVE_STATUS = "RESOLVED_BY_ACTIVE_STATUS"
    RESOLVED_BY_EFFECTIVE_DATE = "RESOLVED_BY_EFFECTIVE_DATE"
    RESOLVED_BY_NEWER_VERSION = "RESOLVED_BY_NEWER_VERSION"
    UNRESOLVED = "UNRESOLVED"


def parse_version_tuple(version_str: Union[str, float, int, None]) -> Tuple:
    """
    Parses a version string into a comparable tuple of integers and tokens.
    e.g. '2026.1' -> (2026, 1)
         'v2.0'   -> (2, 0)
         '1.0'    -> (1, 0)
    """
    if not version_str:
        return (1, 0)
    cleaned = re.sub(r"^[vV]", "", str(version_str).strip())
    tokens = []
    for part in re.split(r"[\.\-_]", cleaned):
        if part.isdigit():
            tokens.append(int(part))
        elif part:
            tokens.append(part.lower())
    return tuple(tokens) if tokens else (1, 0)


def compare_versions(v1: str, v2: str) -> int:
    """
    Compares two version strings.
    Returns 1 if v1 > v2, -1 if v1 < v2, and 0 if v1 == v2.
    """
    t1 = parse_version_tuple(v1)
    t2 = parse_version_tuple(v2)
    if t1 > t2:
        return 1
    elif t1 < t2:
        return -1
    return 0


def parse_date(date_val: Union[str, date, datetime, None]) -> Optional[date]:
    """Safely coerces various date representations into a standard date object."""
    if not date_val:
        return None
    if isinstance(date_val, datetime):
        return date_val.date()
    if isinstance(date_val, date):
        return date_val
    if isinstance(date_val, str):
        date_str = date_val.strip()
        if not date_str:
            return None
        try:
            return date.fromisoformat(date_str)
        except ValueError:
            for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%m/%d/%Y"):
                try:
                    return datetime.strptime(date_str, fmt).date()
                except ValueError:
                    continue
    return None


class DetectedConflict(BaseModel):
    """Itemized factual conflict detected between two evidence chunks."""

    topic: str
    claim_a: str
    source_a: str
    version_a: str
    status_a: str
    chunk_id_a: str
    claim_b: str
    source_b: str
    version_b: str
    status_b: str
    chunk_id_b: str
    conflict_type: str  # numeric_discrepancy, polarity_mismatch, schedule_divergence
    description: str


class ContradictionAnalysisResult(BaseModel):
    """
    Structured outcome of the contradiction analysis and conflict resolution engine.
    """

    status: ContradictionStatus
    has_conflict: bool
    claims: List[str] = Field(default_factory=list)
    conflicting_sources: List[str] = Field(default_factory=list)
    conflicts: List[DetectedConflict] = Field(default_factory=list)
    resolution: ConflictResolutionAction = ConflictResolutionAction.NO_CONFLICT
    is_resolved: bool = True
    resolved_chunks: List[RetrievedChunk] = Field(default_factory=list)
    conflicting_chunks: List[RetrievedChunk] = Field(default_factory=list)
    superseded_chunks: List[RetrievedChunk] = Field(default_factory=list)
    rationale: Optional[str] = None

    def to_report(self) -> ContradictionReport:
        """Converts internal analysis result into the public API ContradictionReport schema."""
        return ContradictionReport(
            status=self.status.value,
            has_conflict=self.has_conflict,
            claims=self.claims,
            conflicting_sources=self.conflicting_sources,
            resolution_status=self.resolution.value,
            resolution_rationale=self.rationale,
        )


class ContradictionService:
    """
    Service responsible for detecting contradictory retrieved evidence across knowledge documents
    and applying deterministic governance rules to resolve conflicts via document versioning
    and lifecycle status.
    """

    def __init__(self, reference_date: Optional[date] = None):
        self.reference_date = reference_date or date.today()

    def extract_factual_sentences(self, text: str) -> List[str]:
        """Splits chunk text into clean candidate factual assertions."""
        cleaned = re.sub(r"\s+", " ", text).strip()
        if not cleaned:
            return []
        sentences = [
            s.strip()
            for s in re.split(r"(?<=[.!?\n])\s+", cleaned)
            if len(s.strip()) >= 12
        ]
        return sentences

    def extract_content_tokens(self, sentence: str) -> Set[str]:
        """Extracts non-stopword tokens of length >= 3 from a sentence."""
        words = re.findall(r"\b[a-zA-Z0-9_\-]{3,}\b", sentence.lower())
        return {w for w in words if w not in STOP_WORDS}

    def extract_numbers(self, sentence: str) -> Set[str]:
        """Extracts numeric values and quantities from a sentence."""
        return set(re.findall(r"\b\d+(?:[\.,]\d+)?\b", sentence))

    def detect_pairwise_sentence_conflict(
        self,
        sent_a: str,
        sent_b: str,
    ) -> Optional[Tuple[str, str, str]]:
        """
        Determines whether two sentences discuss the same policy attribute but contain
        a direct factual conflict (numeric discrepancy, polarity mismatch, or schedule divergence).
        Returns (conflict_type, topic, description) or None if consistent.
        """
        tokens_a = self.extract_content_tokens(sent_a)
        tokens_b = self.extract_content_tokens(sent_b)

        if not tokens_a or not tokens_b:
            return None

        # Check semantic/topic overlap
        shared_tokens = tokens_a & tokens_b
        jaccard = len(shared_tokens) / len(tokens_a | tokens_b)

        # Sentences must share core topic concepts
        if len(shared_tokens) < 2 and jaccard < 0.30:
            return None

        topic = " ".join(sorted(shared_tokens)[:4])

        # Test 1: Numeric Discrepancy on the same topic
        # Example: "Employees receive 20 annual vacation days" vs "Employees receive 25 annual vacation days"
        nums_a = self.extract_numbers(sent_a)
        nums_b = self.extract_numbers(sent_b)

        if nums_a and nums_b and nums_a != nums_b:
            # Check if numbers appear in a shared metric context
            return (
                "numeric_discrepancy",
                topic,
                f"Numeric discrepancy on topic '{topic}': values {sorted(nums_a)} vs {sorted(nums_b)}",
            )

        # Test 2: Polarity / Antonym Mismatch
        # Example: "Deployments are permitted on Fridays" vs "Deployments are prohibited on Fridays"
        lower_a = sent_a.lower()
        lower_b = sent_b.lower()

        has_pos_a = any(w in lower_a for w in POSITIVE_INDICATORS)
        has_neg_a = any(w in lower_a for w in NEGATIVE_INDICATORS)
        has_pos_b = any(w in lower_b for w in POSITIVE_INDICATORS)
        has_neg_b = any(w in lower_b for w in NEGATIVE_INDICATORS)

        if (has_pos_a and has_neg_b) or (has_neg_a and has_pos_b):
            return (
                "polarity_mismatch",
                topic,
                f"Polarity opposition on topic '{topic}': permissive vs restrictive assertions",
            )

        # Test 3: Day/Schedule Divergence
        # Example: "Tuesday through Thursday" vs "Monday through Wednesday"
        days_of_week = {
            "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"
        }
        days_a = {w for w in re.findall(r"\b\w+\b", lower_a) if w in days_of_week}
        days_b = {w for w in re.findall(r"\b\w+\b", lower_b) if w in days_of_week}

        if days_a and days_b and days_a != days_b and len(shared_tokens) >= 2:
            return (
                "schedule_divergence",
                topic,
                f"Schedule divergence on topic '{topic}': {sorted(days_a)} vs {sorted(days_b)}",
            )

        return None

    def analyze_chunks(
        self,
        chunks: List[RetrievedChunk],
    ) -> Tuple[ContradictionStatus, List[DetectedConflict], List[str], List[str]]:
        """
        Examines retrieved chunks across different documents to detect factual contradictions.
        """
        if len(chunks) <= 1:
            return ContradictionStatus.AGREE, [], [], []

        conflicts: List[DetectedConflict] = []
        conflicting_sources_set: Set[str] = set()
        all_claims: List[str] = []

        # Compare pairs across different documents or versions
        for i in range(len(chunks)):
            chunk_a = chunks[i]
            meta_a = chunk_a.metadata or {}
            doc_a_name = meta_a.get("filename") or meta_a.get("document_title") or chunk_a.document_id
            doc_a_version = str(meta_a.get("version") or "1.0")
            doc_a_status = str(meta_a.get("status") or "ACTIVE").upper()

            sents_a = self.extract_factual_sentences(chunk_a.content)
            all_claims.extend(sents_a[:2])

            for j in range(i + 1, len(chunks)):
                chunk_b = chunks[j]
                meta_b = chunk_b.metadata or {}
                doc_b_name = meta_b.get("filename") or meta_b.get("document_title") or chunk_b.document_id
                doc_b_version = str(meta_b.get("version") or "1.0")
                doc_b_status = str(meta_b.get("status") or "ACTIVE").upper()

                # Ignore pairs from the exact same document and version
                if chunk_a.document_id == chunk_b.document_id and doc_a_version == doc_b_version:
                    continue

                sents_b = self.extract_factual_sentences(chunk_b.content)

                for sent_a in sents_a:
                    for sent_b in sents_b:
                        conflict_info = self.detect_pairwise_sentence_conflict(sent_a, sent_b)
                        if conflict_info:
                            c_type, topic, desc = conflict_info
                            conflicts.append(
                                DetectedConflict(
                                    topic=topic,
                                    claim_a=sent_a,
                                    source_a=doc_a_name,
                                    version_a=doc_a_version,
                                    status_a=doc_a_status,
                                    chunk_id_a=chunk_a.chunk_id,
                                    claim_b=sent_b,
                                    source_b=doc_b_name,
                                    version_b=doc_b_version,
                                    status_b=doc_b_status,
                                    chunk_id_b=chunk_b.chunk_id,
                                    conflict_type=c_type,
                                    description=desc,
                                )
                            )
                            conflicting_sources_set.add(doc_a_name)
                            conflicting_sources_set.add(doc_b_name)

        if conflicts:
            return (
                ContradictionStatus.CONFLICT,
                conflicts,
                sorted(conflicting_sources_set),
                list(dict.fromkeys(all_claims))[:8],
            )

        return ContradictionStatus.AGREE, [], [], list(dict.fromkeys(all_claims))[:8]

    def resolve_conflicts(
        self,
        conflicts: List[DetectedConflict],
        chunks: List[RetrievedChunk],
        ref_date: Optional[date] = None,
    ) -> Tuple[ConflictResolutionAction, bool, List[RetrievedChunk], List[RetrievedChunk], List[RetrievedChunk], Optional[str]]:
        """
        Applies deterministic resolution rules to resolve conflicts:
        1. Prefer ACTIVE documents over SUPERSEDED or EXPIRED documents.
        2. Prefer currently effective documents over expired documents.
        3. Prefer strictly newer versions when they supersede older versions.
        4. If conflicts cannot be safely resolved, mark UNRESOLVED.
        """
        current_date = ref_date or self.reference_date

        if not conflicts:
            return (
                ConflictResolutionAction.NO_CONFLICT,
                True,
                chunks,
                [],
                [],
                "No factual contradictions detected among retrieved evidence.",
            )

        # Index chunks by chunk_id
        chunk_by_id = {c.chunk_id: c for c in chunks}
        discard_chunk_ids: Set[str] = set()
        resolution_notes: List[str] = []
        rule_actions: List[ConflictResolutionAction] = []

        for conflict in conflicts:
            chunk_a = chunk_by_id.get(conflict.chunk_id_a)
            chunk_b = chunk_by_id.get(conflict.chunk_id_b)

            if not chunk_a or not chunk_b:
                continue

            meta_a = chunk_a.metadata or {}
            meta_b = chunk_b.metadata or {}

            status_a = str(meta_a.get("status") or conflict.status_a).upper()
            status_b = str(meta_b.get("status") or conflict.status_b).upper()

            eff_a = parse_date(meta_a.get("effective_date"))
            exp_a = parse_date(meta_a.get("expiration_date"))
            eff_b = parse_date(meta_b.get("effective_date"))
            exp_b = parse_date(meta_b.get("expiration_date"))

            ver_a = str(meta_a.get("version") or conflict.version_a)
            ver_b = str(meta_b.get("version") or conflict.version_b)

            doc_a_name = conflict.source_a
            doc_b_name = conflict.source_b

            # -------------------------------------------------------------
            # Deterministic Rule 1: Status Precedence (ACTIVE vs SUPERSEDED/EXPIRED)
            # -------------------------------------------------------------
            if status_a == DocumentStatus.ACTIVE.value and status_b in (
                DocumentStatus.SUPERSEDED.value,
                DocumentStatus.EXPIRED.value,
            ):
                discard_chunk_ids.add(chunk_b.chunk_id)
                note = f"Resolved via ACTIVE status: '{doc_a_name}' is ACTIVE; '{doc_b_name}' is {status_b}."
                resolution_notes.append(note)
                rule_actions.append(ConflictResolutionAction.RESOLVED_BY_ACTIVE_STATUS)
                continue

            if status_b == DocumentStatus.ACTIVE.value and status_a in (
                DocumentStatus.SUPERSEDED.value,
                DocumentStatus.EXPIRED.value,
            ):
                discard_chunk_ids.add(chunk_a.chunk_id)
                note = f"Resolved via ACTIVE status: '{doc_b_name}' is ACTIVE; '{doc_a_name}' is {status_a}."
                resolution_notes.append(note)
                rule_actions.append(ConflictResolutionAction.RESOLVED_BY_ACTIVE_STATUS)
                continue

            # -------------------------------------------------------------
            # Deterministic Rule 2: Temporal Precedence (Effective vs Expired)
            # -------------------------------------------------------------
            is_a_expired = exp_a is not None and exp_a < current_date
            is_b_expired = exp_b is not None and exp_b < current_date

            is_a_effective = (eff_a is None or eff_a <= current_date) and not is_a_expired
            is_b_effective = (eff_b is None or eff_b <= current_date) and not is_b_expired

            if is_a_effective and is_b_expired:
                discard_chunk_ids.add(chunk_b.chunk_id)
                note = f"Resolved via temporal validity: '{doc_a_name}' is currently effective; '{doc_b_name}' expired on {exp_b}."
                resolution_notes.append(note)
                rule_actions.append(ConflictResolutionAction.RESOLVED_BY_EFFECTIVE_DATE)
                continue

            if is_b_effective and is_a_expired:
                discard_chunk_ids.add(chunk_a.chunk_id)
                note = f"Resolved via temporal validity: '{doc_b_name}' is currently effective; '{doc_a_name}' expired on {exp_a}."
                resolution_notes.append(note)
                rule_actions.append(ConflictResolutionAction.RESOLVED_BY_EFFECTIVE_DATE)
                continue

            # -------------------------------------------------------------
            # Deterministic Rule 3: Version Precedence (Newer version supersedes older)
            # -------------------------------------------------------------
            v_cmp = compare_versions(ver_a, ver_b)
            if v_cmp > 0 and (status_b == DocumentStatus.SUPERSEDED.value or status_a == DocumentStatus.ACTIVE.value):
                discard_chunk_ids.add(chunk_b.chunk_id)
                note = f"Resolved via version precedence: '{doc_a_name}' (v{ver_a}) supersedes '{doc_b_name}' (v{ver_b})."
                resolution_notes.append(note)
                rule_actions.append(ConflictResolutionAction.RESOLVED_BY_NEWER_VERSION)
                continue

            if v_cmp < 0 and (status_a == DocumentStatus.SUPERSEDED.value or status_b == DocumentStatus.ACTIVE.value):
                discard_chunk_ids.add(chunk_a.chunk_id)
                note = f"Resolved via version precedence: '{doc_b_name}' (v{ver_b}) supersedes '{doc_a_name}' (v{ver_a})."
                resolution_notes.append(note)
                rule_actions.append(ConflictResolutionAction.RESOLVED_BY_NEWER_VERSION)
                continue

            # -------------------------------------------------------------
            # Rule 4: Unresolvable Direct Conflict
            # -------------------------------------------------------------
            unresolved_note = (
                f"Unresolvable direct conflict between '{doc_a_name}' (v{ver_a}, {status_a}) "
                f"and '{doc_b_name}' (v{ver_b}, {status_b}) on topic '{conflict.topic}'."
            )
            logger.warning(unresolved_note)
            conflicting_chunks = [c for c in chunks if c.chunk_id in (conflict.chunk_id_a, conflict.chunk_id_b)]
            return (
                ConflictResolutionAction.UNRESOLVED,
                False,
                [],
                conflicting_chunks,
                [],
                unresolved_note,
            )

        # All conflicts were successfully resolved by rules
        resolved_chunks = [c for c in chunks if c.chunk_id not in discard_chunk_ids]
        superseded_chunks = [c for c in chunks if c.chunk_id in discard_chunk_ids]
        primary_action = rule_actions[0] if rule_actions else ConflictResolutionAction.RESOLVED_BY_ACTIVE_STATUS
        summary_rationale = "; ".join(dict.fromkeys(resolution_notes))

        logger.info(f"Contradiction resolved successfully ({primary_action.value}): {summary_rationale}")

        return (
            primary_action,
            True,
            resolved_chunks,
            [],
            superseded_chunks,
            summary_rationale,
        )

    def analyze_and_resolve(
        self,
        retrieved_chunks: List[RetrievedChunk],
        ref_date: Optional[date] = None,
    ) -> ContradictionAnalysisResult:
        """
        Coordinates the complete contradiction analysis and conflict resolution workflow.
        """
        status, conflicts, sources, claims = self.analyze_chunks(retrieved_chunks)

        if status == ContradictionStatus.AGREE:
            return ContradictionAnalysisResult(
                status=ContradictionStatus.AGREE,
                has_conflict=False,
                claims=claims,
                conflicting_sources=[],
                conflicts=[],
                resolution=ConflictResolutionAction.NO_CONFLICT,
                is_resolved=True,
                resolved_chunks=retrieved_chunks,
                conflicting_chunks=[],
                superseded_chunks=[],
                rationale="Retrieved knowledge sources agree without factual conflict.",
            )

        action, is_resolved, resolved, conflicting_chunks, superseded, rationale = self.resolve_conflicts(
            conflicts=conflicts,
            chunks=retrieved_chunks,
            ref_date=ref_date,
        )

        return ContradictionAnalysisResult(
            status=ContradictionStatus.CONFLICT,
            has_conflict=True,
            claims=claims,
            conflicting_sources=sources,
            conflicts=conflicts,
            resolution=action,
            is_resolved=is_resolved,
            resolved_chunks=resolved,
            conflicting_chunks=conflicting_chunks or retrieved_chunks,
            superseded_chunks=superseded,
            rationale=rationale,
        )
