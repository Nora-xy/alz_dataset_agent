"""Heuristic (regex-only) intent parser — used for tests/CI when no LLM client is available.

All patterns are deterministic: no LLM calls, no catalog lookups.
The output is a StructuredIntent with parsing_mode='heuristic'.
"""

from __future__ import annotations

import re

from alz_agent.agent.intent_schema import QuestionType, StructuredIntent
from alz_agent.schemas import IdSpace

_POPULATION_PATTERNS: list[tuple[str, str]] = [
    (r"\bmci\b", "mci"),
    (r"\bnormal\b|\bcognitive(?:ly)?\s*normal\b|\bNC\b|\bNL\b", "normal"),
    (r"\bdementia\b", "dementia"),
    (r"\bimpaired.not.mci\b|\bimpaired[\s_-]not[\s_-]mci\b", "impaired-not-mci"),
]

_HORIZON_RE: re.Pattern = re.compile(r"(\d+)\s*[-–]?\s*year")
_MONTH_RE: re.Pattern = re.compile(r"(\d+)\s*[-–]?\s*month")

_MODALITY_PATTERNS: list[tuple[str, str]] = [
    (r"\bcogniti", "cognition"),
    (r"\bmri\b|\bimaging\b|\bvolumet", "mri"),
    (r"\bpet\b|\bamyloid\b|\btau\b", "pet"),
    (r"\bbiomarker\b|\bcsf\b|\bplasma\b|\bfluid\b", "fluid_biomarker"),
    (r"\bvascular\b", "vascular"),
    (r"\bneuropath\b", "neuropath"),
    (r"\bdemograph\b|\bage\b|\bsex\b|\beducation\b", "demographic"),
]


def _parse_heuristic(nl: str) -> StructuredIntent:
    low = nl.lower()
    mentions_adni = bool(re.search(r"\badni\b", low))
    mentions_nacc = bool(re.search(r"\bnacc\b", low))
    if mentions_adni and mentions_nacc:
        raise ValueError("Heuristic mode cannot combine NACC and ADNI in one question")
    id_space = IdSpace.B if mentions_adni else IdSpace.A

    population: list[str] = []
    for pat, label in _POPULATION_PATTERNS:
        if re.search(pat, low, re.IGNORECASE):
            if label not in population:
                population.append(label)
    if not population:
        population = ["mci"]

    target_etiology: list[str] | None = None
    if re.search(r"\b(?:ad|alzheimer)\b", low):
        target_etiology = ["ad"]
        if population == ["mci"] and not any(
            re.search(p, low, re.IGNORECASE) for p, _ in _POPULATION_PATTERNS
        ):
            population = ["dementia"]

    modalities: list[str] = []
    for pat, label in _MODALITY_PATTERNS:
        if re.search(pat, low, re.IGNORECASE) and label not in modalities:
            modalities.append(label)

    _STAT_RE = re.compile(
        r"\baverage\b|\bmean\b|\bmedian\b|\bstd\b|\bstandard deviation\b"
        r"|\bminimum\b|\bmaximum\b",
    )
    asks_stat = bool(_STAT_RE.search(low))

    stat_variable: str | None = None
    stat_type: str = "mean"
    event_anchor: str | None = None

    if asks_stat:
        if re.search(r"\bage\b|\byears old\b", low):
            stat_variable = "age"
        elif re.search(r"\beduc\b|\byears of education\b", low):
            stat_variable = "education"
        elif re.search(r"\bmmse\b", low):
            stat_variable = "mmse"
        elif re.search(r"\bcdr\b", low):
            stat_variable = "cdr"
        if re.search(r"\bmedian\b", low):
            stat_type = "median"
        elif re.search(r"\bstd\b|\bstandard deviation\b", low):
            stat_type = "std"
        elif re.search(r"\bmin\b|\bminimum\b", low):
            stat_type = "min"
        elif re.search(r"\bmax\b|\bmaximum\b", low):
            stat_type = "max"
        if re.search(r"\bfirst diagno|\bwhen.*diagno|\bdiagno.*first|\bat.*diagno", low):
            event_anchor = "first_diagnosis"

    _STAT_ANALYSIS_RE = re.compile(
        r"\bcompare\b|\bdiffer\b|\bdifference\b|\blower\b|\bhigher\b|\bassociat\b"
        r"|\bp.?value\b|\bt.test\b|\bsignificant\b|\bcorrelat\b|\bregress\b"
        r"|\bdo\s+\w+\s+have\b|\bdoes\b.*\bdiffer\b|\bbetween\s+group\b|\bgroup\s+comparison\b"
        r"|\beffect\b.*\bon\b|\bimpact\b|\bpredict\w*\b.*\bvariable\b"
    )
    asks_stat_analysis = bool(_STAT_ANALYSIS_RE.search(low))

    analysis_type: str | None = None
    group_variable: str | None = None
    outcome_variable: str | None = None

    if asks_stat_analysis:
        if re.search(r"\bcorrelat\b", low):
            analysis_type = "correlation"
        elif re.search(r"\bregress\b", low):
            analysis_type = "regression"
        else:
            analysis_type = "group_comparison"

        _GROUP_PATTERNS = [
            (r"\bheart\s*attack\b|\bcvhatt\b|\bmi\s+history\b", "heart attack history"),
            (r"\bafib\b|\batrial\s*fibrill\b", "atrial fibrillation"),
            (r"\bstroke\b|\bcvstroke\b", "stroke history"),
            (r"\bsex\b|\bgender\b", "sex"),
            (r"\bapoe\b", "APOE4 status"),
            (r"\bsmoking\b|\bsmoke\b", "smoking history"),
            (r"\bdiabet\b", "diabetes"),
            (r"\bhypertens\b|\bblood\s*pressure\b", "hypertension"),
            (r"\bt1\b.*\bmri\b|\bmri\b.*\bdata\b|\bany\s+(?:t1|mri)\b|\bimaging\s+data\b", "T1 MRI data"),
        ]
        for pat, label in _GROUP_PATTERNS:
            if re.search(pat, low):
                group_variable = label
                break

        _OUTCOME_PATTERNS = [
            (r"\bmmse\b|\bmini.mental\b|\bnaccmmse\b", "MMSE"),
            (r"\bcdr\b|\bcdrsum\b", "CDR sum of boxes"),
            (r"\bage\b", "age"),
            (r"\beducation\b|\byears\s+of\s+school\b", "education"),
            (r"\bmoca\b", "MoCA"),
        ]
        for pat, label in _OUTCOME_PATTERNS:
            if re.search(pat, low):
                outcome_variable = label
                break

    asks_predict = bool(re.search(r"\bpredict\b|\bfeasib\b|\bpossible\b|\bcan we\b", low))
    asks_size = bool(re.search(r"\bhow many\b|\bsubject\b|\bcount\b|\bsize\b|\bsurvive\b|\bn=\b", low))
    asks_lookup = bool(re.search(r"\bwhat is\b|\bdefin\b|\blookup\b|\bvariable\b|\bdescri", low))

    if asks_stat_analysis and (group_variable or outcome_variable):
        q_type = QuestionType.STATISTICAL_ANALYSIS
    elif asks_stat and stat_variable:
        q_type = QuestionType.AGGREGATE_STAT
    elif asks_predict and asks_size:
        q_type = QuestionType.MODELING_DESIGN
    elif asks_predict:
        q_type = QuestionType.FEASIBILITY
    elif asks_size:
        q_type = QuestionType.COHORT_SIZE
    elif asks_lookup:
        q_type = QuestionType.LOOKUP
    else:
        q_type = QuestionType.MODELING_DESIGN

    _LONGITUDINAL_QTYPES = {QuestionType.FEASIBILITY, QuestionType.MODELING_DESIGN}
    m_yr = _HORIZON_RE.search(low)
    m_mo = _MONTH_RE.search(low) if not m_yr else None
    if m_yr:
        horizon: int | None = int(m_yr.group(1))
    elif m_mo:
        horizon = max(1, round(int(m_mo.group(1)) / 12))
    elif q_type in _LONGITUDINAL_QTYPES:
        horizon = 3
    else:
        horizon = None

    if re.search(r"\bconver", low):
        candidate_label = "conversion to AD" if re.search(r"\bad\b|\balzheimer", low) else "conversion to dementia"
    else:
        candidate_label = "conversion to dementia"

    pre_index = bool(re.search(r"\bpre.index\b|\bbaseline\b|\bbefore\b|\bpre-index\b", low))
    if not pre_index:
        pre_index = True

    return StructuredIntent(
        target_population=population,
        target_etiology=target_etiology,
        candidate_label=candidate_label,
        horizon_years=horizon,
        modalities=modalities,
        question_type=q_type,
        pre_index_only=pre_index,
        id_space=id_space,
        stat_variable=stat_variable,
        stat_type=stat_type,
        event_anchor=event_anchor,
        analysis_type=analysis_type,
        group_variable=group_variable,
        outcome_variable=outcome_variable,
        parsing_mode="heuristic",
    )
