"""Scenario replay simulator for startup-suggestion bundles.

Loads a suggestion bundle produced by ``moldvision predictive bundle`` and
replays synthetic startup-assistant states against its Tier 1 local-search
logic. The simulator is intentionally CLI-oriented and UI-free.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import tempfile
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


@dataclass(frozen=True)
class SimulationMachineParameter:
    name: str
    unit: str
    current_value: float
    suggested_value: float
    confidence: float
    parameter_id: str = ""
    family_id: str = ""
    recipe_label: str = ""


@dataclass(frozen=True)
class SimulationSuggestion:
    triggered_by_metric: str
    triggered_value: float
    threshold_value: float
    parameters: tuple[SimulationMachineParameter, ...]
    summary: str
    urgency: str
    input_state: dict[str, float] = field(default_factory=dict)
    model_id: str = ""
    logic_version: str = ""


@dataclass(frozen=True)
class SuggestionInferenceResult:
    quality_score: float
    defect_risks: Dict[str, float]


@dataclass(frozen=True)
class StartupParameterDefinition:
    parameter_id: str
    display_name: str
    unit: str
    baseline: float
    range_min: float
    range_max: float
    control_feature_keys: tuple[str, ...]
    family_id: str | None = None
    semantic_parameter_id: str | None = None
    page_id: str | None = None
    subpage_id: str | None = None
    slot_id: str | None = None
    canonical_slot_id: str | None = None
    step_mode: str = "absolute"
    preferred_step: float = 1.0
    max_delta: float = 5.0
    observed_support_min: float | None = None
    observed_support_max: float | None = None
    support_margin_ratio: float = 0.05
    decimal_places: int | None = None

    @property
    def range_span(self) -> float:
        return self.range_max - self.range_min


@dataclass(frozen=True)
class StartupControlFamilyDefinition:
    family_id: str
    display_name: str
    family_type: str
    parameter_ids: tuple[str, ...]
    semantic_parameter_id: str | None = None
    page_id: str | None = None
    subpage_id: str | None = None
    family_constraints: dict[str, object] = field(default_factory=dict)
    observed_activation_states: tuple[str, ...] = ()
    observed_observability_states: tuple[str, ...] = ()


@dataclass(frozen=True)
class RuntimeControlFamilyState:
    family_id: str
    family_type: str
    parameter_ids: tuple[str, ...]
    semantic_parameter_id: str | None
    current_values: dict[str, float]
    activation_mask: tuple[int, ...]
    activation_state: str
    observability_state: str
    confidence: float
    source: str
    family_constraints: dict[str, object]


@dataclass(frozen=True)
class RuntimeCandidateProposal:
    parameter_values: dict[str, float]
    family_ids: tuple[str, ...]
    recipe_label: str
    confidence_scale: float = 1.0


@dataclass(frozen=True)
class _SearchCandidate:
    parameter_values: dict[str, float]
    score: float
    recipe_label: str = ""
    confidence_scale: float = 1.0


@dataclass(frozen=True)
class SimulationStepInput:
    step_id: str
    metric_id: str
    metric_value: float
    threshold: float
    current_parameter_values: dict[str, float]
    defect_severities: dict[str, float]
    defect_metrics: dict[str, dict[str, float]]
    expected: dict[str, Any]
    parameter_state_mode: str = "carry"


@dataclass(frozen=True)
class SimulationStepResult:
    step_id: str
    metric_id: str
    metric_value: float
    threshold: float
    triggered: bool
    focus_label: str | None
    baseline_quality_score: float
    predicted_label_signals: dict[str, float]
    suggestion: dict[str, Any] | None
    expectations: dict[str, Any]


class BundleSimulationRuntime:
    def __init__(self, bundle_path: Path) -> None:
        self._bundle_path = Path(bundle_path)
        self._tmp_dir: Optional[Path] = None
        self._manifest: dict[str, Any] = {}
        self._training_meta: dict[str, Any] = {}
        self._sessions: dict[str, Any] = {}
        self._loaded = False
        self._parameter_definitions: tuple[StartupParameterDefinition, ...] = ()
        self._control_family_definitions: tuple[StartupControlFamilyDefinition, ...] = ()
        self._feature_to_parameter_id: dict[str, str] = {}
        self._definition_by_id: dict[str, StartupParameterDefinition] = {}
        self._family_by_parameter_id: dict[str, StartupControlFamilyDefinition] = {}
        self._current_values: dict[str, float] = {}
        self._defect_severities: dict[str, float] = {}
        self._defect_metrics: dict[str, dict[str, float]] = {}
        self._used_feature_keys: tuple[str, ...] = ()
        self._trained_feature_key_set: set[str] = set()
        self._defect_signal_kinds: dict[str, str] = {}

    def load(self) -> None:
        bundle_dir = self._resolve_bundle_dir()
        manifest = self._load_manifest(bundle_dir)
        training_meta = self._load_training_meta(bundle_dir)
        self._validate_bundle_type(manifest)
        self._verify_checksums(bundle_dir, manifest)
        self._sessions = self._load_sessions(bundle_dir, manifest)
        self._manifest = manifest
        self._training_meta = training_meta
        self._parameter_definitions = derive_parameter_definitions(
            self.feature_keys,
            self.imputation_values,
            self.parameter_schema,
        )
        self._control_family_definitions = derive_control_family_definitions(
            self._parameter_definitions,
            self.deployable_control_families or self.control_families,
        )
        self._feature_to_parameter_id = control_feature_mapping(self._parameter_definitions)
        self._definition_by_id = {
            definition.parameter_id: definition for definition in self._parameter_definitions
        }
        self._family_by_parameter_id = self._build_family_lookup()
        self._used_feature_keys = tuple(self.used_feature_keys)
        self._trained_feature_key_set = set(self.trained_feature_keys)
        self._defect_signal_kinds = {
            str(label): str(kind)
            for label, kind in self.defect_signal_kinds.items()
        }
        self._loaded = True

    def unload(self) -> None:
        self._sessions.clear()
        self._training_meta = {}
        self._loaded = False
        if self._tmp_dir is not None and self._tmp_dir.exists():
            shutil.rmtree(self._tmp_dir, ignore_errors=True)
            self._tmp_dir = None

    @property
    def manifest(self) -> dict[str, Any]:
        self._assert_loaded()
        return dict(self._manifest)

    @property
    def feature_keys(self) -> list[str]:
        return list(self._manifest.get("feature_keys", []))

    @property
    def imputation_values(self) -> dict[str, float]:
        return dict(self._manifest.get("imputation_values", {}))

    @property
    def null_strategy(self) -> str:
        return str(self._manifest.get("null_strategy", "mean_impute"))

    @property
    def target_model_specs(self) -> dict[str, dict]:
        return self._normalize_target_model_specs(
            self._manifest.get("target_models", {}),
            self._training_meta.get("cv_metrics", {}),
        )

    @property
    def defect_signal_kinds(self) -> dict[str, str]:
        kinds: dict[str, str] = {}
        for target_name, spec in self.target_model_specs.items():
            if not str(target_name).startswith("defect_"):
                continue
            kinds[str(target_name).removeprefix("defect_")] = str(
                spec.get("signal_kind") or "binary_defect_presence"
            )
        return kinds

    @property
    def used_feature_keys(self) -> list[str]:
        cv_metrics = self._training_meta.get("cv_metrics")
        if not isinstance(cv_metrics, dict):
            return self.feature_keys
        feature_key_set = set(self.feature_keys)
        used: list[str] = []
        seen: set[str] = set()
        for metric in cv_metrics.values():
            if not isinstance(metric, dict):
                continue
            for key in metric.get("used_feature_keys", ()):
                token = str(key).strip()
                if not token or token not in feature_key_set or token in seen:
                    continue
                used.append(token)
                seen.add(token)
        return used or self.feature_keys

    @property
    def trained_feature_keys(self) -> list[str]:
        raw = self._manifest.get("trained_feature_keys")
        if isinstance(raw, list):
            return [str(key) for key in raw if str(key).strip()]
        raw = self._training_meta.get("trained_feature_keys")
        if isinstance(raw, list):
            return [str(key) for key in raw if str(key).strip()]
        return self.feature_keys

    @property
    def parameter_schema(self) -> list[dict]:
        raw = self._manifest.get("parameter_schema")
        return list(raw) if isinstance(raw, list) else []

    @property
    def control_families(self) -> list[dict]:
        raw = self._manifest.get("control_families")
        return list(raw) if isinstance(raw, list) else []

    @property
    def deployable_control_families(self) -> list[dict]:
        raw = self._manifest.get("deployable_control_families")
        return list(raw) if isinstance(raw, list) else []

    @property
    def quality_weights(self) -> dict[str, float]:
        return dict(self._manifest.get("quality_weights", {}))

    def simulate_step(self, step: SimulationStepInput) -> SimulationStepResult:
        self._assert_loaded()
        self._current_values = map_ui_to_internal_values(
            step.current_parameter_values,
            self._parameter_definitions,
        )
        self._defect_severities = {
            _canonical_defect_label(label): float(value)
            for label, value in step.defect_severities.items()
        }
        self._defect_metrics = {
            _canonical_defect_label(label): {
                str(metric): float(value)
                for metric, value in metrics.items()
            }
            for label, metrics in step.defect_metrics.items()
        }

        triggered = step.metric_value > step.threshold
        baseline_result = self.run_inference(self._build_feature_vector())
        focus_label = self._target_defect_label(step.metric_id, baseline_result)
        suggestion: SimulationSuggestion | None = None
        if triggered:
            suggestion = self.get_suggestion(step.metric_id, step.metric_value, step.threshold)

        expectations = evaluate_expectations(step.expected, triggered, focus_label, suggestion)
        suggestion_payload = None
        if suggestion is not None:
            suggestion_payload = {
                "summary": suggestion.summary,
                "urgency": suggestion.urgency,
                "parameters": [asdict(parameter) for parameter in suggestion.parameters],
            }

        return SimulationStepResult(
            step_id=step.step_id,
            metric_id=step.metric_id,
            metric_value=step.metric_value,
            threshold=step.threshold,
            triggered=triggered and suggestion is not None,
            focus_label=focus_label,
            baseline_quality_score=baseline_result.quality_score,
            predicted_label_signals=dict(sorted(baseline_result.defect_risks.items())),
            suggestion=suggestion_payload,
            expectations=expectations,
        )

    def get_suggestion(
        self,
        metric_id: str,
        metric_value: float,
        threshold: float,
    ) -> Optional[SimulationSuggestion]:
        if metric_value <= threshold:
            return None
        excess_ratio = (metric_value - threshold) / max(threshold, 1e-9)
        urgency = _compute_urgency(excess_ratio)
        base_vector = self._build_feature_vector()
        baseline_result = self.run_inference(base_vector)
        focus_label = self._target_defect_label(metric_id, baseline_result)
        candidates = self._local_search(base_vector, baseline_result, focus_label)
        if not candidates:
            return None
        parameters = self._build_machine_parameters(candidates, excess_ratio)
        if not parameters:
            return None
        return SimulationSuggestion(
            triggered_by_metric=metric_id,
            triggered_value=metric_value,
            threshold_value=threshold,
            parameters=tuple(parameters),
            summary=self._build_summary(baseline_result),
            urgency=urgency,
            input_state=dict(zip(self.feature_keys, base_vector)),
            model_id=str(self._manifest.get("bundle_id", "unknown")),
            logic_version=str(self._manifest.get("model_version", "unknown")),
        )

    def run_inference(self, feature_vector: List[float]) -> SuggestionInferenceResult:
        import numpy as np

        x = np.array([feature_vector], dtype=np.float32)
        quality_score = 0.5
        defect_risks: Dict[str, float] = {}
        for target_name, session in self._sessions.items():
            input_name = session.get_inputs()[0].name
            outputs = session.run(None, {input_name: x})
            target_spec = self.target_model_specs.get(target_name, {})
            model_type = str(target_spec.get("model_type") or "classification")
            if target_name == "quality_score":
                quality_score = float(np.clip(self._extract_regression_value(outputs), 0.0, 1.0))
            else:
                defect_label = _canonical_defect_label(target_name.removeprefix("defect_"))
                if model_type == "classification":
                    value = self._extract_classification_probability(outputs)
                else:
                    value = self._extract_regression_value(outputs)
                defect_risks[defect_label] = float(np.clip(value, 0.0, 1.0))
        return SuggestionInferenceResult(quality_score=quality_score, defect_risks=defect_risks)

    def _build_feature_vector(self) -> List[float]:
        vector: List[float] = []
        default_values = {
            definition.parameter_id: definition.baseline
            for definition in self._parameter_definitions
        }
        for key in self.feature_keys:
            param_id = self._feature_to_parameter_id.get(key)
            if param_id is not None and param_id in self._current_values:
                vector.append(float(self._current_values[param_id]))
            elif param_id is not None:
                if self.null_strategy == "native_missing":
                    vector.append(float("nan"))
                else:
                    vector.append(float(default_values.get(param_id, self.imputation_values.get(key, 0.0))))
            else:
                runtime_value = self._runtime_feature_value(key)
                if runtime_value is not None:
                    vector.append(runtime_value)
                elif self.null_strategy == "native_missing":
                    vector.append(float("nan"))
                else:
                    vector.append(float(self.imputation_values.get(key, 0.0)))
        return vector

    def _runtime_feature_value(self, feature_key: str) -> float | None:
        if feature_key.startswith("defect_state."):
            return self._defect_state_feature_value(feature_key)
        stem, dot, stat = feature_key.partition(".")
        if not dot:
            return None
        param_id = self._parameter_id_for_feature_stem(stem)
        if param_id is None:
            return None
        is_present = param_id in self._current_values
        if stat == "present":
            return 1.0 if is_present else 0.0
        if stat == "effective_coverage_ratio":
            return 1.0 if is_present else (float("nan") if self.null_strategy == "native_missing" else 0.0)
        return None

    def _defect_state_feature_value(self, feature_key: str) -> float | None:
        parts = str(feature_key).split(".")
        if len(parts) != 3:
            return None
        _, label, metric = parts
        canonical_label = _canonical_defect_label(label)
        metrics = self._defect_metrics.get(canonical_label)
        if metrics is None:
            if metric == "component_severity":
                return float(self._defect_severities.get(canonical_label, 0.0))
            return float("nan") if self.null_strategy == "native_missing" else 0.0
        if metric == "component_severity":
            return float(metrics.get(metric, self._defect_severities.get(canonical_label, 0.0)))
        return float(metrics.get(metric, float("nan") if self.null_strategy == "native_missing" else 0.0))

    def _target_defect_label(
        self,
        metric_id: str,
        baseline: SuggestionInferenceResult,
    ) -> str | None:
        explicit = self._metric_to_label(metric_id)
        if explicit:
            return explicit
        if self._defect_severities:
            return max(self._defect_severities.items(), key=lambda item: item[1])[0]
        if baseline.defect_risks:
            return max(baseline.defect_risks.items(), key=lambda item: item[1])[0]
        return None

    def _metric_to_label(self, metric_id: str) -> str | None:
        token = _canonical_defect_label(metric_id)
        if token in self._defect_signal_kinds:
            return token
        for suffix in (
            "component_severity",
            "duration_ratio",
            "defect_burden_per_frame",
            "detection_frame_ratio",
            "signal",
        ):
            marker = f"_{suffix}"
            if token.endswith(marker):
                candidate = token[: -len(marker)]
                if candidate in self._defect_signal_kinds:
                    return candidate
        return None

    def _local_search(
        self,
        base_vector: List[float],
        baseline_result: SuggestionInferenceResult,
        focus_label: str | None,
    ) -> List[_SearchCandidate]:
        results: List[_SearchCandidate] = []
        definitions = self._ordered_searchable_definitions()
        baseline_signal = self._optimization_signal(baseline_result, focus_label)
        family_states = build_runtime_family_states(
            self._control_family_definitions,
            self._definition_by_id,
            self._current_values or current_parameter_defaults(self._parameter_definitions),
        )
        coupled_groups = build_coupled_family_groups(family_states)
        coupled_group_by_family_id = {
            family_id: group for group in coupled_groups for family_id in group
        }
        consumed_parameter_ids: set[str] = set()
        consumed_group_ids: set[tuple[str, ...]] = set()
        for definition in definitions:
            if definition.parameter_id in consumed_parameter_ids:
                continue
            family = self._family_by_parameter_id.get(definition.parameter_id)
            if family is not None:
                coupled_group = coupled_group_by_family_id.get(family.family_id)
                if coupled_group is not None and coupled_group not in consumed_group_ids:
                    candidate = self._search_coupled_family_group(
                        coupled_group,
                        family_states,
                        base_vector,
                        baseline_result,
                        baseline_signal,
                        focus_label,
                    )
                    if candidate is not None:
                        results.append(candidate)
                        consumed_group_ids.add(coupled_group)
                        for family_id in coupled_group:
                            coupled_family = next(
                                (
                                    item
                                    for item in self._control_family_definitions
                                    if item.family_id == family_id
                                ),
                                None,
                            )
                            if coupled_family is not None:
                                consumed_parameter_ids.update(coupled_family.parameter_ids)
                        continue
            if family is not None and family.family_type in {"atomic", "partially_controllable"} and len(family.parameter_ids) > 1:
                family_state = family_states.get(family.family_id)
                if family_state is None:
                    continue
                if self._current_values and not all(
                    parameter_id in self._current_values for parameter_id in family.parameter_ids
                ):
                    continue
                candidate = self._search_atomic_family(
                    family_state,
                    base_vector,
                    baseline_result,
                    baseline_signal,
                    focus_label,
                )
                consumed_parameter_ids.update(family.parameter_ids)
                if candidate is not None:
                    results.append(candidate)
                continue
            candidate = self._search_single_definition(
                definition,
                base_vector,
                baseline_result,
                baseline_signal,
                focus_label,
            )
            consumed_parameter_ids.add(definition.parameter_id)
            if candidate is not None:
                results.append(candidate)
        results.sort(key=lambda candidate: candidate.score, reverse=True)
        selected: list[_SearchCandidate] = []
        used_parameter_ids: set[str] = set()
        for candidate in results:
            candidate_parameter_ids = set(candidate.parameter_values)
            if candidate_parameter_ids & used_parameter_ids:
                continue
            selected.append(candidate)
            used_parameter_ids.update(candidate_parameter_ids)
            if len(selected) >= 5:
                break
        return selected

    def _search_single_definition(
        self,
        definition: StartupParameterDefinition,
        base_vector: List[float],
        baseline_result: SuggestionInferenceResult,
        baseline_signal: float,
        focus_label: str | None,
    ) -> _SearchCandidate | None:
        current_val = self._current_values.get(definition.parameter_id, definition.baseline)
        best_score = 0.0
        best_suggested = current_val
        for perturbed in search_candidate_values(definition, current_val):
            perturbed = self._clamp_to_empirical_support(definition, perturbed)
            perturbed_vector = self._apply_param_to_vector(base_vector, definition.parameter_id, perturbed)
            result = self.run_inference(perturbed_vector)
            delta_q = result.quality_score - baseline_result.quality_score
            signal_gain = baseline_signal - self._optimization_signal(result, focus_label)
            candidate_score = delta_q + signal_gain
            if candidate_score > best_score:
                best_score = candidate_score
                best_suggested = perturbed
        if best_score <= 0.0:
            return None
        recipe_label = "increase setting" if best_suggested > current_val else "decrease setting"
        return _SearchCandidate(
            parameter_values={definition.parameter_id: best_suggested},
            score=best_score,
            recipe_label=recipe_label,
        )

    def _search_atomic_family(
        self,
        family_state: RuntimeControlFamilyState,
        base_vector: List[float],
        baseline_result: SuggestionInferenceResult,
        baseline_signal: float,
        focus_label: str | None,
    ) -> _SearchCandidate | None:
        best_score = 0.0
        best_candidate: _SearchCandidate | None = None
        proposals = enumerate_family_recipe_candidates(family_state, self._definition_by_id)
        for proposal in proposals:
            parameter_values = self._clamp_candidate_parameter_values(dict(proposal.parameter_values))
            if not parameter_values:
                continue
            result = self.run_inference(self._apply_params_to_vector(base_vector, parameter_values))
            delta_q = result.quality_score - baseline_result.quality_score
            signal_gain = baseline_signal - self._optimization_signal(result, focus_label)
            candidate_score = delta_q + signal_gain
            if candidate_score > best_score:
                best_score = candidate_score
                best_candidate = _SearchCandidate(
                    parameter_values=parameter_values,
                    score=float(candidate_score),
                    recipe_label=proposal.recipe_label,
                    confidence_scale=proposal.confidence_scale,
                )
        return best_candidate if best_candidate is not None and best_score > 0.0 else None

    def _search_coupled_family_group(
        self,
        coupled_group: tuple[str, ...],
        family_states: Mapping[str, RuntimeControlFamilyState],
        base_vector: List[float],
        baseline_result: SuggestionInferenceResult,
        baseline_signal: float,
        focus_label: str | None,
    ) -> _SearchCandidate | None:
        best_score = 0.0
        best_candidate: _SearchCandidate | None = None
        proposals = enumerate_coupled_group_candidates(coupled_group, family_states, self._definition_by_id)
        for proposal in proposals:
            parameter_values = self._clamp_candidate_parameter_values(dict(proposal.parameter_values))
            if not parameter_values:
                continue
            result = self.run_inference(self._apply_params_to_vector(base_vector, parameter_values))
            delta_q = result.quality_score - baseline_result.quality_score
            signal_gain = baseline_signal - self._optimization_signal(result, focus_label)
            candidate_score = delta_q + signal_gain
            if candidate_score > best_score:
                best_score = candidate_score
                best_candidate = _SearchCandidate(
                    parameter_values=parameter_values,
                    score=float(candidate_score),
                    recipe_label=proposal.recipe_label,
                    confidence_scale=proposal.confidence_scale,
                )
        return best_candidate if best_candidate is not None and best_score > 0.0 else None

    def _apply_param_to_vector(
        self,
        base_vector: List[float],
        parameter_id: str,
        new_value: float,
    ) -> List[float]:
        v = list(base_vector)
        for index, key in enumerate(self.feature_keys):
            if self._feature_to_parameter_id.get(key) == parameter_id:
                v[index] = new_value
        return v

    def _apply_params_to_vector(
        self,
        base_vector: List[float],
        parameter_values: dict[str, float],
    ) -> List[float]:
        updated = list(base_vector)
        for parameter_id, new_value in parameter_values.items():
            updated = self._apply_param_to_vector(updated, parameter_id, new_value)
        return updated

    def _clamp_candidate_parameter_values(self, parameter_values: dict[str, float]) -> dict[str, float]:
        clamped: dict[str, float] = {}
        for parameter_id, suggested_value in parameter_values.items():
            definition = self._definition_by_id.get(parameter_id)
            if definition is None:
                continue
            current_value = float(self._current_values.get(parameter_id, definition.baseline))
            bounded = self._clamp_to_empirical_support(definition, suggested_value)
            if abs(bounded - current_value) <= 1e-9:
                continue
            clamped[parameter_id] = bounded
        return clamped

    def _clamp_to_empirical_support(
        self,
        definition: StartupParameterDefinition,
        value: float,
    ) -> float:
        support_min = definition.observed_support_min
        support_max = definition.observed_support_max
        if support_min is None or support_max is None or support_max <= support_min:
            bounded = self._fallback_support_bound(definition, value, support_min)
        else:
            span = float(support_max) - float(support_min)
            margin = max(0.0, span * max(0.0, float(definition.support_margin_ratio or 0.0)))
            low = max(definition.range_min, float(support_min) - margin)
            high = min(definition.range_max, float(support_max) + margin)
            bounded = max(low, min(high, float(value)))
        if definition.decimal_places is not None:
            return round(bounded, int(definition.decimal_places))
        return float(bounded)

    def _fallback_support_bound(
        self,
        definition: StartupParameterDefinition,
        value: float,
        support_anchor: float | None,
    ) -> float:
        anchor = float(support_anchor) if support_anchor is not None else float(definition.baseline)
        ratio = max(0.0, float(definition.support_margin_ratio or 0.0))
        range_span = max(0.0, float(definition.range_max) - float(definition.range_min))
        range_margin = range_span * ratio
        if definition.step_mode == "relative":
            preferred_margin = abs(anchor) * abs(float(definition.preferred_step or 0.0))
            max_delta_margin = abs(anchor) * abs(float(definition.max_delta or 0.0))
        else:
            preferred_margin = abs(float(definition.preferred_step or 0.0))
            max_delta_margin = abs(float(definition.max_delta or 0.0))
        fallback_margin = max(range_margin, preferred_margin, max_delta_margin * 0.5)
        if fallback_margin <= 1e-9:
            return max(definition.range_min, min(definition.range_max, float(value)))
        low = max(definition.range_min, anchor - fallback_margin)
        high = min(definition.range_max, anchor + fallback_margin)
        return max(low, min(high, float(value)))

    def _ordered_searchable_definitions(self) -> tuple[StartupParameterDefinition, ...]:
        trainable = tuple(
            definition
            for definition in self._parameter_definitions
            if definition.control_feature_keys
        )
        deployable_parameter_ids = {
            parameter_id
            for family in self._control_family_definitions
            for parameter_id in family.parameter_ids
        }
        if deployable_parameter_ids:
            trainable = tuple(
                definition
                for definition in trainable
                if definition.parameter_id in deployable_parameter_ids
            )
        if self._trained_feature_key_set:
            aligned = tuple(
                definition
                for definition in trainable
                if any(key in self._trained_feature_key_set for key in definition.control_feature_keys)
            )
            trainable = aligned or trainable
        prioritized = prioritized_parameter_definitions(trainable, self._defect_severities)
        seen_parameter_ids = {definition.parameter_id for definition in prioritized}
        definitions = tuple(prioritized) + tuple(
            definition for definition in trainable if definition.parameter_id not in seen_parameter_ids
        )
        if self._current_values:
            definitions = tuple(
                definition for definition in definitions if definition.parameter_id in self._current_values
            )
        return definitions

    def _build_machine_parameters(
        self,
        candidates: List[_SearchCandidate],
        excess_ratio: float,
    ) -> List[SimulationMachineParameter]:
        parameters: List[SimulationMachineParameter] = []
        total_gain = sum(candidate.score for candidate in candidates) or 1.0
        parameter_order = {
            definition.parameter_id: index
            for index, definition in enumerate(self._ordered_searchable_definitions())
        }
        for candidate in candidates:
            base_confidence = candidate.score / total_gain
            urgency_boost = min(1.0, excess_ratio / 2.0)
            confidence = min(
                1.0,
                base_confidence * (1.0 + urgency_boost) * max(candidate.confidence_scale, 0.0),
            )
            for parameter_id in sorted(candidate.parameter_values, key=lambda item: parameter_order.get(item, len(parameter_order))):
                suggested = candidate.parameter_values[parameter_id]
                definition = self._definition_by_id.get(parameter_id)
                if definition is None:
                    continue
                current_val = self._current_values.get(parameter_id, definition.baseline)
                parameters.append(
                    SimulationMachineParameter(
                        name=definition.display_name,
                        unit=definition.unit,
                        current_value=round(current_val, 3),
                        suggested_value=round(suggested, 3),
                        confidence=round(confidence, 3),
                        parameter_id=parameter_id,
                        family_id=definition.family_id or "",
                        recipe_label=candidate.recipe_label,
                    )
                )
        return parameters

    def _optimization_signal(self, result: SuggestionInferenceResult, focus_label: str | None) -> float:
        if focus_label:
            return float(result.defect_risks.get(focus_label, 0.0))
        return self._weighted_defect_risk(result)

    def _weighted_defect_risk(self, result: SuggestionInferenceResult) -> float:
        weights = self._risk_priority_weights(result.defect_risks)
        return sum(weights.get(label, 0.0) * float(result.defect_risks.get(label, 0.0)) for label in weights)

    def _risk_priority_weights(self, defect_risks: Dict[str, float]) -> Dict[str, float]:
        active = {str(label): max(0.0, float(severity)) for label, severity in self._defect_severities.items() if float(severity) > 0.0}
        if active:
            total = sum(active.values()) or 1.0
            return {label: value / total for label, value in active.items()}
        manifest_weights = {
            str(label): max(0.0, float(weight))
            for label, weight in self.quality_weights.items()
            if float(weight) > 0.0
        }
        if manifest_weights:
            total = sum(manifest_weights.values()) or 1.0
            return {label: value / total for label, value in manifest_weights.items()}
        if not defect_risks:
            return {}
        uniform = 1.0 / float(len(defect_risks))
        return {str(label): uniform for label in defect_risks}

    def _build_summary(self, baseline: SuggestionInferenceResult) -> str:
        dominant = max(baseline.defect_risks, key=baseline.defect_risks.get) if baseline.defect_risks else None
        qs = baseline.quality_score
        if dominant and baseline.defect_risks[dominant] > 0.5:
            label = dominant.replace("_", " ").capitalize()
            signal_kind = self._defect_signal_kinds.get(dominant, "binary_defect_presence")
            if signal_kind == "binary_defect_presence":
                leading_text = f"{label} risk is elevated (p={baseline.defect_risks[dominant]:.0%}). "
            elif signal_kind == "defect_burden":
                leading_text = f"{label} burden is elevated ({baseline.defect_risks[dominant]:.2f}). "
            elif signal_kind == "duration_ratio":
                leading_text = f"{label} frame ratio is elevated ({baseline.defect_risks[dominant]:.2f}). "
            else:
                leading_text = f"{label} signal is elevated ({baseline.defect_risks[dominant]:.2f}). "
            return leading_text + f"Baseline quality score: {qs:.2f}. Adjusting the listed parameters is projected to improve part quality."
        improvement_goal = "reduce defect risk" if self._uses_binary_defect_signals() else "reduce predicted defect burden"
        return f"Baseline quality score: {qs:.2f}. Adjusting the listed parameters is projected to {improvement_goal}."

    def _uses_binary_defect_signals(self) -> bool:
        if not self._defect_signal_kinds:
            return True
        return all(kind == "binary_defect_presence" for kind in self._defect_signal_kinds.values())

    def _parameter_id_for_feature_stem(self, stem: str) -> str | None:
        if stem in self._definition_by_id:
            return stem
        for parameter_id in sorted(self._definition_by_id, key=len, reverse=True):
            if stem.startswith(f"{parameter_id}:"):
                return parameter_id
        return None

    def _build_family_lookup(self) -> dict[str, StartupControlFamilyDefinition]:
        lookup: dict[str, StartupControlFamilyDefinition] = {}
        for family in self._control_family_definitions:
            for parameter_id in family.parameter_ids:
                lookup[parameter_id] = family
        return lookup

    def _resolve_bundle_dir(self) -> Path:
        if self._bundle_path.is_dir():
            return self._bundle_path
        if self._bundle_path.suffix == ".sugbundle" or zipfile.is_zipfile(self._bundle_path):
            self._tmp_dir = Path(tempfile.mkdtemp(prefix="sugbundle_"))
            with zipfile.ZipFile(self._bundle_path, "r") as zf:
                zf.extractall(self._tmp_dir)
            candidates = [c for c in self._tmp_dir.iterdir() if c.is_dir()]
            if len(candidates) == 1:
                return candidates[0]
            return self._tmp_dir
        raise FileNotFoundError(f"Bundle not found or unrecognised format: {self._bundle_path}")

    @staticmethod
    def _load_manifest(bundle_dir: Path) -> dict:
        manifest_path = bundle_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"manifest.json not found in bundle: {bundle_dir}")
        with open(manifest_path, encoding="utf-8") as fh:
            return json.load(fh)

    @staticmethod
    def _load_training_meta(bundle_dir: Path) -> dict:
        meta_path = bundle_dir / "training_meta.json"
        if not meta_path.exists():
            return {}
        with open(meta_path, encoding="utf-8") as fh:
            return json.load(fh)

    @staticmethod
    def _validate_bundle_type(manifest: dict) -> None:
        if manifest.get("bundle_type") != "startup_suggestion":
            raise ValueError(
                f"Expected bundle_type='startup_suggestion', got '{manifest.get('bundle_type')}'."
            )

    @staticmethod
    def _sha256_file(path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    def _verify_checksums(self, bundle_dir: Path, manifest: dict) -> None:
        checksums: Dict[str, str] = manifest.get("checksums", {})
        for filename, expected in checksums.items():
            path = bundle_dir / filename
            if not path.exists():
                raise FileNotFoundError(f"Bundle file missing: {filename}")
            actual = self._sha256_file(path)
            if actual != expected:
                raise ValueError(f"Checksum mismatch for {filename}")

    @staticmethod
    def _load_sessions(bundle_dir: Path, manifest: dict) -> dict:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError(
                "onnxruntime is required to simulate suggestion bundles. Install with: pip install onnxruntime"
            ) from exc
        sessions = {}
        for target_name, spec in BundleSimulationRuntime._normalize_target_model_specs(
            manifest.get("target_models", {}),
            {},
        ).items():
            filename = spec["filename"]
            model_path = bundle_dir / filename
            if not model_path.exists():
                raise FileNotFoundError(f"ONNX model file missing: {filename}")
            sess_opts = ort.SessionOptions()
            sess_opts.log_severity_level = 3
            sessions[target_name] = ort.InferenceSession(str(model_path), sess_options=sess_opts)
        return sessions

    @staticmethod
    def _normalize_target_model_specs(target_models: Any, cv_metrics: Any) -> Dict[str, dict]:
        if not isinstance(target_models, dict):
            return {}
        metrics = cv_metrics if isinstance(cv_metrics, dict) else {}
        normalized: Dict[str, dict] = {}
        for raw_target_name, raw_spec in target_models.items():
            target_name = str(raw_target_name).strip()
            if not target_name:
                continue
            metric_spec = metrics.get(target_name) if isinstance(metrics.get(target_name), dict) else {}
            if isinstance(raw_spec, dict):
                filename = str(raw_spec.get("filename") or "").strip()
                model_type = str(raw_spec.get("model_type") or metric_spec.get("model_type") or BundleSimulationRuntime._legacy_model_type(target_name)).strip()
                source_target = str(raw_spec.get("source_target") or metric_spec.get("source_target") or BundleSimulationRuntime._legacy_source_target(target_name, model_type)).strip()
                signal_kind = str(raw_spec.get("signal_kind") or metric_spec.get("signal_kind") or BundleSimulationRuntime._legacy_signal_kind(target_name, model_type)).strip()
                signal_role = str(raw_spec.get("signal_role") or metric_spec.get("signal_role") or "optimization").strip()
            else:
                filename = str(raw_spec).strip()
                model_type = str(metric_spec.get("model_type") or BundleSimulationRuntime._legacy_model_type(target_name)).strip()
                source_target = str(metric_spec.get("source_target") or BundleSimulationRuntime._legacy_source_target(target_name, model_type)).strip()
                signal_kind = str(metric_spec.get("signal_kind") or BundleSimulationRuntime._legacy_signal_kind(target_name, model_type)).strip()
                signal_role = str(metric_spec.get("signal_role") or "optimization").strip()
            if not filename:
                continue
            normalized[target_name] = {
                "filename": filename,
                "model_type": model_type,
                "source_target": source_target,
                "signal_kind": signal_kind,
                "signal_role": signal_role,
            }
        return normalized

    @staticmethod
    def _legacy_model_type(target_name: str) -> str:
        return "regression" if target_name == "quality_score" else "classification"

    @staticmethod
    def _legacy_source_target(target_name: str, model_type: str) -> str:
        if target_name == "quality_score":
            return "y_quality_score"
        defect_label = str(target_name).removeprefix("defect_")
        if model_type == "classification":
            return f"y_defect_{defect_label}"
        return f"y_burden_{defect_label}"

    @staticmethod
    def _legacy_signal_kind(target_name: str, model_type: str) -> str:
        if target_name == "quality_score":
            return "quality_score"
        if model_type == "classification":
            return "binary_defect_presence"
        return "defect_burden"

    @staticmethod
    def _extract_regression_value(outputs: List[Any]) -> float:
        import numpy as np
        values = np.asarray(outputs[0]).reshape(-1)
        if values.size == 0:
            raise ValueError("Regression model returned no outputs.")
        return float(values[0])

    @staticmethod
    def _extract_classification_probability(outputs: List[Any]) -> float:
        import numpy as np
        if len(outputs) < 2:
            raise ValueError("Classification model did not return probability outputs.")
        probabilities = np.asarray(outputs[1])
        if probabilities.ndim == 1:
            if probabilities.size < 2:
                raise ValueError("Classification probability output is malformed.")
            return float(probabilities[1])
        if probabilities.ndim >= 2 and probabilities.shape[-1] >= 2:
            reshaped = probabilities.reshape(-1, probabilities.shape[-1])
            return float(reshaped[0][1])
        raise ValueError("Classification probability output is malformed.")

    def _assert_loaded(self) -> None:
        if not self._loaded:
            raise RuntimeError("Call load() before simulating steps.")


def load_simulation_steps(
    path: Path,
    *,
    default_metric_id: str,
    default_threshold: float,
) -> list[SimulationStepInput]:
    raw_steps = _load_json_or_jsonl(path)
    if not raw_steps:
        raise ValueError("Scenario file contains zero steps.")
    steps: list[SimulationStepInput] = []
    for index, raw in enumerate(raw_steps, start=1):
        if not isinstance(raw, dict):
            raise ValueError(f"Scenario step {index} is not an object.")
        step_id = str(raw.get("step_id") or f"step_{index:03d}")
        metric_id = str(raw.get("metric_id") or default_metric_id).strip()
        if not metric_id:
            raise ValueError(f"{step_id}: metric_id is required.")
        metric_value = _coerce_float(raw.get("metric_value"), f"{step_id}: metric_value")
        threshold = _coerce_float(raw.get("threshold", default_threshold), f"{step_id}: threshold")
        current_parameter_values = _coerce_float_mapping(
            raw.get("current_parameter_values"),
            f"{step_id}: current_parameter_values",
            required=True,
        )
        defect_metrics = _normalize_defect_metrics(raw.get("defect_metrics"), step_id)
        defect_severities = _normalize_defect_severities(raw.get("defect_severities"), defect_metrics)
        expected = raw.get("expected") if isinstance(raw.get("expected"), dict) else {}
        parameter_state_mode = str(raw.get("parameter_state_mode") or "carry").strip().lower()
        if parameter_state_mode not in {"carry", "replace", "merge"}:
            raise ValueError(
                f"{step_id}: parameter_state_mode must be one of carry, replace, merge."
            )
        steps.append(
            SimulationStepInput(
                step_id=step_id,
                metric_id=metric_id,
                metric_value=metric_value,
                threshold=threshold,
                current_parameter_values=current_parameter_values,
                defect_severities=defect_severities,
                defect_metrics=defect_metrics,
                expected=expected,
                parameter_state_mode=parameter_state_mode,
            )
        )
    return steps


def simulate_bundle_scenarios(
    bundle_path: Path,
    scenario_path: Path,
    *,
    default_metric_id: str = "duration_ratio",
    default_threshold: float = 0.10,
    closed_loop_assume_apply: bool = False,
) -> list[SimulationStepResult]:
    runtime = BundleSimulationRuntime(bundle_path)
    runtime.load()
    try:
        steps = load_simulation_steps(
            scenario_path,
            default_metric_id=default_metric_id,
            default_threshold=default_threshold,
        )
        results: list[SimulationStepResult] = []
        carried_values: dict[str, float] | None = None
        for index, step in enumerate(steps):
            effective_values = dict(step.current_parameter_values)
            if closed_loop_assume_apply and index > 0 and carried_values is not None:
                if step.parameter_state_mode == "replace":
                    effective_values = dict(step.current_parameter_values)
                elif step.parameter_state_mode == "merge":
                    effective_values = dict(carried_values)
                    effective_values.update(step.current_parameter_values)
                else:  # carry
                    effective_values = dict(carried_values)
            effective_step = SimulationStepInput(
                step_id=step.step_id,
                metric_id=step.metric_id,
                metric_value=step.metric_value,
                threshold=step.threshold,
                current_parameter_values=effective_values,
                defect_severities=step.defect_severities,
                defect_metrics=step.defect_metrics,
                expected=step.expected,
                parameter_state_mode=step.parameter_state_mode,
            )
            result = runtime.simulate_step(effective_step)
            results.append(result)
            carried_values = dict(effective_values)
            if closed_loop_assume_apply and result.suggestion is not None:
                for parameter in result.suggestion.get("parameters", ()):
                    parameter_id = str(parameter.get("parameter_id", "")).strip()
                    if not parameter_id:
                        continue
                    carried_values[parameter_id] = float(parameter.get("suggested_value"))
        return results
    finally:
        runtime.unload()


def evaluate_expectations(
    expected: dict[str, Any],
    triggered: bool,
    focus_label: str | None,
    suggestion: SimulationSuggestion | None,
) -> dict[str, Any]:
    failures: list[str] = []
    if not expected:
        return {"checked": False, "passed": True, "failures": failures}
    if "must_trigger" in expected:
        if bool(expected["must_trigger"]) != bool(suggestion is not None and triggered):
            failures.append(f"must_trigger expected {bool(expected['must_trigger'])}, got {bool(suggestion is not None and triggered)}")
    if "must_not_trigger" in expected:
        expected_value = bool(expected["must_not_trigger"])
        actual_value = not bool(suggestion is not None and triggered)
        if expected_value != actual_value:
            failures.append(f"must_not_trigger expected {expected_value}, got {actual_value}")
    if "focus_label" in expected:
        wanted = _canonical_defect_label(str(expected["focus_label"]))
        if wanted != (focus_label or ""):
            failures.append(f"focus_label expected '{wanted}', got '{focus_label or ''}'")
    if "suggested_parameter_ids_any" in expected:
        wanted = {
            str(item).strip()
            for item in expected.get("suggested_parameter_ids_any", ())
            if str(item).strip()
        }
        actual = {
            parameter.parameter_id
            for parameter in (suggestion.parameters if suggestion is not None else ())
        }
        if wanted and not (wanted & actual):
            failures.append(f"expected any suggested parameter in {sorted(wanted)}, got {sorted(actual)}")
    if "max_suggestions" in expected:
        actual_count = len(suggestion.parameters) if suggestion is not None else 0
        if actual_count > int(expected["max_suggestions"]):
            failures.append(f"max_suggestions expected <= {int(expected['max_suggestions'])}, got {actual_count}")
    return {"checked": True, "passed": not failures, "failures": failures}


def format_simulation_report(results: Sequence[SimulationStepResult]) -> str:
    lines: list[str] = []
    for result in results:
        state = "TRIGGERED" if result.suggestion is not None else "NO_SUGGESTION"
        lines.append(
            f"[{result.step_id}] {result.metric_id}={result.metric_value:.3f} "
            f"threshold={result.threshold:.3f} -> {state}"
        )
        if result.focus_label:
            lines.append(f"  Focus label: {result.focus_label}")
        lines.append(f"  Baseline quality: {result.baseline_quality_score:.2f}")
        if result.predicted_label_signals:
            joined = "  ".join(
                f"{label}={value:.2f}" for label, value in sorted(result.predicted_label_signals.items())
            )
            lines.append(f"  Predicted label signals: {joined}")
        if result.suggestion is not None:
            lines.append("  Suggestion:")
            for parameter in result.suggestion["parameters"]:
                current_value = float(parameter["current_value"])
                suggested_value = float(parameter["suggested_value"])
                base = max(abs(current_value), 1e-6)
                delta_pct = ((suggested_value - current_value) / base) * 100.0
                lines.append(
                    f"    {parameter['name']}: {current_value:.3f} -> {suggested_value:.3f} "
                    f"({delta_pct:+.1f}%)"
                )
            lines.append(f"  Summary: {result.suggestion['summary']}")
        if result.expectations.get("checked"):
            lines.append(
                "  Expectations: "
                + ("PASS" if result.expectations.get("passed") else "FAIL")
            )
            for failure in result.expectations.get("failures", ()):
                lines.append(f"    - {failure}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def results_to_jsonable(results: Sequence[SimulationStepResult]) -> list[dict[str, Any]]:
    return [asdict(result) for result in results]


def _load_json_or_jsonl(path: Path) -> list[Any]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    payload = json.loads(text)
    if isinstance(payload, list):
        return payload
    raise ValueError("Scenario JSON must be an array of step objects.")


def _coerce_float(value: Any, field_name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be numeric.") from exc


def _coerce_float_mapping(value: Any, field_name: str, *, required: bool) -> dict[str, float]:
    if value is None:
        if required:
            raise ValueError(f"{field_name} is required.")
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be an object.")
    out: dict[str, float] = {}
    for key, raw in value.items():
        token = str(key).strip()
        if not token:
            continue
        out[token] = _coerce_float(raw, f"{field_name}.{token}")
    return out


def _normalize_defect_metrics(raw: Any, step_id: str) -> dict[str, dict[str, float]]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"{step_id}: defect_metrics must be an object.")
    out: dict[str, dict[str, float]] = {}
    for label, metrics in raw.items():
        canonical_label = _canonical_defect_label(str(label))
        if not isinstance(metrics, dict):
            raise ValueError(f"{step_id}: defect_metrics.{label} must be an object.")
        out[canonical_label] = {
            str(metric): _coerce_float(value, f"{step_id}: defect_metrics.{label}.{metric}")
            for metric, value in metrics.items()
        }
    return out


def _normalize_defect_severities(
    raw: Any,
    defect_metrics: Mapping[str, Mapping[str, float]],
) -> dict[str, float]:
    if isinstance(raw, dict):
        return {
            _canonical_defect_label(str(label)): _coerce_float(value, f"defect_severities.{label}")
            for label, value in raw.items()
        }
    derived: dict[str, float] = {}
    for label, metrics in defect_metrics.items():
        derived[label] = float(metrics.get("component_severity", 0.0))
    return derived


def _compute_urgency(excess_ratio: float) -> str:
    if excess_ratio < 0.15:
        return "info"
    if excess_ratio < 0.40:
        return "warning"
    return "critical"


def _canonical_defect_label(label: str) -> str:
    return str(label).strip().lower().replace(" ", "_").replace("-", "_")


def current_parameter_defaults(
    definitions: Iterable[StartupParameterDefinition],
) -> dict[str, float]:
    return {definition.parameter_id: definition.baseline for definition in definitions}


def control_feature_mapping(definitions: Iterable[StartupParameterDefinition]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for definition in definitions:
        for key in definition.control_feature_keys:
            mapping[key] = definition.parameter_id
    return mapping


def map_ui_to_internal_values(
    ui_values: dict[str, float],
    definitions: Iterable[StartupParameterDefinition],
) -> dict[str, float]:
    updated = dict(ui_values)
    for definition in definitions:
        if definition.parameter_id in updated:
            continue
        prefix = _parameter_prefix(definition.parameter_id)
        semantic = definition.semantic_parameter_id
        for candidate in (semantic, prefix):
            if not candidate:
                continue
            if candidate in ui_values:
                updated[definition.parameter_id] = ui_values[candidate]
                break
    return updated


def derive_parameter_definitions(
    feature_keys: Sequence[str],
    imputation_values: dict[str, float],
    manifest_schema: Sequence[dict] | None = None,
) -> tuple[StartupParameterDefinition, ...]:
    from_manifest = _from_manifest_schema(manifest_schema)
    if from_manifest:
        return from_manifest
    grouped: dict[str, list[str]] = {}
    for key in feature_keys:
        prefix = _parameter_prefix(key)
        grouped.setdefault(prefix, []).append(key)
    definitions: list[StartupParameterDefinition] = []
    for parameter_id, keys in grouped.items():
        control_keys = _select_control_feature_keys(keys)
        if not control_keys:
            continue
        baseline = float(imputation_values.get(control_keys[0], 0.0))
        range_min, range_max = _heuristic_range(baseline)
        preferred_step, max_delta = _step_policy(
            parameter_id=parameter_id,
            range_min=range_min,
            range_max=range_max,
            baseline=baseline,
            decimal_places=None,
        )
        definitions.append(
            StartupParameterDefinition(
                parameter_id=parameter_id,
                display_name=_humanize_parameter_id(parameter_id),
                unit="setpoint",
                baseline=baseline,
                range_min=range_min,
                range_max=range_max,
                control_feature_keys=control_keys,
                family_id=parameter_id,
                semantic_parameter_id=parameter_id,
                preferred_step=preferred_step,
                max_delta=max_delta,
            )
        )
    return tuple(definitions)


def derive_control_family_definitions(
    definitions: Sequence[StartupParameterDefinition],
    manifest_control_families: Sequence[dict] | None = None,
) -> tuple[StartupControlFamilyDefinition, ...]:
    if manifest_control_families:
        from_manifest = _family_definitions_from_manifest(definitions, manifest_control_families)
        if from_manifest:
            return from_manifest
    grouped: dict[str, list[StartupParameterDefinition]] = {}
    for definition in definitions:
        family_id = str(
            definition.family_id
            or definition.semantic_parameter_id
            or _parameter_prefix(definition.parameter_id)
        ).strip()
        grouped.setdefault(family_id, []).append(definition)
    out: list[StartupControlFamilyDefinition] = []
    for family_id in sorted(grouped):
        members = sorted(
            grouped[family_id],
            key=lambda definition: _slot_sort_key(definition.canonical_slot_id or definition.slot_id),
        )
        lead = members[0]
        family_type = "atomic" if len(members) > 1 else "single_slot"
        out.append(
            StartupControlFamilyDefinition(
                family_id=family_id,
                display_name=_humanize_parameter_id(family_id),
                family_type=family_type,
                parameter_ids=tuple(definition.parameter_id for definition in members),
                semantic_parameter_id=lead.semantic_parameter_id or _parameter_prefix(lead.parameter_id),
                page_id=lead.page_id,
                subpage_id=lead.subpage_id,
                family_constraints=_default_family_constraints(
                    family_id=family_id,
                    family_type=family_type,
                    members=members,
                ),
            )
        )
    return tuple(out)


def search_candidate_values(
    definition: StartupParameterDefinition,
    current_value: float,
) -> tuple[float, ...]:
    current = float(current_value)
    candidates: list[float] = []
    if definition.step_mode == "relative":
        preferred = abs(float(definition.preferred_step or 0.05))
        max_delta = abs(float(definition.max_delta or max(preferred, 0.15)))
        delta_scales = [preferred, min(max_delta, preferred * 2.0), max_delta]
        for delta in delta_scales:
            for sign in (-1.0, 1.0):
                perturbed = current * (1.0 + sign * delta)
                perturbed = max(definition.range_min, min(definition.range_max, perturbed))
                if abs(perturbed - current) > 1e-9 and perturbed not in candidates:
                    candidates.append(perturbed)
        return tuple(candidates)
    preferred = abs(float(definition.preferred_step or 1.0))
    max_delta = abs(float(definition.max_delta or preferred))
    deltas = sorted({round(preferred, 9), round(min(max_delta, preferred * 2.0), 9), round(max_delta, 9)})
    for delta in deltas:
        if delta <= 0.0:
            continue
        for sign in (-1.0, 1.0):
            perturbed = current + sign * delta
            perturbed = max(definition.range_min, min(definition.range_max, perturbed))
            perturbed = _round_to_precision(perturbed, definition.decimal_places)
            if abs(perturbed - current) > 1e-9 and perturbed not in candidates:
                candidates.append(perturbed)
    return tuple(candidates)


def prioritized_parameter_definitions(
    definitions: Sequence[StartupParameterDefinition],
    defect_severities: dict[str, float] | None,
) -> tuple[StartupParameterDefinition, ...]:
    if not definitions:
        return ()
    if not defect_severities:
        return tuple(definitions)
    scores: dict[str, float] = {}
    for definition in definitions:
        token = str(definition.semantic_parameter_id or _parameter_prefix(definition.parameter_id)).lower()
        for defect_label, severity in defect_severities.items():
            if float(severity) <= 0.0:
                continue
            if defect_label == "flash" and ("pressure" in token or "speed" in token or "clamp" in token):
                scores[definition.parameter_id] = scores.get(definition.parameter_id, 0.0) + float(severity)
            elif defect_label == "sink_mark" and ("hold" in token or "pressure" in token or "temperature" in token):
                scores[definition.parameter_id] = scores.get(definition.parameter_id, 0.0) + float(severity)
            elif defect_label == "burn_mark" and ("speed" in token or "temperature" in token):
                scores[definition.parameter_id] = scores.get(definition.parameter_id, 0.0) + float(severity)
            elif defect_label == "weld_line" and ("temperature" in token or "pressure" in token or "speed" in token):
                scores[definition.parameter_id] = scores.get(definition.parameter_id, 0.0) + float(severity)
    ranked = sorted(
        definitions,
        key=lambda definition: (-scores.get(definition.parameter_id, 0.0), definition.parameter_id),
    )
    return tuple(ranked)


def build_runtime_family_states(
    control_families: Sequence[StartupControlFamilyDefinition],
    definition_by_id: Mapping[str, StartupParameterDefinition],
    current_values: Mapping[str, float],
) -> dict[str, RuntimeControlFamilyState]:
    states: dict[str, RuntimeControlFamilyState] = {}
    for family in control_families:
        family_values: dict[str, float] = {}
        activation_mask: list[int] = []
        for parameter_id in family.parameter_ids:
            definition = definition_by_id.get(parameter_id)
            if definition is None:
                continue
            if parameter_id in current_values:
                family_values[parameter_id] = float(current_values[parameter_id])
                activation_mask.append(1)
            else:
                family_values[parameter_id] = float(definition.baseline)
                activation_mask.append(0)
        if not family_values:
            continue
        if activation_mask and all(value == 1 for value in activation_mask):
            activation_state = "active"
        elif any(activation_mask):
            activation_state = "partial"
        else:
            activation_state = "inactive"
        observed = set(str(value).strip() for value in family.observed_observability_states if str(value).strip())
        if "observed" in observed:
            observability_state = "observed"
        elif "partial" in observed:
            observability_state = "partial"
        elif "unknown" in observed or not observed:
            observability_state = "unknown"
        else:
            observability_state = "unobserved"
        states[family.family_id] = RuntimeControlFamilyState(
            family_id=family.family_id,
            family_type=family.family_type,
            parameter_ids=tuple(family_values),
            semantic_parameter_id=family.semantic_parameter_id,
            current_values=family_values,
            activation_mask=tuple(activation_mask),
            activation_state=activation_state,
            observability_state=observability_state,
            confidence=_family_state_confidence(activation_state, observability_state),
            source="assumed",
            family_constraints=dict(family.family_constraints or {}),
        )
    return states


def build_coupled_family_groups(
    family_states: Mapping[str, RuntimeControlFamilyState],
) -> tuple[tuple[str, ...], ...]:
    seen: set[tuple[str, ...]] = set()
    groups: list[tuple[str, ...]] = []
    for family_id, state in family_states.items():
        linked_raw = state.family_constraints.get("linked_family_ids") or []
        linked = sorted({family_id, *[str(item).strip() for item in linked_raw if str(item).strip() in family_states]})
        if len(linked) < 2:
            continue
        group = tuple(linked)
        if group in seen:
            continue
        seen.add(group)
        groups.append(group)
    return tuple(groups)


def enumerate_family_recipe_candidates(
    family_state: RuntimeControlFamilyState,
    definition_by_id: Mapping[str, StartupParameterDefinition],
) -> tuple[RuntimeCandidateProposal, ...]:
    if family_state.family_type not in {"atomic", "partially_controllable"} or len(family_state.parameter_ids) <= 1:
        return _single_parameter_candidates(family_state, definition_by_id)
    if family_state.activation_state == "inactive":
        return ()
    allowed = _allowed_recipe_types(family_state.family_constraints, grouped=True)
    proposals: list[RuntimeCandidateProposal] = []
    current = [family_state.current_values[parameter_id] for parameter_id in family_state.parameter_ids]
    preferred = [max(0.0, float(definition_by_id[parameter_id].preferred_step or 0.0)) for parameter_id in family_state.parameter_ids]
    front_weights = _profile_weights(len(current), reverse=False)
    tail_weights = _profile_weights(len(current), reverse=True)
    for direction in (-1.0, 1.0):
        if "shift_level" in allowed:
            proposals.append(_proposal_from_vector(family_state, definition_by_id, _shift_profile(current, preferred, direction, [1.0] * len(current)), "raise whole profile" if direction > 0 else "lower whole profile"))
        if "front_load" in allowed:
            proposals.append(_proposal_from_vector(family_state, definition_by_id, _shift_profile(current, preferred, direction, front_weights), "load earlier steps" if direction > 0 else "unload earlier steps"))
        if "tail_load" in allowed:
            proposals.append(_proposal_from_vector(family_state, definition_by_id, _shift_profile(current, preferred, direction, tail_weights), "load later steps" if direction > 0 else "unload later steps"))
    if "flatten_profile" in allowed:
        proposals.append(_proposal_from_vector(family_state, definition_by_id, _flatten_profile(current, preferred), "flatten profile"))
    return _dedupe_valid_proposals(proposals, {family_state.family_id: family_state}, definition_by_id)


def enumerate_coupled_group_candidates(
    group_family_ids: Sequence[str],
    family_states: Mapping[str, RuntimeControlFamilyState],
    definition_by_id: Mapping[str, StartupParameterDefinition],
) -> tuple[RuntimeCandidateProposal, ...]:
    states = [family_states[family_id] for family_id in group_family_ids if family_id in family_states]
    if len(states) < 2 or any(state.activation_state != "active" for state in states):
        return ()
    common_recipes = set(_allowed_recipe_types(states[0].family_constraints, grouped=True))
    for state in states[1:]:
        common_recipes &= set(_allowed_recipe_types(state.family_constraints, grouped=True))
    common_recipes.discard("single_step")
    if not common_recipes:
        common_recipes = {"shift_level"}
    proposals: list[RuntimeCandidateProposal] = []
    for recipe_type in sorted(common_recipes):
        for direction in (-1.0, 1.0):
            merged: dict[str, float] = {}
            labels: list[str] = []
            for state in states:
                current = [state.current_values[parameter_id] for parameter_id in state.parameter_ids]
                preferred = [max(0.0, float(definition_by_id[parameter_id].preferred_step or 0.0)) for parameter_id in state.parameter_ids]
                if recipe_type == "front_load":
                    vector = _shift_profile(current, preferred, direction, _profile_weights(len(current), reverse=False))
                    labels.append("load earlier steps" if direction > 0 else "unload earlier steps")
                elif recipe_type == "tail_load":
                    vector = _shift_profile(current, preferred, direction, _profile_weights(len(current), reverse=True))
                    labels.append("load later steps" if direction > 0 else "unload later steps")
                else:
                    vector = _shift_profile(current, preferred, direction, [1.0] * len(current))
                    labels.append("raise paired profiles" if direction > 0 else "lower paired profiles")
                proposal = _proposal_from_vector(state, definition_by_id, vector, labels[-1])
                merged.update(proposal.parameter_values)
            confidence_scale = min(state.confidence for state in states)
            proposals.append(
                RuntimeCandidateProposal(
                    parameter_values=merged,
                    family_ids=tuple(state.family_id for state in states),
                    recipe_label=labels[0],
                    confidence_scale=confidence_scale,
                )
            )
    scoped_states = {state.family_id: state for state in states}
    return _dedupe_valid_proposals(proposals, scoped_states, definition_by_id)


def _single_parameter_candidates(
    family_state: RuntimeControlFamilyState,
    definition_by_id: Mapping[str, StartupParameterDefinition],
) -> tuple[RuntimeCandidateProposal, ...]:
    proposals: list[RuntimeCandidateProposal] = []
    for parameter_id in family_state.parameter_ids:
        definition = definition_by_id.get(parameter_id)
        if definition is None:
            continue
        current_value = family_state.current_values[parameter_id]
        for perturbed in search_candidate_values(definition, current_value):
            recipe_label = "increase setting" if float(perturbed) > current_value else "decrease setting"
            proposals.append(
                RuntimeCandidateProposal(
                    parameter_values={parameter_id: float(perturbed)},
                    family_ids=(family_state.family_id,),
                    recipe_label=recipe_label,
                    confidence_scale=family_state.confidence,
                )
            )
    return _dedupe_valid_proposals(proposals, {family_state.family_id: family_state}, definition_by_id)


def _dedupe_valid_proposals(
    proposals: Sequence[RuntimeCandidateProposal],
    family_states: Mapping[str, RuntimeControlFamilyState],
    definition_by_id: Mapping[str, StartupParameterDefinition],
) -> tuple[RuntimeCandidateProposal, ...]:
    out: list[RuntimeCandidateProposal] = []
    seen: set[tuple[tuple[str, float], ...]] = set()
    for proposal in proposals:
        if not validate_candidate_proposal(proposal, family_states, definition_by_id):
            continue
        signature = tuple(sorted((parameter_id, round(float(value), 6)) for parameter_id, value in proposal.parameter_values.items()))
        if signature in seen:
            continue
        seen.add(signature)
        out.append(proposal)
    return tuple(out)


def validate_candidate_proposal(
    proposal: RuntimeCandidateProposal,
    family_states: Mapping[str, RuntimeControlFamilyState],
    definition_by_id: Mapping[str, StartupParameterDefinition],
) -> bool:
    if not proposal.parameter_values:
        return False
    for parameter_id, value in proposal.parameter_values.items():
        definition = definition_by_id.get(parameter_id)
        if definition is None:
            return False
        if float(value) < definition.range_min - 1e-9 or float(value) > definition.range_max + 1e-9:
            return False
    for family_id in proposal.family_ids:
        state = family_states.get(family_id)
        if state is None:
            return False
        if state.activation_state == "inactive":
            return False
    return True


def _allowed_recipe_types(constraints: Mapping[str, object], *, grouped: bool) -> tuple[str, ...]:
    raw = constraints.get("allowed_recipe_types")
    if isinstance(raw, list) and raw:
        return tuple(str(item) for item in raw if str(item).strip())
    return ("shift_level", "front_load", "tail_load", "flatten_profile") if grouped else ("single_step",)


def _proposal_from_vector(
    family_state: RuntimeControlFamilyState,
    definition_by_id: Mapping[str, StartupParameterDefinition],
    values: Sequence[float],
    recipe_label: str,
) -> RuntimeCandidateProposal:
    parameter_values: dict[str, float] = {}
    for parameter_id, value in zip(family_state.parameter_ids, values):
        definition = definition_by_id[parameter_id]
        current_value = family_state.current_values[parameter_id]
        clipped = max(definition.range_min, min(definition.range_max, float(value)))
        rounded = _round_value(clipped, definition.decimal_places)
        if abs(rounded - current_value) <= 1e-9:
            continue
        parameter_values[parameter_id] = rounded
    return RuntimeCandidateProposal(
        parameter_values=parameter_values,
        family_ids=(family_state.family_id,),
        recipe_label=recipe_label,
        confidence_scale=family_state.confidence,
    )


def _shift_profile(current: Sequence[float], preferred_steps: Sequence[float], direction: float, weights: Sequence[float]) -> list[float]:
    out: list[float] = []
    for value, preferred_step, weight in zip(current, preferred_steps, weights):
        delta = direction * max(float(preferred_step), 1e-6) * float(weight)
        out.append(float(value) + delta)
    return out


def _flatten_profile(current: Sequence[float], preferred_steps: Sequence[float]) -> list[float]:
    target = sum(float(value) for value in current) / max(1, len(current))
    out: list[float] = []
    for value, preferred_step in zip(current, preferred_steps):
        delta = max(float(preferred_step), 1e-6)
        if abs(target - float(value)) <= delta:
            out.append(target)
        elif target > float(value):
            out.append(float(value) + delta)
        else:
            out.append(float(value) - delta)
    return out


def _profile_weights(length: int, *, reverse: bool) -> list[float]:
    if length <= 1:
        return [1.0]
    span = max(1, length - 1)
    weights = [1.0 - (0.75 * index / span) for index in range(length)]
    return list(reversed(weights)) if reverse else weights


def _family_state_confidence(activation_state: str, observability_state: str) -> float:
    if activation_state == "active" and observability_state == "observed":
        return 0.85
    if activation_state == "active" and observability_state == "partial":
        return 0.75
    if activation_state == "active":
        return 0.65
    if activation_state == "partial":
        return 0.55
    return 0.40


def _parameter_prefix(feature_key: str) -> str:
    return feature_key.split(":", 1)[0] if ":" in feature_key else feature_key


def _select_control_feature_keys(feature_keys: Sequence[str]) -> tuple[str, ...]:
    control_stat_priority = ("setpoint_end", "setpoint", "actual.last", "actual.mean", "actual")
    scored: list[tuple[int, str]] = []
    for key in feature_keys:
        stat = key.split(".", 1)[1] if "." in key else ""
        for priority, wanted in enumerate(control_stat_priority):
            if stat == wanted:
                scored.append((priority, key))
                break
    if not scored:
        # Some lightweight bundles carry bare feature keys with no stat suffix
        # (for example "Iniettare_Q_1"). Treat those as directly controllable
        # features instead of dropping the entire parameter mapping.
        bare = tuple(str(key) for key in feature_keys if str(key).strip())
        return bare
    best_priority = min(priority for priority, _ in scored)
    return tuple(key for priority, key in scored if priority == best_priority)


def _heuristic_range(baseline: float) -> tuple[float, float]:
    span = max(abs(baseline) * 0.5, 1.0)
    range_min = baseline - span
    range_max = baseline + span
    if baseline >= 0.0:
        range_min = max(0.0, range_min)
    if range_max <= range_min:
        range_max = range_min + max(1.0, abs(baseline) * 0.1)
    return (range_min, range_max)


def _step_policy(*, parameter_id: str, range_min: float, range_max: float, baseline: float, decimal_places: int | None) -> tuple[float, float]:
    span = max(0.0, float(range_max) - float(range_min))
    quantum = 10.0 ** (-int(decimal_places)) if decimal_places is not None and decimal_places >= 0 else 1.0
    token = str(parameter_id).lower()
    if any(key in token for key in ("temp", "temperature", "barrel", "melt", "mold")):
        preferred = max(quantum, 1.0)
        max_delta = min(max(preferred * 5.0, 5.0), span * 0.10 if span > 0 else preferred * 5.0)
    elif any(key in token for key in ("pressure", "press", "back_pressure", "clamp")):
        preferred = max(quantum, 5.0 if span <= 300 else 10.0)
        max_delta = min(max(preferred * 5.0, 30.0), span * 0.10 if span > 0 else preferred * 5.0)
    elif any(key in token for key in ("time", "cool", "hold_time")):
        preferred = max(quantum, 0.1 if decimal_places not in (None, 0) else 1.0)
        max_delta = min(max(preferred * 5.0, 1.0), span * 0.10 if span > 0 else preferred * 5.0)
    elif any(key in token for key in ("speed", "flow", "dose", "dosaggio", "iniettare", "vite", "screw")):
        preferred = max(quantum, max(1.0, round(span * 0.02, 3)))
        max_delta = min(max(preferred * 4.0, 5.0), span * 0.10 if span > 0 else preferred * 4.0)
    else:
        preferred = max(quantum, max(1.0 if abs(baseline) >= 10.0 else quantum, round(span * 0.02, 3)))
        max_delta = min(max(preferred * 4.0, preferred), span * 0.10 if span > 0 else preferred * 4.0)
    if max_delta < preferred:
        max_delta = preferred
    return (round(float(preferred), 6), round(float(max_delta), 6))


def _round_to_precision(value: float, decimal_places: int | None) -> float:
    if decimal_places is None:
        return value
    return round(value, int(decimal_places))


def _round_value(value: float, decimal_places: int | None) -> float:
    if decimal_places is None:
        return float(value)
    return round(float(value), int(decimal_places))


def _humanize_parameter_id(parameter_id: str) -> str:
    parts = [segment.strip() for segment in parameter_id.replace("/", "|").split("|")]
    cleaned: list[str] = []
    for part in parts:
        token = part.replace("_", " ").strip()
        if not token:
            continue
        if len(token) <= 2 or token.isupper():
            cleaned.append(token.upper())
        else:
            cleaned.append(token.capitalize())
    return " / ".join(cleaned) or parameter_id


def _from_manifest_schema(manifest_schema: Sequence[dict] | None) -> tuple[StartupParameterDefinition, ...]:
    if not manifest_schema:
        return ()
    definitions: list[StartupParameterDefinition] = []
    for item in manifest_schema:
        if not isinstance(item, dict):
            continue
        trained_keys_raw = item.get("trained_control_feature_keys")
        if isinstance(trained_keys_raw, (list, tuple)):
            control_keys = tuple(str(key) for key in trained_keys_raw if key)
        else:
            control_keys = tuple(str(key) for key in item.get("control_feature_keys", ()) if key)
        parameter_id = str(item.get("parameter_id", "")).strip()
        if not parameter_id:
            continue
        baseline = float(item.get("baseline", 0.0))
        range_min = float(item.get("range_min", baseline))
        range_max = float(item.get("range_max", baseline))
        if range_max <= range_min:
            range_min, range_max = _heuristic_range(baseline)
        decimal_places = item.get("decimal_places")
        if decimal_places is not None:
            decimal_places = int(decimal_places)
        definitions.append(
            StartupParameterDefinition(
                parameter_id=parameter_id,
                display_name=str(item.get("display_name", "")).strip() or _humanize_parameter_id(parameter_id),
                unit=str(item.get("unit", "")).strip() or "setpoint",
                baseline=baseline,
                range_min=range_min,
                range_max=range_max,
                control_feature_keys=control_keys,
                family_id=str(item.get("family_id", "")).strip() or None,
                semantic_parameter_id=(
                    str(item.get("canonical_parameter_id", "")).strip()
                    or str(item.get("semantic_parameter_id", "")).strip()
                    or _parameter_prefix(parameter_id)
                ),
                page_id=str(item.get("page_id", "")).strip() or None,
                subpage_id=str(item.get("subpage_id", "")).strip() or None,
                slot_id=str(item.get("slot_id", "")).strip() or None,
                canonical_slot_id=str(item.get("canonical_slot_id", "")).strip() or None,
                step_mode=str(item.get("step_mode", "absolute")).strip() or "absolute",
                preferred_step=float(item.get("preferred_step", 1.0)),
                max_delta=float(item.get("max_delta", item.get("preferred_step", 1.0))),
                observed_support_min=(float(item["observed_support_min"]) if item.get("observed_support_min") is not None else None),
                observed_support_max=(float(item["observed_support_max"]) if item.get("observed_support_max") is not None else None),
                support_margin_ratio=float(item.get("support_margin_ratio", 0.05)),
                decimal_places=decimal_places,
            )
        )
    return tuple(definitions)


def _family_definitions_from_manifest(
    definitions: Sequence[StartupParameterDefinition],
    manifest_control_families: Sequence[dict],
) -> tuple[StartupControlFamilyDefinition, ...]:
    definition_by_id = {definition.parameter_id: definition for definition in definitions}
    out: list[StartupControlFamilyDefinition] = []
    for item in manifest_control_families:
        if not isinstance(item, dict):
            continue
        family_id = str(item.get("family_id", "")).strip()
        if not family_id:
            continue
        raw_constraints = dict(item.get("family_constraints") or {})
        members_raw = item.get("ordered_members") or []
        if not isinstance(members_raw, list):
            members_raw = []
        parameter_ids = [
            str(member.get("parameter_id", "")).strip()
            for member in members_raw
            if isinstance(member, dict) and str(member.get("parameter_id", "")).strip() in definition_by_id
        ]
        controllable_member_ids = [
            str(parameter_id).strip()
            for parameter_id in raw_constraints.get("controllable_member_parameter_ids", ())
            if str(parameter_id).strip() in definition_by_id
        ]
        if controllable_member_ids:
            allowed = set(controllable_member_ids)
            parameter_ids = [parameter_id for parameter_id in parameter_ids if parameter_id in allowed]
        if not parameter_ids:
            continue
        lead = definition_by_id[parameter_ids[0]]
        family_type = str(item.get("family_type", "")).strip() or ("atomic" if len(parameter_ids) > 1 else "single_slot")
        defaults = _default_family_constraints(
            family_id=family_id,
            family_type=family_type,
            members=[definition_by_id[param_id] for param_id in parameter_ids],
        )
        out.append(
            StartupControlFamilyDefinition(
                family_id=family_id,
                display_name=str(item.get("display_name", "")).strip() or _humanize_parameter_id(family_id),
                family_type=family_type,
                parameter_ids=tuple(parameter_ids),
                semantic_parameter_id=(
                    str(item.get("semantic_parameter_id", "")).strip()
                    or lead.semantic_parameter_id
                    or _parameter_prefix(lead.parameter_id)
                ),
                page_id=str(item.get("page_id", "")).strip() or lead.page_id,
                subpage_id=str(item.get("subpage_id", "")).strip() or lead.subpage_id,
                family_constraints=_merge_family_constraints(defaults, raw_constraints),
                observed_activation_states=tuple(str(value) for value in item.get("observed_activation_states", ()) if str(value).strip()),
                observed_observability_states=tuple(str(value) for value in item.get("observed_observability_states", ()) if str(value).strip()),
            )
        )
    return tuple(out)


def _merge_family_constraints(defaults: dict[str, object], override: dict[str, object]) -> dict[str, object]:
    merged = dict(defaults)
    for key, value in override.items():
        if isinstance(value, list):
            seen: list[object] = []
            base = merged.get(key)
            if isinstance(base, list):
                for item in base:
                    if item not in seen:
                        seen.append(item)
            for item in value:
                if item not in seen:
                    seen.append(item)
            merged[key] = seen
            continue
        merged[key] = value
    return merged


def _default_family_constraints(
    *,
    family_id: str,
    family_type: str,
    members: Sequence[StartupParameterDefinition],
) -> dict[str, object]:
    lead = members[0]
    semantic_id = str(lead.semantic_parameter_id or _parameter_prefix(lead.parameter_id) or family_id).strip()
    token = semantic_id.lower()
    is_grouped = len(members) > 1
    constraints: dict[str, object] = {
        "ordered_slots": is_grouped,
        "dynamic_activation": is_grouped,
        "activation_semantics": "prefix" if is_grouped else "single",
        "shape_preserving_only": is_grouped,
        "allowed_recipe_types": (
            ["shift_level", "front_load", "tail_load", "flatten_profile"]
            if family_type in {"atomic", "partially_controllable"} and is_grouped
            else ["single_step"]
        ),
    }
    if any(key in token for key in ("time", "position")) and is_grouped:
        constraints["monotonicity"] = "nondecreasing"
    if token in {"pressure_injection", "injection_pressure", "injection_speed"}:
        constraints["coupled_group_id"] = "fill_profile"
        constraints["linked_family_ids"] = ["injection_speed"] if token in {"pressure_injection", "injection_pressure"} else ["pressure_injection"]
    elif token in {"hold_pressure", "hold_time"}:
        constraints["coupled_group_id"] = "packing_profile"
        constraints["linked_family_ids"] = ["hold_time"] if token == "hold_pressure" else ["hold_pressure"]
    return constraints


def _slot_sort_key(token: str | None) -> tuple:
    raw = str(token or "").strip().lower()
    if not raw:
        return ("",)
    key: list[object] = []
    current = ""
    for char in raw:
        if char.isdigit():
            if current and not current[-1].isdigit():
                key.append(current)
                current = ""
            current += char
        else:
            if current and current[-1].isdigit():
                key.append(int(current))
                current = ""
            current += char
    if current:
        key.append(int(current) if current.isdigit() else current)
    return tuple(key)
