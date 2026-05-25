from __future__ import annotations

import json


def test_load_simulation_steps_derives_defect_severities(tmp_path) -> None:
    from moldvision.predictive.simulator import load_simulation_steps

    scenario_path = tmp_path / "scenario.json"
    scenario_path.write_text(
        json.dumps(
            [
                {
                    "step_id": "shot_01",
                    "metric_value": 0.42,
                    "current_parameter_values": {"injection_pressure": 850.0},
                    "defect_metrics": {
                        "Flash": {
                            "component_severity": 0.31,
                            "duration_ratio": 0.42,
                        }
                    },
                }
            ]
        ),
        encoding="utf-8",
    )

    steps = load_simulation_steps(
        scenario_path,
        default_metric_id="duration_ratio",
        default_threshold=0.10,
    )

    assert len(steps) == 1
    assert steps[0].metric_id == "duration_ratio"
    assert steps[0].threshold == 0.10
    assert steps[0].defect_severities == {"flash": 0.31}
    assert steps[0].defect_metrics["flash"]["duration_ratio"] == 0.42


def test_load_simulation_steps_accepts_jsonl(tmp_path) -> None:
    from moldvision.predictive.simulator import load_simulation_steps

    scenario_path = tmp_path / "scenario.jsonl"
    scenario_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "metric_id": "duration_ratio",
                        "metric_value": 0.20,
                        "current_parameter_values": {"injection_pressure": 850.0},
                        "defect_metrics": {},
                    }
                ),
                json.dumps(
                    {
                        "metric_id": "component_severity",
                        "metric_value": 0.50,
                        "current_parameter_values": {"injection_pressure": 860.0},
                        "defect_metrics": {},
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )

    steps = load_simulation_steps(
        scenario_path,
        default_metric_id="duration_ratio",
        default_threshold=0.10,
    )

    assert [step.step_id for step in steps] == ["step_001", "step_002"]
    assert [step.metric_id for step in steps] == ["duration_ratio", "component_severity"]


def test_evaluate_expectations_checks_focus_label_and_parameters() -> None:
    from moldvision.predictive.simulator import (
        SimulationMachineParameter,
        SimulationSuggestion,
        evaluate_expectations,
    )

    suggestion = SimulationSuggestion(
        triggered_by_metric="duration_ratio",
        triggered_value=0.4,
        threshold_value=0.1,
        parameters=(
            SimulationMachineParameter(
                name="Injection Pressure",
                unit="bar",
                current_value=850.0,
                suggested_value=810.0,
                confidence=0.8,
                parameter_id="injection_pressure",
            ),
        ),
        summary="test",
        urgency="warning",
    )

    outcome = evaluate_expectations(
        {
            "must_trigger": True,
            "focus_label": "Flash",
            "suggested_parameter_ids_any": ["injection_pressure", "hold_pressure"],
            "max_suggestions": 2,
        },
        triggered=True,
        focus_label="flash",
        suggestion=suggestion,
    )

    assert outcome["checked"] is True
    assert outcome["passed"] is True
    assert outcome["failures"] == []


def test_format_simulation_report_renders_expectation_failure() -> None:
    from moldvision.predictive.simulator import SimulationStepResult, format_simulation_report

    report = format_simulation_report(
        [
            SimulationStepResult(
                step_id="shot_01",
                metric_id="duration_ratio",
                metric_value=0.4,
                threshold=0.1,
                triggered=False,
                focus_label="flash",
                baseline_quality_score=0.55,
                predicted_label_signals={"flash": 0.33},
                suggestion=None,
                expectations={
                    "checked": True,
                    "passed": False,
                    "failures": ["must_trigger expected True, got False"],
                },
            )
        ]
    )

    assert "[shot_01] duration_ratio=0.400 threshold=0.100 -> NO_SUGGESTION" in report
    assert "Expectations: FAIL" in report
    assert "must_trigger expected True, got False" in report
