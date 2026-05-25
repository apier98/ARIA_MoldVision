# Predictive Simulator

`moldvision predictive simulate` replays synthetic startup-assistant states
against a trained startup-suggestion bundle before the bundle is published to
MoldPilot.

The simulator is intended for the ARIA operator who is producing suggestion
bundles and wants to verify:

- which defect label the bundle will focus on
- whether a watched metric would trigger a suggestion
- which machine parameters would be suggested
- whether the bundle behaves as expected on known startup situations

## Command

```powershell
moldvision predictive simulate `
  --bundle runs\suggest-v1\deploy\startup-suggestion-v1.0.0.sugbundle `
  --scenario scenarios\startup_replay.json `
  --output-format text `
  --json-out runs\suggest-v1\simulation_report.json `
  --closed-loop-assume-apply
```

Accepted bundle inputs:

- unpacked bundle directory
- packed `.sugbundle`

Accepted scenario inputs:

- `.json` array of replay steps
- `.jsonl` one replay step per line

## Scenario shape

Each replay step mirrors the live information used by the MoldPilot Startup
Assistant:

```json
{
  "step_id": "shot_01",
  "metric_id": "duration_ratio",
  "metric_value": 0.42,
  "threshold": 0.10,
  "current_parameter_values": {
    "injection_pressure": 850.0,
    "melt_temperature": 230.0
  },
  "defect_metrics": {
    "flash": {
      "component_severity": 0.31,
      "duration_ratio": 0.42,
      "multiplicity_term": 0.08
    }
  },
  "expected": {
    "must_trigger": true,
    "focus_label": "flash",
    "suggested_parameter_ids_any": ["injection_pressure", "injection_speed"]
  }
}
```

Notes:

- `metric_id` defaults to `duration_ratio` if omitted
- `threshold` defaults to `0.10` if omitted
- `defect_severities` is optional; when missing it is derived from
  `defect_metrics[*].component_severity`
- `expected` is optional and is used for regression-style validation
- When `--closed-loop-assume-apply` is used, the simulator carries forward the
  previous step's parameter state and applies suggested parameter updates
  automatically. Use this to simulate an operator who applies the assistant's
  recommended setpoints.
- Per step, `parameter_state_mode` can be used to control how the step's
  `current_parameter_values` interacts with the carried-forward state when
  `--closed-loop-assume-apply` is enabled:
  - `carry` (default): ignore `current_parameter_values` after step 1 and use the carried state
  - `replace`: use `current_parameter_values` exactly for this step
  - `merge`: start from carried state and override keys present in `current_parameter_values`

## Output

The default text output is operator-oriented. For each replay step it prints:

- watched metric value and threshold
- chosen focus label
- baseline predicted quality
- predicted per-label signals from the bundle
- suggested parameter changes
- expectation pass/fail, if provided

The simulator exits with:

- `0` when replay succeeded and no expectations failed
- `2` when the bundle or scenario is invalid
- `3` when replay succeeded but one or more expectations failed

## Scope

The first version simulates the bundle's Tier 1 logic only.

It does **not** emulate:

- Tier 0 rule-based fallback
- Tier 2 online GP adaptation
- the MoldPilot UI workflow

This is deliberate: the tool is for validating the offline prior that
MoldVision produces before it is deployed.
