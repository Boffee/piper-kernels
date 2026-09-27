# Contributor and review guidance

## Validation contract

New development and code review must follow the public
[validation contract](README.md#validation-contract). It covers inference, dispatch,
compiler/fake paths, and weight wrapper construction and lifecycle operations.

Review changes for implicit tensor-content validation, synchronization, validation kernels,
and allocations that violate this contract. The absence of runtime rejection for numerical
values outside documented preconditions is intentional, not by itself a bug. Missing
metadata checks and incorrect results for valid inputs remain review findings. Preserve
guards that are part of the numerical algorithm, such as handling all-zero dynamic inputs.

## Performance tuning

Real model shapes and sequence lengths are benchmark anchors, not dispatch contracts.
Keep kernels and scheduling rules general. Do not gate optimizations on exact model
dimensions or on the minimum/maximum sequence lengths used in a benchmark sweep.
Use general schedules or rules justified by hardware and workload properties, and
validate them across representative nearby shapes.

## Execution plans and schedules

An `ExecutionPlan` contains the complete resolved choices for one operator invocation:
implementation, preparation/fusion, memory access, and launch settings. It may run one
kernel or several. A `Schedule` describes how work is organized, such as tiles, warps,
pipeline stages, or chunk sizes. Extract a named schedule only when it is independently
selected, reused, or tuned; flat execution plans are valid.

- Define immutable `<Operation>ExecutionPlan` and `<Stage>Schedule` values in `_plan.py`.
  Vendor-specific plan subclasses may use the vendor name. Keep configuration validation
  and reporting with these types, and target/shape selection in `policy.py` or `_policy.py`.
- Use `select_execution_plan(...)` for policy selection from host metadata, and
  `default_execution_plan(...)` for helpers that resolve operand metadata before selection.
- Name runtime parameters and stored plan fields `execution_plan`; local variables may
  use `plan`. Name independently selected schedule parameters `schedule` or `<stage>_schedule`.
- Benchmark metadata uses `execution_plan` for complete choices and `schedule` for an
  individual schedule. Document changes to serialized benchmark fields.
- Keep type and policy modules importable without Triton. Preserve compiler source tracking
  when moving definitions. Follow the validation contract for all metadata checks.
