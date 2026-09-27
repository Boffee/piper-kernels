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
