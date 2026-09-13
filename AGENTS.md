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
