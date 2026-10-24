# piper-kernels

Instructions for coding agents working in this repository.

Start with [README.md](README.md) for the public entry point and topic map.
[Development](docs/development.md) covers architecture, validation, and testing.
API docstrings describe individual contracts; the guides explain how APIs work
together.

## Engineering

- Lead with the problem: what goes wrong, for whom, how often, and how badly.
  Design for production needs and foreseeable uses. Hypothetical problems
  usually need a note or a contract before they need a mechanism.
- Minimize how much a reader must understand to make a correct change.
  Make ownership and invariants clear, and hide implementation details behind
  useful interfaces. Consider the whole design: broader changes, including
  to internal contracts, are welcome when they reduce special cases, coupling,
  or duplicated knowledge. Remove what they supersede. Preserve unrelated
  behavior; make intended behavior changes explicit.
- State caller contracts in docs and docstrings. Validate supported metadata
  and configuration at responsible boundaries. Within those boundaries, rely
  on established contracts: avoid checking the same requirement redundantly or
  adding defensive machinery for misuse the contract already forbids. Follow
  the validation contract below for numerical tensor contents.
- Use strong types and let pyright prove what it can. Contain loose types at
  boundaries and explain why they are needed.
- Base performance decisions on hardware and workload properties, supported by
  representative measurements. Keep specialized assumptions explicit; see
  [performance tuning](docs/development.md#performance-tuning).
- Keep documentation clear, concise, and coherent with the current library.
  Rework the organization and examples when concepts change, consolidate
  overlap, and remove obsolete material. Prefer one authoritative explanation
  with links from other contexts. Release history belongs in `CHANGELOG.md`.

## Contracts to preserve

- [Attention](docs/attention.md) and [weights](docs/weights.md) describe public
  shapes, numerical behavior, mutation, and backend support. Check the relevant
  contract before changing an operator or compiler rewrite.
- [Validation](docs/development.md#validation-contract) applies to inference,
  dispatch, compiler/fake paths, and weight storage wrapping and lifecycle.
  Do not add implicit tensor-content validation or readbacks to these paths.
  Preserve required metadata checks and algorithmic guards for valid inputs;
  numerical preconditions do not promise runtime rejection.
- The [execution-plan guide](docs/development.md#execution-plans-and-schedules)
  covers backend conventions, optional imports, and compiler source tracking.
  Read it before changing backend selection or moving kernel definitions.

Follow the [code conventions](docs/development.md#code-conventions) for naming,
types, and internal interfaces. Keep documentation with its
[owner](docs/development.md#documentation-ownership).

## Reviews

Explain the failure, its impact, and where it occurs. Judge the resulting design
and documentation, including ownership and duplicated behavior. Distinguish
failures for supported inputs and missing required metadata checks from violations
of documented numerical preconditions. A new capability needs a foreseeable use;
an API documentation finding should show how the omission causes a caller to act
wrongly or misinterpret the result.

## Checks

Follow [writing tests](docs/development.md#writing-tests) when changing coverage.
Use the [development checks](docs/development.md#checks) for tests, lint,
formatting, types, and hardware regressions. See [VERSIONING.md](VERSIONING.md)
for compatibility and releases.
