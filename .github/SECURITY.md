# Security Policy

## Supported versions

Lacuna is a research tool under active development; only the latest released
version on [PyPI](https://pypi.org/project/lacuna-pockets/) and the `main`
branch receive fixes.

| Version | Supported |
|---------|-----------|
| 0.3.x   | Yes |
| < 0.3   | No |

## Scope

Lacuna runs locally against structure files (PDB/mmCIF) you provide or it
downloads; it is not a network service and does not handle user credentials
or remote sessions. Reports in scope include things like:

- Arbitrary code execution via a crafted structure file or CLI input
- Unsafe deserialization (including anything loaded via `pickle` or similar)
- Path traversal or unsafe file writes from untrusted input
- Vulnerable dependencies pulled in by `pyproject.toml`

## Reporting a vulnerability

Preferred: use GitHub's private vulnerability reporting for this repository
(**Security** tab -> **Report a vulnerability**).

Alternative: email clayton@tamu.edu with a description of the issue and
steps to reproduce. Please don't open a public issue for suspected security
problems until there's been a chance to assess and, if needed, fix them.

This is a solo-maintained project, so response times are best-effort, not
guaranteed on an SLA. You'll get an acknowledgment and, if the report is
valid, a fix or mitigation plan.
