---
trigger: always_on
---

# Core Project Rules

System rules and constraints for Antigravity agents operating in this workspace.

---

## 1. Environment & Stack Constraints
* **Language/Runtime:** Python 3.11+, Node.js 20 LTS
* **Formatting:** Enforce strict PEP 8 compliance for Python; use Prettier for web assets.
* **Imports:** Always prefer explicit imports over wildcard `from module import *` patterns.

---

## 2. Execution Guardrails
* **Test Isolation:** Never execute unit tests that mutate live database states. Use the defined SQLite memory mock fixtures.
* **Code Modification:** When refactoring, do not strip existing code comments or inline documentation unless explicitly requested.
* **Errors:** If a terminal command returns a non-zero exit code during an execution step, halt the trajectory immediately and present the stack trace before attempting an automated fix.

---

## 3. Artifact Handover Preference
* **Documentation:** All public-facing modules must contain complete docstrings summarizing parameters, return types, and exceptions.
* **Plans:** Generate a brief execution checklist inside the Agent Manager before running complex, multi-file structural refactors.