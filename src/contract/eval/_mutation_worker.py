import contextlib
import io
import json
from pathlib import Path
import socket
import sys
import traceback

import icontract


class LimitedText(io.StringIO):
    def write(self, value):
        if self.tell() + len(value) > 1024 * 1024:
            raise RuntimeError("Program output exceeded the 1 MiB capture limit")
        return super().write(value)


def main():
    request = json.loads(Path(sys.argv[1]).read_text())
    namespace = {"__name__": "__main__", "__file__": request["program"]}
    kinds = {}
    seen_target = None
    reached = {"require": 0, "ensure": 0}
    failed = []
    violations = {}

    def profile(frame, event, value):
        nonlocal seen_target
        target = namespace.get(request["target"])
        if callable(target) and target is not seen_target:
            seen_target = target
            for kind, contracts in [
                ("ensure", getattr(target, "__postconditions__", [])),
                (
                    "require",
                    [
                        item
                        for group in getattr(target, "__preconditions__", [])
                        for item in group
                    ],
                ),
            ]:
                for contract in contracts:
                    code = getattr(contract.condition, "__code__", None)
                    if code is not None:
                        kinds[code] = kind
        kind = kinds.get(frame.f_code)
        if kind and event == "call":
            reached[kind] += 1
        elif kind and event == "return" and value is False:
            failed.append(kind)
        if (
            event == "return"
            and frame.f_code.co_name == "_create_violation_error"
            and frame.f_globals.get("__name__") == "icontract._checkers"
            and isinstance(value, BaseException)
        ):
            condition = getattr(frame.f_locals.get("contract"), "condition", None)
            violations[id(value)] = (
                value,
                kinds.get(getattr(condition, "__code__", None)),
            )

    def deny_network(*args, **kwargs):
        raise RuntimeError("Network access is disabled during mutation evaluation")

    socket.socket.connect = deny_network
    socket.socket.connect_ex = deny_network
    socket.create_connection = deny_network
    socket.getaddrinfo = deny_network
    stdout, stderr = LimitedText(), LimitedText()
    state = "normal"
    exit_code = 0
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        sys.stdin = io.StringIO(request["input"])
        try:
            source = Path(request["program"]).read_text()
            if request["trace_contracts"]:
                sys.setprofile(profile)
            exec(compile(source, request["program"], "exec"), namespace)
        except SystemExit as error:
            exit_code = (
                error.code
                if isinstance(error.code, int)
                else (0 if error.code is None else 1)
            )
            state = "normal" if exit_code == 0 else "program_failure"
        except icontract.errors.ViolationError as error:
            recorded = violations.get(id(error))
            kind = recorded[1] if recorded and recorded[0] is error else None
            state = (
                "postcondition_violation"
                if kind == "ensure"
                else (
                    "precondition_violation"
                    if kind == "require"
                    else "unattributed_contract_violation"
                )
            )
            exit_code = 1
            traceback.print_exc()
        except BaseException:
            state = "program_failure"
            exit_code = 1
            traceback.print_exc()
        finally:
            sys.setprofile(None)
    print(
        json.dumps(
            {
                "state": state,
                "stdout": stdout.getvalue(),
                "stderr": stderr.getvalue(),
                "program_exit_code": exit_code,
                "predicate_calls": reached,
                "failed_predicate_kinds": failed,
            }
        )
    )


if __name__ == "__main__":
    main()
