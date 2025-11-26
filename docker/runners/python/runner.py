import json
import importlib.util
import sys
import traceback
from pathlib import Path

def load_function(module_path: Path, fn_name: str):
    spec = importlib.util.spec_from_file_location("user_code", module_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["user_code"] = mod
    assert spec.loader
    spec.loader.exec_module(mod)
    return getattr(mod, fn_name)


def main():
    workdir = Path("/home/runner/app")
    tests_path = workdir / "tests.json"
    code_path = workdir / "user_code.py"
    result = {"tests": [], "error": None}
    try:
        data = json.loads(tests_path.read_text(encoding="utf-8"))
        fn_name = data.get("function_name")
        cases = data.get("tests", [])
        fn = load_function(code_path, fn_name)
        for idx, case in enumerate(cases):
            inputs = case.get("input", [])
            try:
                out = fn(*inputs)
                result["tests"].append({"name": case.get("name", f"test_{idx+1}"), "passed": out == case.get("expected"), "output": out, "expected": case.get("expected")})
            except Exception as exc:
                result["tests"].append({"name": case.get("name", f"test_{idx+1}"), "passed": False, "error": str(exc), "trace": traceback.format_exc()})
    except Exception as exc:
        result["error"] = str(exc)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
