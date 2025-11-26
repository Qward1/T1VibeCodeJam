import json
import subprocess
import tempfile
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional


DOCKER_CONFIG = {
    "python": {"image": "interview-runner-python", "ext": "py"},
    "javascript": {"image": "interview-runner-js", "ext": "js"},
    "fullstack": {"image": "interview-runner-fullstack", "ext": "js"},
    "cpp": {"image": "interview-runner-cpp", "ext": "cpp"},
}


def run_code_in_docker(language: str, code: str, function_name: str, tests: List[Dict[str, Any]], timeout: int = 10):
    lang = (language or "python").lower()
    cfg = DOCKER_CONFIG.get(lang)
    if not cfg:
        raise ValueError(f"Unsupported language: {language}")
    image = cfg["image"]
    ext = cfg["ext"]
    with tempfile.TemporaryDirectory() as tmpdir:
        workdir = Path(tmpdir)
        code_path = workdir / f"user_code.{ext}"
        tests_path = workdir / "tests.json"
        code_path.write_text(code, encoding="utf-8")
        tests_path.write_text(json.dumps({"function_name": function_name, "tests": tests}, ensure_ascii=False), encoding="utf-8")
        cmd = [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{workdir}:/home/runner/app",
            image,
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout)
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr or proc.stdout)
        try:
            data = json.loads(proc.stdout.strip() or "{}")
        except Exception as exc:
            raise RuntimeError(f"Failed to parse runner output: {proc.stdout}") from exc
        return data


def _run_code_locally_python(code: str, function_name: str, tests: List[Dict[str, Any]], timeout: int = 5) -> Dict[str, Any]:
    """
    Упрощённый локальный раннер для python на случай отсутствия Docker.
    Запускает код пользователя в отдельном процессе python, читает tests.json и печатает JSON результатов.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        workdir = Path(tmpdir)
        code_path = workdir / "user_code.py"
        tests_path = workdir / "tests.json"
        runner_path = workdir / "runner.py"
        code_path.write_text(code, encoding="utf-8")
        tests_path.write_text(json.dumps({"function_name": function_name, "tests": tests}, ensure_ascii=False), encoding="utf-8")
        runner_code = f"""
import json, importlib.util, sys, traceback
from pathlib import Path

def load_function(module_path: Path, fn_name: str):
    spec = importlib.util.spec_from_file_location("user_code", module_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["user_code"] = mod
    assert spec.loader
    spec.loader.exec_module(mod)
    return getattr(mod, fn_name)

def main():
    workdir = Path("{workdir.as_posix()}")
    tests_path = workdir / "tests.json"
    code_path = workdir / "user_code.py"
    result = {{"tests": [], "error": None}}
    try:
        data = json.loads(tests_path.read_text(encoding="utf-8"))
        fn_name = data.get("function_name")
        cases = data.get("tests", [])
        fn = load_function(code_path, fn_name)
        for idx, case in enumerate(cases):
            inputs = case.get("input", [])
            try:
                out = fn(*inputs) if isinstance(inputs, list) else fn(inputs)
                result["tests"].append({{"name": case.get("name", f"test_{{idx+1}}"), "passed": out == case.get("expected"), "output": out, "expected": case.get("expected")}})
            except Exception as exc:
                result["tests"].append({{"name": case.get("name", f"test_{{idx+1}}"), "passed": False, "error": str(exc), "trace": traceback.format_exc()}})
    except Exception as exc:
        result["error"] = str(exc)
    print(json.dumps(result, ensure_ascii=False))

if __name__ == "__main__":
    main()
"""
        runner_path.write_text(runner_code, encoding="utf-8")
        proc = subprocess.run(
            [sys.executable, "-u", str(runner_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            env={**os.environ},
        )
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr or proc.stdout)
        try:
            return json.loads(proc.stdout.strip() or "{}")
        except Exception as exc:
            raise RuntimeError(f"Failed to parse local runner output: {proc.stdout}") from exc


def run_code_with_fallback(language: str, code: str, function_name: str, tests: List[Dict[str, Any]], timeout: int = 10, allow_local: bool = True) -> Dict[str, Any]:
    """
    Сначала пробуем Docker. Если Docker недоступен и язык python — пробуем локальный раннер.
    """
    result = {}
    try:
        result = run_code_in_docker(language, code, function_name, tests, timeout=timeout)
    except Exception as exc:
        msg = str(exc).lower()
        docker_down = "docker" in msg and ("pipe" in msg or "daemon" in msg or "connect" in msg)
        if allow_local and docker_down and language.lower() in ("python", "py"):
            result = _run_code_locally_python(code, function_name, tests, timeout=timeout)
        else:
            raise
    try:
        rtests = result.get("tests", [])
        for idx, orig in enumerate(tests):
            if idx < len(rtests):
                rtests[idx]["input"] = rtests[idx].get("input", orig.get("input"))
                if "expected" not in rtests[idx]:
                    rtests[idx]["expected"] = orig.get("expected")
        result["tests"] = rtests
    except Exception:
        pass
    return result
