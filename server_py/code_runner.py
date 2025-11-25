import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Tuple


class CodeRunError(Exception):
    pass


LANG_CONFIG = {
    "python": {"ext": ".py", "cmd": ["python3", "main.py"]},
    "py": {"ext": ".py", "cmd": ["python3", "main.py"]},
    "javascript": {"ext": ".js", "cmd": ["node", "main.js"]},
    "js": {"ext": ".js", "cmd": ["node", "main.js"]},
}


def run_code(language: str, code: str, tests: List[Dict[str, str]], timeout: int = 5) -> Tuple[List[Dict], str, str]:
    lang_key = language.lower()
    if lang_key not in LANG_CONFIG:
        raise CodeRunError(f"Unsupported language: {language}")
    config = LANG_CONFIG[lang_key]
    ext = config["ext"]
    cmd = config["cmd"]

    results: List[Dict] = []
    aggregate_stdout = ""
    aggregate_stderr = ""

    for idx, test in enumerate(tests):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            src = tmp_path / f"main{ext}"
            src.write_text(code, encoding="utf-8")
            start = time.time()
            try:
                proc = subprocess.run(
                    cmd,
                    cwd=tmp_path,
                    input=str(test.get("input", "")),
                    text=True,
                    capture_output=True,
                    timeout=timeout,
                )
                runtime_ms = int((time.time() - start) * 1000)
                stdout = proc.stdout
                stderr = proc.stderr
                aggregate_stdout += stdout
                aggregate_stderr += stderr
                actual_output = stdout.strip()
                expected_output = str(test.get("output", "")).strip()
                status = "passed" if actual_output == expected_output and proc.returncode == 0 else "failed"
                results.append(
                    {
                        "test_id": idx + 1,
                        "status": status,
                        "actual_output": stdout,
                        "expected_output": test.get("output", ""),
                        "runtime_ms": runtime_ms,
                        "returncode": proc.returncode,
                        "stderr": stderr,
                    }
                )
            except subprocess.TimeoutExpired:
                results.append(
                    {
                        "test_id": idx + 1,
                        "status": "timeout",
                        "actual_output": "",
                        "expected_output": test.get("output", ""),
                        "runtime_ms": int((time.time() - start) * 1000),
                        "returncode": None,
                        "stderr": "Время выполнения превышено",
                    }
                )
    return results, aggregate_stdout, aggregate_stderr
