import subprocess
from pathlib import Path

RUNNERS = [
    {"name": "interview-runner-python", "path": "docker/runners/python"},
    {"name": "interview-runner-js", "path": "docker/runners/javascript"},
    {"name": "interview-runner-cpp", "path": "docker/runners/cpp"},
    {"name": "interview-runner-fullstack", "path": "docker/runners/fullstack"},
]


def main():
    root = Path(__file__).resolve().parents[1]
    for runner in RUNNERS:
        ctx = root / runner["path"]
        cmd = ["docker", "build", "-t", runner["name"], str(ctx)]
        print("Building", runner["name"], "from", ctx)
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
