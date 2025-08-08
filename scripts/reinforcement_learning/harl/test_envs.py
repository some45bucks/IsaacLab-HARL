import subprocess
import json
import re

# ANSI color codes
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BOLD = "\033[1m"
RESET = "\033[0m"

ANSI_ESCAPE = re.compile(r'\x1B[@-_][0-?]*[ -/]*[@-~]')

def load_configs(json_path="test_envs.json"):
    with open(json_path, "r") as f:
        return json.load(f)

def run_config(name, script_path, args, successes, failures):
    command = ["python3", script_path] + args
    print(f"\n{YELLOW}Running: {BOLD}{name}{RESET}")
    print(f"{BOLD}Command:{RESET} {' '.join(command)}")

    try:
        subprocess.run(command, capture_output=True, text=True, check=True)
        success_str = f"SUCCESS, ENV: {name}"
        len_str = len(success_str)
        print(GREEN, BOLD)
        print("=" * len_str)
        print(success_str)
        print("=" * len_str + RESET)
        successes.append(name)
    except subprocess.CalledProcessError as e:
        clean_err = ANSI_ESCAPE.sub('', e.stderr)
        failure_str = f"FAILURE, ENV: {name}"
        len_str = len(failure_str)
        print(RED, BOLD, "=" * len_str)
        print(clean_err)
        print(failure_str)
        print("=" * len_str + RESET)
        failures.append((name, clean_err))

def print_summary(successes, failures):
    print(f"\n{BOLD}===== SUMMARY =====")
    print(f"{GREEN}Successes: {len(successes)}")
    for name in successes:
        print(f"  - {name}")

    print(f"\n{RED}Failures: {len(failures)}")
    for name, _ in failures:
        print(f"  - {name}")
    print(RESET)
    
def main():
    configs = load_configs()
    successes = []
    failures = []

    for config in configs:
        run_config(config["name"], config["script"], config["args"], successes, failures)

    print_summary(successes, failures)

if __name__ == "__main__":
    main()
