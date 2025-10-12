import yaml
import subprocess
import subprocess


def load_config(config_path):
    """
    Loads a YAML config file and returns its contents as a dictionary.
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_git_commit():
    """Get the current git commit hash."""
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
    except Exception:
        commit = "unknown"
    return commit


def get_last_commit_for_path(path):
    """Return (commit_hash, commit_timestamp) for the most recent change in path."""
    result = subprocess.run(
        ["git", "log", "-n", "1", "--format=%H %ct", "--", path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True
    )
    if not result.stdout.strip():
        return None, None  # path might be untracked
    commit, timestamp = result.stdout.strip().split()
    return commit, int(timestamp)


def get_experiment_commit_hash(paths):
    """Return the commit hash of the *latest* commit among given paths."""
    commits = []
    for path in paths:
        commit, timestamp = get_last_commit_for_path(path)
        if commit:
            commits.append((timestamp, commit))
    if not commits:
        return None
    latest_commit = max(commits, key=lambda x: x[0])[1]
    return latest_commit
