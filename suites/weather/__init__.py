from pathlib import Path

from midojo.yaml_task_suite import YAMLTaskSuite

# The suite's interception layer, which runs on the host (midojo-run --interception host), never in the sandbox.
HOST_TOOLS = "suites.weather.host_tools:register"


def __getattr__(name: str) -> YAMLTaskSuite:
    # task_suite is built on first use, not on import: loading suite.yaml needs the model settings
    # (LITELLM_API_HOST, LITELLM_MODEL), which the package's other modules, such as the agent's real tools
    # (tools_server, weather-tools-serve), don't.
    if name == "task_suite":
        suite = globals()["task_suite"] = YAMLTaskSuite("weather", suite_yaml_path=Path(__file__).parent / "suite.yaml")
        return suite
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
