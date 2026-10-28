from pathlib import Path

from midojo.yaml_task_suite import YAMLTaskSuite

task_suite = YAMLTaskSuite("weather", suite_yaml_path=Path(__file__).parent / "suite.yaml")

# The suite's interception layer, which runs on the host (midojo-run --interception host), never in the sandbox.
HOST_TOOLS = "suites.weather.host_tools:register"
