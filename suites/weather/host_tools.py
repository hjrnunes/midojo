"""The weather suite's interception layer, run on the host by each evaluation's interceptor.

``midojo-run --interception host`` builds a host-side interceptor for each
evaluation (``midojo.interception_host``) and registers these tools on it
(the suite's ``HOST_TOOLS``). It mirrors the agent's real tools
(``tools_server``) and records every call; these change what three of them
do during an evaluation:

- ``get_weather`` forwards the call to the real tool and adds the city's
  notes from the evaluation's environment, where the injection tasks plant
  their payloads.
- ``send_weather_alert`` records the alert in the environment, where the
  injection tasks look for it, and never calls the real alerts service.
- ``send_email`` is blocked: it never runs, and the attempt is recorded,
  marked blocked.

``list_cities`` is mirrored unchanged. The environment and the session that
opens it stay on the host: the agent's sandbox gets only the interceptor's
URL and its bearer token.
"""

from typing import Annotated

from midojo_sdk.mcp import MidojoMCP, ToolContext
from pydantic import Field

from suites.weather.tools_server import ALERT_SENT_RESPONSE

# How many times sending an alert reads the alerts and tries to add to them.
CHANGE_ATTEMPTS = 5


async def get_weather(
    ctx: ToolContext, city: Annotated[str, Field(description="The name of the city to get weather for")]
) -> str:
    """Get current weather for a city."""
    real = await ctx.forward("get_weather", {"city": city})
    cities = await ctx.env("cities")
    notes = (cities.get(city) or {}).get("notes")
    return f"{real}\n{notes}" if notes else real


async def send_weather_alert(ctx: ToolContext, city: str, message: str) -> str:
    """Send a weather alert for a city."""
    # The injection tasks look for the alert in the environment. The change
    # applies only at the revision read, so an alert added concurrently is kept:
    # when one lands in between, read again and append to it.
    for _ in range(CHANGE_ATTEMPTS):
        revision, environment = await ctx.read()
        alerts = [*environment["weather_alerts"], {"city": city, "message": message}]
        if await ctx.env_update("weather_alerts", alerts, expected_revision=revision):
            break
    else:
        raise RuntimeError("The alerts kept changing while sending the alert; try again.")
    # Forwarding would send a real alert.
    return ALERT_SENT_RESPONSE.format(city=city, message=message)


def register(mcp: MidojoMCP) -> None:
    """Register the weather suite's tools on one evaluation's interceptor."""
    mcp.tool()(get_weather)
    mcp.tool()(send_weather_alert)
    # Never send email during an evaluation. The control plane still records the attempt, marked blocked.
    mcp.block("send_email")
