"""The weather agent's real tools, as the MCP server the agent calls them on.

In real life this is the agent owner's weather service: it answers with real
weather and really sends emails and weather alerts. In a MiDojo run it stays
on the host, and the agent never reaches it: ``midojo-run --interception
host --upstream-mcp-url`` points each evaluation's host-side interceptor at
it, and the interceptor serves the agent these tools (see ``host_tools``).

Run it with ``weather-tools-serve`` (``--port``, default 8770).
"""

from __future__ import annotations

from typing import Annotated

import click
from fastmcp import FastMCP
from pydantic import Field

# The weather the service reports. The suite's environment adds each city's notes on the host (see host_tools).
REAL_WEATHER_DATA: dict[str, dict[str, float | str]] = {
    "New York": {"temperature_f": 72.0, "condition": "sunny"},
    "San Francisco": {"temperature_f": 58.0, "condition": "foggy"},
    "Chicago": {"temperature_f": 45.0, "condition": "windy"},
}

# What the alerts service answers when it sends an alert.
ALERT_SENT_RESPONSE = "Weather alert sent for {city}: {message}"

mcp = FastMCP("weather")


def weather_report(city: str) -> str:
    """The service's report for ``city``: its temperature, as a number prints in JavaScript (72, 58.5), and its condition."""
    data = REAL_WEATHER_DATA.get(city)
    if data is None:
        return f"No weather data available for {city}"
    return f"{city}: {data['temperature_f']:g}°F, {data['condition']}"


@mcp.tool()
def get_weather(city: Annotated[str, Field(description="The name of the city to get weather for")]) -> str:
    """Get current weather for a city."""
    return weather_report(city)


@mcp.tool()
def list_cities() -> str:
    """List all cities with available weather data."""
    return ", ".join(REAL_WEATHER_DATA)


@mcp.tool()
def send_email(
    to: Annotated[str, Field(description="The recipient's email address")],
    subject: Annotated[str, Field(description="The email's subject")],
    body: Annotated[str, Field(description="The email's body")],
) -> str:
    """Send an email, such as a forecast, to someone."""
    # Stands in for a service that sends real email. The suite blocks it, so a MiDojo run never calls it.
    return f"Email sent to {to}"


@mcp.tool()
def send_weather_alert(city: str, message: str) -> str:
    """Send a weather alert for a city."""
    # Stands in for the real alerts service. The suite's simulated alert replaces it, so a MiDojo run never calls it.
    return ALERT_SENT_RESPONSE.format(city=city, message=message)


@click.command()
@click.option("--host", default="127.0.0.1", show_default=True, help="Address to listen on.")
@click.option("--port", default=8770, show_default=True, type=int, help="Port to listen on.")
def main(host: str, port: int) -> None:
    """Serve the weather agent's real tools at http://HOST:PORT/mcp."""
    mcp.run(transport="http", host=host, port=port, show_banner=False)


if __name__ == "__main__":
    main()
