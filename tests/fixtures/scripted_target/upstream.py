"""The scripted upstream: the real tools behind the interceptor, with fixed results."""

from typing import Annotated, Literal

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from pydantic import Field

WEATHER = {"Boston": "41°F, rain", "New York": "72°F, sunny"}


def scripted_upstream(calls: list[tuple[str, dict]] | None = None) -> FastMCP:
    """The upstream; each call it runs is appended to ``calls`` when given."""
    mcp = FastMCP("scripted-weather")

    def ran(name: str, **args) -> None:
        if calls is not None:
            calls.append((name, args))

    @mcp.tool(output_schema=None)
    def get_weather(city: str) -> str:
        """Current weather for a city."""
        ran("get_weather", city=city)
        return WEATHER[city]

    @mcp.tool(output_schema=None)
    def list_alerts() -> str:
        """Active weather alerts. Takes no arguments."""
        ran("list_alerts")
        return "no alerts"

    @mcp.tool()
    def forecast(city: str, days: Annotated[int, Field(ge=1, le=7)] = 3, units: Literal["C", "F"] = "F") -> dict:
        """A multi-day forecast, as structured content."""
        ran("forecast", city=city, days=days, units=units)
        return {"city": city, "units": units, "highs": [50 + day for day in range(days)]}

    @mcp.tool(output_schema=None)
    def book_shelter(city: str) -> str:
        """Always fails: there are no shelters."""
        ran("book_shelter", city=city)
        raise ToolError(f"no shelters available in {city}")

    return mcp
