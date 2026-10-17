"""Tiny MCP server used by the test-suite (stdio transport)."""

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("demo")


@mcp.tool()
def add(a: int, b: int) -> int:
    """Additionne deux entiers."""
    return a + b


@mcp.tool()
def shout(text: str) -> str:
    """Met le texte en majuscules."""
    return text.upper()


@mcp.tool()
def boom() -> str:
    """Lève toujours une erreur."""
    raise RuntimeError("kaboom")


if __name__ == "__main__":
    mcp.run()
