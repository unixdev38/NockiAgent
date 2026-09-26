"""Run the Nocki Agent API server:  python -m app.api"""

from __future__ import annotations

import uvicorn

from app.config import get_settings
from app.logging_config import setup_logging


def main() -> None:
    settings = get_settings()
    setup_logging(settings.log_level.value)
    uvicorn.run(
        "app.api.server:app",
        host=settings.api_host,
        port=settings.api_port,
        log_level=settings.log_level.value.lower(),
    )


if __name__ == "__main__":
    main()
