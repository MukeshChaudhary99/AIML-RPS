import uvicorn

from src.api import app
from src.config import load_app_config


def main():
    config = load_app_config()
    uvicorn.run(
        app,
        host=config.api.host,
        port=config.api.port,
        log_level=config.api.log_level.lower(),
    )


if __name__ == "__main__":
    main()
