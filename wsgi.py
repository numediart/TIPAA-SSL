import logging

from app import init_app

logger = logging.getLogger(__name__)

app = init_app()
debug = True

if __name__ == "__main__":
    if debug is True:
        logger.warning("Running in debug mode")
    app.run(debug=debug, host="127.0.0.1", port=8001)
