import os

def is_running_in_docker() -> bool:
    """
    Checks if the application is running inside a Docker container.
    """
    return os.path.exists('/.dockerenv')
