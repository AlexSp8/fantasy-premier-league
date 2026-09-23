
from pathlib import Path

def file_exists(file_path: str | Path | None) -> bool:
    """Check if a file exists in a local directory.
    Args:
        file_path: The path to the file.
    Returns:
        A boolean indicating file existence."""

    return (file_path is not None) and (Path(file_path).exists())
