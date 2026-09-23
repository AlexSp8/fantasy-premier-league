
from typing import Any
from pathlib import Path
import json
import csv
import logging

import pandas as pd

from src.data import file_utils


logger = logging.getLogger(__name__)

def store_data_to_json(
    data: dict[str, Any] | list[dict[str, Any]], file_path: Path | str,
    overwrite: bool = False
    ) -> None:
    """Store data to a JSON file.
    Args:
        data: The data dictionary or list of dictionaries to be saved.
        file_path: The path to the JSON file.
        overwrite (optional): Boolean indicating whether to overwrite an existing file.
    Raises:
        OSError: If the file cannot be written to disk."""

    target_path = Path(file_path)

    if (not overwrite) and file_utils.file_exists(file_path=target_path):
        logger.warning(f"File {target_path.resolve()} already exists. File not saved!")
        return None

    # Create missing parent directories without error raise if the folder exists
    target_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(file=target_path, mode='w', encoding="utf-8") as f:
            json.dump(obj=data, fp=f, indent=4)
        logger.info(f"Data stored to JSON file: {target_path.resolve()}")
    except OSError as e:
        raise OSError(f"Disk write error at {target_path.resolve()}") from e

def save_list_dict_to_csv(
    data_list_dict: list[dict[str, Any]], file_path: Path | str,
    fieldnames: list[str] | None = None, overwrite: bool = False
    ) -> None:
    """Save a list of data dictionaries to a CSV file.
    Args:
        data_list_dict: The list of data dictionaries to be saved.
        file_path: The path to the CSV file.
        fieldnames (optional): List of column names to write.
        overwrite (optional): Boolean indicating whether to overwrite an existing file.
    Raises:
        OSError: If the file cannot be written to disk."""

    target_path = Path(file_path)

    if (not overwrite) and file_utils.file_exists(file_path=target_path):
        logger.warning(f"File {target_path.resolve()} already exists. File not saved!")
        return None

    if len(data_list_dict) == 0:
        logger.warning("The list of data dictionaries is empty. File not saved!")
        return None

    target_path.parent.mkdir(parents=True, exist_ok=True)

    # Infer fieldnames from all keys in all rows of the dictionary if not provided
    if fieldnames is None:
        all_keys = [key for row in data_list_dict for key in row.keys()]
        fieldnames = list(dict.fromkeys(all_keys))

    try:
        with open(file=target_path, mode="w", newline="", encoding="utf-8") as f:
            # extrasaction="ignore" prevents errors if dictionaries have extra keys
            writer = csv.DictWriter(f=f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(data_list_dict)
        logger.info(f"Data stored to CSV file: {target_path.resolve()}")
    except OSError as e:
        raise OSError(f"Disk write error at {target_path.resolve()}") from e

def save_df_to_csv(df: pd.DataFrame, file_path: Path | str, overwrite: bool = False) -> None:
    """Save a pd.DataFrame to a CSV file.
    Args:
        df: The pd.DataFrame to be saved.
        file_path: The path to the CSV file.
        overwrite (optional): Boolean indicating whether to overwrite an existing file.
    Raises:
        OSError: If the file cannot be written to disk."""

    target_path = Path(file_path)

    if (not overwrite) and file_utils.file_exists(file_path=target_path):
        logger.warning(f"File {target_path.resolve()} already exists. File not saved!")
        return None

    target_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        df.to_csv(target_path, index=False)
        logger.info(f"DataFrame stored to CSV file: {target_path.resolve()}")
    except OSError as e:
        raise OSError(f"Disk write error at {target_path.resolve()}") from e
