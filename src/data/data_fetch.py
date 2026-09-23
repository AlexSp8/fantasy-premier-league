
from typing import Any
from pathlib import Path
import json
import urllib.error
import logging

import pandas as pd
import requests

from src.data import file_utils


logger = logging.getLogger(__name__)

def fetch_json_data(
    url: str,  file_path: str | Path | None = None
    ) -> dict[str, Any] | list[dict[str, Any]]:
    """Fetch JSON data from URL or existing file.
    Args:
        url: The endpoint URL.
        file_path (optional): The path to the JSON file.
    Returns:
        The parsed JSON data as a dictionary or list of dictionaries."""

    if file_utils.file_exists(file_path=file_path):
        return fetch_json_data_from_file(file_path)

    return fetch_json_data_from_url(url)

def fetch_json_data_from_url(url: str) -> dict[str, Any] | list[dict[str, Any]]:
    """Fetch JSON data from a URL.
    Args:
        url: The endpoint URL.
    Returns:
        The parsed JSON data as a dictionary or list of dictionaries.
    Raises:
        ConnectionError: If connection to the url cannot be established.
        ValueError: If the json file cannot be decoded."""

    data_dict: dict[str, Any] | list[dict[str, Any]] = {}
    try:
        response: requests.Response = requests.get(url)
        response.raise_for_status()
        # logger.info(f"Response status: {response.status_code}")
        data_dict = response.json()
        logger.info(f"JSON data fetched from url: {url}")
    except requests.exceptions.RequestException as e:
        raise ConnectionError(f"Failed to fetch data from {url}") from e
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON returned from {url}") from e

    return data_dict

def fetch_json_data_from_file(file_path: str | Path) -> dict[str, Any] | list[dict[str, Any]]:
    """Fetch JSON data from a JSON file.
    Args:
        file_path: The path to the JSON file.
    Returns:
        The parsed JSON data as a dictionary or list of dictionaries.
    Raises:
        FileNotFoundError: If the file is not found.
        ValueError: If the json file cannot be decoded."""

    target_path = Path(file_path)

    data_dict: dict[str, Any] | list[dict[str, Any]] = {}
    try:
        with open(file=target_path, mode='r', encoding="utf-8") as f:
            data_dict = json.load(fp=f)
        logger.info(f"JSON data fetched from file: {target_path.resolve()}")
    except FileNotFoundError as e:
        raise FileNotFoundError(f"File does not exist at {target_path.resolve()}") from e
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in file: {target_path.resolve()}") from e

    return data_dict

def fetch_csv_data(url: str, file_path: str | Path | None = None) -> pd.DataFrame:
    """Fetch CSV data from URL or existing file.
    Args:
        url: The endpoint URL.
        file_path (optional): The path to the CSV file.
    Returns:
        The parsed CSV data as a pd.DataFrame."""

    if file_utils.file_exists(file_path=file_path):
        return fetch_csv_data_from_file(file_path)

    return fetch_csv_data_from_url(url)

def fetch_csv_data_from_url(url: str) -> pd.DataFrame:
    """Fetch CSV data from a URL.
    Args:
        url: The endpoint URL.
    Returns:
        The parsed CSV data as a pd.DataFrame.
    Raises:
        ConnectionError: If connection to the url cannot be established."""

    df = pd.DataFrame()
    try:
        df = pd.read_csv(url)
        logger.info(f"CSV data fetched from url: {url}")
    except (urllib.error.URLError, pd.errors.ParserError) as e:
        raise ConnectionError(f"Failed to fetch data from {url}") from e

    return df

def fetch_csv_data_from_file(file_path: str | Path) -> pd.DataFrame:
    """Fetch CSV data from a CSV file.
    Args:
        file_path: The path to the CSV file.
    Returns:
        The parsed CSV data as a pd.DataFrame.
    Raises:
        FileNotFoundError: If the file is not found.
        ValueError: If the csv file cannot be decoded."""

    target_path = Path(file_path)

    df = pd.DataFrame()
    try:
        df = pd.read_csv(file_path)
        logger.info(f"CSV data fetched from file: {target_path.resolve()}")
    except FileNotFoundError as e:
        raise FileNotFoundError(f"File does not exist at {target_path.resolve()}") from e
    except pd.errors.ParserError as e:
        raise ValueError(f"Failed to fetch data from {target_path.resolve()}") from e

    return df

def fetch_multiple_csv_data(
    prefix: str, suffix: str, identifiers: list[str],
    file_paths: list[str] | list[Path] | None = None,
    ) -> dict[str, pd.DataFrame]:
    """Fetch CSV data from multiple URLs or existing files
    with common prefix/suffix and unique identifiers.
    Args:
        prefix: The prefix to the CSV identifiers.
        suffix: The suffix to the CSV identifiers.
        identifiers: The list of CSV name identifiers.
        file_paths (optional): The list of paths to the CSV files.
    Returns:
        The parsed CSV data as a dictionary of pd.DataFrame.
    Raises:
        ValueError: If a csv file cannot be decoded or if there is a length mismatch
            between identifiers and file_paths.
        ConnectionError: If connection to a url cannot be established.
        FileNotFoundError: If a file is not found."""

    files = file_paths
    if file_paths is None:
        files = [None]*len(identifiers)
    elif len(identifiers) != len(file_paths):
        raise ValueError("Length mismatch between identifiers and file_paths!")

    urls = [(prefix+i+suffix) for i in identifiers]

    df_dict: dict[str, pd.DataFrame] = {}
    for i, url, file_path in zip(identifiers, urls, files):
        try:
            df_dict[i] = fetch_csv_data(url=url, file_path=file_path)
        except (ConnectionError, FileNotFoundError, ValueError) as e:
            logger.warning(f"Skipped identifier {i}. Reason: {e}")
            continue

    return df_dict
