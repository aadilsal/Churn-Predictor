"""Download Telco Customer Churn dataset."""

import hashlib
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

from src.utils.config import config
from src.utils.logging import logger


def download_telco_churn_dataset(
    output_path: Optional[Path] = None, force_download: bool = False
) -> Path:
    """Download the Telco Customer Churn dataset.

    Args:
        output_path: Path to save the dataset (default: data/raw/telco_churn.csv)
        force_download: Force re-download even if file exists

    Returns:
        Path to downloaded dataset
    """
    if output_path is None:
        output_path = config.data_raw_path / "telco_churn.csv"

    # Create directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Check if file already exists
    if output_path.exists() and not force_download:
        logger.info(f"Dataset already exists at {output_path}")
        return output_path

    # Dataset URL
    dataset_url = config.get(
        "data.dataset_url",
        "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv",
    )

    logger.info(f"Downloading dataset from {dataset_url}")

    try:
        # Download dataset
        response = requests.get(dataset_url, timeout=30)
        response.raise_for_status()

        # Save to file
        with open(output_path, "wb") as f:
            f.write(response.content)

        logger.info(f"Dataset downloaded successfully to {output_path}")

        # Verify it's a valid CSV
        df = pd.read_csv(output_path)
        logger.info(f"Dataset shape: {df.shape}")
        logger.info(f"Columns: {list(df.columns)}")

        # Calculate and log file hash for versioning
        file_hash = _calculate_file_hash(output_path)
        logger.info(f"Dataset hash (MD5): {file_hash}")

        return output_path

    except requests.RequestException as e:
        logger.error(f"Failed to download dataset: {e}")
        raise
    except pd.errors.ParserError as e:
        logger.error(f"Downloaded file is not a valid CSV: {e}")
        raise


def _calculate_file_hash(file_path: Path) -> str:
    """Calculate MD5 hash of a file.

    Args:
        file_path: Path to file

    Returns:
        MD5 hash string
    """
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def main() -> None:
    """Main function to download dataset."""
    logger.info("Starting dataset download...")
    output_path = download_telco_churn_dataset()
    logger.info(f"Dataset ready at: {output_path}")


if __name__ == "__main__":
    main()
