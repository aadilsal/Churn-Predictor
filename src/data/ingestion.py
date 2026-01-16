"""Data ingestion pipeline."""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from src.data.download_data import download_telco_churn_dataset
from src.data.preprocessing import clean_telco_dataset, validate_data_quality
from src.utils.config import config
from src.utils.logging import logger


def ingest_data(
    force_download: bool = False, save_processed: bool = True
) -> pd.DataFrame:
    """Complete data ingestion pipeline.

    This function:
    1. Downloads raw data (if needed)
    2. Loads raw data
    3. Validates data quality
    4. Cleans data
    5. Saves processed data
    6. Generates data quality report

    Args:
        force_download: Force re-download of raw data
        save_processed: Save processed data to disk

    Returns:
        Cleaned dataframe
    """
    logger.info("=" * 80)
    logger.info("STARTING DATA INGESTION PIPELINE")
    logger.info("=" * 80)

    # Step 1: Download raw data
    raw_data_path = download_telco_churn_dataset(force_download=force_download)

    # Step 2: Load raw data
    logger.info(f"Loading raw data from {raw_data_path}")
    df_raw = pd.read_csv(raw_data_path)
    logger.info(f"Loaded {len(df_raw)} records with {len(df_raw.columns)} columns")

    # Step 3: Validate raw data quality
    logger.info("Validating raw data quality...")
    raw_quality_report = validate_data_quality(df_raw)

    # Step 4: Clean data
    df_clean = clean_telco_dataset(df_raw)

    # Step 5: Validate cleaned data quality
    logger.info("Validating cleaned data quality...")
    clean_quality_report = validate_data_quality(df_clean)

    # Step 6: Save processed data
    if save_processed:
        processed_path = config.data_processed_path / "telco_churn_processed.csv"
        processed_path.parent.mkdir(parents=True, exist_ok=True)

        df_clean.to_csv(processed_path, index=False)
        logger.info(f"Saved processed data to {processed_path}")

    # Step 7: Generate and save quality report
    _save_quality_report(raw_quality_report, clean_quality_report)

    logger.info("=" * 80)
    logger.info("DATA INGESTION PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("=" * 80)

    return df_clean


def _save_quality_report(
    raw_report: dict, clean_report: dict, output_dir: Optional[Path] = None
) -> None:
    """Save data quality report.

    Args:
        raw_report: Quality report for raw data
        clean_report: Quality report for cleaned data
        output_dir: Directory to save report (default: data/processed)
    """
    if output_dir is None:
        output_dir = config.data_processed_path

    output_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "timestamp": datetime.now().isoformat(),
        "raw_data_quality": raw_report,
        "cleaned_data_quality": clean_report,
    }

    # Save as JSON
    json_path = output_dir / "data_quality_report.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    logger.info(f"Saved data quality report to {json_path}")

    # Save human-readable summary
    summary_path = output_dir / "data_quality_summary.txt"
    with open(summary_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("DATA QUALITY REPORT\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Generated: {report['timestamp']}\n\n")

        f.write("RAW DATA:\n")
        f.write(f"  Total Records: {raw_report['total_records']}\n")
        f.write(f"  Total Features: {raw_report['total_features']}\n")
        f.write(f"  Duplicate Records: {raw_report['duplicate_records']}\n")
        f.write(f"  Missing Values: {len(raw_report['missing_values'])} columns\n")

        if raw_report["missing_values"]:
            f.write("\n  Missing Value Details:\n")
            for col, count in raw_report["missing_values"].items():
                pct = (count / raw_report["total_records"]) * 100
                f.write(f"    - {col}: {count} ({pct:.2f}%)\n")

        f.write("\n" + "-" * 80 + "\n\n")

        f.write("CLEANED DATA:\n")
        f.write(f"  Total Records: {clean_report['total_records']}\n")
        f.write(f"  Total Features: {clean_report['total_features']}\n")
        f.write(f"  Duplicate Records: {clean_report['duplicate_records']}\n")
        f.write(f"  Missing Values: {len(clean_report['missing_values'])} columns\n")

        if clean_report["missing_values"]:
            f.write("\n  Remaining Missing Values:\n")
            for col, count in clean_report["missing_values"].items():
                pct = (count / clean_report["total_records"]) * 100
                f.write(f"    - {col}: {count} ({pct:.2f}%)\n")

        f.write("\n" + "=" * 80 + "\n")

    logger.info(f"Saved data quality summary to {summary_path}")


def load_processed_data(file_name: str = "telco_churn_processed.csv") -> pd.DataFrame:
    """Load processed data from disk.

    Args:
        file_name: Name of processed data file

    Returns:
        Processed dataframe
    """
    file_path = config.data_processed_path / file_name

    if not file_path.exists():
        logger.warning(f"Processed data not found at {file_path}")
        logger.info("Running ingestion pipeline...")
        return ingest_data()

    logger.info(f"Loading processed data from {file_path}")
    df = pd.read_csv(file_path)
    logger.info(f"Loaded {len(df)} records")

    return df


def main() -> None:
    """Main function to run data ingestion."""
    df = ingest_data()
    logger.info(f"Final dataset shape: {df.shape}")
    logger.info(f"Churn rate: {df['Churn'].mean():.2%}")


if __name__ == "__main__":
    main()
