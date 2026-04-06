"""bio_ml_agent.dataset_catalog — re-export from legacy."""
from bio_ml_agent.legacy.dataset_catalog import (
    DATASET_CATALOG,
    calculate_path_hash,
    format_catalog_for_prompt,
    get_categories,
    get_dataset_info,
    get_dataset_version,
    list_datasets,
    load_dataset,
    search_datasets,
    verify_dataset_integrity,
)

__all__ = [
    "DATASET_CATALOG",
    "list_datasets",
    "get_dataset_info",
    "load_dataset",
    "search_datasets",
    "get_categories",
    "format_catalog_for_prompt",
    "calculate_path_hash",
    "get_dataset_version",
    "verify_dataset_integrity",
]
