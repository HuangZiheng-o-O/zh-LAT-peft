# Understanding the Caching Mechanism

This project utilizes a two-level caching system to accelerate dataset loading and preprocessing. Understanding this is crucial for avoiding long data processing times, especially when moving the project to a new machine.

## Level 1: Hugging Face Cache (Raw Data)

This is the standard cache used by the Hugging Face `datasets` and `transformers` libraries.

*   **Purpose:** To store raw model weights, tokenizers, and dataset files downloaded from the Hugging Face Hub.
*   **Location:** The location is determined by environment variables set in the training scripts (e.g., `gla_round_new.sh`). In this project, it is hardcoded to:
    ```bash
    /home/user/mzs_h/data/hf_cache
    ```
*   **Effect:** When you run a script for the first time, it downloads the necessary files from the internet and stores them here. Subsequent runs will use the files from this cache, avoiding re-downloads. This is why you don't see download progress bars on the second run.

## Level 2: Local Preprocessing Cache (Processed Data)

This is a custom caching layer specific to this project.

*   **Purpose:** To store the result of the time-consuming data preprocessing step. After the raw data is loaded from the Hugging Face cache, the scripts perform significant processing (tokenization, formatting, etc.) on every single example. The output of this work is saved as a single `.pkl` (pickle) file.
*   **Location:** This cache is stored in a `data` directory relative to the project's `mamba-peft` sub-directory. The exact path to the pickle file is constructed based on the dataset name and split, for example:
    ```
    mamba-peft/data/nyu-mll_glue/cache_qqp-tvt_train.pkl
    ```
*   **Effect:** The "Parallel processing" progress bar you see is this preprocessing step in action. If the corresponding `.pkl` file is found in the `mamba-peft/data` directory, this entire step is skipped, and the data is loaded instantly from the pickle file. If not, the processing is re-run from scratch, which can take hours.

## How to Correctly Move the Project to a New Machine

To completely avoid data reprocessing on a new machine, you must copy **both** caches:

1.  **Copy the Hugging Face Cache:**
    *   **Source:** `/home/user/mzs_h/data/hf_cache` from your old machine.
    *   **Destination:** `/home/user/mzs_h/data/hf_cache` on your new machine. The path must be identical because it is hardcoded in the scripts.

2.  **Copy the Local Preprocessing Cache:**
    *   **Source:** The `data` directory located inside the `mamba-peft` folder on your old machine.
    *   **Destination:** The `mamba-peft` folder on your new machine.

By ensuring both the raw data cache and the processed data cache are in their correct locations, you can start training runs immediately without any lengthy preprocessing delays.
