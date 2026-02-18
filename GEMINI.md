# Gemini CLI Instructions for cellmech Project

This document outlines the operational guidelines and project-specific conventions for the Gemini CLI when interacting with the `cellmech` project.

## Project Overview
The `cellmech` project appears to be a Python-based library focused on cell mechanics, likely involving image processing, data analysis, and potentially GUI applications (given `pyqt6`). It uses Poetry for dependency management and `pytest` for testing. Jupyter notebooks (`demonstration.ipynb`, `tractionforce.ipynb`) suggest an exploratory or demonstrative aspect to the project.

## Key Directories and Their Purpose

*   `cellmech/`: Core source code for the `cellmech` library.
    *   `cellmech/imgproc/`: Contains image processing modules (e.g., `bead_detection.py`, `cell_detection.py`).
*   `images/`: Stores image assets, likely used for testing, examples, or as input for image processing tasks.
    *   `images/beads/`: Bead-related images.
    *   `images/cell_boundary/`: Cell boundary images.
*   `tests/`: Contains unit and integration tests for the project.
*   `__pycache__/`, `.git/`, `.pytest_cache/`, `dist/`: Standard environment/build-related directories, should generally be ignored during code modifications unless specifically troubleshooting build or test issues.

## Key Files and Their Purpose

*   `pyproject.toml`: Project configuration, dependency management (Poetry), and build system definition.
*   `poetry.lock`: Locks the exact versions of project dependencies.
*   `README.md`: Project description and basic usage instructions.
*   `LICENSE`: Project licensing information (MIT).
*   `demonstration.ipynb`, `tractionforce.ipynb`: Jupyter notebooks likely containing examples, analyses, or tutorials.
*   `force_field.npy`, `force_points.npy`: NumPy array files, likely containing pre-computed data.
*   `kernel_graph.py`: A Python script, likely related to kernel operations or graph structures.

## Technologies and Conventions

*   **Language:** Python 3.12+
*   **Dependency Management:** Poetry. Always use `poetry add` or `poetry install` for dependency management.
*   **Testing Framework:** `pytest`. Tests are located in the `tests/` directory. To run tests, use `poetry run pytest`.
*   **Coding Style:** Adhere to PEP 8. While no explicit linter is defined in `pyproject.toml`, aim for clean, readable, and idiomatic Python code consistent with existing patterns.
*   **Image Processing:** `opencv-python`, `pillow`.
*   **Numerical Operations:** `numpy`, `scipy`.
*   **Plotting:** `seaborn`.
*   **GUI:** `pyqt6`.

## Operating Instructions for Gemini CLI

1.  **Before Modifying Code:** Always read relevant files, especially those in the same directory or imported by the target file, to understand local conventions, existing logic, and dependencies.
2.  **When Adding Features/Fixing Bugs:**
    *   Prioritize adding or updating tests in the `tests/` directory to cover the changes.
    *   Ensure changes are consistent with the existing codebase's style and structure.
    *   Verify changes by running `poetry run pytest` (if applicable) and any relevant Jupyter notebooks.
3.  **Dependency Management:** If new Python packages are required, use `poetry add <package_name>`.
4.  **Avoid Assumptions:** Do not assume the presence of libraries or specific configurations without verifying `pyproject.toml` or existing code.
5.  **Output:** Be concise and direct in responses. Provide minimal output unless detailed explanation is requested.
6.  **Security:** Always adhere to security best practices; never expose sensitive information.

This `GEMINI.md` file serves as a dynamic guide. It should be updated as new insights into the project's structure, conventions, or tooling emerge.