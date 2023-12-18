# Embryo Classification Challenge - Kaggle 2023 README

## Overview

Welcome to the Embryo Classification Challenge, part of the World Championship in Data Science & Artificial Intelligence 2023. This project focuses on using deep learning techniques to classify embryo images into 'good' or 'not good' categories at day-3 and day-5 stages.

## Challenge Details

- **Dataset:** Provided by Hung Vuong Hospital, comprising labeled images of embryos at day-3 and day-5.
- **Objective:** Develop a deep learning model for accurate embryo classification to aid fertility specialists.

## Project Structure

- `data/`: Folder containing the dataset.
- `notebooks/`: Jupyter notebooks for data exploration, model development, and evaluation.
- `src/`: Source code for model training and evaluation.
- `requirements.txt`: Dependencies required to run the code.

## Usage

1. Ensure you have the necessary dependencies installed. Run:
    ```bash
    pip install -r requirements.txt
    ```

2. Explore the Jupyter notebooks in the `notebooks/` directory for detailed analysis and model development.

3. Execute the model training script:
    ```bash
    python src/train_model.py
    ```

4. Evaluate the model:
    ```bash
    python src/evaluate_model.py
    ```

5. Make predictions on new data using the trained model:
    ```bash
    python src/predict.py
    ```

## Evaluation

The model is evaluated based on the F1 score, balancing precision and recall for accurate embryo quality classification.

## Acknowledgment

I extend my gratitude to Hung Vuong Hospital for providing the dataset, making this research endeavor possible.

## Contact

For inquiries, contact [Your Name] at [your.email@example.com](mailto:your.email@example.com).

Happy coding!
