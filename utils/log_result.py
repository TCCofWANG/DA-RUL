import json
import csv
import os
# Function to log a single experiment's results
def log_experiment_results(file_path, hyperparams, result):
    """
    Log the results of a single experiment.

    Args:
    - file_path (str): Path to the log file.
    - hyperparams (dict): Dictionary of hyperparameters used in the experiment.
    - result (dict): Dictionary of the results/metrics from the experiment.
    """
    # Combine hyperparameters and results
    experiment_log = {"hyperparameters": hyperparams, "results": result}

    # Open the log file and append the experiment results
    with open(file_path, 'a') as file:
        # Convert the combined dictionary to a JSON string for readability
        file.write(json.dumps(experiment_log) + '\n')

def log_results_to_csv(file_path, header, data):
    """
    Logs results into a CSV file. If the file doesn't exist, it creates it and adds the header.

    Parameters:
    - file_path: str, the path to the CSV file.
    - header: list, the header row containing the column names.
    - data: list, the data to be logged, where each item in the list is a row of data.
    """
    file_exists = os.path.isfile(file_path)

    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)

        # If the file does not exist, write the header first
        if not file_exists:
            writer.writerow(header)

        # Write the data rows
        for row in data:
            writer.writerow(row)


def log_args_and_metrics_to_csv(file_path, args, **metrics):
    """
    Logs the command-line arguments and additional metrics to a CSV file.

    Parameters:
    - file_path: str, the path to the CSV file.
    - args: Namespace, the parsed command-line arguments.
    - metrics: dict, additional metrics to log, e.g., RMSE.
    """
    # Convert Namespace to a dictionary and update with metrics
    args_dict = metrics
    args_dict.update(vars(args))

    # Determine if the file exists to decide on writing the header
    file_exists = os.path.isfile(file_path)

    with open(file_path, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=args_dict.keys())

        # If the file does not exist, write the header
        if not file_exists:
            writer.writeheader()

        # Write the data row
        writer.writerow(args_dict)