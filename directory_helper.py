import os


def create_train_val_dirs(root_path):
    """
    Creates directories for the train and test sets

    Args:
      root_path (string) - the base directory path to create subdirectories from

    Returns:
      None
    """
    os.makedirs(os.path.join(root_path, 'training'))
    os.makedirs(os.path.join(root_path, 'validation'))
    for number in range(0, 10):
        os.makedirs(os.path.join(root_path, 'training', str(number)))
        os.makedirs(os.path.join(root_path, 'validation', str(number)))


def get_filepaths(directory):
    file_paths = []  # List which will store all the full filepaths.

    # Walk the tree.
    for root, directories, files in os.walk(directory):
        for filename in files:
            # Join the two strings in order to form the full filepath.
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)  # Add it to the list.

    return file_paths  # Self-explanatory.
