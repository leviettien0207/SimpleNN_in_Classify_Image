import kaggle
import constant


def download_dataset():
    """
    This function is to connect api kaggle then download
    :return:
    """
    # Authenticate personal API token
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(constant.DATASET_NAME_ON_KAGGLE, unzip=True)
