import os
import tarfile
import zipfile
from urllib.request import urlopen

IMDB_URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
IMDB_ARCHIVE_NAME = "aclImdb_v1.tar.gz"


def get_datasets_folder():
    here = os.path.dirname(__file__)
    notebooks = os.path.join(here, 'notebooks')
    datasets_folder = os.path.abspath(os.path.join(notebooks, 'datasets'))
    datasets_archive = os.path.abspath(os.path.join(notebooks, 'datasets.zip'))

    if not os.path.exists(datasets_folder):
        if os.path.exists(datasets_archive):
            print("Extracting " + datasets_archive)
            with zipfile.ZipFile(datasets_archive) as zf:
                root = os.path.abspath(notebooks)
                for member in zf.namelist():  # block path traversal ("zip slip")
                    target = os.path.abspath(os.path.join(root, member))
                    if os.path.commonpath([root, target]) != root:
                        raise ValueError("Unsafe path in archive: %r" % member)
                zf.extractall(root)
            assert os.path.exists(datasets_folder)
        else:
            print("Creating datasets folder: " + datasets_folder)
            os.makedirs(datasets_folder)
    else:
        print("Using existing dataset folder:" + datasets_folder)
    return datasets_folder


def check_imdb(datasets_folder):
    print("\nChecking availability of the IMDb dataset")
    archive_path = os.path.join(datasets_folder, IMDB_ARCHIVE_NAME)
    imdb_path = os.path.join(datasets_folder, 'IMDb')

    train_path = os.path.join(imdb_path, 'aclImdb', 'train')
    test_path = os.path.join(imdb_path, 'aclImdb', 'test')

    if not os.path.exists(imdb_path):
        if not os.path.exists(archive_path):
            print("Downloading dataset from %s (84.1MB)" % IMDB_URL)
            with urlopen(IMDB_URL, timeout=60) as response, open(archive_path, 'wb') as out:
                out.write(response.read())
        else:
            print("Found archive: " + archive_path)

        print("Extracting %s to %s" % (archive_path, imdb_path))

        # filter="data" blocks path traversal and unsafe members (Python 3.12+/3.11.4+)
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(path=imdb_path, filter="data")
        os.remove(archive_path)

    print("Checking that the IMDb train & test directories exist...")
    assert os.path.exists(train_path)
    assert os.path.exists(test_path)
    print("=> Success!")


if __name__ == "__main__":
    datasets_folder = get_datasets_folder()
    check_imdb(datasets_folder)

    print("\nLoading Labeled Faces Data (~200MB)")
    from sklearn.datasets import fetch_lfw_people
    fetch_lfw_people(min_faces_per_person=70, resize=0.4,
                     data_home=datasets_folder)
    print("=> Success!")
