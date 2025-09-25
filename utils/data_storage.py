import joblib
import os


def save_all_vectorizers(tfidf_data, bow_data, embeddings_data, targets, path="../pkls/", filename="all_vectorizers.pkl"):
    """
    Function to save all vectorizers and vector representations
    :param tfidf_data: tuple (train, test) of tifidf
    :param bow_data: tuple (train, test) of bow
    :param embeddings_data: tuple (train, test) of embeddings
    :param path: path of folder
    :param filename:
    :return: file with all vectorizers and vector representations
    """
    try:
        folder = os.path.join(path)
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f" Folder {folder} created")

        data = {
            "tfidf": {"train": tfidf_data[0], "test": tfidf_data[1]},
            "bow": {"train": bow_data[0], "test": bow_data[1]},
            "embeddings": {"train": embeddings_data[0], "test": embeddings_data[1]},
            "targets": {"train": targets[0], "test": targets[1]}
        }
        joblib.dump(data, path + filename)
        print(f"✅ Saved all vectorizers and vector representations to {path + filename}")
    except Exception as e:
        print("❌ Unexpected error while saving:", e)
    except PermissionError:
        print("❌ Permission Denied: Can not write to specified folder")
    except OSError as e:
        print("❌ OS error while saving:", e)


def load_all_vectorizers(path="../pkls/", filename="all_vectorizers.pkl"):
    """
    Function to load all vectorizers and vector representations
    :param path: path of folder
    :param filename: filename
    :return: the Xtrain, Xtest and Ytrain, Ytest of all vectorizers and vector representations
    """
    filepath = os.path.join(path, filename)
    try:
        if not os.path.exists(path + filename):
            raise FileNotFoundError (f"❌ File {filepath} not found")
        data = joblib.load(filepath)
        print("Succesfully loaded all vectorizers and vector representations")
        return (
            data["tfidf"]["train"], data["tfidf"]["test"],
            data["bow"]["train"], data["bow"]["test"],
            data["embeddings"]["train"], data["embeddings"]["test"],
            data["targets"]["train"], data["targets"]["test"]
        )

    except FileNotFoundError as e:
        print(f"❌ File {filename} not found")
        return None, None, None, None, None, None, None, None
    except Exception as e:
        print("❌ Unexpected error while saving:", e)
        return None, None, None, None, None, None, None, None


if __name__ == "__main__":
    obj = joblib.load("../pkls/all_vectorizers.pkl")
    print(type(obj))
    if isinstance(obj, dict):
        print("keys:", obj.keys())
    elif isinstance(obj, (list, tuple)):
        print("length:", len(obj))