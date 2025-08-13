import os
from tqdm import tqdm

if __name__ == "__main__":
    for directory in os.listdir("../../data"):
        if not directory.startswith("session"):
            continue

        for file_name in tqdm(os.listdir(os.path.join("../../data", directory))):
            if "sim" in file_name:
                continue
            relative_path = os.path.join(directory, file_name)
            os.system(
                "python ./trace_simulate.py "
                + os.path.join("../../data", relative_path)
            )
