import os
import re
import shutil
import tarfile
import requests

from midi2audio import FluidSynth
from music21 import converter
from sklearn.model_selection import train_test_split

GRANDSTAFF_PATH = "./grandstaff"
SOUND_FONT = "./data/SGM-v2.01-YamahaGrand-Guit-Bass-v2.7.sf2"


########################################################## DOWNLOAD DATASET


def download_and_extract_grandstaff_dataset():
    """Download and extract the GRANDSTAFF dataset."""
    file_path = "grandstaff.tgz"
    extract_path = GRANDSTAFF_PATH
    os.makedirs(extract_path, exist_ok=True)

    # Download dataset
    response = requests.get(url="https://grfia.dlsi.ua.es/musicdocs/grandstaff.tgz")
    with open(file_path, "wb") as file:
        file.write(response.content)
    # Extract dataset
    with tarfile.open(file_path, "r:gz") as tar:
        tar.extractall(extract_path)
    # Remove tar file
    os.remove(file_path)


########################################################## PARSE NEW DATASET FOLDER STRUCTURE


def parse_grandstaff_dataset():
    """
    Parse the new folder structure of the GRANDSTAFF dataset.
    The new folder structure is as follows:
    grandstaff
    ├── composer (beethoven, chopin, hummel, joplin, mozart, scarlatti-d)
    │   ├── img
    │   ├── img_distorted
    │   ├── krn
    │   ├── bekrn
    │   └── wav
    """
    for composer in os.listdir(GRANDSTAFF_PATH):
        old_composer_path = os.path.join(GRANDSTAFF_PATH, composer)
        new_composer_path = os.path.join(GRANDSTAFF_PATH, composer + "_parsed")

        # Create a new folder structure for the composer
        os.makedirs(os.path.join(new_composer_path, "wav"), exist_ok=True)
        os.makedirs(os.path.join(new_composer_path, "krn"), exist_ok=True)
        os.makedirs(os.path.join(new_composer_path, "bekrn"), exist_ok=True)
        os.makedirs(os.path.join(new_composer_path, "img"), exist_ok=True)
        os.makedirs(os.path.join(new_composer_path, "img_distorted"), exist_ok=True)

        # Move files to the new folder structure
        for foldername, subfolders, filenames in os.walk(old_composer_path):
            for filename in filenames:
                if filename.startswith("."):
                    continue
                new_filename = "_".join(
                    foldername.replace(old_composer_path, "").split("/")[1:]
                    + [filename]
                )
                if filename.endswith(".bekrn"):
                    shutil.move(
                        os.path.join(foldername, filename),
                        os.path.join(new_composer_path, "bekrn", new_filename),
                    )
                elif filename.endswith(".krn"):
                    shutil.move(
                        os.path.join(foldername, filename),
                        os.path.join(new_composer_path, "krn", new_filename),
                    )
                elif filename.endswith("_distorted.jpg"):
                    shutil.move(
                        os.path.join(foldername, filename),
                        os.path.join(new_composer_path, "img_distorted", new_filename),
                    )
                elif filename.endswith(".jpg"):
                    shutil.move(
                        os.path.join(foldername, filename),
                        os.path.join(new_composer_path, "img", new_filename),
                    )
                else:
                    continue

        # Remove the old folder structure
        shutil.rmtree(old_composer_path)

        # Remove the "_parsed" suffix from the composer folder
        os.rename(new_composer_path, old_composer_path)


########################################################## CONVERT KRNS TO WAVS


def krn2wav():
    """
    Convert all krn files () to wav files.
    Save the wav files in the corresponding wav folder of each composer.
    Save the errors in a text file for each composer. Path: ./grandstaff/errors/composer.txt
    """
    os.makedirs(os.path.join(GRANDSTAFF_PATH, "errors"), exist_ok=True)

    fs = FluidSynth(sample_rate=22050, sound_font=SOUND_FONT)
    for composer in os.listdir(GRANDSTAFF_PATH):
        if composer == "errors" or composer.startswith("."):
            continue
        print(f"Generating wav files for {composer}")
        composer_path = os.path.join(GRANDSTAFF_PATH, composer)

        errors = []
        for id, krn_file in enumerate(os.listdir(os.path.join(composer_path, "krn"))):
            # krn to midi
            try:
                krn_stream = converter.parse(
                    os.path.join(composer_path, "krn", krn_file)
                )
            except Exception as err:
                errors.append(krn_file + "\t" + str(type(err)) + "\t" + str(err))
                # Remove the file and all its corresponding files
                os.remove(os.path.join(composer_path, "krn", krn_file))
                os.remove(
                    os.path.join(
                        composer_path,
                        "img",
                        krn_file.replace(".krn", ".jpg"),
                    )
                )
                os.remove(
                    os.path.join(
                        composer_path,
                        "img_distorted",
                        krn_file.replace(".krn", "_distorted.jpg"),
                    )
                )
                os.remove(
                    os.path.join(
                        composer_path,
                        "bekrn",
                        krn_file.replace(".krn", ".bekrn"),
                    )
                )
                continue
            midi_file = os.path.join(composer_path, "krn", krn_file + ".mid")
            _ = krn_stream.write("midi", fp=midi_file)

            # midi to wav
            wav_file = os.path.join(
                composer_path, "wav", krn_file.replace(".krn", ".wav")
            )
            fs.midi_to_audio(midi_file, wav_file)
            os.remove(midi_file)

        # Save errors
        if len(errors) == 0:
            print(f"All {id+1} files were converted to wav.")
        else:
            print(f"{len(errors)} out of {id+1} files could not be converted to wav.")
            errors = "\n".join(errors)
            with open(os.path.join("./grandstaff/errors", f"{composer}.txt"), "w") as f:
                f.write(errors)
            print(f"Errors saved to ./grandstaff/errors/{composer}.txt")


########################################################## CREATE PARTITIONS


def check_and_create_partitions():
    """
    Check if partitions have been created for each composer.
    If not, create them.
    """
    for composer in os.listdir(GRANDSTAFF_PATH):
        if composer == "partitions" or composer == "errors" or composer.startswith("."):
            continue

        partition_folder = os.path.join(GRANDSTAFF_PATH, "partitions", composer)
        if not os.path.exists(partition_folder):
            create_composer_partitions()
            break
        else:
            train = os.path.join(partition_folder, "train.txt")
            val = os.path.join(partition_folder, "val.txt")
            test = os.path.join(partition_folder, "test.txt")
            if (
                not os.path.exists(train)
                or not os.path.exists(val)
                or not os.path.exists(test)
            ):
                create_composer_partitions()
                break


def create_composer_partitions():
    """
    Create train, val and test partitions for each composer.
    Save the partitions in the corresponding partitions folder of each composer.
    Path: ./grandstaff/partitions/composer/{train, val, test}.txt
    """

    def extract_org_name(name):
        """
        Utility function to extract the original name of a file.
        Examples:
            keyboard-sonatas_L337K336_maj2_down_m-73-76 -> keyboard-sonatas_L337K336_m-73-76
            keyboard-sonatas_L340K476_original_m-51-56 -> keyboard-sonatas_L340K476_m-51-56
            piano-sonatas_sonata01-1_min3_down_m-49-53 -> piano-sonatas_sonata01-1_m-49-53
            piano-sonatas_sonata16-1_original_m-121-126 -> piano-sonatas_sonata16-1_m-121-126
        """
        return re.sub(r"_(maj\d+|min\d+|original|up|down)", "", name)

    partitions_path = os.path.join(GRANDSTAFF_PATH, "partitions")
    os.makedirs(partitions_path, exist_ok=True)

    for composer in os.listdir(GRANDSTAFF_PATH):
        if (
            composer == "partitions"
            or composer == "errors"
            or composer == "vocabs"
            or composer.startswith(".")
        ):
            continue

        partition_folder = os.path.join(partitions_path, composer)
        os.makedirs(partition_folder, exist_ok=True)

        composer_path = os.path.join(GRANDSTAFF_PATH, composer)
        samples = [
            f.split(".wav")[0]
            for f in os.listdir(os.path.join(composer_path, "wav"))
            if f.endswith(".wav") and not f.startswith(".")
        ]

        # Test samples should be "original" samples that are not transposed
        # That is, we cannot have a original sample in the test set and
        # the same but transposed (maj2, min3, etc.) sample in the train set
        test = [s for s in samples if "original" in s]
        org_test = [extract_org_name(s) for s in test]
        train_val = []
        for s in samples:
            if s in test:
                continue
            org_name = extract_org_name(s)
            if org_name not in org_test:
                train_val.append(s)

        train, val = train_test_split(train_val, test_size=0.2, random_state=42)

        for partition, samples in zip(["train", "val", "test"], [train, val, test]):
            with open(
                os.path.join(partition_folder, f"{partition}.txt"), "w"
            ) as partition_file:
                partition_file.write("\n".join(samples))


def create_grandstaff_partitions():
    """
    Create train, val and test partitions for the GRANDSTAFF dataset.
    Use the partitions of each composer.
    Save the partitions in the corresponding partitions folder of the GRANDSTAFF dataset.
    """
    partitions_path = os.path.join(GRANDSTAFF_PATH, "partitions")
    grandstaff_partitions_path = os.path.join(partitions_path, "grandstaff")
    os.makedirs(grandstaff_partitions_path, exist_ok=True)

    for composer in os.listdir(partitions_path):
        if composer == "grandstaff" or composer.startswith("."):
            continue
        for partition in ["train", "val", "test"]:
            with open(
                os.path.join(partitions_path, composer, f"{partition}.txt"), "r"
            ) as partition_file:
                samples = partition_file.read().splitlines()
                samples = [f"{composer}\t{s}" for s in samples]
                with open(
                    os.path.join(grandstaff_partitions_path, f"{partition}.txt"), "a"
                ) as grandstaff_partition_file:
                    grandstaff_partition_file.write("\n".join(samples) + "\n")


if __name__ == "__main__":
    print("Downloading and extracting GRANDSTAFF dataset...")
    download_and_extract_grandstaff_dataset()
    print("Parsing GRANDSTAFF dataset...")
    parse_grandstaff_dataset()
    print("Converting krn files to wav files...")
    krn2wav()
    print("Creating partitions...")
    check_and_create_partitions()
    create_grandstaff_partitions()
    print("Done!")
