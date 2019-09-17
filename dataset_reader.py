import os
import time

BASE_FULL_TEXT_DIR = "../Fake.br-Corpus/full_texts/"
BASE_NORMALIZED_TEXT_DIR = "../Fake.br-Corpus/size_normalized_texts/"
FULL_DIR_TRUE = os.path.join(BASE_FULL_TEXT_DIR, "true")
FULL_DIR_FAKE = os.path.join(BASE_FULL_TEXT_DIR, "fake")
NORM_DIR_TRUE = os.path.join(BASE_NORMALIZED_TEXT_DIR, "true")
NORM_DIR_FAKE = os.path.join(BASE_NORMALIZED_TEXT_DIR, "fake")
META_TRUE = os.path.join(BASE_FULL_TEXT_DIR, "true-meta-information")
META_FAKE = os.path.join(BASE_FULL_TEXT_DIR, "fake-meta-information")


def read_text(file):
    f = open(file, "r")
    text = f.read()
    f.close()
    return text


def read_dir(dir):
    if not os.path.isdir(dir):
        return
    files = os.listdir(dir)
    dir_copy = {}
    for file in files:
        dir_copy[file] = read_text(os.path.join(dir, file))
    return dir_copy


def meta_information():
    return read_dir(META_TRUE), read_dir(META_FAKE)


def full_text():
    return read_dir(FULL_DIR_TRUE), read_dir(FULL_DIR_FAKE)


def norm_text():
    return read_dir(NORM_DIR_TRUE), read_dir(NORM_DIR_FAKE)


if __name__ == "__main__":
    t = time.process_time()
    meta_true_info, meta_false_info = meta_information()
    full_true_info, full_false_info = full_text()
    norm_true_info, norm_false_info = norm_text()
    if len(meta_true_info) == 0 or len(meta_false_info) == 0 \
            or len(meta_true_info) != len(meta_false_info):
        print("Meta reader error")
    if len(full_true_info) == 0 or len(full_false_info) == 0 \
            or len(full_true_info) != len(full_false_info):
        print("Full text dir error")
    if len(norm_true_info) == 0 or len(norm_false_info) == 0 \
            or len(norm_true_info) != len(norm_false_info):
        print("Norm text dir error")
    if len(meta_true_info) != len(full_true_info) \
            or len(full_true_info) != len(norm_true_info):
        print("Directories has differnt number of files")
    elapsed_time = time.process_time() - t
    print(elapsed_time)
    print("Number of documents by dir: "+ str(len(meta_true_info)))
