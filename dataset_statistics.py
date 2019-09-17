from dataset_reader import *
from utils import *

meta_true_info, meta_false_info = meta_information()
full_true_info, full_false_info = full_text()
norm_true_info, norm_false_info = norm_text()


if __name__ == "__main__":
    print(meta_true_info['1529-meta.txt'].split('\n')[META_DESCRIPTION.index('percentage of news with speeling errors')])
    print()