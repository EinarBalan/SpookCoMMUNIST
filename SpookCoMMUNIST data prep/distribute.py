import os
import shutil
import random


def distribute_files(src):
    # Define the paths
    source_folder = os.path.join('./spookcommunist/all', src)
    test_folder = os.path.join('./spookcommunist/test', src)
    train_folder = os.path.join('./spookcommunist/train', src)
    val_folder = os.path.join('./spookcommunist/val', src)

    # Create the destination folders if they don't exist
    os.makedirs(test_folder, exist_ok=True)
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(val_folder, exist_ok=True)

    # Get a list of all files in the 'all' folder
    all_files = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]

    # Shuffle the list to ensure random distribution
    random.shuffle(all_files)

    # Calculate the number of files for each folder
    total_files = len(all_files)
    num_test = int(total_files * .45)
    num_train = int(total_files * .45)
    num_val = total_files - num_test - num_train

    # Distribute the files
    for i, file in enumerate(all_files):
        src_path = os.path.join(source_folder, file)
        if i < num_test:
            dst_path = os.path.join(test_folder, file)
        elif i < num_test + num_train:
            dst_path = os.path.join(train_folder, file)
        else:
            dst_path = os.path.join(val_folder, file)
        shutil.move(src_path, dst_path)

    print(f"Distributed {num_test} files to test, {num_train} files to train, and {num_val} files to val.")
    
    
distribute_files("communist/sepia")
distribute_files("communist/woods")
distribute_files("spook/sepia")
distribute_files("spook/woods")
