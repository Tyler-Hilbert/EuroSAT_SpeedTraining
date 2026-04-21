# Vibe coded with Gemini

from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt
import random

def main():
    ds = load_satellite_dataset(shuffle=True)
    label_counts, label_examples = get_counts_and_examples(ds)
    print_label_counts(label_counts, ds)
    display_examples(label_examples, ds)
    print_images_metadata(ds)

# Loads the dataset from HuggingFace
def load_satellite_dataset(shuffle):
    print ('Loading dataset.')
    ds = load_dataset("nielsr/eurosat-demo", split="train")

    if shuffle:
        print ('Shuffling dataset')
        random_seed = random.randint(0, 10000)
        ds = ds.shuffle(seed=random_seed)
    return ds

# Iterates over `ds` and returns count along with example for each label
def get_counts_and_examples(ds):
    print(f'Iterating over dataset.')
    label_counts = {}
    label_examples = {}
    for row in ds:
        label_idx = row['label']
        
        # If first of class
        if label_idx not in label_examples:
            label_examples[label_idx] = row['image']
            label_counts[label_idx] = 0

        # Increment count
        label_counts[label_idx] += 1

    return label_counts, label_examples

# Prints count of each `label_counts` from label idx in 'ds
def print_label_counts(label_counts, ds):
    print ('Count of each label:')
    print(f"{'ID':<5} | {'Label Name':<25} | {'Count'}")
    print("-" * 45)

    for idx, name in enumerate(ds.features['label'].names):
        count = label_counts[idx]
        print(f"{idx:<5} | {name:<25} | {count}")

# Displays the example for each label
def display_examples(label_examples, ds):
    print ('Displaying examples for each label.')

    label_names = ds.features['label'].names
    num_labels = len(label_names)
    if num_labels != 10:
        print (f'error: ds has {num_labels} labels instead of 10.')
        exit()

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()

    # Sort by label index so the display order is consistent (0-9)
    for i, label_idx in enumerate(sorted(label_examples.keys())):
        axes[i].imshow(label_examples[label_idx])
        axes[i].set_title(f"{label_idx}: {label_names[label_idx]}")
        axes[i].axis('off')

    plt.tight_layout()
    print ('**Close plot to continue**')
    plt.show()

# Prints metadata about each of the images in `ds`
# Prints if there is more than 1 shape or dtype
# TODO formatting of table isn't perfect
def print_images_metadata(ds):
    print ('Previewing metadata for each image (data may be shuffled):')

    shape_set = set()
    dtype_set = set()
    label_names = ds.features['label'].names

    # Iterate over images
    for row_num, row in enumerate(ds):
        # Print header every 25 rows
        if row_num % 25 == 0:
            print ('----------------------------------------------------------------')
            print ('| shape       | dtype |min | max | mean | label_id | label_name |')
            print ('----------------------------------------------------------------')


        # Print row
        label_id = row['label']
        label_name = label_names[label_id]
        img_array = np.array(row['image'])

        shape_set.add(img_array.shape)
        dtype_set.add(str(img_array.dtype))

        print (f'| {img_array.shape} | {img_array.dtype} | {np.min(img_array)} | {np.max(img_array)} | {np.mean(img_array)} | {label_id} | {label_name} |')

    # Print if there is only 1 shape and 1 dtype, or if there are multiple
    if len(shape_set) == 1:
        print(f"All images have the same shape.")
    else:
        print(f"WARNING: Multiple shapes detected!")
    if len(dtype_set) == 1:
        print(f"All images have the same dtype.")
    else:
        print(f"WARNING: Multiple dtypes detected!")

if __name__ == "__main__":
    main()