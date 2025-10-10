# Preprocessing of Datasets

This folder contains code for preprocessing datasets and converting them into a standardized format for easier loading and use.

Configuration is done directly in the respective python file. Then execute like this
```bash
python nerdd.py
# or
python armasuisse.py # converts the F-UAV-D dataset 
```

## Output of preprocessing

The prepocessing creates a folder structure like this:
```sh
├── train
│   ├── 2024_01_10_112814_drone_000
│   │   ├── events
│   │   │   ├── event_0.npz
│   │   │   ├── event_100.npz
│   │   │   ├── event_101.npz
│   │   ├── labels
│   │   │   ├── label_0.npz
│   │   │   ├── label_100.npz
│   │   │   ├── label_101.npz
│   ├── 2024_01_10_112814_drone_002
│   │   ├── events
│   │   └── labels
├── val
│   ├── 2024_01_10_112814_drone_001
│   │   ├── events
│   │   └── labels
│   ├── 2024_01_10_112957_drone_1_repeat_001
│   │   ├── events
│   │   └── labels
│   │
```

Each subfolder under `train/` and `val/` contains a sequence of stacked histograms along with their corresponding labels. 
For example, `event_0.npz` contains a single stacked histogram, and the matching labels are stored in `label_0.npz`. 
This structure allows for straightforward pairing between events and their labels.

## event_X.npz
We can access the data in event_X.npz like this:

```python
# this loads the npz compressed file
data = np.load("event_X.npz")

# the pure data is a StackedHistogram as a structured numpy array with shape (bins, height, width) 
data = data[list(data.keys())[0]]
# or as we know the first key is arr_0
data = data['arr_0']
```
## label_X.npz
We can access the data in labels_X.npz like this:
```python
# this loads the npz compressed file
data = np.load("label_X.npz")

# this is a structured numpy array with
# dtype=[('t', 'i8'), ('x', 'f4'), ('y', 'f4'), ('w', 'f4'), ('h', 'f4'), ('class_id', 'u1'), ('class_confidence', 'f4')]
# it can contain multiple bounding boxes
data = data[list(data.keys())[0]]
# or as we know the first key is arr_0
data = data['arr_0']
```

## Matching labels and creating sequences

Since each folder is expected to contain only sequential data, we can construct sequences by simply relying on the file system’s ordering.
This makes sequence generation straightforward and efficient.

```python
def create_sequences(path: Path, sequence_length: int):
    seq_list = list()
    event_folder = Path("events")
    label_folder = Path("labels")
    frame_folder = Path("rgbs")

    for dir in os.listdir(path):
        event_path = path / dir / event_folder
        label_path = path / dir / label_folder
        rgb_path = path / dir / frame_folder

        event_files = os.listdir(event_path)
        label_files = os.listdir(label_path)
        rgb_files = os.listdir(rgb_path)

        event_files.sort(key=lambda item: (len(item), item))
        label_files.sort(key=lambda item: (len(item), item))
        rgb_files.sort(key=lambda item: (len(item), item))
        assert_msg = f"event_len({len(event_files)}) != label_len({len(label_files)}) != rgb_len({len(rgb_files)}) for\n{event_path},\n{label_path} and\n{rgb_path}"
        assert len(event_files) > 0, f"event_files empty, {event_path}"
        assert len(label_files) > 0, f"event_files empty, {label_path}"
        assert len(rgb_files) > 0, f"event_files empty, {rgb_path}"
        assert len(event_files) == len(label_files) == len(rgb_files), assert_msg


        # we start at sequence_length to not run out of elements at the end of the files
        for index in range(sequence_length, len(event_files), sequence_length):
            start = index - sequence_length
            event_list = [event_path / event_file for event_file in event_files[start:start+sequence_length]]
            label_list = [label_path / label_file for label_file in label_files[start:start+sequence_length]]
            frame_list = [rgb_path / rgb_file for rgb_file in rgb_files[start:start+sequence_length]]

            sequence = Sequence(event_list, label_list, frame_list)
            seq_list.append(sequence)

    return seq_list

```
