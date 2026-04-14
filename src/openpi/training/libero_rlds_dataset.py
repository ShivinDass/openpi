"""
RLDS-based data loader for DROID.
While openpi typically uses LeRobot's data loader, it is not currently scalable enough for larger datasets like DROID.
Thus, we provide a data loader example here that uses the RLDS data format.
The data loader also applies a few DROID-specific data filters / transformations.
"""

from collections.abc import Sequence
import dataclasses
from enum import Enum
from enum import auto
import json
import logging
from pathlib import Path

import tqdm

import openpi.shared.download as download


class DroidActionSpace(Enum):
    """Action space for DROID dataset."""

    JOINT_POSITION = auto()
    JOINT_VELOCITY = auto()


@dataclasses.dataclass
class RLDSDataset:
    name: str
    version: str
    weight: float
    filter_dict_path: str | None = None


class LiberoRldsDataset:
    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        datasets: Sequence[RLDSDataset],
        *,  # Force keyword-only arguments
        shuffle: bool = True,
        action_chunk_size: int = 50,
        action_space=None,
        # Reduce this if you are running out of memory, but careful -- below ~100k shuffling is not sufficiently random.
        shuffle_buffer_size: int = 100_000,
        num_parallel_reads: int = -1,  # -1 == tf.data.AUTOTUNE -- hack to not import tf at top level
        num_parallel_calls: int = -1,  # -1 == tf.data.AUTOTUNE -- hack to not import tf at top level
        compute_length_on_init: bool = False,  # Whether to compute dataset length during initialization
    ):
        # Import tensorflow here to not make it mandatory in case RLDS data loader is not used.
        import dlimp as dl
        import tensorflow as tf
        import tensorflow_datasets as tfds

        # Configure Tensorflow with *no GPU devices* (to prevent clobber with PyTorch / JAX)
        tf.config.set_visible_devices([], "GPU")

        # Ensure dataset weights sum to 1.0
        assert sum(dataset.weight for dataset in datasets) == 1.0, "Dataset weights must sum to 1.0"

        def prepare_single_dataset(dataset_cfg: RLDSDataset):
            # ds_name, version = dataset_name.split(":")
            ds_name, version = dataset_cfg.name, dataset_cfg.version
            builder = tfds.builder(ds_name, data_dir=data_dir, version=version)
            dataset = dl.DLataset.from_rlds(
                builder, split="train", shuffle=shuffle, num_parallel_reads=num_parallel_reads
            )

            # Repeat dataset so we never run out of data.
            if not compute_length_on_init:
                dataset = dataset.repeat()

            # Load the filter dictionary if provided.
            # The filter dictionary is a JSON file that maps episode keys to ranges of frames to sample
            # (e.g.,
            # {
            #     "<episode key>": [[0, 100], [200, 300]]
            # }
            # means keep frames 0-99 and 200-299).

            def restructure(traj):
                """Reformat observation and action keys, sample language instruction."""
                traj_len = tf.shape(traj["action"])[0]
                indices = tf.as_string(tf.range(traj_len))

                traj_index = None
                task_index = None
                if "traj_index" in traj:
                    traj_index = traj["traj_index"]
                if "task_index" in traj:
                    task_index = traj["task_index"]

                # flip last action to pi0 convention
                actions = traj['action']
                # actions[:, -1] = 1-2*(actions[:, -1])
                actions = tf.concat(
                        (
                            actions[:, :6],
                            1-2*actions[:, -1:],
                        ),
                        axis=-1,
                    )

                if traj_index is not None:
                    traj["traj_index"] = traj_index[:traj_len]
                if task_index is not None:
                    traj["task_index"] = task_index[:traj_len]

                # Data filtering:
                # Compute a uniquely-identifying step ID by concatenating the recording folderpath, file path,
                # and each step's time step index. This will index into the filter hash table, and if it returns true,
                # then the frame passes the filter.
                # step_id = (
                #     traj["traj_metadata"]["episode_metadata"]["recording_folderpath"]
                #     + "--"
                #     + traj["traj_metadata"]["episode_metadata"]["file_path"]
                #     + "--"
                #     + indices
                # )

                return {
                    "actions": actions,
                    "observation": {
                        "image": traj["observation"]["image"],
                        "wrist_image": traj["observation"]["wrist_image"],
                        "state": traj["observation"]["state"],
                    },
                    "prompt": traj["language_instruction"],
                    # "step_id": step_id,
                }

            dataset = dataset.traj_map(restructure, num_parallel_calls)

            def chunk_actions(traj):
                """Splits episode into action chunks."""
                traj_len = tf.shape(traj["actions"])[0]

                # For each step in the trajectory, construct indices for the next n actions
                action_chunk_indices = tf.broadcast_to(
                    tf.range(action_chunk_size)[None],
                    [traj_len, action_chunk_size],
                ) + tf.broadcast_to(
                    tf.range(traj_len)[:, None],
                    [traj_len, action_chunk_size],
                )

                # Cap to length of the sequence --> final chunks will repeat the last action
                # This makes sense, since we are using absolute joint + gripper position actions
                action_chunk_indices = tf.minimum(action_chunk_indices, traj_len - 1)

                # Gather the actions for each chunk
                traj["actions"] = tf.gather(traj["actions"], action_chunk_indices)
                return traj

            dataset = dataset.traj_map(chunk_actions, num_parallel_calls)

            # Flatten: map from trajectory dataset to dataset of individual action chunks
            dataset = dataset.flatten(num_parallel_calls=num_parallel_calls)

            # Decode images: RLDS saves encoded images, only decode now for efficiency
            def decode_images(traj):
                traj["observation"]["image"] = tf.io.decode_image(
                    traj["observation"]["image"], expand_animations=False, dtype=tf.uint8
                )
                traj["observation"]["wrist_image"] = tf.io.decode_image(
                    traj["observation"]["wrist_image"], expand_animations=False, dtype=tf.uint8
                )
                return traj

            dataset = dataset.frame_map(decode_images, num_parallel_calls)
            
            # NOTE: Count samples BEFORE repeat() to get the true finite dataset size
            if compute_length_on_init:
                logging.info(f"Computing length for {ds_name}:{version}...")
                sample_count = 0
                for _ in tqdm.tqdm(dataset.as_numpy_iterator()):
                    sample_count += 1
                    if sample_count % 1000 == 0:
                        logging.info(f"  {ds_name}: counted {sample_count} samples...")
                logging.info(f"  {ds_name}: {sample_count} samples")
                dataset = dataset.repeat()
            else:
                sample_count = None
            
            return dataset, sample_count

        logging.info(f"Preparing {len(datasets)} datasets...")
        logging.info("-" * 50)
        for dataset in datasets:
            logging.info(f"    {dataset.name}:{dataset.version} with weight {dataset.weight:.2f}")
        logging.info("-" * 50)
        results = [prepare_single_dataset(dataset) for dataset in datasets]
        all_datasets = [result[0] for result in results]
        sample_counts = [result[1] for result in results]
        weights = [dataset.weight for dataset in datasets]
        
        # Compute total dataset size if we counted samples
        self._num_samples = None
        if compute_length_on_init and all(count is not None for count in sample_counts):
            total = sum(sample_counts)
            logging.info(f"Total samples across all datasets: {total}")
            self._num_samples = total

        final_dataset = dl.DLataset.sample_from_datasets(all_datasets, weights=weights)
        final_dataset = final_dataset.shuffle(shuffle_buffer_size)
        final_dataset = final_dataset.batch(batch_size)
        # Note =>> Seems to reduce memory usage without affecting speed?
        final_dataset = final_dataset.with_ram_budget(1)

        self.dataset = final_dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        yield from self.dataset.as_numpy_iterator()

    def __len__(self):
        return 669043 #self._num_samples
