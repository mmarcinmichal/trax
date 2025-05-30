# coding=utf-8
# Copyright 2022 The Trax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Write Summaries from JAX for use with Tensorboard.

See jaxboard_demo.py for example usage.
"""
import io
import struct
import time
import warnings
import wave

import matplotlib as mpl

# Necessary to prevent attempted Tk import:
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    mpl.use("Agg")
# pylint: disable=g-import-not-at-top
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


# pylint: disable=g-direct-tensorflow-import
from tensorflow.core.util import event_pb2
from tensorflow.python.summary.writer.event_file_writer import EventFileWriter

# pylint: enable=g-direct-tensorflow-import


def _pack_images(images, rows, cols):
    """Helper utility to make a tiled field of images from numpy arrays.

    Args:
      images: Image tensor in shape [N, W, H, C].
      rows: Number of images per row in tiled image.
      cols: Number of images per column in tiled image.

    Returns:
      A tiled image of shape [W * rows, H * cols, C].
      Truncates incomplete rows.
    """
    shape = np.shape(images)
    width, height, depth = shape[-3:]
    images = np.reshape(images, (-1, width, height, depth))
    batch = np.shape(images)[0]
    rows = np.minimum(rows, batch)
    cols = np.minimum(batch // rows, cols)
    images = images[: rows * cols]
    images = np.reshape(images, (rows, cols, width, height, depth))
    images = np.transpose(images, [0, 2, 1, 3, 4])
    images = np.reshape(images, [rows * width, cols * height, depth])
    return images


class SummaryWriter:
    """Saves data in event and summary protos for tensorboard."""

    def __init__(self, log_dir, enable=True):
        """Create a new SummaryWriter.

        Args:
          log_dir: path to record tfevents files in.
          enable: bool: if False don't actually write or flush data.  Used in
            multihost training.
        """
        # If needed, create log_dir directory as well as missing parent directories.
        if not tf.io.gfile.isdir(log_dir):
            tf.io.gfile.makedirs(log_dir)

        self._event_writer = EventFileWriter(log_dir, 10, 120, None)
        self._step = 0
        self._closed = False
        self._enabled = enable

    def add_summary(self, summary, step):
        if not self._enabled:
            return
        event = event_pb2.Event(summary=summary)
        event.wall_time = time.time()
        if step is not None:
            event.step = int(step)
        self._event_writer.add_event(event)

    def close(self):
        """Close SummaryWriter. Final!"""
        if not self._closed:
            self._event_writer.close()
            self._closed = True
            del self._event_writer

    def __del__(self):  # safe?
        # TODO(afrozm): Sometimes this complains with
        #  `TypeError: 'NoneType' object is not callable`
        try:
            self.close()
        except Exception:  # pylint: disable=broad-except
            pass

    def flush(self):
        if not self._enabled:
            return
        self._event_writer.flush()

    def scalar(self, tag, value, step=None):
        """Saves scalar value.

        Args:
          tag: str: label for this data
          value: int/float: number to log
          step: int: training step
        """
        value = float(np.array(value))
        if step is None:
            step = self._step
        else:
            self._step = step
        summary = tf.compat.v1.Summary(
            value=[tf.compat.v1.Summary.Value(tag=tag, simple_value=value)]
        )
        self.add_summary(summary, step)

    def image(self, tag, image, step=None):
        """Saves RGB image summary from np.ndarray [H,W], [H,W,1], or [H,W,3].

        Args:
          tag: str: label for this data
          image: ndarray: [H,W], [H,W,1], [H,W,3] save image in greyscale or colors/
          step: int: training step
        """
        image = np.array(image)
        if step is None:
            step = self._step
        else:
            self._step = step
        if len(np.shape(image)) == 2:
            image = image[:, :, np.newaxis]
        if np.shape(image)[-1] == 1:
            image = np.repeat(image, 3, axis=-1)
        image_strio = io.BytesIO()
        plt.imsave(image_strio, image, format="png")
        image_summary = tf.compat.v1.Summary.Image(
            encoded_image_string=image_strio.getvalue(),
            colorspace=3,
            height=image.shape[0],
            width=image.shape[1],
        )
        summary = tf.compat.v1.Summary(
            value=[tf.compat.v1.Summary.Value(tag=tag, image=image_summary)]
        )
        self.add_summary(summary, step)

    def images(self, tag, images, step=None, rows=None, cols=None):
        """Saves (rows, cols) tiled images from np.ndarray.

        If either rows or cols aren't given, they are determined automatically
        from the size of the image batch, if neither are given a long column
        of images is produced. This truncates the image batch rather than padding
        if it doesn't fill the final row.

        Args:
          tag: str: label for this data
          images: ndarray: [N,H,W,1] or [N,H,W,3] to tile in 2d
          step: int: training step
          rows: int: number of rows in tile
          cols: int: number of columns in tile
        """
        images = np.array(images)
        if step is None:
            step = self._step
        else:
            self._step = step
        n_images = np.shape(images)[0]
        if rows is None and cols is None:
            rows = 1
            cols = n_images
        elif rows is None:
            rows = n_images // cols
        elif cols is None:
            cols = n_images // rows
        tiled_images = _pack_images(images, rows, cols)
        self.image(tag, tiled_images, step=step)

    def plot(self, tag, mpl_plt, step=None, close_plot=True):
        """Saves matplotlib plot output to summary image.

        Args:
          tag: str: label for this data
          mpl_plt: matplotlib stateful pyplot object with prepared plotting state
          step: int: training step
          close_plot: bool: automatically closes plot
        """
        if step is None:
            step = self._step
        else:
            self._step = step
        fig = mpl_plt.get_current_fig_manager()
        img_w, img_h = fig.canvas.get_width_height()
        image_buf = io.BytesIO()
        mpl_plt.savefig(image_buf, format="png")
        image_summary = tf.compat.v1.Summary.Image(
            encoded_image_string=image_buf.getvalue(),
            colorspace=4,  # RGBA
            height=img_h,
            width=img_w,
        )
        summary = tf.compat.v1.Summary(
            value=[tf.compat.v1.Summary.Value(tag=tag, image=image_summary)]
        )
        self.add_summary(summary, step)
        if close_plot:
            mpl_plt.close()

    def audio(self, tag, audiodata, step=None, sample_rate=44100):
        """Saves audio.

        NB: single channel only right now.

        Args:
          tag: str: label for this data
          audiodata: ndarray [Nsamples,]: data between (-1.0,1.0) to save as wave
          step: int: training step
          sample_rate: sample rate of passed in audio buffer
        """
        audiodata = np.array(audiodata)
        if step is None:
            step = self._step
        else:
            self._step = step
        audiodata = np.clip(np.squeeze(audiodata), -1, 1)
        if audiodata.ndim != 1:
            raise ValueError("Audio data must be 1D.")
        sample_list = (32767.0 * audiodata).astype(int).tolist()
        wio = io.BytesIO()
        wav_buf = wave.open(wio, "wb")
        wav_buf.setnchannels(1)
        wav_buf.setsampwidth(2)
        wav_buf.setframerate(sample_rate)
        enc = b"".join([struct.pack("<h", v) for v in sample_list])
        wav_buf.writeframes(enc)
        wav_buf.close()
        encoded_audio_bytes = wio.getvalue()
        wio.close()
        audio = tf.compat.v1.Summary.Audio(
            sample_rate=sample_rate,
            num_channels=1,
            length_frames=len(sample_list),
            encoded_audio_string=encoded_audio_bytes,
            content_type="audio/wav",
        )
        summary = tf.compat.v1.Summary(
            value=[tf.compat.v1.Summary.Value(tag=tag, audio=audio)]
        )
        self.add_summary(summary, step)

    def histogram(self, tag, values, bins, step=None):
        """Saves histogram of values.

        Args:
          tag: str: label for this data
          values: ndarray: will be flattened by this routine
          bins: number of bins in histogram, or array of bins for np.histogram
          step: int: training step
        """
        if step is None:
            step = self._step
        else:
            self._step = step
        values = np.array(values)
        bins = np.array(bins)
        values = np.reshape(values, -1)
        counts, limits = np.histogram(values, bins=bins)
        # boundary logic
        cum_counts = np.cumsum(np.greater(counts, 0, dtype=np.int32))
        start, end = np.searchsorted(cum_counts, [0, cum_counts[-1] - 1], side="right")
        start, end = int(start), int(end) + 1
        counts = (
            counts[start - 1 : end]
            if start > 0
            else np.concatenate([[0], counts[:end]])
        )
        limits = limits[start : end + 1]
        sum_sq = values.dot(values)
        histo = tf.compat.v1.HistogramProto(
            min=values.min(),
            max=values.max(),
            num=len(values),
            sum=values.sum(),
            sum_squares=sum_sq,
            bucket_limit=limits.tolist(),
            bucket=counts.tolist(),
        )
        summary = tf.compat.v1.Summary(
            value=[tf.compat.v1.Summary.Value(tag=tag, histo=histo)]
        )
        self.add_summary(summary, step)

    def text(self, tag, textdata, step=None):
        """Saves a text summary.

        Args:
          tag: str: label for this data
          textdata: string, or 1D/2D list/numpy array of strings
          step: int: training step
        Note: markdown formatting is rendered by tensorboard.
        """
        if step is None:
            step = self._step
        else:
            self._step = step
        smd = tf.compat.v1.SummaryMetadata(
            plugin_data=tf.compat.v1.SummaryMetadata.PluginData(plugin_name="text")
        )
        if isinstance(textdata, (str, bytes)):
            tensor = tf.make_tensor_proto(
                values=[textdata.encode(encoding="utf_8")], shape=(1,)
            )
        else:
            textdata = np.array(textdata)  # convert lists, jax arrays, etc.
            datashape = np.shape(textdata)
            if len(datashape) == 1:
                tensor = tf.make_tensor_proto(
                    values=[td.encode(encoding="utf_8") for td in textdata],
                    shape=(datashape[0],),
                )
            elif len(datashape) == 2:
                tensor = tf.make_tensor_proto(
                    values=[
                        td.encode(encoding="utf_8") for td in np.reshape(textdata, -1)
                    ],
                    shape=(datashape[0], datashape[1]),
                )
        summary = tf.compat.v1.Summary(
            value=[tf.compat.v1.Summary.Value(tag=tag, metadata=smd, tensor=tensor)]
        )
        self.add_summary(summary, step)


# Copied from gin/tf/utils.py:GinConfigSaverHook
def markdownify_operative_config_str(string):
    """Convert an operative config string to markdown format."""

    # TODO(b/37527917): Total hack below. Implement more principled formatting.
    def process(line):
        """Convert a single line to markdown format."""
        if not line.startswith("#"):
            return "    " + line

        line = line[2:]
        if line.startswith("===="):
            return ""
        if line.startswith("None"):
            return "    # None."
        if line.endswith(":"):
            return "#### " + line
        return line

    output_lines = []
    for line in string.splitlines():
        procd_line = process(line)
        if procd_line is not None:
            output_lines.append(procd_line)

    return "\n".join(output_lines)
