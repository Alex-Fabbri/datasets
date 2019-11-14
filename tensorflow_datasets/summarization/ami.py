"""AMI Meeting Corpus"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json

import tensorflow as tf
import tensorflow_datasets.public_api as tfds

_CITATION = """
@INPROCEEDINGS{Mccowan05theami,
    author = {I. Mccowan and G. Lathoud and M. Lincoln and A. Lisowska and W. Post and D. Reidsma and P. Wellner},
    title = {The AMI Meeting Corpus},
    booktitle = {In: Proceedings Measuring Behavior 2005, 5th International Conference on Methods and Techniques
    in Behavioral Research. L.P.J.J. Noldus, F. Grieco, L.W.S. Loijens and P.H. Zimmerman (Eds.), Wageningen: Noldus Information Technology},
    year = {2005}
}
"""

_DESCRIPTION = """
The AMI Meeting Corpus consists of 100 hours of meeting recordings in English. The annotator was asked to write an
abstractive summary of a meeting and extract a subset of the meeting's dialogue acts linked with each sentence in the abstractive summary. 
"""

_URL = "https://drive.google.com/uc?export=download&id=1HFIs5e-aCO--AS_U4hAMbOzRrHw0nKap"

_EXAMPLE = "example"

class Ami(tfds.core.GeneratorBasedBuilder):
  """AMI Meeting Corpus Summarization Dataset"""

  VERSION = tfds.core.Version('1.0.0')

  def _info(self):
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            _EXAMPLE: tfds.features.Text(),
        }),
        homepage='http://groups.inf.ed.ac.uk/ami/corpus/',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager):
    """Returns SplitGenerators."""
    extract_path = os.path.join(
        dl_manager.download_and_extract(_URL), "ami_data")
    return [
        tfds.core.SplitGenerator(
            name=tfds.Split.VALIDATION,
            gen_kwargs={"path": os.path.join(extract_path, "valid")},
        ),
        tfds.core.SplitGenerator(
            name=tfds.Split.TEST,
            gen_kwargs={"path": os.path.join(extract_path, "test")},
        ),
    ]

  def _generate_examples(self, path=None):
    """Yields examples."""
    with tf.io.gfile.GFile(
      os.path.join(path + ".json")) as inputf:
      print(path)
      print(inputf)
      json_data = json.load(inputf)
      for count, key in enumerate(json_data):
        # there is a lot of info in the json_data dict
        # leaving the dict, which can be loading with
        # json.loads for now
        ex = json.dumps(json_data[key])
        yield count, {_EXAMPLE: ex}