"""Test for AMI Meeting Corpus dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow_datasets import testing
from tensorflow_datasets.summarization import ami


class AmiTest(testing.DatasetBuilderTestCase):
  DATASET_CLASS = ami.Ami
  SPLITS = {
      "validation": 3,  # Number of fake train example
      "test": 1,  # Number of fake test example
  }
  DL_EXTRACT_RESULT = ""

if __name__ == "__main__":
  testing.test_main()

