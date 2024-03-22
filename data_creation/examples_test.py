# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for examples.py.

Run command:
```
python -m data_creation.examples_test
```
"""

from absl.testing import absltest

# pylint: disable=g-bad-import-order
from data_creation import examples
# pylint: enable=g-bad-import-order

_FORMAT = f"""
PROMPT:
{examples.PLACEHOLDERS.prompt}

TOPIC:
{examples.PLACEHOLDERS.topic}
"""


class ExamplesTest(absltest.TestCase):

  def test_fill_format_base_concepts(self) -> None:
    constructed_examples = examples.fill_format(
        _FORMAT, examples=examples.CONCEPT_EXAMPLES
    )

    self.assertEqual(len(constructed_examples), len(examples.CONCEPT_EXAMPLES))

    for example in constructed_examples:
      for placeholder in examples.PLACEHOLDERS.list_placeholders():
        self.assertNotIn(placeholder, example)
      self.assertNotIn('\n\n\n', example)

  def test_fill_format_base_objects(self) -> None:
    constructed_examples = examples.fill_format(
        _FORMAT, examples=examples.OBJECT_EXAMPLES
    )
    self.assertEqual(len(constructed_examples), len(examples.OBJECT_EXAMPLES))

    for example in constructed_examples:
      for placeholder in examples.PLACEHOLDERS.list_placeholders():
        self.assertNotIn(placeholder, example)

      self.assertNotIn('\n\n\n', example)


if __name__ == '__main__':
  absltest.main()
