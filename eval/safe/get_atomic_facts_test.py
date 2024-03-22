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
"""Tests for get_atomic_facts.py.

Run command:
```
python -m eval.safe.get_atomic_facts_test
```
"""

from unittest import mock

from absl.testing import absltest

# pylint: disable=g-bad-import-order
from common import modeling
from eval.safe import get_atomic_facts
# pylint: enable=g-bad-import-order

_TEST_SENTENCE = 'This is a test sentence.'

_NUM_SENTENCES = 5
TEST_SENTENCES = [f'{_TEST_SENTENCE} #{i}' for i in range(_NUM_SENTENCES)]
_TEST_ATOMIC_FACTS = [sentence.split(' ') for sentence in TEST_SENTENCES]
_TUPLIZED_DATA = zip(TEST_SENTENCES, _TEST_ATOMIC_FACTS)
_DATA_AS_DICTS = [
    {'sentence': sentence, 'atomic_facts': atomic_facts}
    for sentence, atomic_facts in _TUPLIZED_DATA
]

_TEST_RESPONSE = 'This is a test response.'
_TEST_MODEL = modeling.FakeModel()


class GetAtomicFactsTest(absltest.TestCase):

  def test_convert_atomic_facts_to_dicts(self) -> None:
    actual_output = get_atomic_facts.convert_atomic_facts_to_dicts(
        outputted_facts=zip(TEST_SENTENCES, _TEST_ATOMIC_FACTS)
    )
    self.assertEqual(actual_output, _DATA_AS_DICTS)

  @mock.patch('third_party.factscore.atomic_facts.AtomicFactGenerator')
  @mock.patch('eval.safe.get_atomic_facts.convert_atomic_facts_to_dicts')
  def test_main(
      self,
      mock_convert_atomic_facts_to_dicts: mock.Mock,
      mock_atomic_fact_generator: mock.Mock
  ) -> None:
    test_atomic_fact_generator = mock.Mock()
    mock_atomic_fact_generator.return_value = test_atomic_fact_generator
    test_atomic_fact_generator.run.return_value = _TUPLIZED_DATA, None
    mock_convert_atomic_facts_to_dicts.return_value = _DATA_AS_DICTS
    num_claims = sum([len(d['atomic_facts']) for d in _DATA_AS_DICTS])
    expected_output = {
        'num_claims': num_claims,
        'sentences_and_atomic_facts': _TUPLIZED_DATA,
        'all_atomic_facts': _DATA_AS_DICTS,
    }
    actual_output = get_atomic_facts.main(
        response=_TEST_RESPONSE, model=_TEST_MODEL
    )
    self.assertEqual(actual_output, expected_output)
    mock_atomic_fact_generator.assert_called_once()
    test_atomic_fact_generator.run.assert_called_once()
    mock_convert_atomic_facts_to_dicts.assert_called_once()


if __name__ == '__main__':
  absltest.main()
