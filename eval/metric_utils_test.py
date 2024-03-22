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
"""Tests for metric_utils.py.

Run command:
```
python -m eval.metric_utils_test
```
"""

import math

from absl.testing import absltest

# pylint: disable=g-bad-import-order
from eval import metric_utils
# pylint: enable=g-bad-import-order

_SUPPORTED = 10
_NOT_SUPPORTED = 8
_TEST_MAX_CLAIMS = 100

_TEST_NUMBER = 12345.6789


class MetricUtilsTest(absltest.TestCase):

  def test_calculate_metrics_base(self) -> None:
    actual_f1 = metric_utils.calculate_metrics(
        supported=_SUPPORTED,
        not_supported=_NOT_SUPPORTED,
        max_claims=_TEST_MAX_CLAIMS,
    )
    self.assertGreaterEqual(actual_f1, 0)
    self.assertLessEqual(actual_f1, 1)

  def test_calculate_metrics_invalid_max_claims(self) -> None:
    self.assertRaises(
        ValueError,
        metric_utils.calculate_metrics,
        supported=_SUPPORTED,
        not_supported=_NOT_SUPPORTED,
        max_claims=0,
    )
    self.assertRaises(
        ValueError,
        metric_utils.calculate_metrics,
        supported=_SUPPORTED,
        not_supported=_NOT_SUPPORTED,
        max_claims=-1,
    )

  def test_calculate_metrics_zero_claims(self) -> None:
    actual_f1 = metric_utils.calculate_metrics(
        supported=0, not_supported=_NOT_SUPPORTED, max_claims=_TEST_MAX_CLAIMS
    )
    self.assertEqual(actual_f1, 0)

  def test_calculate_metrics_negative_input(self) -> None:
    self.assertRaises(
        ValueError,
        metric_utils.calculate_metrics,
        supported=-1,
        not_supported=_NOT_SUPPORTED,
        max_claims=_TEST_MAX_CLAIMS,
    )
    self.assertRaises(
        ValueError,
        metric_utils.calculate_metrics,
        supported=_SUPPORTED,
        not_supported=-1,
        max_claims=_TEST_MAX_CLAIMS,
    )

  def test_round_to_sigfigs(self) -> None:
    self.assertAlmostEqual(
        metric_utils.round_to_sigfigs(_TEST_NUMBER, num_sigfigs=1), 10000
    )
    self.assertAlmostEqual(
        metric_utils.round_to_sigfigs(_TEST_NUMBER, num_sigfigs=3), 12300
    )
    self.assertAlmostEqual(
        metric_utils.round_to_sigfigs(_TEST_NUMBER, num_sigfigs=6), 12345.7
    )
    self.assertAlmostEqual(
        metric_utils.round_to_sigfigs(-_TEST_NUMBER, num_sigfigs=3), -12300
    )
    self.assertAlmostEqual(
        metric_utils.round_to_sigfigs(-_TEST_NUMBER, num_sigfigs=6), -12345.7
    )
    self.assertAlmostEqual(
        metric_utils.round_to_sigfigs(_TEST_NUMBER % 1, num_sigfigs=2), 0.68
    )
    self.assertAlmostEqual(
        metric_utils.round_to_sigfigs(_TEST_NUMBER % 1, num_sigfigs=3), 0.679
    )
    self.assertEqual(metric_utils.round_to_sigfigs(9999, 1), 10000)
    self.assertEqual(metric_utils.round_to_sigfigs(9999, 2), 10000)

  def test_round_to_sigfigs_unnecessary(self) -> None:
    self.assertEqual(metric_utils.round_to_sigfigs(_TEST_NUMBER, -1), 0)
    self.assertEqual(metric_utils.round_to_sigfigs(_TEST_NUMBER, -0), 0)
    self.assertEqual(metric_utils.round_to_sigfigs(0, 3), 0)
    self.assertEqual(
        metric_utils.round_to_sigfigs(_TEST_NUMBER, num_sigfigs=10),
        _TEST_NUMBER,
    )
    self.assertTrue(
        math.isnan(metric_utils.round_to_sigfigs(float('nan'), num_sigfigs=3))
    )

if __name__ == '__main__':
  absltest.main()
