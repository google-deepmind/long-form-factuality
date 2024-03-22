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
"""Tests for utils.py.

Run command:
```
python -m common.utils_test
```
"""

import copy
import os
import types
from unittest import mock

from absl.testing import absltest

# pylint: disable=g-bad-import-order
from common import utils
# pylint: enable=g-bad-import-order

_TEST_NUM_ITEMS = 10
_TEST_LIST = list(range(_TEST_NUM_ITEMS))
_TEST_DICT = {f'key_{i}': f'value_{i}' for i in range(_TEST_NUM_ITEMS)}

_TEST_RANDOM_SEED = 1

_TEST_STRING = 'SAMPLE STRING'
_TEST_FILEPATH = '/some/test/path/'
_SQUARE_BRACKET_STRINGS = [f'[{i}]' for i in range(_TEST_NUM_ITEMS)]


class UtilsTest(absltest.TestCase):

  @mock.patch('common.utils.print_info')
  @mock.patch('os._exit')
  def test_stop_all_execution_base(
      self, mock_exit: mock.Mock, mock_print_info: mock.Mock
  ) -> None:
    utils.stop_all_execution(stop_flag=True)
    mock_exit.assert_called_once()
    mock_print_info.assert_called_once()

  @mock.patch('common.utils.print_info')
  @mock.patch('os._exit')
  def test_stop_all_execution_not_necessary(
      self, mock_exit: mock.Mock, mock_print_info: mock.Mock
  ) -> None:
    utils.stop_all_execution(stop_flag=False)
    mock_exit.assert_not_called()
    mock_print_info.assert_not_called()

  @mock.patch('random.seed')
  @mock.patch('random.sample')
  def test_random_selection_base(
      self, mock_sample: mock.Mock, mock_seed: mock.Mock
  ) -> None:
    test_list = copy.deepcopy(_TEST_LIST)
    num_to_select = len(test_list) - 1
    mock_sample.return_value = copy.deepcopy(_TEST_LIST)[:num_to_select]
    assert num_to_select > 0
    actual_output = utils.random_selection(
        test_list, num_to_select=num_to_select, random_seed=_TEST_RANDOM_SEED
    )
    self.assertLen(actual_output, num_to_select)
    mock_seed.assert_called_once_with(_TEST_RANDOM_SEED)
    mock_sample.assert_called_once_with(test_list, num_to_select)

  @mock.patch('random.seed')
  @mock.patch('random.sample')
  def test_random_selection_unnecessary(
      self, mock_sample: mock.Mock, mock_seed: mock.Mock
  ) -> None:
    test_list = copy.deepcopy(_TEST_LIST)
    actual_output = utils.random_selection(
        in_list=test_list, num_to_select=-1, random_seed=_TEST_RANDOM_SEED
    )
    self.assertEqual(actual_output, test_list)
    test_list = copy.deepcopy(_TEST_LIST)
    actual_output = utils.random_selection(
        in_list=test_list,
        num_to_select=len(test_list),
        random_seed=_TEST_RANDOM_SEED,
    )
    self.assertEqual(actual_output, test_list)
    mock_sample.assert_not_called()
    mock_seed.assert_not_called()

  def test_join_segments(self) -> None:
    segment_1 = 'SAMPLE PREAMBLE'
    segment_2 = [f'FEW SHOT EXAMPLE #{i}' for i in range(_TEST_NUM_ITEMS)]
    segment_3 = 'SAMPLE POSTAMBLE'
    separator = '\n\n\n'
    joined_segments = utils.join_segments(
        segment_1, segment_2, segment_3, separator=separator
    )
    self.assertEqual(joined_segments.count(separator), len(segment_2) + 1)

    for segment in [segment_1] + segment_2 + [segment_3]:
      self.assertIn(segment, joined_segments)

  def test_strip_string(self) -> None:
    self.assertEqual(utils.strip_string('\n' + _TEST_STRING), _TEST_STRING)
    self.assertEqual(utils.strip_string('\n\n' + _TEST_STRING), _TEST_STRING)
    self.assertEqual(utils.strip_string('  ' + _TEST_STRING), _TEST_STRING)
    self.assertEqual(utils.strip_string('\n ' + _TEST_STRING), _TEST_STRING)
    self.assertEqual(utils.strip_string(_TEST_STRING + '\n'), _TEST_STRING)
    self.assertEqual(utils.strip_string(_TEST_STRING + '\n\n'), _TEST_STRING)
    self.assertEqual(utils.strip_string(_TEST_STRING + '\n '), _TEST_STRING)
    self.assertEqual(utils.strip_string(_TEST_STRING + '  '), _TEST_STRING)
    self.assertEqual(utils.strip_string(f' {_TEST_STRING} '), _TEST_STRING)

  def test_extract_first_square_brackets_base(self) -> None:
    test_string = '\n'.join(_SQUARE_BRACKET_STRINGS)
    self.assertEqual(
        _SQUARE_BRACKET_STRINGS[0][1:-1],
        utils.extract_first_square_brackets(test_string),
    )

  def test_extract_first_code_block_base(self) -> None:
    test_string = f'```\n{_TEST_STRING}\n```'
    self.assertEqual(_TEST_STRING, utils.extract_first_code_block(test_string))

  def test_extract_first_code_block_with_language(self) -> None:
    test_string = f'```python\n{_TEST_STRING}\n```'
    self.assertEqual(
        _TEST_STRING,
        utils.extract_first_code_block(test_string, ignore_language=True),
    )
    self.assertNotEqual(
        _TEST_STRING,
        utils.extract_first_code_block(test_string, ignore_language=False),
    )

  def test_extract_first_code_block_no_code_block(self) -> None:
    self.assertEmpty(utils.extract_first_code_block(_TEST_STRING))

  def test_to_readable_json(self) -> None:
    actual_output = utils.to_readable_json(_TEST_DICT)
    self.assertTrue(actual_output.startswith('```json'))
    self.assertTrue(actual_output.endswith('```'))

    for key, value in _TEST_DICT.items():
      self.assertIn(key, actual_output)
      self.assertIn(value, actual_output)

  def test_recursive_to_saveable_base(self) -> None:
    test_objects = [None, _TEST_NUM_ITEMS, float(_TEST_NUM_ITEMS), _TEST_STRING]

    for each in test_objects:
      self.assertEqual(utils.recursive_to_saveable(each), str(each))

  def test_recursive_to_saveable_dict_input(self) -> None:
    actual_output = utils.recursive_to_saveable(_TEST_DICT)

    for key, value in _TEST_DICT.items():
      self.assertIn(key, actual_output)
      self.assertEqual(str(value), actual_output[key])

  def test_recursive_to_saveable_list_input(self) -> None:
    test_objects = [None, _TEST_NUM_ITEMS, float(_TEST_NUM_ITEMS), _TEST_STRING]
    actual_output = utils.recursive_to_saveable(test_objects)

    for each in test_objects:
      self.assertIn(str(each), actual_output)

  def test_get_attributes(self) -> None:
    mock_module = types.ModuleType('SAMPLE MODULE')
    mock_module.random_seed = _TEST_RANDOM_SEED
    mock_module.string = _TEST_STRING
    mock_module.list = copy.deepcopy(_TEST_LIST)
    mock_module.dict = copy.deepcopy(_TEST_DICT)
    expected_output = {
        'random_seed': str(_TEST_RANDOM_SEED),
        'string': _TEST_STRING,
        'list': [str(item) for item in _TEST_LIST],
        'dict': {str(key): str(value) for key, value in _TEST_DICT.items()},
    }
    actual_output = utils.get_attributes(mock_module)
    self.assertEqual(actual_output, expected_output)

  @mock.patch('common.utils.open_file_wrapped')
  @mock.patch('json.load')
  def test_read_json(
      self, mock_json_load: mock.Mock, mock_open: mock.Mock
  ) -> None:
    mock_json_load.return_value = _TEST_DICT
    actual_output = utils.read_json(filepath=_TEST_FILEPATH)
    self.assertEqual(actual_output, _TEST_DICT)
    mock_open.assert_called_once_with(_TEST_FILEPATH, mode='r')
    mock_json_load.assert_called_once()

  @mock.patch('common.utils.open_file_wrapped')
  @mock.patch('common.utils.make_directory_wrapped')
  @mock.patch('json.dump')
  def test_save_json(
      self,
      mock_json_dump: mock.Mock,
      mock_make_directory_wrapped: mock.Mock,
      mock_open: mock.Mock,
  ) -> None:
    utils.save_json(filepath=_TEST_FILEPATH, json_obj=_TEST_DICT)
    mock_make_directory_wrapped.assert_called_once_with(_TEST_FILEPATH)
    mock_json_dump.assert_called_once()
    mock_open.assert_called_once_with(_TEST_FILEPATH, mode='w')

  @mock.patch('common.utils.open_file_wrapped')
  @mock.patch('json.loads')
  def test_read_from_jsonlines(
      self, mock_json_loads: mock.Mock, mock_open: mock.Mock
  ) -> None:
    mock_open.return_value.__enter__.return_value = [
        str(_TEST_DICT) for _ in range(_TEST_NUM_ITEMS)
    ]
    mock_json_loads.return_value = _TEST_DICT
    actual_output = utils.read_from_jsonlines(filepath=_TEST_FILEPATH)
    self.assertLen(actual_output, _TEST_NUM_ITEMS)
    self.assertEqual(
        actual_output,
        [copy.deepcopy(_TEST_DICT) for _ in range(_TEST_NUM_ITEMS)],
    )
    mock_open.assert_called_once_with(_TEST_FILEPATH, mode='r')
    mock_json_loads.assert_called()

  @mock.patch('common.utils.open_file_wrapped')
  @mock.patch('common.utils.make_directory_wrapped')
  @mock.patch('json.dumps')
  def test_write_to_jsonlines(
      self,
      mock_json_dumps: mock.Mock,
      mock_make_directory_wrapped: mock.Mock,
      mock_open: mock.Mock,
  ) -> None:
    test_jsonl = [copy.deepcopy(_TEST_DICT) for _ in range(_TEST_NUM_ITEMS)]
    utils.write_to_jsonlines(test_jsonl, filepath=_TEST_FILEPATH)
    mock_make_directory_wrapped.assert_called_once_with(_TEST_FILEPATH)
    mock_open.assert_called_once_with(_TEST_FILEPATH + '.jsonl', mode='w')
    mock_json_dumps.assert_called_with(_TEST_DICT)

  @mock.patch('common.utils.open_file_wrapped')
  @mock.patch('common.utils.make_directory_wrapped')
  @mock.patch('os.path.dirname')
  def test_save_buffer(
      self,
      mock_dirname: mock.Mock,
      mock_make_dir: mock.Mock,
      mock_open: mock.Mock,
  ) -> None:
    mock_file = mock_open.return_value.__enter__.return_value
    mock_file.write.return_value = None
    mock_buffer = mock.Mock()
    mock_buffer.seek.return_value = None
    mock_buffer.getvalue.return_value = _TEST_STRING
    mock_buffer.close.return_value = None
    mock_dirname.return_value = _TEST_FILEPATH
    utils.save_buffer(filepath=_TEST_FILEPATH, buffer=mock_buffer)
    mock_make_dir.assert_called_once_with(_TEST_FILEPATH)
    mock_open.assert_called_once_with(_TEST_FILEPATH, mode='wb')
    mock_file.write.assert_called_once_with(_TEST_STRING)
    mock_buffer.seek.assert_called_once_with(0)
    mock_buffer.getvalue.assert_called_once_with()
    mock_buffer.close.assert_called_once_with()

  @mock.patch('os.get_terminal_size')
  @mock.patch('builtins.print')
  def test_clear_line(
      self, mock_print: mock.Mock, mock_get_terminal_size: mock.Mock
  ) -> None:
    mock_get_terminal_size.return_value = os.terminal_size((100, 100))
    utils.clear_line()
    mock_get_terminal_size.assert_called_once_with()
    mock_print.assert_called_once()

  @mock.patch('os.get_terminal_size')
  @mock.patch('builtins.print')
  def test_print_divider(
      self, mock_print: mock.Mock, mock_get_terminal_size: mock.Mock
  ) -> None:
    mock_get_terminal_size.return_value = os.terminal_size((100, 100))
    utils.print_divider()
    mock_get_terminal_size.assert_called_once_with()
    mock_print.assert_called_once()

  @mock.patch('termcolor.cprint')
  def test_print_color(self, mock_cprint: mock.Mock) -> None:
    test_color = 'red'
    utils.print_color(message=_TEST_STRING, color=test_color)
    mock_cprint.assert_called_once_with(_TEST_STRING, test_color)

  @mock.patch('common.utils.print_color')
  @mock.patch('common.utils.clear_line')
  def test_print_step_errors(
      self, mock_clear_line: mock.Mock, mock_print_color: mock.Mock
  ) -> None:
    utils.print_step_errors(step_name=_TEST_STRING, success_rate=0.0)
    mock_clear_line.assert_called_once_with()
    mock_print_color.assert_called_once()

  @mock.patch('common.utils.print_color')
  @mock.patch('common.utils.clear_line')
  def test_print_info_base(
      self, mock_clear_line: mock.Mock, mock_print_color: mock.Mock
  ) -> None:
    utils.print_info(message=_TEST_STRING, add_punctuation=True)
    mock_clear_line.assert_called_once_with()
    mock_print_color.assert_called_once()

  @mock.patch('common.utils.print_color')
  @mock.patch('common.utils.clear_line')
  def test_print_info_no_message(
      self, mock_clear_line: mock.Mock, mock_print_color: mock.Mock
  ) -> None:
    utils.print_info(message='', add_punctuation=True)
    mock_clear_line.assert_not_called()
    mock_print_color.assert_not_called()

  @mock.patch('common.utils.print_color')
  @mock.patch('common.utils.clear_line')
  @mock.patch('common.utils.strip_string')
  def test_maybe_print_error_base(
      self,
      mock_strip_string: mock.Mock,
      mock_clear_line: mock.Mock,
      mock_print_color: mock.Mock,
  ) -> None:
    mock_strip_string.return_value = _TEST_STRING
    utils.maybe_print_error(
        message=_TEST_STRING, additional_info=_TEST_STRING, verbose=True
    )
    mock_clear_line.assert_called_once_with()
    mock_print_color.assert_called_once_with(
        f'ERROR: {_TEST_STRING}\n{_TEST_STRING}', color='red'
    )

  @mock.patch('common.utils.print_color')
  @mock.patch('common.utils.clear_line')
  @mock.patch('common.utils.strip_string')
  def test_maybe_print_error_no_message(
      self,
      mock_strip_string: mock.Mock,
      mock_clear_line: mock.Mock,
      mock_print_color: mock.Mock,
  ) -> None:
    mock_strip_string.return_value = ''
    utils.maybe_print_error(message='', additional_info='', verbose=True)
    mock_clear_line.assert_not_called()
    mock_print_color.assert_not_called()

  @mock.patch('common.utils.print_color')
  @mock.patch('common.utils.clear_line')
  @mock.patch('common.utils.strip_string')
  def test_maybe_print_error_not_verbose(
      self,
      mock_strip_string: mock.Mock,
      mock_clear_line: mock.Mock,
      mock_print_color: mock.Mock,
  ) -> None:
    mock_strip_string.return_value = _TEST_STRING
    utils.maybe_print_error(
        message=_TEST_STRING, additional_info=_TEST_STRING, verbose=False
    )
    mock_clear_line.assert_called_once_with()
    mock_print_color.assert_called_once_with(
        f'ERROR: {_TEST_STRING}', color='red'
    )

  @mock.patch('common.utils.print_color')
  @mock.patch('common.utils.clear_line')
  @mock.patch('common.utils.strip_string')
  def test_maybe_print_error_exception(
      self,
      mock_strip_string: mock.Mock,
      mock_clear_line: mock.Mock,
      mock_print_color: mock.Mock,
  ) -> None:
    mock_strip_string.return_value = _TEST_STRING
    utils.maybe_print_error(
        message=ValueError(_TEST_STRING), additional_info='', verbose=False
    )
    mock_clear_line.assert_called_once_with()
    mock_print_color.assert_called_once_with(
        f'ValueError: {_TEST_STRING}', color='red'
    )

  @mock.patch('builtins.print')
  @mock.patch('termcolor.colored')
  @mock.patch('os.get_terminal_size')
  def test_print_progress_base(
      self,
      mock_get_terminal_size: mock.Mock,
      mock_colored: mock.Mock,
      mock_print: mock.Mock,
  ) -> None:
    num_columns = len(_TEST_STRING) * len(_TEST_STRING)
    assert num_columns >= len(_TEST_STRING) + 7
    mock_get_terminal_size.return_value = os.terminal_size(
        (num_columns, num_columns)
    )
    utils.print_progress(sentence=_TEST_STRING, progress=1, out_of=10)
    mock_get_terminal_size.assert_called_once_with()
    mock_colored.assert_called_once()
    mock_print.assert_called_once()

  @mock.patch('builtins.print')
  @mock.patch('termcolor.colored')
  @mock.patch('os.get_terminal_size')
  def test_print_progress_start(
      self,
      mock_get_terminal_size: mock.Mock,
      mock_colored: mock.Mock,
      mock_print: mock.Mock,
  ) -> None:
    num_columns = len(_TEST_STRING) * len(_TEST_STRING)
    assert num_columns >= len(_TEST_STRING) + 7
    mock_get_terminal_size.return_value = os.terminal_size(
        (num_columns, num_columns)
    )
    utils.print_progress(sentence=_TEST_STRING, progress=0, out_of=10)
    mock_get_terminal_size.assert_called_once_with()
    mock_colored.assert_called_once()
    mock_print.assert_called()
    mock_print.assert_has_calls([()])

  @mock.patch('builtins.print')
  @mock.patch('termcolor.colored')
  @mock.patch('os.get_terminal_size')
  def test_print_progress_end(
      self,
      mock_get_terminal_size: mock.Mock,
      mock_colored: mock.Mock,
      mock_print: mock.Mock,
  ) -> None:
    num_columns = len(_TEST_STRING) * len(_TEST_STRING)
    assert num_columns >= len(_TEST_STRING) + 7
    mock_get_terminal_size.return_value = os.terminal_size(
        (num_columns, num_columns)
    )
    utils.print_progress(sentence=_TEST_STRING, progress=10, out_of=10)
    mock_get_terminal_size.assert_called_once_with()
    mock_colored.assert_called_once()
    mock_print.assert_called()
    mock_print.assert_has_calls([()])

  @mock.patch('builtins.print')
  @mock.patch('termcolor.colored')
  @mock.patch('os.get_terminal_size')
  def test_print_progress_small_terminal(
      self,
      mock_get_terminal_size: mock.Mock,
      mock_colored: mock.Mock,
      mock_print: mock.Mock,
  ) -> None:
    mock_get_terminal_size.return_value = os.terminal_size((1, 1))
    utils.print_progress(sentence=_TEST_STRING, progress=1, out_of=10)
    mock_get_terminal_size.assert_called_once_with()
    mock_colored.assert_called_once_with(_TEST_STRING + ': 1/10 ', 'green')
    mock_print.assert_called()

  @mock.patch('common.utils.print_divider')
  @mock.patch('builtins.print')
  @mock.patch('os.get_terminal_size')
  def test_print_side_by_side_base(
      self,
      mock_get_terminal_size: mock.Mock,
      mock_print: mock.Mock,
      mock_print_divider: mock.Mock,
  ) -> None:
    mock_get_terminal_size.return_value = os.terminal_size((100, 100))
    side_1 = [_TEST_STRING for _ in range(_TEST_NUM_ITEMS)]
    side_2 = [_TEST_STRING for _ in range(_TEST_NUM_ITEMS)]
    utils.print_side_by_side(
        side_1, side_2, headers=(_TEST_STRING, _TEST_STRING)
    )
    mock_get_terminal_size.assert_called_once_with()
    mock_print_divider.assert_called_with()
    mock_print.assert_called()

  @mock.patch('common.utils.print_divider')
  @mock.patch('builtins.print')
  @mock.patch('os.get_terminal_size')
  def test_print_side_by_side_uneven_lists(
      self,
      mock_get_terminal_size: mock.Mock,
      mock_print: mock.Mock,
      mock_print_divider: mock.Mock,
  ) -> None:
    mock_get_terminal_size.return_value = os.terminal_size((100, 100))
    side_1 = [_TEST_STRING for _ in range(_TEST_NUM_ITEMS)]
    side_2 = [_TEST_STRING for _ in range(_TEST_NUM_ITEMS * 2)]
    utils.print_side_by_side(
        side_1, side_2, headers=(_TEST_STRING, _TEST_STRING)
    )
    mock_get_terminal_size.assert_called_once_with()
    mock_print_divider.assert_called_with()
    mock_print.assert_called()


if __name__ == '__main__':
  absltest.main()
