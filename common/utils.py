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
"""Shared utility functions."""

import copy
import io
import json
import os
import random
import re
import string
import types
from typing import Any

import termcolor


################################################################################
#                              RUNTIME CONTROLS                                #
################################################################################
def stop_all_execution(stop_flag: bool) -> None:
  """Immediately stops all execution."""
  if stop_flag:
    print_info('Stopping execution...')
    os._exit(1)  # pylint: disable=protected-access


################################################################################
#                              LIST MANIPULATION                               #
################################################################################
def random_selection(
    in_list: list[Any], num_to_select: int, random_seed: int
) -> list[Any]:
  """Selects a random subset of the elements in a list."""
  if num_to_select <= 0 or num_to_select >= len(in_list):
    return copy.deepcopy(in_list)

  random.seed(random_seed)
  return random.sample(in_list, num_to_select)


################################################################################
#                             STRING MANIPULATION                              #
################################################################################
def join_segments(*args: str | list[str], separator: str = '\n\n\n') -> str:
  """Joins an unspecified number of strings using the separator."""
  all_segments = []

  for arg in args:
    if isinstance(arg, list):
      all_segments.extend(arg)
    else:
      all_segments.append(strip_string(str(arg)))

  return strip_string(separator.join(all_segments))


def strip_string(s: str) -> str:
  """Strips a string of newlines and spaces."""
  return s.strip(' \n')


def extract_first_square_brackets(input_string: str) -> str:
  """Extracts the contents of the FIRST string between square brackets."""
  raw_result = re.findall(r'\[.*?\]', input_string, flags=re.DOTALL)

  if raw_result:
    return raw_result[0][1:-1]
  else:
    return ''


def extract_first_code_block(
    input_string: str, ignore_language: bool = False
) -> str:
  """Extracts the contents of a string between the first code block (```)."""
  if ignore_language:
    pattern = re.compile(r'```(?:\w+\n)?(.*?)```', re.DOTALL)
  else:
    pattern = re.compile(r'```(.*?)```', re.DOTALL)

  match = pattern.search(input_string)
  return strip_string(match.group(1)) if match else ''


################################################################################
#                            OBJECT CONVERSIONS                                #
################################################################################
def to_readable_json(json_obj: dict[Any, Any], sort_keys: bool = False) -> str:
  """Converts a json object to a readable string."""
  return f'```json\n{json.dumps(json_obj, indent=2, sort_keys=sort_keys)}\n```'


def recursive_to_saveable(value: Any) -> Any:
  """Converts a value to a saveable value."""
  if isinstance(value, dict):
    return {k: recursive_to_saveable(v) for k, v in value.items()}
  elif isinstance(value, list):
    return [recursive_to_saveable(v) for v in value]
  else:
    return str(value)


def get_attributes(package: types.ModuleType) -> dict[str, Any]:
  """Gets all attributes of a package as a dictionary."""
  result = {}

  for attr_name in dir(package):
    if not attr_name.startswith('__'):
      attr_value = getattr(package, attr_name)
      result[attr_name] = recursive_to_saveable(attr_value)

  return result


################################################################################
#                               FILE HANDLING                                  #
################################################################################
def open_file_wrapped(filepath: str, **kwargs) -> Any:
  return open(filepath, **kwargs)


def file_exists_wrapped(filepath: str, **kwargs) -> bool:
  return os.path.exists(filepath, **kwargs)


def make_directory_wrapped(filepath: str, **kwargs) -> None:
  folder_name = '/'.join(filepath.split('/')[:-1])
  os.makedirs(folder_name, exist_ok=True, **kwargs)


def listdir_wrapped(filepath: str, **kwargs) -> list[str]:
  return os.listdir(filepath, **kwargs)


def read_json(filepath: str) -> dict[Any, Any]:
  """Reads in the json file at the filepath."""
  with open_file_wrapped(filepath, mode='r') as f:
    return json.load(f)


def save_json(filepath: str, json_obj: dict[Any, Any]) -> None:
  """Saves the json object at the filepath."""
  make_directory_wrapped(filepath)

  with open_file_wrapped(filepath, mode='w') as f:
    json.dump(json_obj, f)


def read_from_jsonlines(filepath: str) -> list[dict[Any, Any]]:
  """Reads in the jsonlines file at the filepath."""
  with open_file_wrapped(filepath, mode='r') as f:
    dict_list = [json.loads(line) for line in f]
    return dict_list


def write_to_jsonlines(dict_list: list[dict[Any, Any]], filepath: str) -> None:
  """Write a list of dictionaries to a jsonlines file."""
  make_directory_wrapped(filepath)

  if not filepath.endswith('.jsonl'):
    filepath += '.jsonl'

  with open_file_wrapped(filepath, mode='w') as file:
    for item in dict_list:
      file.write(json.dumps(item) + '\n')


def save_buffer(buffer: io.BytesIO, filepath: str) -> None:
  make_directory_wrapped(filepath)
  buffer.seek(0)

  with open_file_wrapped(filepath, mode='wb') as f:
    f.write(buffer.getvalue())

  buffer.close()


################################################################################
#                                   PRINTING                                   #
################################################################################
def clear_line() -> None:
  """Clears the current line."""
  print(' ' * os.get_terminal_size().columns, end='\r')


def print_divider() -> None:
  """Prints a dividing line as wide as the terminal."""
  print('-' * os.get_terminal_size().columns)


def print_color(message: str, color: str) -> None:
  """Prints a message with a color."""
  termcolor.cprint(message, color)


def print_step_errors(step_name: str, success_rate: float) -> None:
  """Prints the success rate of a step."""
  message = f'{step_name} success rate: {round(success_rate * 100, 1)}%'
  clear_line()
  print_color(message, 'green')


def print_info(message: str, add_punctuation: bool = True) -> None:
  """Prints the message with an INFO: preamble and colored green."""
  if not message:
    return

  if add_punctuation:
    message = (
        f'{message}.' if message[-1] not in string.punctuation else message
    )
  clear_line()
  print_color(f'INFO: {message}', color='green')


def maybe_print_error(
    message: str | Exception | None,
    additional_info: str = '',
    verbose: bool = False
) -> None:
  """Prints the error message with additional info if flag is True."""
  if not strip_string(str(message)):
    return

  error = type(message).__name__ if isinstance(message, Exception) else 'ERROR'
  message = f'{error}: {str(message)}'
  message += f'\n{additional_info}' if verbose else ''
  clear_line()
  print_color(message, color='red')


def print_progress(sentence: str, progress: int, out_of: int) -> None:
  """Prints the given sentences with a progress bar."""
  if progress == 0:
    print()

  sentence += f': {progress}/{out_of} '
  num_remaining = os.get_terminal_size().columns - len(sentence) - 2

  if num_remaining >= 5:
    num_fill = int(float(progress / out_of) * num_remaining) or 1
    fill = '=' * num_fill
    fill = fill[1:] + '>' if fill else '>'
    empty = ' ' * (num_remaining - num_fill)
    sentence = f'{sentence}[{fill}{empty}]'

  print(termcolor.colored(sentence, 'green'), end='\r')

  if progress == out_of:
    print()


def print_side_by_side(
    list1: list[str], list2: list[str], headers: tuple[str, str]
) -> None:
  """Prints two lists for side-by-side comparison."""
  def split_to_chunks(s: str, width: int) -> list[str]:
    """Split string into chunks of specified width, preserving ANSI escapes."""
    ansi_escape = re.compile(r'(\x1b[^m]*m)')
    lines, chunks = s.split('\n'), []

    for line in lines:
      current_chunk, current_length = '', 0

      # Splitting the line based on ANSI escape codes
      fragments = ansi_escape.split(line)

      for fragment in fragments:
        if ansi_escape.fullmatch(fragment):
          current_chunk += fragment
        else:
          for char in fragment:
            if current_length + len(char) > width:
              # Wrap to next chunk
              chunks.append(current_chunk)

              # Re-apply escape codes from previous line if they exist
              if len(ansi_escape.findall(current_chunk)) >= 1:
                current_chunk = '' + ansi_escape.findall(current_chunk)[-1]
              else:
                current_chunk = ''

              current_length = 0

            current_chunk += char
            current_length += len(char)

      if current_chunk:
        chunks.append(current_chunk)

    return chunks

  def pad_string_with_ansi(s: str, width: int) -> str:
    """Pad a string containing ANSI escape codes to the specified width."""
    ansi_escape = re.compile(r'(\x1b[^m]*m)')  # Extract text without ANSI
    text_without_ansi = ansi_escape.sub('', s)

    padding_needed = width - len(text_without_ansi)
    padded_string = s + ' ' * padding_needed

    return padded_string

  terminal_width = os.get_terminal_size().columns
  per_column_width = (terminal_width - 3) // 2
  list1, list2 = [headers[0]] + list1, [headers[1]] + list2

  for item1, item2 in zip(list1, list2):
    print_divider()
    chunks1 = split_to_chunks(item1, per_column_width)
    chunks2 = split_to_chunks(item2, per_column_width)
    max_chunks = max(len(chunks1), len(chunks2))

    # Equalize number of chunks for both sides by adding empty strings
    while len(chunks1) < max_chunks:
      chunks1.append('' * per_column_width)
    while len(chunks2) < max_chunks:
      chunks2.append('')

    for c1, c2 in zip(chunks1, chunks2):
      c1 = pad_string_with_ansi(c1, per_column_width)
      c2 = pad_string_with_ansi(c2, per_column_width)
      print(
          f'\x1b[0m{c1:<{per_column_width}}\x1b[0m |'
          f' \x1b[0m{c2:<{per_column_width}}'
      )

  print_divider()
