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
"""Tests for query_serper.py.

Run command:
```
python -m eval.safe.query_serper_test
```
"""

import copy
from unittest import mock

from absl.testing import absltest

# pylint: disable=g-bad-import-order
from eval.safe import query_serper
# pylint: enable=g-bad-import-order

_TEST_SERPER_API_KEY = 'test-serper-api-key'
_TEST_GL = 'US'
_TEST_HL = 'EN'
_TEST_K = 3
_TEST_TBS = 'TEST TBS'
_TEST_SEARCH_TYPE = 'search'

_TEST_QUERY = 'What is 1 + 1?'
_TEST_GOOGLE_SEARCH_RESULT = {
    'searchParameters': {
        'q': 'Lanny Flaherty nationality',
        'gl': 'us',
        'hl': 'en',
        'num': 3,
        'type': 'search',
        'engine': 'google',
    },
    'answerBox': {
        'title': 'Lanny Flaherty / Nationality', 'answer': 'American'
    },
    'knowledgeGraph': {
        'title': 'Lanny Flaherty',
        'type': 'American actor',
        'description': 'Lanny Flaherty is an American actor.',
        'descriptionSource': 'Wikipedia',
        'descriptionLink': 'en.wikipedia.org/wiki/Lanny_Flaherty',
        'attributes': {
            'Born': '1942 (age 81 years), Pontotoc, MS',
            'Nationality': 'American',
        },
    },
    'organic': [
        {
            'title': 'Lanny Flaherty - Wikipedia',
            'link': 'en.wikipedia.org/wiki/Lanny_Flaherty',
            'snippet': (
                'Lanny Flaherty (born July 27, 1942) is an American actor.'
                ' Lanny Flaherty. Born, (1942-07-27) July 27, 1942 (age 81).'
                ' Pontotoc, Mississippi, U.S..'
            ),
            'sitelinks': [
                {
                    'title': 'Career',
                    'link': 'en.wikipedia.org/wiki/Lanny_Flaherty#Career',
                }, {
                    'title': 'Filmography',
                    'link': 'en.wikipedia.org/wiki/Lanny_Flaherty#Filmography',
                }, {
                    'title': 'Film',
                    'link': 'en.wikipedia.org/wiki/Lanny_Flaherty#Film',
                }
            ],
            'position': 1
        }, {
            'title': 'Lanny Flaherty - IMDb',
            'link': 'www.imdb.com/name/nm0280890/',
            'snippet': (
                'Lanny Flaherty was born on 27 July 1942 in Pontotoc,'
                ' Mississippi, USA. He is an actor, known for Signs (2002),'
                " Men in Black 3 (2012) and Miller's Crossing ..."
            ),
            'position': 2
        }, {
            'title': 'Lanny Flaherty - The Movie Database (TMDB)',
            'link': (
                'www.themoviedb.org/person/3204-lanny-flaherty?language=en-US'
            ),
            'snippet': (
                'Lanny Flaherty (born July 27, 1942) is an American actor.'
            ),
            'position': 3
        }
    ],
    'relatedSearches': [
        {'query': 'Lanny Flaherty net worth'},
        {'query': 'Loriann Hart Flaherty husband'},
        {'query': 'Lanny Flaherty Blood In Blood Out'}
    ]
}

_TEST_RESPONSE = 'The answer is 2.'

_TEST_SERPER_API = query_serper.SerperAPI(
    serper_api_key=_TEST_SERPER_API_KEY,
    gl=_TEST_GL,
    hl=_TEST_HL,
    k=_TEST_K,
    tbs=_TEST_TBS,
    search_type=_TEST_SEARCH_TYPE,
)


class QuerySerperTest(absltest.TestCase):

  def test_init_base(self) -> None:
    serper_api = query_serper.SerperAPI(
        serper_api_key=_TEST_SERPER_API_KEY,
        gl=_TEST_GL,
        hl=_TEST_HL,
        k=_TEST_K,
        tbs=_TEST_TBS,
        search_type=_TEST_SEARCH_TYPE,
    )
    self.assertEqual(serper_api.serper_api_key, _TEST_SERPER_API_KEY)
    self.assertEqual(serper_api.gl, _TEST_GL)
    self.assertEqual(serper_api.hl, _TEST_HL)
    self.assertEqual(serper_api.k, _TEST_K)
    self.assertEqual(serper_api.tbs, _TEST_TBS)
    self.assertEqual(serper_api.search_type, _TEST_SEARCH_TYPE)
    self.assertIsInstance(serper_api.result_key_for_type, dict)
    self.assertNotEmpty(serper_api.result_key_for_type)

  @mock.patch('eval.safe.query_serper.SerperAPI._google_serper_api_results')
  @mock.patch('eval.safe.query_serper.SerperAPI._parse_results')
  def test_run_base(
      self,
      mock_parse_results: mock.Mock,
      mock_google_serper_api_results: mock.Mock,
  ) -> None:
    test_serper_api = copy.deepcopy(_TEST_SERPER_API)
    test_serper_api.run(query=_TEST_QUERY)
    mock_google_serper_api_results.return_value = _TEST_GOOGLE_SEARCH_RESULT
    mock_parse_results.return_value = _TEST_RESPONSE
    actual_output = test_serper_api.run(query=_TEST_QUERY)
    self.assertEqual(actual_output, _TEST_RESPONSE)
    mock_google_serper_api_results.assert_called_with(
        _TEST_QUERY,
        gl=test_serper_api.gl,
        hl=test_serper_api.hl,
        num=test_serper_api.k,
        tbs=test_serper_api.tbs,
        search_type=test_serper_api.search_type,
    )
    mock_parse_results.assert_called_with(_TEST_GOOGLE_SEARCH_RESULT)

  def test_run_no_api_key(self) -> None:
    test_serper_api = copy.deepcopy(_TEST_SERPER_API)
    test_serper_api.serper_api_key = ''
    self.assertRaises(AssertionError, test_serper_api.run, query=_TEST_QUERY)

  @mock.patch('requests.post')
  def test_google_serper_api_results(self, mock_post: mock.Mock) -> None:
    test_serper_api = copy.deepcopy(_TEST_SERPER_API)
    response = mock.Mock()
    response.json.return_value = _TEST_GOOGLE_SEARCH_RESULT
    mock_post.return_value = response
    actual_output = test_serper_api._google_serper_api_results(
        search_term=_TEST_QUERY,
        gl=test_serper_api.gl,
        hl=test_serper_api.hl,
        num=test_serper_api.k,
        tbs=test_serper_api.tbs,
        search_type=test_serper_api.search_type,
    )
    self.assertEqual(actual_output, _TEST_GOOGLE_SEARCH_RESULT)
    mock_post.assert_called_once()
    response.raise_for_status.assert_called_once_with()
    response.json.assert_called_once_with()

  def test_parse_snippets_base(self) -> None:
    test_serper_api = copy.deepcopy(_TEST_SERPER_API)
    test_serper_api.k = _TEST_K
    google_search_results = copy.deepcopy(_TEST_GOOGLE_SEARCH_RESULT)
    answer = google_search_results['answerBox']['answer']
    google_search_results['answerBox']['snippet'] = answer
    google_search_results['answerBox']['snippetHighlighted'] = answer
    title = google_search_results['knowledgeGraph']['title']
    entity_type = google_search_results['knowledgeGraph']['type']
    description = google_search_results['knowledgeGraph']['description']
    attributes = google_search_results['knowledgeGraph']['attributes']
    expected_output = [
        answer,
        answer,
        answer,
        f'{title}: {entity_type}.',
        description,
        f'{title} {"Born"}: {attributes["Born"]}.',
        f'{title} {"Nationality"}: {attributes["Nationality"]}.',
    ] + [r['snippet'] for r in google_search_results['organic'][:_TEST_K]]
    actual_output = test_serper_api._parse_snippets(google_search_results)
    self.assertEqual(actual_output, expected_output)

  def test_parse_snippets_no_answerbox(self) -> None:
    test_serper_api = copy.deepcopy(_TEST_SERPER_API)
    test_serper_api.k = _TEST_K
    google_search_results = copy.deepcopy(_TEST_GOOGLE_SEARCH_RESULT)
    google_search_results.pop('answerBox')
    kg = google_search_results['knowledgeGraph']
    expected_output = (
        [f"{kg['title']}: {kg['type']}.", kg['description']]
        + [
            f"{kg['title']} {attribute}: {value}."
            for attribute, value in kg['attributes'].items()
        ]
        + [r['snippet'] for r in google_search_results['organic'][:_TEST_K]]
    )
    actual_output = test_serper_api._parse_snippets(google_search_results)
    self.assertEqual(actual_output, expected_output)

  def test_snippets_no_results(self) -> None:
    test_serper_api = copy.deepcopy(_TEST_SERPER_API)
    actual_output = test_serper_api._parse_snippets(results={})
    self.assertLen(actual_output, 1)
    self.assertEqual(actual_output[0], query_serper.NO_RESULT_MSG)

  @mock.patch('eval.safe.query_serper.SerperAPI._parse_snippets')
  def test_parse_results(self, mock_parse_snippets: mock.Mock) -> None:
    mock_parse_snippets.return_value = _TEST_RESPONSE.split(' ')
    test_serper_api = copy.deepcopy(_TEST_SERPER_API)
    actual_output = test_serper_api._parse_results(_TEST_GOOGLE_SEARCH_RESULT)
    self.assertEqual(actual_output, _TEST_RESPONSE)
    mock_parse_snippets.assert_called_once_with(_TEST_GOOGLE_SEARCH_RESULT)


if __name__ == '__main__':
  absltest.main()
