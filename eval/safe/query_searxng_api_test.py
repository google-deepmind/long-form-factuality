"""Tests for searxng_api.py.

This test suite provides comprehensive testing for the SearxNG API wrapper class,
combining detailed test coverage with extensive mocking for external dependencies.

Testing Strategy:
    1. Unit Tests:
       - Individual method testing
       - Configuration validation
       - Error handling
       - Edge cases
    
    2. Mock Implementation:
       - External API call simulation
       - Response data control
       - Error condition testing
       - Network behavior simulation

Libraries Used:
    - unittest.mock: Mock object library for simulating external dependencies
    - absltest: Google's testing framework for enhanced assertions
    - copy: Deep copying of test data to prevent cross-test contamination
    - requests: HTTP client library (mocked in tests)

Mocking Strategy:
    The test suite uses several types of mocks:
    1. Method Mocks: Replace class methods for controlled testing
    2. HTTP Mocks: Simulate API responses without network calls
    3. Response Mocks: Provide predefined response data
    4. Error Mocks: Simulate various error conditions

Run command:
```
python -m eval.safe.searxng_api_test
```
"""

import copy
import json
from unittest import mock
import requests
from absl.testing import absltest
from typing import Any, Dict, List, Optional, Union

# Update this import to match your project structure
from eval.safe import searxng_api

# Test constants representing typical configuration values
_TEST_BASE_URL = 'https://test-searxng-instance.com'
_TEST_API_KEY = 'test-api-key'
_TEST_LANGUAGE = 'en'
_TEST_K = 3
_TEST_TIME_RANGE = 'day'
_TEST_CATEGORIES = ['general']
_TEST_SEARCH_FORMAT = 'json'
_TEST_QUERY = 'What is 1 + 1?'

# Comprehensive mock response data matching SearxNG API format
_TEST_SEARXNG_RESULT = {
    'query': 'Lanny Flaherty nationality',
    'number_of_results': 3,
    'results': [
        {
            'title': 'Lanny Flaherty - Wikipedia',
            'url': 'https://en.wikipedia.org/wiki/Lanny_Flaherty',
            'content': ('Lanny Flaherty (born July 27, 1942) is an American actor.'
                       ' Born in Pontotoc, Mississippi, U.S..'),
            'publisher': 'Wikipedia',
            'highlighted': 'American actor Lanny Flaherty',
            'category': 'general',
            'score': 0.95,
            'engine': 'wikipedia',
            'parsed_url': {
                'scheme': 'https',
                'netloc': 'en.wikipedia.org',
                'path': '/wiki/Lanny_Flaherty'
            }
        },
        {
            'title': 'Lanny Flaherty - IMDb',
            'url': 'https://www.imdb.com/name/nm0280890/',
            'content': ('Lanny Flaherty was born on 27 July 1942 in Pontotoc,'
                       ' Mississippi, USA. He is an actor, known for Signs (2002)'),
            'publisher': 'IMDb',
            'category': 'general',
            'score': 0.85
        },
        {
            'title': 'Lanny Flaherty - TMDB',
            'url': 'https://www.themoviedb.org/person/3204-lanny-flaherty',
            'content': 'Lanny Flaherty (born July 27, 1942) is an American actor.',
            'publisher': 'TMDB',
            'category': 'general',
            'score': 0.75
        }
    ],
    'answers': ['American actor'],
    'corrections': [],
    'infoboxes': [{
        'infobox': 'Knowledge Graph',
        'content': {
            'title': 'Lanny Flaherty',
            'type': 'Actor',
            'birth': '1942',
            'nationality': 'American'
        }
    }],
    'suggestions': ['Lanny Flaherty movies', 'Lanny Flaherty biography'],
    'unresponsive_engines': []
}

_TEST_RESPONSE = 'Lanny Flaherty is an American actor.'

class MockResponse:
    """Custom mock response class for simulating HTTP responses.
    
    Attributes:
        status_code: HTTP status code
        _json_data: Data to return from json() method
        _raise_error: Whether to raise an error on raise_for_status()
    """
    
    def __init__(self, status_code: int, json_data: Dict[str, Any], raise_error: bool = False):
        self.status_code = status_code
        self._json_data = json_data
        self._raise_error = raise_error

    def json(self) -> Dict[str, Any]:
        """Return JSON response data."""
        return self._json_data

    def raise_for_status(self) -> None:
        """Simulate response status checking."""
        if self._raise_error:
            raise requests.exceptions.HTTPError(f"HTTP Error: {self.status_code}")

class SearxNGAPITest(absltest.TestCase):
    """Comprehensive test suite for the SearxNGAPI class.
    
    This test suite combines unit tests with mock-based integration testing
    to verify all aspects of the SearxNG API wrapper functionality.
    
    Test Categories:
        1. Initialization Tests
        2. API Call Tests
        3. Response Parsing Tests
        4. Error Handling Tests
        5. Integration Tests
    """

    def setUp(self) -> None:
        """Set up test fixtures before each test method.
        
        Creates:
            1. Fresh API instance
            2. Mock response objects
            3. Test data copies
        """
        self.test_api = searxng_api.SearxNGAPI(
            base_url=_TEST_BASE_URL,
            api_key=_TEST_API_KEY,
            language=_TEST_LANGUAGE,
            k=_TEST_K,
            time_range=_TEST_TIME_RANGE,
            categories=_TEST_CATEGORIES,
            search_format=_TEST_SEARCH_FORMAT,
        )
        self.test_data = copy.deepcopy(_TEST_SEARXNG_RESULT)
        self.mock_response = MockResponse(200, self.test_data)

    def create_mock_response(
        self,
        status_code: int = 200,
        json_data: Optional[Dict[str, Any]] = None,
        raise_error: bool = False
    ) -> MockResponse:
        """Create a mock response with specific configuration.
        
        Args:
            status_code: HTTP status code to simulate
            json_data: Data to return from json() method
            raise_error: Whether to raise an error on raise_for_status()
            
        Returns:
            Configured MockResponse object
        """
        return MockResponse(
            status_code=status_code,
            json_data=json_data or self.test_data,
            raise_error=raise_error
        )

    def test_init_base(self) -> None:
        """Test basic initialization of the SearxNGAPI class."""
        api = searxng_api.SearxNGAPI(
            base_url=_TEST_BASE_URL,
            api_key=_TEST_API_KEY,
            language=_TEST_LANGUAGE,
            k=_TEST_K,
            time_range=_TEST_TIME_RANGE,
            categories=_TEST_CATEGORIES,
            search_format=_TEST_SEARCH_FORMAT,
        )
        
        # Verify all initialization parameters
        self.assertEqual(api.base_url, _TEST_BASE_URL)
        self.assertEqual(api.api_key, _TEST_API_KEY)
        self.assertEqual(api.language, _TEST_LANGUAGE)
        self.assertEqual(api.k, _TEST_K)
        self.assertEqual(api.time_range, _TEST_TIME_RANGE)
        self.assertEqual(api.categories, _TEST_CATEGORIES)
        self.assertEqual(api.search_format, _TEST_SEARCH_FORMAT)

    def test_init_minimal(self) -> None:
        """Test initialization with minimal parameters."""
        api = searxng_api.SearxNGAPI(base_url=_TEST_BASE_URL)
        
        # Verify default values
        self.assertIsNone(api.api_key)
        self.assertEqual(api.language, 'en')
        self.assertEqual(api.k, 1)
        self.assertEqual(api.categories, ['general'])
        self.assertEqual(api.search_format, 'json')

    @mock.patch('requests.get')
    def test_api_call_base(self, mock_get: mock.Mock) -> None:
        """Test basic API call functionality.
        
        Tests:
            1. Correct URL formation
            2. Parameter handling
            3. Response processing
            4. Result parsing
        """
        # Setup mock response
        mock_get.return_value = self.create_mock_response()
        
        # Perform API call
        result = self.test_api._searxng_api_results(_TEST_QUERY)
        
        # Verify API call
        mock_get.assert_called_once()
        call_args = mock_get.call_args
        
        # Check URL and parameters
        self.assertEqual(call_args.kwargs['params']['q'], _TEST_QUERY)
        self.assertEqual(call_args.kwargs['params']['language'], _TEST_LANGUAGE)
        self.assertEqual(call_args.kwargs['params']['format'], _TEST_SEARCH_FORMAT)
        
        # Verify result
        self.assertEqual(result, self.test_data)

    def test_parse_snippets_comprehensive(self) -> None:
        """Test comprehensive snippet parsing."""
        snippets = self.test_api._parse_snippets(self.test_data)
        
        # Verify all result components are parsed
        for result in self.test_data['results'][:self.test_api.k]:
            self.assertIn(result['title'], snippets)
            self.assertIn(result['content'], snippets)
            self.assertIn(f"Source: {result['publisher']}", snippets)
            
            if 'highlighted' in result:
                self.assertIn(result['highlighted'], snippets)

    @mock.patch('requests.get')
    def test_retry_logic(self, mock_get: mock.Mock) -> None:
        """Test API retry logic for failed requests."""
        # Configure mock to fail twice then succeed
        mock_get.side_effect = [
            requests.exceptions.RequestException("Network error"),
            requests.exceptions.RequestException("Timeout"),
            self.create_mock_response()
        ]
        
        # Execute API call
        result = self.test_api._searxng_api_results(_TEST_QUERY)
        
        # Verify retry behavior
        self.assertEqual(mock_get.call_count, 3)
        self.assertEqual(result, self.test_data)

    def test_error_handling_comprehensive(self) -> None:
        """Test comprehensive error handling scenarios."""
        error_scenarios = [
            (400, "Bad Request"),
            (401, "Unauthorized"),
            (403, "Forbidden"),
            (404, "Not Found"),
            (500, "Server Error"),
            (503, "Service Unavailable")
        ]
        
        for status_code, error_message in error_scenarios:
            with self.subTest(status=status_code):
                with mock.patch('requests.get') as mock_get:
                    mock_get.return_value = self.create_mock_response(
                        status_code=status_code,
                        raise_error=True
                    )
                    
                    with self.assertRaises(requests.exceptions.HTTPError):
                        self.test_api._searxng_api_results(_TEST_QUERY)

    def test_results_limit(self) -> None:
        """Test that results are limited to specified k value."""
        # Create test data with more results than k
        extended_data = copy.deepcopy(self.test_data)
        extended_data['results'].extend(extended_data['results'])
        
        snippets = self.test_api._parse_snippets(extended_data)
        
        # Count unique titles in snippets
        titles = [s for s in snippets if any(
            r['title'] in s for r in extended_data['results']
        )]
        
        self.assertLessEqual(len(titles), self.test_api.k)

    def test_custom_parameters(self) -> None:
        """Test handling of custom search parameters."""
        custom_params = {
            'time_range': 'week',
            'category': 'news',
            'engines': 'google,bing',
            'language': 'fr'
        }
        
        with mock.patch('requests.get') as mock_get:
            mock_get.return_value = self.create_mock_response()
            
            self.test_api._searxng_api_results(
                _TEST_QUERY,
                **custom_params
            )
            
            # Verify custom parameters were included
            call_params = mock_get.call_args.kwargs['params']
            for key, value in custom_params.items():
                self.assertIn(key, call_params)
                self.assertEqual(call_params[key], value)

if __name__ == '__main__':
    absltest.main()