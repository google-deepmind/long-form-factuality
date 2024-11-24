"""Class for querying the SearxNG API."""

import random
import time
from typing import Any, Optional, Literal
import requests

class SearxNGAPI:
    """Class for querying the SearxNG API."""

    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        language: str = 'en',
        k: int = 1,
        time_range: Optional[str] = None,
        categories: Optional[list[str]] = None,
        search_format: Literal['json', 'rss', 'csv'] = 'json',
    ):
        """Initialize SearxNG API client.
        
        Args:
            base_url: The base URL of your SearxNG instance
            api_key: Optional API key if your instance requires it
            language: Language code for results (e.g. 'en', 'fr')
            k: Number of results to return
            time_range: Optional time filter ('day', 'week', 'month', 'year')
            categories: List of categories to search (e.g. ['general', 'news'])
            search_format: Response format ('json', 'rss', 'csv')
        """
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.language = language
        self.k = k
        self.time_range = time_range
        self.categories = categories or ['general']
        self.search_format = search_format

    def run(self, query: str, **kwargs: Any) -> str:
        """Run query through SearxNG and parse result."""
        results = self._searxng_api_results(
            query,
            format=self.search_format,
            **kwargs,
        )
        return self._parse_results(results)

    def _searxng_api_results(
        self,
        search_term: str,
        max_retries: int = 20,
        **kwargs: Any,
    ) -> dict[Any, Any]:
        """Run query through SearxNG."""
        headers = {}
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'

        params = {
            'q': search_term,
            'format': self.search_format,
            'language': self.language,
            'categories': ','.join(self.categories),
            'pageno': 1,
            'results': self.k,
        }

        if self.time_range:
            params['time_range'] = self.time_range

        # Add any additional parameters passed through kwargs
        params.update({key: value for key, value in kwargs.items() if value is not None})

        response, num_fails, sleep_time = None, 0, 0
        search_url = f"{self.base_url}/search"
        

        while not response and num_fails < max_retries:
            try:
                response = requests.get(
                    search_url,
                    headers=headers,
                    params=params
                )
            except AssertionError as e:
                raise e
            except Exception:  # pylint: disable=broad-exception-caught
                response = None
                num_fails += 1
                sleep_time = min(sleep_time * 2, 600)
                sleep_time = random.uniform(1, 10) if not sleep_time else sleep_time
                time.sleep(sleep_time)

        if not response:
            raise ValueError('Failed to get result from SearxNG API')

        response.raise_for_status()
        return response.json()

    def _parse_snippets(self, results: dict[Any, Any]) -> list[str]:
        """Parse results from SearxNG response."""
        snippets = []

        # Process results
        for result in results.get('results', [])[:self.k]:
            # Add title and content if available
            if result.get('title'):
                snippets.append(result['title'])
            if result.get('content'):
                snippets.append(result['content'])
            
            # Add any additional result metadata
            if result.get('publisher'):
                snippets.append(f"Source: {result['publisher']}")
            
            # Add any highlighted content
            if result.get('highlighted'):
                snippets.append(result['highlighted'])

        # Handle case where no results are found
        if not snippets:
            return ['No search results found']

        return snippets

    def _parse_results(self, results: dict[Any, Any]) -> str:
        """Combine parsed snippets into a single string."""
        return ' '.join(self._parse_snippets(results))

    @property
    def supported_categories(self) -> list[str]:
        """Return list of supported search categories."""
        return [
            'general',
            'news',
            'files',
            'images',
            'videos',
            'music',
            'social media',
            'map',
            'science',
            'it'
        ]