"""Rates a single atomic fact for accuracy using either Serper or SearxNG API."""

import dataclasses
import re
from typing import Any, Literal, Optional

# pylint: disable=g-bad-import-order
from common import modeling
from common import shared_config
from common import utils
from eval.safe import config as safe_config
from eval.safe import query_serper
from eval.safe import searxng_api
# pylint: enable=g-bad-import-order

SUPPORTED_LABEL = 'Supported'
NOT_SUPPORTED_LABEL = 'Not Supported'

_STATEMENT_PLACEHOLDER = '[STATEMENT]'
_KNOWLEDGE_PLACEHOLDER = '[KNOWLEDGE]'
_NEXT_SEARCH_FORMAT = f"""\
Instructions:
1. You have been given a STATEMENT and some KNOWLEDGE points.
2. Your goal is to try to find evidence that either supports or does not \
support the factual accuracy of the given STATEMENT.
3. To do this, you are allowed to issue ONE search query that you think \
will allow you to find additional useful evidence.
4. Your query should aim to obtain new information that does not appear in the \
KNOWLEDGE. This new information should be useful for determining the factual \
accuracy of the given STATEMENT.
5. Format your final query by putting it in a markdown code block.

KNOWLEDGE:
{_KNOWLEDGE_PLACEHOLDER}

STATEMENT:
{_STATEMENT_PLACEHOLDER}
"""
_FINAL_ANSWER_FORMAT = f"""\
Instructions:
1. You have been given a STATEMENT and some KNOWLEDGE points.
2. Determine whether the given STATEMENT is supported by the given KNOWLEDGE. \
The STATEMENT does not need to be explicitly supported by the KNOWLEDGE, but \
should be strongly implied by the KNOWLEDGE.
3. Before showing your answer, think step-by-step and show your specific \
reasoning. As part of your reasoning, summarize the main points of the \
KNOWLEDGE.
4. If the STATEMENT is supported by the KNOWLEDGE, be sure to show the \
supporting evidence.
5. After stating your reasoning, restate the STATEMENT and then determine your \
final answer based on your reasoning and the STATEMENT.
6. Your final answer should be either "{SUPPORTED_LABEL}" or \
"{NOT_SUPPORTED_LABEL}". Wrap your final answer in square brackets.

KNOWLEDGE:
{_KNOWLEDGE_PLACEHOLDER}

STATEMENT:
{_STATEMENT_PLACEHOLDER}
"""


@dataclasses.dataclass()
class SearchResult:
    """Represents a search result from any search API."""
    query: str
    result: str
    search_type: Literal['serper', 'searxng']


@dataclasses.dataclass()
class FinalAnswer:
    """Represents the final fact-checking answer."""
    response: str
    answer: str


def call_search(
    search_query: str,
    search_type: str = safe_config.search_type,
    num_searches: int = safe_config.num_searches,
    serper_api_key: Optional[str] = shared_config.serper_api_key,
    searxng_url: Optional[str] = shared_config.searxng_url,
    searxng_api_key: Optional[str] = shared_config.searxng_api_key,
    search_postamble: str = '',  # ex: 'site:https://en.wikipedia.org'
) -> tuple[str, Literal['serper', 'searxng']]:
    """Call search API to get the search result.
    
    Args:
        search_query: The search query string
        search_type: Type of search API to use ('serper' or 'searxng')
        num_searches: Number of search results to return
        serper_api_key: API key for Serper (if using Serper)
        searxng_url: Base URL for SearxNG instance (if using SearxNG)
        searxng_api_key: API key for SearxNG (if using SearxNG)
        search_postamble: Optional string to append to search query
        
    Returns:
        Tuple of (search results string, search type used)
        
    Raises:
        ValueError: If search type is not supported or required credentials missing
    """
    search_query += f' {search_postamble}' if search_postamble else ''

    if search_type == 'serper':
        if not serper_api_key:
            raise ValueError('Serper API key required for Serper search type')
        serper_searcher = query_serper.SerperAPI(serper_api_key, k=num_searches)
        return serper_searcher.run(search_query, k=num_searches), 'serper'

    elif search_type == 'searxng':
        if not searxng_url:
            raise ValueError('SearxNG URL required for SearxNG search type')
        searxng_searcher = searxng_api.SearxNGAPI(
            base_url=searxng_url,
            api_key=searxng_api_key,  # Optional
            k=num_searches
        )
        return searxng_searcher.run(search_query), 'searxng'

    else:
        raise ValueError(f'Unsupported search type: {search_type}')


def maybe_get_next_search(
    atomic_fact: str,
    past_searches: list[SearchResult],
    model: modeling.Model,
    search_type: str = safe_config.search_type,
    debug: bool = safe_config.debug_safe,
) -> Optional[SearchResult]:
    """Get the next query from the model.
    
    Args:
        atomic_fact: The fact to check
        past_searches: List of previous search results
        model: The language model to use
        search_type: Type of search API to use
        debug: Whether to print debug information
        
    Returns:
        SearchResult object if successful, None otherwise
    """
    knowledge = '\n'.join([s.result for s in past_searches])
    knowledge = 'N/A' if not knowledge else knowledge
    full_prompt = _NEXT_SEARCH_FORMAT.replace(_STATEMENT_PLACEHOLDER, atomic_fact)
    full_prompt = full_prompt.replace(_KNOWLEDGE_PLACEHOLDER, knowledge)
    full_prompt = utils.strip_string(full_prompt)
    model_response = model.generate(full_prompt, do_debug=debug)
    query = utils.extract_first_code_block(model_response, ignore_language=True)

    if model_response and query:
        result, search_type_used = call_search(query, search_type=search_type)
        return SearchResult(query=query, result=result, search_type=search_type_used)

    return None


def maybe_get_final_answer(
    atomic_fact: str,
    searches: list[SearchResult],
    model: modeling.Model,
    debug: bool = safe_config.debug_safe,
) -> Optional[FinalAnswer]:
    """Get the final answer from the model.
    
    Args:
        atomic_fact: The fact to check
        searches: List of search results
        model: The language model to use
        debug: Whether to print debug information
        
    Returns:
        FinalAnswer object if successful, None otherwise
    """
    knowledge = '\n'.join([search.result for search in searches])
    full_prompt = _FINAL_ANSWER_FORMAT.replace(
        _STATEMENT_PLACEHOLDER, atomic_fact
    )
    full_prompt = full_prompt.replace(_KNOWLEDGE_PLACEHOLDER, knowledge)
    full_prompt = utils.strip_string(full_prompt)
    model_response = model.generate(full_prompt, do_debug=debug)
    answer = utils.extract_first_square_brackets(model_response)
    answer = re.sub(r'[^\w\s]', '', answer).strip()

    if model_response and answer in [SUPPORTED_LABEL, NOT_SUPPORTED_LABEL]:
        return FinalAnswer(response=model_response, answer=answer)

    return None


def check_atomic_fact(
    atomic_fact: str,
    rater: modeling.Model,
    search_type: str = safe_config.search_type,
    max_steps: int = safe_config.max_steps,
    max_retries: int = safe_config.max_retries,
    debug: bool = safe_config.debug_safe,
) -> tuple[Optional[FinalAnswer], dict[str, Any]]:
    """Check if the given atomic fact is supported.
    
    Args:
        atomic_fact: The fact to check
        rater: The language model to use for rating
        search_type: Type of search API to use
        max_steps: Maximum number of search steps
        max_retries: Maximum number of retries per step
        debug: Whether to print debug information
        
    Returns:
        Tuple of (FinalAnswer object or None, search results dictionary)
    """
    search_results = []

    for _ in range(max_steps):
        next_search, num_tries = None, 0

        while not next_search and num_tries <= max_retries:
            next_search = maybe_get_next_search(
                atomic_fact, 
                search_results, 
                rater,
                search_type=search_type
            )
            num_tries += 1

        if next_search is None:
            utils.maybe_print_error('Unsuccessful parsing for `next_search`')
            break
        else:
            search_results.append(next_search)

    # Organize search results by search type
    search_dicts = {
        'searches': [dataclasses.asdict(s) for s in search_results],
        'search_types_used': list(set(s.search_type for s in search_results))
    }
    
    final_answer, num_tries = None, 0

    while not final_answer and num_tries <= max_retries:
        num_tries += 1
        final_answer = maybe_get_final_answer(
            atomic_fact, searches=search_results, model=rater, debug=debug
        )

    if final_answer is None:
        utils.maybe_print_error('Unsuccessful parsing for `final_answer`')

    return final_answer, search_dicts