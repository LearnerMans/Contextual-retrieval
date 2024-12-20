import asyncio
import aiohttp
from aiohttp import ClientSession, ClientTimeout
from readability import Document
from bs4 import BeautifulSoup
import logging
import time
from typing import List, Dict, Any
import json
import os
from urllib.parse import urlparse
import argparse
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s:%(name)s: %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger('WebScraper')

# Constants
MAX_CONCURRENT_REQUESTS = 10
REQUEST_TIMEOUT = 30  # seconds
MAX_RETRIES = 3
RETRY_BACKOFF_FACTOR = 2  # Exponential backoff factor

async def fetch(session: ClientSession, url: str, semaphore: asyncio.Semaphore) -> str:
    """
    Fetch the content of a URL using aiohttp with retries and exponential backoff.

    Args:
        session (ClientSession): The aiohttp session to use for making requests.
        url (str): The URL to fetch.
        semaphore (asyncio.Semaphore): Semaphore to limit concurrent requests.

    Returns:
        str: The HTML content of the page.

    Raises:
        Exception: If all retries fail.
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) ' +
                      'AppleWebKit/537.36 (KHTML, like Gecko) ' +
                      'Chrome/58.0.3029.110 Safari/537.3'
    }
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            async with semaphore:
                logger.debug(f'Fetching URL: {url} (Attempt {attempt})')
                async with session.get(url, headers=headers, timeout=ClientTimeout(total=REQUEST_TIMEOUT)) as response:
                    if response.status != 200:
                        raise aiohttp.ClientResponseError(
                            status=response.status,
                            message=f'Unexpected status code: {response.status}',
                            request_info=response.request_info,
                            history=response.history
                        )
                    content = await response.text()
                    logger.info(f'Successfully fetched URL: {url}')
                    return content
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            logger.warning(f'Error fetching {url}: {e} (Attempt {attempt})')
            if attempt == MAX_RETRIES:
                logger.error(f'Failed to fetch {url} after {MAX_RETRIES} attempts.')
                raise
            else:
                backoff = RETRY_BACKOFF_FACTOR ** (attempt - 1)
                logger.info(f'Waiting for {backoff} seconds before retrying...')
                await asyncio.sleep(backoff)
    raise Exception(f'All retries failed for URL: {url}')

def extract_main_content(html: str, url: str) -> str:
    """
    Extract the main content from HTML using readability-lxml.

    Args:
        html (str): The raw HTML content.
        url (str): The URL of the page (used for base URL resolution).

    Returns:
        str: The extracted main HTML content.
    """
    try:
        doc = Document(html)
        summary_html = doc.summary(html_partial=True)
        soup = BeautifulSoup(summary_html, 'lxml')
        # Optional: Clean up the HTML further if needed
        # For example, remove scripts or styles
        for script_or_style in soup(['script', 'style']):
            script_or_style.decompose()
        formatted_html = soup.prettify()
        logger.debug(f'Extracted main content for URL: {url}')
        return formatted_html
    except Exception as e:
        logger.error(f'Error extracting main content from {url}: {e}')
        return ''

async def scrape_url(session: ClientSession, url: str, semaphore: asyncio.Semaphore) -> Dict[str, Any]:
    """
    Scrape a single URL and extract its main content.

    Args:
        session (ClientSession): The aiohttp session to use for making requests.
        url (str): The URL to scrape.
        semaphore (asyncio.Semaphore): Semaphore to limit concurrent requests.

    Returns:
        Dict[str, Any]: A dictionary with 'url' and 'text_content'.
    """
    try:
        html = await fetch(session, url, semaphore)
        content = extract_main_content(html, url)
        return {'url': url, 'text_content': content}
    except Exception as e:
        logger.error(f'Failed to scrape {url}: {e}')
        return {'url': url, 'text_content': ''}

async def scrape_websites(urls: List[str]) -> List[Dict[str, Any]]:
    """
    Scrape multiple websites concurrently and extract their main content.

    Args:
        urls (List[str]): A list of URLs to scrape.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries with 'url' and 'text_content'.
    """
    results = []
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT_REQUESTS, ssl=False)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [scrape_url(session, url, semaphore) for url in urls]
        for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Scraping URLs"):
            result = await f
            results.append(result)
    return results

def save_to_json(data: List[Dict[str, Any]], output_file: str):
    """
    Save the scraped data to a JSON file.

    Args:
        data (List[Dict[str, Any]]): The list of scraped data.
        output_file (str): The path to the output JSON file.
    """
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        logger.info(f'Successfully saved data to {output_file}')
    except Exception as e:
        logger.error(f'Error saving data to {output_file}: {e}')

def load_urls_from_file(file_path: str) -> List[str]:
    """
    Load URLs from a text file, one URL per line.

    Args:
        file_path (str): Path to the text file containing URLs.

    Returns:
        List[str]: A list of URLs.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            urls = [line.strip() for line in f if line.strip()]
        logger.info(f'Loaded {len(urls)} URLs from {file_path}')
        return urls
    except Exception as e:
        logger.error(f'Error reading URLs from {file_path}: {e}')
        return []

def parse_html_to_hierarchy(html: str) -> List[Dict[str, Any]]:
    """
    Parse HTML content and convert it into a hierarchical JSON structure without HTML tags.

    Args:
        html (str): The HTML content to parse.

    Returns:
        List[Dict[str, Any]]: A list representing the hierarchical structure.
    """
    soup = BeautifulSoup(html, 'lxml')

    # Define the heading tags in order
    heading_tags = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']

    # Initialize the root of the hierarchy
    hierarchy = []
    stack = [{'level': 0, 'children': hierarchy}]

    for element in soup.body.descendants:
        if element.name in heading_tags:
            level = int(element.name[1])
            title = element.get_text(strip=True)

            # Create a new section
            section = {'title': title, 'content': [], 'children': []}

            # Adjust the stack based on heading level
            while stack and level <= stack[-1]['level']:
                stack.pop()

            # Add the new section to the current parent
            stack[-1]['children'].append(section)

            # Push the new section to the stack
            stack.append({'level': level, 'children': section['children']})
        elif element.name == 'p':
            text = element.get_text(strip=True)
            if text:
                stack[-1]['children'].append({'paragraph': text})
        elif element.name in ['ul', 'ol']:
            # Handle lists
            list_type = 'ordered_list' if element.name == 'ol' else 'unordered_list'
            items = [li.get_text(strip=True) for li in element.find_all('li')]
            if items:
                stack[-1]['children'].append({list_type: items})
        # You can add more handlers for other tags if needed

    return hierarchy

def convert_hierarchy_to_text(hierarchy: List[Dict[str, Any]], indent: int = 0) -> str:
    """
    Convert the hierarchical JSON structure into a formatted text string.

    Args:
        hierarchy (List[Dict[str, Any]]): The hierarchical structure.
        indent (int): Current indentation level.

    Returns:
        str: Formatted text preserving the hierarchy.
    """
    text = ""
    indent_str = "    " * indent  # 4 spaces per indent level

    for item in hierarchy:
        if 'title' in item:
            # Add heading with indentation
            text += f"{indent_str}{item['title']}\n"
            text += f"{indent_str}{'-' * len(item['title'])}\n\n"
            # Recursively add child content
            text += convert_hierarchy_to_text(item['children'], indent + 1)
        elif 'paragraph' in item:
            text += f"{indent_str}{item['paragraph']}\n\n"
        elif 'unordered_list' in item:
            for li in item['unordered_list']:
                text += f"{indent_str}- {li}\n"
            text += "\n"
        elif 'ordered_list' in item:
            for idx, li in enumerate(item['ordered_list'], 1):
                text += f"{indent_str}{idx}. {li}\n"
            text += "\n"
        # Handle other types if added

    return text

def transform_content(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Transform the scraped data to remove HTML tags and preserve hierarchy.

    Args:
        data (List[Dict[str, Any]]): The list of scraped data with 'url' and 'text_content'.

    Returns:
        List[Dict[str, Any]]: Transformed data with 'url' and 'page_content'.
    """
    transformed = []
    for entry in data:
        url = entry.get('url', '')
        html_content = entry.get('text_content', '')
        if html_content:
            hierarchy = parse_html_to_hierarchy(html_content)
            page_content = convert_hierarchy_to_text(hierarchy)
        else:
            page_content = ""
        transformed.append({'url': url, 'page_content': page_content})
    return transformed

def save_hierarchy_to_json(data: List[Dict[str, Any]], output_file: str):
    """
    Save the transformed hierarchical data to a JSON file.

    Args:
        data (List[Dict[str, Any]]): The list of transformed data.
        output_file (str): The path to the output JSON file.
    """
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        logger.info(f'Successfully saved hierarchical data to {output_file}')
    except Exception as e:
        logger.error(f'Error saving hierarchical data to {output_file}: {e}')

def main(urls: List[str], output_file: str):
    """
    The main function to execute the scraping and transformation process.

    Args:
        urls (List[str]): A list of URLs to scrape.
        output_file (str): Path to the output JSON file.
    """
    start_time = time.time()
    logger.info('Starting the scraping process...')
    scraped_data = asyncio.run(scrape_websites(urls))
    logger.info('Scraping completed. Starting transformation...')
    transformed_data = transform_content(scraped_data)
    save_hierarchy_to_json(transformed_data, output_file)
    end_time = time.time()
    logger.info(f'Process completed in {end_time - start_time:.2f} seconds.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Asynchronous Web Scraper that outputs hierarchical JSON without HTML.')
    parser.add_argument(
        '-u', '--urls',
        nargs='*',
        help='List of URLs to scrape. If not provided, the script will look for a file with URLs.'
    )
    parser.add_argument(
        '-f', '--file',
        type=str,
        help='Path to a text file containing URLs (one per line). Used if --urls is not provided.'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='scraped_data_hierarchical.json',
        help='Path to the output JSON file.'
    )
    args = parser.parse_args()

    # Determine the list of URLs
    if args.urls:
        url_list = args.urls
    elif args.file:
        url_list = load_urls_from_file(args.file)
    else:
        # Default example URLs if none provided
        with open('urls.json', 'r') as file:
        # Load the JSON data
            url_list = json.load(file)
            logger.info('No URLs provided. Using default example URLs.')
        logger.info('No URLs provided. Using default example URLs.')

    if not url_list:
        logger.error('No URLs to scrape. Exiting.')
        exit(1)

    main(url_list, args.output)
