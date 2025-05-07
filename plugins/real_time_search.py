# Поиск информации онлайн
import requests
from bs4 import BeautifulSoup

class RealTimeSearch:
    def __init__(self):
        self.sources = [
            "https://api.search.deepseek.com/v1",
            "https://www.googleapis.com/customsearch/v1"
        ]

    def search(self, query, source="all"):
        results = []
        if source in ["deepseek", "all"]:
            results += self._deepseek_search(query)
        if source in ["google", "all"]:
            results += self._google_search(query)
        return results

    def _deepseek_search(self, query):
        try:
            response = requests.post(
                f"{self.sources[0]}/search",
                json={"query": query},
                timeout=5
            )
            return response.json().get('results', [])
        except:
            return []

    def _google_search(self, query):
        try:
            html = requests.get(
                f"https://www.google.com/search?q={query}",
                headers={'User-Agent': 'Mozilla/5.0'}
            ).text
            soup = BeautifulSoup(html, 'html.parser')
            return [h3.text for h3 in soup.find_all('h3')[:3]]
        except:
            return []
