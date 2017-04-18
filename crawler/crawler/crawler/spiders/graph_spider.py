import scrapy
import pickle
from scrapy.item import Item, Field
from scrapy import signals
from scrapy.spiders import Rule, CrawlSpider
from scrapy.linkextractors import LinkExtractor
from scrapy.exceptions import CloseSpider

class Listing(Item):
    pid = Field()
    url = Field()
    html = Field()

class GraphSpider(CrawlSpider):
    name = "graph"
    allowed_domains = ["cs.ucl.ac.uk"]
    deny = ['\.tar.gz', '\.gz', '\.txt', '\.jnt', '\.wmf', '\.7z']
    start_urls = ['http://www.cs.ucl.ac.uk/']
    rules = [
        Rule(LinkExtractor(allow='cs.ucl.ac.uk/.*', deny=(deny)), callback='parse_item', follow=True)
    ]

    completed = False

    # Increments page ID when encountering new URL
    page_counter = 0

    # Increments when a page has been crawled
    crawled_counter = 0

    pids = {}

    adjacency = []

    # Closing
    @classmethod
    def from_crawler(cls, crawler, *args, **kwargs):
        spider = super(GraphSpider, cls).from_crawler(crawler, *args, **kwargs)
        crawler.signals.connect(spider.spider_closed, signal=signals.spider_closed)
        return spider

    # When spider ends, save the adjacency obj
    def spider_closed(self, spider):
        with open('adjacency.pkl', 'wb') as f:
            pickle.dump(self.adjacency, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Override parse_start_url to catch the homepage
    def parse_start_url(self, response):
        return self.parse_item(response)

    # Parse page
    def parse_item(self, response):

        # Limit number of pages scraped (mainly use for debug)
        if(self.crawled_counter >= 500000):
            if self.completed:
                pass

            self.completed = True
            raise CloseSpider ('Obtained subset')
        else:
            page = Listing()
            page_url = response.url

            # If we haven't seen this page, initialise it
            if page_url not in self.pids:
                self.pids[page_url] = self.page_counter
                self.adjacency.append({"outlinks":[],"inlinks":[]})
                self.page_counter += 1

            # Create item
            page['pid'] = self.pids[page_url]
            page['url'] = page_url
            page['html'] = response.body.decode("utf-8", "replace")

            self.crawled_counter += 1

            if self.crawled_counter % 1000 == 0:
                print("... %d ..." % (self.crawled_counter))

            # Extract all links for building adjacency matrix
            for link in LinkExtractor(allow=(self.allowed_domains),deny=(self.deny)).extract_links(response):
                link_url = link.url

                # If link not seen before, initialise with next pid
                if link_url not in self.pids:
                    self.pids[link_url] = self.page_counter
                    self.adjacency.append({"outlinks":[],"inlinks":[]})
                    self.page_counter += 1

                # Append to adjacency dicts
                self.adjacency[self.pids[link_url]]["inlinks"].append(self.pids[page_url])
                self.adjacency[self.pids[page_url]]["outlinks"].append(self.pids[link_url])

        yield page
