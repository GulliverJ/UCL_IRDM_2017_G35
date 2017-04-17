import scrapy
from scrapy.item import Item, Field
from scrapy.spiders import Rule, CrawlSpider
from scrapy.linkextractors import LinkExtractor
from scrapy.exceptions import CloseSpider

class Listing(Item):
    pid = Field()
    url = Field()
    html = Field()
    links = Field()

class GraphSpider(CrawlSpider):
    name = "graph"
    allowed_domains = ["cs.ucl.ac.uk"]
    start_urls = ['http://www.cs.ucl.ac.uk/home']
    rules = [
        Rule(LinkExtractor(allow='cs.ucl.ac.uk/.+'), callback='parse_item', follow=True)
    ]
    page_counter = 1
    pids = {'http://www.cs.ucl.ac.uk/home':0}
    fully_crawled = 0

    def parse_item(self, response):
        
        if(self.fully_crawled >= 500):
            raise CloseSpider ('Obtained subset')
        else:
            print(self.fully_crawled)
            page = Listing()
            page_url = response.url
            if page_url not in self.pids:
                self.pids[page_url] = self.page_counter
                self.page_counter += 1
            page['pid'] = self.pids[page_url]
            page['url'] = page_url
            page['html'] = response.body
            page['links'] = []
            self.fully_crawled += 1
            for link in LinkExtractor(allow=(self.allowed_domains)).extract_links(response):
                link_url = link.url
                if link_url not in self.pids:
                    self.pids[link_url] = self.page_counter
                    self.page_counter += 1
                link_pid = self.pids[link_url]
                page['links'].append(link_pid)
        yield page
