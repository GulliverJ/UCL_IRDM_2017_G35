import scrapy
from scrapy.item import Item, Field
from scrapy.spiders import Rule, CrawlSpider
from scrapy.linkextractors import LinkExtractor
from scrapy.exceptions import CloseSpider

import json

class Listing(Item):
    pid = Field()
    url = Field()
    html = Field()


class GraphSpider(CrawlSpider):
    name = "graph"
    allowed_domains = ["cs.ucl.ac.uk"]
    start_urls = ['http://www.cs.ucl.ac.uk/home']
    rules = [
        Rule(LinkExtractor(allow='cs.ucl.ac.uk/.+'), callback='parse_item', follow=True)
    ]
    page_counter = 0

    def parse_item(self, response):
        
        if(self.page_counter >= 10000):
            raise CloseSpider('Obtained subset')
        else:
            #print("URL", type(response.url))
            #print("BODY", type(response.body.decode("utf-8")))
            page = Listing()
            page["pid"] = self.page_counter
            page["url"] = response.url
            #page["html"] = response.body.decode("utf-8")
            page["html"] = str(response.body, "utf-8", errors="replace")
            self.page_counter += 1

        yield page
