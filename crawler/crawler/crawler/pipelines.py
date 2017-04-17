# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: http://doc.scrapy.org/en/latest/topics/item-pipeline.html


class CrawlerPipeline(object):
    def process_item(self, item, spider):
        with open('./pages/%s.json' % item['pid'], "w+b") as f:
            f.write(str(item).encode())
        return item
