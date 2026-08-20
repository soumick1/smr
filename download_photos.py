from icrawler.builtin import BingImageCrawler

# Keep 'root_dir' as the key, and put your desired path as the value
crawler = BingImageCrawler(storage={'root_dir': '/home/soumick/smr/nature_photos'})

crawler.crawl(
    keyword='naturalistic nature landscape',
    max_num=15,
    filters={'license': 'creativecommons'}
)
