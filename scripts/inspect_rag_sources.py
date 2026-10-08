"""Read-only discovery of official page links and public visitor API endpoints."""
import re
import urllib.request
from bs4 import BeautifulSoup


def main():
    for url in ['https://www.njmuseum.com/static/js/index.13221380.js',
                'https://www.wuzhizhou.com/javascripts/information/list.js',
                'https://www.cqmetro.cn/yyfw/yydt/']:
        text=urllib.request.urlopen(url,timeout=10).read().decode('utf-8',errors='replace')
        print(url)
        if 'njmuseum' in url:
            print(re.findall(r'c.post\(d\+"([^"]+)"',text))
        elif url.endswith('.js'):
            print(text[:15000])
        else:
            soup=BeautifulSoup(text,'html.parser')
            print([(a.get_text(' ',strip=True),a['href']) for a in soup.find_all('a',href=True)][:80])


if __name__=='__main__':main()
