from bs4 import BeautifulSoup, SoupStrainer
import requests
import re
import time
import pandas as pd

home_page = 'https://www.isna.ir/'
# A list of information related to each news article.
news_data = []

def get_html(url: str, only=None) -> tuple :
    """Obtain the title and the BeautifulSoup object of a webpage with the requests library.
    
    Arguments:
    url -- url of the desired webpage.
    Keyword arguments:
    only -- A SoupStrainer object to filter the output of BeautifulSoup constructor. (default: None)
    Outputs:
    title, html -- A tuple of title of the webpage and the BeuatifulSoup object.
    """

    # Use a context manager to close the connection after obtainig the BeautifulSoup object.
    with requests.get(url) as news_page:
        # The reqular expression fetches the title tag of the webpage and .group(1) gets rid of html tags.
        title = re.search(r'\<title\>(.*)\s*\</title\>', news_page.text).group(1)
        html = BeautifulSoup(news_page.text, "html.parser", parse_only=only)
    return title, html

def get_archive_link() -> str:
    """Get the archive url of ISNA website."""

    # Limit our html scope as the archive link is in the header tag.
    only_header_tag = SoupStrainer('header')
    _, home_soup = get_html(home_page, only= only_header_tag)
    # Search all 'li' tags in the header BeautifulSoup object and check if 'archive' is in their 'href' attribute or not. Then,
    # return the archive link
    for li in home_soup.find_all('li'):
        if li[u'data-id'][0] == '10':
            link = re.search(r'archive', li.a['href'])
        if link:
            return link.string

def get_news_links(archive: str, page_num: int = 10) -> list:
    """Obtain news urls from archive.
    
    Arguments:
    archive -- The url of the archive web page.
    Keyword arguments:
    page_num -- The number of pages of the archive to crawl. (default: 10)
    Outputs:
    news_link -- A list of news urls.
    Obtaining urls from the picture of the news title one by one and ignore 'photo reports' as they don't have any text."""

    news_links = []
    for _ in range(page_num):
        _, archive_soup = get_html(archive, only=only_main_tag)
        # Search for the part of the page that news titles are set, ''div' class="items"'.
        for div in archive_soup.find_all("div"):
            if div[u'class'][0] == 'items':
                # Iterate over pictures and obtain the urls included in them. Also, skip urls starting with '/photo' as they are
                # just picure reports.
                for figure in div.find_all('figure'):
                    if re.search(r'^/photo', figure.a['href']):
                        continue
                    news_links.append(home_page + figure.a['href'])
                break
        # Search for the 'next page' button in the 'footer' tag.
        footer = archive_soup.find('footer')
        for a in footer.find_all('a'):
            if re.search(r'بعدی', a.text):
                # Changing the url we want to crawl on.
                archive = home_page + a['href']
                break
        # Set a small delay between crawling each archive page to avoid being detected as a malicious bot.
        time.sleep(2)
    return news_links


def extract_li_information(soup) -> tuple:
    """Extract all the article information in the 'li' tags
    
    Arguemnts:
    soup -- The BeautifulSoup object of the news article webpage.
    Outputs:
    li_out -- A tuple containing weekday, date, time, category, source, journalist(if exists), and secretary of a news article."""

    # This feature is set to None and change later if there is any journalist for the article. Some articles just have
    # secretary.
    news_journalist = None
    for li in soup.find_all('li'):
        # Search for weekday, date and time of publication in 'i' tags of 'li' tags.
        if re.search(r'calendar', li.i[u'class'][1]):
            # Extracting relevant information after finding them.
            for span in li.find_all('span'):
                if span[u'class'][0] == 'title-meta':
                    news_week_day = span.text
                elif span[u'class'][0] == 'text-meta':
                    (news_date, news_time) = span.text.split('/')
                    break
            # Go to the next iteration to find the next infromation
            continue
        # Same as above, search for the category of the news article in 'i' tags of 'li' tags.
        elif re.search(r'folder', li.i[u'class'][1]):
            for span in li.find_all('span'):
                if span[u'class'][0] == 'text-meta':
                    news_category = span.text
                    break
            continue
        elif re.search(r'microphone$', li.i[u'class'][1]):
            for span in li.find_all('span'):
                if span[u'class'][0] == 'title-meta':
                    # If there is any external source for the news article, we continue to extract it. Otherwise, the news
                    # article is from an ISNA journalist and source is set to 'Journalist'.
                    if span.text == 'منبع :':
                        continue
                    else:
                        news_source = 'Journalist'
                        break
                elif span[u'class'][0] == 'text-meta':
                    news_source = span.text
                    break
            continue
        # Extract the journalist, if any.
        elif re.search(r'edit$', li.i[u'class'][1]):
            news_journalist = li.strong.text
            continue
        elif re.search(r'check\-square$', li.i[u'class'][1]):
            news_secretary = li.strong.text
            # continue
            break
    li_out = (news_week_day, news_date, news_time, news_category, news_source, news_journalist, news_secretary)
    return li_out

def extract_news_content(soup) -> str:
    """Extract the body of a news article
    
    Arguments:
    soup -- The BeautifulSoup object of the news webpage.
    Outputs:
    news_content -- The body of the news article."""

    # Create an empty string.
    news_content = str()
    # Add the content of each parapraph to the string. As the information are seperated as parapraphs in the website.
    for p in soup.find_all('p'):
        news_content += ' ' + p.text
    return news_content

def extract_news_tags(soup) -> str:
    """Extract the tags of a news article
    
    Arguments:
    soup -- The BeautifulSoup object of the news webpage.
    Outputs:
    news_tags -- The tags of the news article."""

    news_tags = []
    # Search 'footer' tag for the tags of the news article.
    news_footer = soup.find('footer')
    # Add tags of the news article to the tags list from 'li' tags of the 'footer' tag.
    for li in news_footer.ul:
        # Skip \n in the html code of the tags section.
        if li.text != '\n':
            news_tags.append(li.text)
    return news_tags
    

def extract_data(news_links: list):
    """A generator that extract information from each news article webpage.
    
    Arguments:
    news_links -- The list of news urls
    Outputs:
    Yield information related to news articles.
    Extract useful information from html code of each news article, namely body, title, category, source, journalist, secretary,
    tags of the news article and weekday, date, and time of the publication. The output of each iteration is a list related to
    a news article."""

    for url in news_links:
        # The title of articles are the same as the title of the html code. 
        news_title, news_html = get_html(url, only=only_main_tag)
        # Extract all the information in 'li' tags of the article's html code.
        (news_week_day, news_date, news_time, news_category, news_source, news_journalist,
        news_secretary) = extract_li_information(news_html)
        # Extract the body of the news article.
        news_content = extract_news_content(news_html)
        # Extract tags.
        news_tags = extract_news_tags(news_html)
        yield [news_content, news_title, news_category, news_source, news_journalist, 
                news_secretary, news_tags, news_week_day, news_date, news_time]
        # A delay to avoid beign caught as a robot.
        time.sleep(2)

archive_link = get_archive_link()
# Define a filter for BeautifulSoup constructor to only include 'main' tag.
only_main_tag = SoupStrainer("main")
news_links = get_news_links(archive_link, 18)
for data in extract_data(news_links):
    news_data.append(data)

# Create a dictionary of the obtained date to be used to create a CSV file.
news_dict = {'Content': [data[0] for data in news_data],
             'Title': [data[1] for data in news_data],
             'Category': [data[2] for data in news_data],
             'Source': [data[3] for data in news_data],
             'Journalist': [data[4] for data in news_data],
             'Secretary': [data[5] for data in news_data],
             'Tags':[data[6] for data in news_data],
             'Week_Day': [data[7] for data in news_data],
             'Date':[data[8] for data in news_data],
             'Time':[data[9] for data in news_data],
             'URL': news_links}

df = pd.DataFrame(news_dict)
df.to_csv('Raw_ISNA_Dataset.csv', index=False)