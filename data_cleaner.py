import matplotlib.pyplot as plt
import pandas as pd
import arabic_reshaper                      # Is used to display Persian words correctly in plots
from bidi.algorithm import get_display      # Is used to display Persian words correctly in plots
import requests
import re
from parsivar import Normalizer, Tokenizer, FindStems


week_day_translation_table = {'شنبه': 'Saturday',
'یکشنبه': 'Sunday',
'دوشنبه': 'Monday',
'سه‌شنبه': 'Tuesday',
'چهارشنبه': 'Wednesday',
'پنجشنبه': 'Thursday',
'جمعه': 'Friday'
}
number_translation_table = str.maketrans('۱۲۳۴۵۶۷۸۹۰', '1234567890')

def tag_cleaner(df):
    """Clean 'Tags' column.
    
    Arguments:
    df -- Data Frame"""

    # Tags column is stored as a string instead of a list when the data was stored as a CSV file. Now we transform it to a
    # list again.
    for i, tags in enumerate(df["Tags"]):
        # Replace 'nan' values with an empty string.
        if type(tags) == float:
            tags = ''
        # Get rid of ' [ ] strings.
        t = re.sub("['\[\]]", "", tags)
        # Seperate tags and get rid of commas and the heading white spaces between tags.
        clean_tags = t.split(', ')
        df.loc[i,'Tags'] = clean_tags

def time_cleaner(df):
    """Clean 'Time' column by translating numbers to English
    
    Argumnets:
    df -- Data Frame"""

    # Translating numbers to English.
    for i, time in enumerate(df["Time"]):
        clean_time = time.translate(number_translation_table)
        df.loc[i, 'Time'] = clean_time

def date_cleaner(df):
    """Clean 'Date' column by translating numbers to English and then normalising dates.
    
    Arguments:
    df -- Data Frame"""
    
    # Translating date numbers to English and normalizing them.
    date_normalizer = Normalizer(date_normalizing_needed = True)
    for i, date in enumerate(df["Date"]):
        clean_date = date.translate(number_translation_table)
        clean_date = date_normalizer.normalize(clean_date)
        df.loc[i, 'Date'] = clean_date

def week_day_cleaner(df):
    """Clean 'Week_Day' column
    
    Argumnets:
    df -- Data Frame"""

    # Remove / from week day and translate it to English.
    for i, day in enumerate(df['Week_Day']):
        clean_day = week_day_translation_table[re.search(r'(.*) /', day).group(1)]
        df.loc[i, 'Week_Day'] = clean_day

def category_cleaner(df):
    """Clean 'Category' column.
    
    Argumnets:
    df -- Data Frame"""
    # Cleaning categories by removing redundant white spaces and replacing '\u200c' with spaces.
    for i, category in enumerate(df["Category"]):
        clean_category = re.search('\s*(.*)\s*', category).group(1)
        clean_category = re.sub('\u200c', ' ', clean_category)
        df.loc[i, 'Category'] = clean_category

def title_cleaner(df):
    """Clean 'Title' column
    
    Arguments:
    df -- Data Frame"""

    # Cleaning the titles by removing the news agency's name from them.
    for i, title in enumerate(df["Title"]):
        clean_title = re.search(r'(.*) - ایسنا', title).group(1)
        df.loc[i, "Title"] = clean_title


def plot_zipf_heap(words: list, dirty: bool=True) -> dict:
    """Plot Zipf's and Heap's plots and return the dictionary of words and their frequencies.
    
    Arguments:
    words -- List of all words in the data set.
    dirty -- Type of the data set, i.e. before or after cleaning
    Outputs:
    words_dict -- Dictionary of all the words in the data set as keys and their frequencies as values."""
    
    if dirty:
        input_type = 'Dirty'
    else:
        input_type = 'Clean'
    # Create an empty dictionary to store words and their frequencies.
    words_dict = dict()
    # x and y axes values of the Heap's Law plot.
    heap_x, heap_y = [], []
    # Number of types and tokens.
    types, tokens  = 0, 0
    # Fill the words_dict.
    for word in words:
        if words_dict.get(word, 0) == 0:
            types += 1
            tokens += 1
            words_dict[word] = 1
        else:
            words_dict[word] += 1
            tokens += 1
        # For every 100 tokens, add the number of tokens and types to the related axis value.
        if tokens % 100 == 0:
            heap_x.append(tokens)
            heap_y.append(types)

    # Sort our words_dict based on words frequencies.
    words_dict = {key:value for key, value in sorted(words_dict.items(), key=lambda x: x[1], reverse=True)}
    # Plot Zipf's Law on log scale.
    plt.plot(range(1, len(words_dict.keys()) + 1), words_dict.values())
    plt.title(f"{input_type} Zipf's Law(Log Scale)")
    plt.xlabel('Rankings')
    plt.ylabel('Frequency')
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig(f"{input_type} Zipf's Law(Log Scale).png", bbox_inches='tight')
    plt.show()
    
    # Plot Heap's Law.
    plt.plot(heap_x, heap_y)
    plt.title(f"{input_type} Heap's Law")
    plt.xlabel('Number of Tokens')
    plt.ylabel('Number of Types')
    plt.xticks(rotation=30)
    plt.xscale('linear')
    plt.yscale('linear')
    plt.savefig(f"{input_type} Heap's Law.png", bbox_inches='tight')
    plt.show()
    return words_dict

def obtain_stop_words() -> tuple:
    """Obtain a list of Persian stop words.
    
    Outputs:
    stop_words -- The tuple of stop words.
    First, I used a popular stop words list available online on GitHub, mentioned below. However, after analysing the effect of using that list,
    I realised that it removes most of the verbs since it includes auxilary verbs. This is because most Persian verbs 
    use auxilary verbs. Therefore, I decided to translate NLTK English stop words list I found on GitHub. Also, at the end of
    each news article, there is 'انتهای پیام', which means 'end of the message. I included these two words in my stop words list
    to remove them from news articles' bodies."""

    # Getting Persian stop words list.
    # res = requests.get("https://raw.githubusercontent.com/kharazi/persian-stopwords/refs/heads/master/persian")
    # stop_words = res.text.split()

    # I decided to create a stop words list by translating stop words of https://gist.github.com/sebleier/554280 into Persian.
    stop_words = ("من",
                "خودم",
                "خود",
                "ما",
                "خودمون",
                "تو",
                "شما",
                "خودتون",
                "خودت",
                "او",
                "وی",
                "اون",
                "خودش",
                "اونها",
                "ها",
                "های",
                "چی",
                "کدوم",
                "کی",
                "این",
                "آن",
                "اینها",
                "آنها",
                "یه",
                "یک",
                "و",
                "اما",
                "را",
                "اگر",
                "اگه",
                "یا",
                "اگرچه",
                "چونکه",
                "چون",
                "تا",
                "حین",
                "از",
                "در",
                "با",
                "به",
                "برای",
                "درباره",
                "علیه",
                "مقابل",
                "بین",
                "قبل",
                "بعد",
                "بالا",
                "زیر",
                "پایین",
                "خارج",
                "بیرون",
                "سپس",
                "وقتی",
                "اینجا",
                "اینجا",
                "آنجا",
                "اونجا",
                "چرا",
                "چطور",
                "چگونه",
                "همه",
                "هیچ",
                "هر",
                "که",
                "بر",
                "کم",
                "بیشتر",
                "کمتر",
                "بقیه",
                "تعدادی",
                "مثل",
                "مانند",
                "نه",
                "فقط",
                "یکسان",
                "بنابراین",
                "همچنین",
                "خیلی",
                "حالا",
                "انتهای",
                "پیام")
    return stop_words

def obtain_clean_words(words: list) -> list:
    """Obtain the list of clean words after applying word processing.
    
    Arguments:
    words -- List of all words in the data set.
    Outputs:
    clean_words -- List of all words after applying cleaning processes.
    I used parsivar library's FindStems class to convert words to their stems."""

    clean_words = []
    stop_words = obtain_stop_words()
    my_stemmer = FindStems()
    for word in words:
        # Excluding stop words.
        if word not in stop_words:
            # Remove the name of the news agency as it appears at the beginning of each news article.
            if word != 'ایسنا':
                # Remove punctuations that stick to words.
                clean_word = re.sub("[./:،؛,\!@#$%^\&\*\)\(_؟]", "", word)
                # Apply the stemmer
                clean_words.append(my_stemmer.convert_to_stem(clean_word))
    return clean_words

def obtain_words(content: str) -> list:
    """Obtain words of each news article.
     
    Arguments:
    content -- Body of the news article.
    Outputs:
    words -- List of the news aricle's words.
    parsivar library's Tokenizer class is used to obtain tokens."""

    my_tokenizer = Tokenizer()
    words = []
    for word in my_tokenizer.tokenize_words(content):
        words.append(word)
    return words

def obtain_all_words_clean_contents(df) -> tuple:
    """Obtain all dirty words of the dataset and clean words of each news article.
    
    Arguments:
    df -- Data Frame.
    Outputs:
    all_words -- The list of all words.
    clean_contents -- The list of contents with cleaned words."""

    all_words = []
    clean_contents = []
    contents = df['Content']
    for content in contents:
        words = obtain_words(content)
        clean_contents.append(obtain_clean_words(words))
        for word in words:
            all_words.append(word)
    return all_words, clean_contents

def feature_dot_plot(feature):
    """Plot dot plots of the given feature'.
    
    Arguments:
    feature -- Name of the desired data frame's column"""

    top_10 = df.value_counts(feature)[:10].items()
    x = []
    y = []
    for value in top_10:
        # Convert Persian words to a correct printable form 
        x.append(get_display(arabic_reshaper.reshape(value[0])))
        y.append(value[1])

    plt.plot(x, y, 'o')
    plt.xticks(rotation=30, ha='right')
    plt.xlabel(feature)
    plt.ylabel("Frequencies")
    plt.title(f"Top 10 Values of {feature}")
    plt.savefig(f"{feature}.png", bbox_inches='tight')
    plt.show()
    
def plot_word_freq(words: list, freq: list, dirty:bool = True, info:str =''):
    """Plot words vs their frequencies
    
    Arguments:
    words -- The list of words.
    freq -- The list of frequencies.
    dirty -- The type of input words.
    info -- The explanation of the plot."""
    # Convert Persian words to a correct printable form
    x = [get_display(arabic_reshaper.reshape(w)) for w in words]
    plt.bar(x, freq)
    if dirty:
        input_type = "Dirty"
        plt.text(.7, 10000, info, c='red')
    else:
        input_type = "Clean"
    plt.title(f"{input_type} Top 10 Words Frequencies")
    plt.xlabel('Frequent Words')
    plt.xticks(rotation=30, ha='right')
    plt.ylabel('Frequencies')
    plt.savefig(f"{input_type} Top 10 Words Frequencies.png", bbox_inches='tight')
    plt.show()

def create_output_file(clean_contents: list, categories: list):
    """Create Clean_ISNA_Dataset.csv to be used by category_predictor.py
    
    Arguments:
    clean_contents -- The list of contents with clean words.
    categories -- The list of news articles' categories."""

    clean_bodies = []
    for content in clean_contents:
        clean_bodies.append(' '.join(content))
    
    clean_data_dict = {'Content': clean_bodies,
                       'Category': categories}
    clean_df = pd.DataFrame(clean_data_dict)
    clean_df.to_csv('Clean_ISNA_Dataset.csv', index=False)

def plot_df_analysis(df):
    """Plot the number of each categorical feature's unique values
    
    Argument:
    df -- Data frame"""

    unique_values = []
    features = ['Date', 'Week_Day', "Secretary", "Journalist", "Category", "Source"]
    for feature in features:
        unique_values.append(len(df[feature].unique()))
    plt.bar(features, unique_values)
    plt.title("Number of Unique Values of each Categorical Feature")
    plt.xlabel('Categorical Features')
    plt.ylabel('Number of Unique Values')
    plt.xticks(rotation=30)
    plt.savefig('unique.png', bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    df = pd.read_csv('Raw_ISNA_Dataset.csv')

    tag_cleaner(df)
    time_cleaner(df)
    date_cleaner(df)
    week_day_cleaner(df)
    category_cleaner(df)
    title_cleaner(df)

    all_words, clean_contents = obtain_all_words_clean_contents(df)
    clean_words = []
    for content in clean_contents:
        for word in content:
            clean_words.append(word)

    # Create the ouput file
    create_output_file(clean_contents, df['Category'])

    dirty_words_dict = plot_zipf_heap(all_words, True)
    clean_words_dict = plot_zipf_heap(clean_words, False)

    # The number of types before and after cleaning.
    print("The number of dirty types:", len(dirty_words_dict), '\n' , 
    "The number of clean types:", len(clean_words_dict))

    print("The number of all dirty words:", len(all_words), '\n' \
    'The number of all clean words:', len(clean_words))


    dirty_words, dirty_freqs = list(dirty_words_dict.keys())[:10], list(dirty_words_dict.values())[:10]
    clean_words, clean_freqs = list(clean_words_dict.keys())[:10], list(clean_words_dict.values())[:10]
    plot_word_freq(dirty_words, dirty_freqs, dirty=True, info='all of the top 10 frequent words of dirty words are stop words')
    plot_word_freq(clean_words, clean_freqs, dirty=False)

    # Change the data types of our columns to save memory and facilitate our work in implementing ML algorithms.
    df = df.astype({'Content': 'str', 'Title': 'str', 'Category': 'category', 'Source': 'category', 'Journalist': 'category', 
                    'Tags': 'object', 'Week_Day': 'category', 'Date': 'category', 'Time': 'category', 'URL': 'str',
                    'Secretary': 'category'})


    # Now lets take a look at distribution of our categorical features:

    feature_dot_plot('Secretary')
    feature_dot_plot('Journalist')
    feature_dot_plot('Category')
    # It shows that 'Journalist is the dominant value of the 'Source' feature.
    feature_dot_plot('Source')

    # Plot categorical features vs their frequencies.
    plot_df_analysis(df)

    # Obtain the time span of our datset
    print("The first article was published on:", df.loc[0, 'Time'], df.loc[0, 'Week_Day'], df.loc[0, 'Date'])
    print("The last article was published on:", df.loc[533, 'Time'], df.loc[533, 'Week_Day'], df.loc[533, 'Date'])