from utils import *
import pandas  as pd
import datetime
import matplotlib.pyplot as plt
from urllib.parse import urlparse
from wordcloud import WordCloud
import multidict as multidict
import re

def date_histogram():
    true_hist = create_date_histogram(meta_true_info)
    false_hist = create_date_histogram(meta_false_info)

    df = pd.Series(true_hist)

    plt.bar(range(len(df)), df.values, align='center')

    plt.xticks(range(len(df)), df.index.values, size='small', rotation='45', fontsize=16)
    xlocs = [i for i in range(0, len(df.values) + 1)]

    for i, v in enumerate(df.values):
        if len(str(v)) > 3:
            plt.text(xlocs[i] - 0.35, v + 12, str(v), fontsize=14)
        elif len(str(v)) > 1:
            plt.text(xlocs[i] - 0.25, v + 12, str(v), fontsize=14)
        else:
            plt.text(xlocs[i] - 0.1, v + 12, str(v), fontsize=14)
    plt.show()


def take_data(file, index):
    return file.split('\n')[META_DESCRIPTION.index(index)]


def create_date_histogram(base):
    serie = []
    hist = {}
    for file in base.values():
        date = take_data(file, 'date of publication').split(' ')
        while "" in date:
            date.remove("")
        if len(date) > 2:
            date = convert_date(date)
        else:
            date = date[0].replace('-', '/').split("/")
            if len(date[2]) != 4:
                if len(date[0]) == 4:
                    date[2] = date[0]
                else:
                    date[2] = '20' + str(date[2])
            date = date_format(date[1], date[2])
        serie.append(date)
    serie = sorted(serie, key=lambda x: datetime.datetime.strptime(x, '%m° sem/%Y'))
    for date in serie:
        if date not in hist.keys():
            hist[date] = 1
        else:
            hist[date] += 1
    return hist


def convert_date(date):
    months = ['janeiro', 'fevereiro', 'março', 'abril', 'maio', 'junho', 'julho', 'agosto',
              'setembro', 'outubro', 'novembro', 'dezembro']
    i = -1
    for month in months:
        if month in date:
            i = months.index(month) + 1
    year = date[len(date) - 1]
    return date_format(i, date[len(date) - 1])


def date_format(month, year):
    if year == '0201':
        year = '2010'
    if int(month) > 6:
        return '2° sem' + '/' + str(year)
    else:
        return '1° sem' + '/' + str(year)


def categories_statistics(base):
    categories = {}
    for file in base.values():
        category = take_data(file, 'category')
        if category not in categories.keys():
            categories[category] = 1
        else:
            categories[category] += 1
    return categories


def unknow_per_category(base):
    dict = {}
    for file in base.values():
        if take_data(file, 'author') == 'None':
            category = take_data(file, 'category')
            if category not in dict.keys():
                dict[category] = 1
            else:
                dict[category] += 1
    return dict


def authors_statistics():
    true = unknow_per_category(meta_true_info)
    fake = unknow_per_category(meta_false_info)
    print("Number of unknow authors of true news per category")
    print(true)
    print("Number of true news per category")
    print(categories_statistics(meta_true_info))
    print("Number of unknow authors of false news per category")
    print(fake)
    print("Number of fake news per category")
    print(categories_statistics(meta_false_info))


def links(base):
    dict = {}
    for file in base.values():
        link = '{uri.scheme}://{uri.netloc}/'.format(uri=urlparse(take_data(file, 'link')))
        if link not in dict.keys():
            dict[link] = 1
        else:
            dict[link] += 1
    print(dict)
    return dict


def texts_in_categories(meta_base):
    dict = {}
    for file in meta_base:
        category = take_data(meta_base[file], 'category')
        file = file.split('-')[0]
        if category not in dict.keys():
            dict[category] = [file + '.txt']
        else:
            dict[category].append(file + '.txt')
    return dict


def words_in_categories(meta_base, base, suffix):
    files_names = texts_in_categories(meta_base)
    dict = {}
    for category in files_names.keys():
        fullTermsDict = multidict.MultiDict()
        words = {}
        for file in files_names[category]:
            for word in base[file].split():
                word = re.sub(r'[^\w\s]', '', word).lower()
                if word not in stop_words and word.isalpha():
                    val = words.get(word, 0)
                    words[word] = val + 1
        for key in words:
            fullTermsDict.add(key, words[key])
        dict[category] = fullTermsDict
    for category in dict:
        tree_map(dict[category], suffix + '_' + category)


def words_in_base(base, name):
    fullTermsDict = multidict.MultiDict()
    words = {}
    for file in base:
        for word in base[file].split():
            word = re.sub(r'[^\w\s]', '', word).lower()
            if word not in stop_words and word.isalpha():
                val = words.get(word, 0)
                words[word] = val + 1
    for key in words:
        fullTermsDict.add(key, words[key])
    tree_map(fullTermsDict, name)


def tree_map(dict, title):
    # Accent, Accent_r, Blues, Blues_r, BrBG, BrBG_r, BuGn, BuGn_r, BuPu, BuPu_r, CMRmap, CMRmap_r, Dark2, Dark2_r, GnBu, GnBu_r, Greens, Greens_r, Greys, Greys_r, OrRd, OrRd_r, Oranges, Oranges_r, PRGn, PRGn_r, Paired, Paired_r, Pastel1, Pastel1_r, Pastel2, Pastel2_r, PiYG, PiYG_r, PuBu, PuBuGn, PuBuGn_r, PuBu_r, PuOr, PuOr_r, PuRd, PuRd_r, Purples, Purples_r, RdBu, RdBu_r, RdGy, RdGy_r, RdPu, RdPu_r, RdYlBu, RdYlBu_r, RdYlGn, RdYlGn_r, Reds, Reds_r, Set1, Set1_r, Set2, Set2_r, Set3, Set3_r, Spectral, Spectral_r, Wistia, Wistia_r, YlGn, YlGnBu, YlGnBu_r, YlGn_r, YlOrBr, YlOrBr_r, YlOrRd, YlOrRd_r, afmhot, afmhot_r, autumn, autumn_r, binary, binary_r, bone, bone_r, brg, brg_r, bwr, bwr_r, cividis, cividis_r, cool, cool_r, coolwarm, coolwarm_r, copper, copper_r, cubehelix, cubehelix_r, flag, flag_r, gist_earth, gist_earth_r, gist_gray, gist_gray_r, gist_heat, gist_heat_r, gist_ncar, gist_ncar_r, gist_rainbow, gist_rainbow_r, gist_stern, gist_stern_r, gist_yarg, gist_yarg_r, gnuplot, gnuplot2, gnuplot2_r, gnuplot_r, gray, gray_r, hot, hot_r, hsv, hsv_r, inferno, inferno_r, jet, jet_r, magma, magma_r, nipy_spectral, nipy_spectral_r, ocean, ocean_r, pink, pink_r, plasma, plasma_r, prism, prism_r, rainbow, rainbow_r, seismic, seismic_r, spring, spring_r, summer, summer_r, tab10, tab10_r, tab20, tab20_r, tab20b, tab20b_r, tab20c, tab20c_r, terrain, terrain_r, twilight, twilight_r, twilight_shifted, twilight_shifted_r, viridis, viridis_r, winter, winter_r
    wc = WordCloud(background_color="black", max_words=350, width=500, height=500, margin=1, max_font_size=45,
                   colormap='winter')
    wc.generate_from_frequencies(frequencies=dict)
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.savefig(title + '.png', bbox_inches='tight')


if __name__ == "__main__":
    words_in_base(full_true_info,"true-words")
    words_in_base(full_false_info,"false-words")
