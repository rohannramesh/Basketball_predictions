import numpy as np
from IPython.core.debugger import set_trace
import re
import difflib
import pandas as pd
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import requests


def isfloat(value):
    """
    Check to see if value passed in is a float

    :param value:
    :return: True or false
    """
    try:
        float(value)
        return True
    except ValueError:
        return False


def get_soups(websites, curr_year):
    """
    Scrape site and get the soups for a given year
    :param websites: list with all websites interested in scraping
    :param curr_year: the year you want to extract - simple replace of default year (2012)
    :return: all_soups: the soup from each website
    """
    all_soups = {}
    for i in range(0, len(websites)):
        curr_site = websites[i]
        curr_year_site = curr_site.replace("2012", str(curr_year))
        page = requests.get(curr_year_site)
        all_soups[str(i)] = BeautifulSoup(page.text, 'lxml')
    return all_soups


def get_data_table(soup):
    """
    Pass the soup as extracted from BeautifulSoup from a webpage that either has
    single player or team statistics stored in a table
    All pages scraped had only one table on the page
    Pull every single row and every single column

    :param soup:
    :return: data, headers
    data = list with all data from a single player or team as a row
    headers = the description for each column
    """
    data = []
    headers = []
    # get table
    table = soup.find('table')
    table_body = table.find('tbody')
    rows = table_body.find_all('tr')
    # to actually extract the data for each year
    for curr_row in rows:
        all_col_that_row = curr_row.find_all('td')
        if len(all_col_that_row) > 0:
            data_curr_row = []
            for curr_col in all_col_that_row:
                # if isnumeric convert to float for later math
                if isfloat(curr_col.text):
                    data_curr_row.append(float(curr_col.text))
                else:
                    data_curr_row.append(curr_col.text)
            data.append(data_curr_row)
    table_head = table.find('thead')
    table_head_row = table_head.find('tr')
    th_col = table_head_row.find_all('th')
    for curr_head in th_col:
        headers.append(curr_head.text)
    return data, headers


def get_stats_player(dataset, headers, year, stat="all", name="all", pos='all'):
    """
    Allows you to grab any particular statistic for a given year for single, multiple, or all players
    at any specific position. The default request is you want all statistics for all players
    at all positions
    :param dataset: A dict where each key is a year and has all the data for all years
    :param headers: The labels we match to know which columns (statistic) to extract from that data
    :param year: The year
    :param stat: The statistic or statistics you want - can provide multiple
    :param name: The player name or names interested in
    :param pos: The position you are interested in
    :return: final_output: list of stats you selected for the relevant player
    output can be players, teams, or positions
    """
    if isinstance(year, str):
        curr_data = dataset[year]
        curr_headers = headers[year]
    else:
        curr_data = dataset[str(year)]
        curr_headers = headers[str(year)]
    # if want all stats see what that should be
    if stat == "all":
        stat = [curr_headers[i] for i in range(0, len(curr_data[0])) if isfloat(curr_data[0][i])]
    # check if type of data is a list and if not make it one
    if isinstance(stat, list) == False:
        stat = [stat]
    # these are the columns to use based off the stats selected
    col_to_use = []
    for curr_category in stat:
        col_to_use.append(curr_headers.index(curr_category))
    # for indexing purposes lets grab all of the names and all of the positions
    all_names = []
    all_pos = []
    all_teams = []
    for curr_row in range(0, np.shape(curr_data)[0]):
        if "Player" in curr_headers:
            # get all the names
            all_names.append(curr_data[curr_row][curr_headers.index("Player")])
            # get all the positions
            all_pos.append(curr_data[curr_row][curr_headers.index("Pos")])
        # bc team exists in all dicts
        all_teams.append(curr_data[curr_row][curr_headers.index("Tm")])
    # if gave stat as player or pos then just give this vector output
    # weird way to use the function and not recommended
    if stat == ['Player']:
        final_output = [all_names]
        return final_output
    elif stat == ['Pos']:
        final_output = [all_pos]
        return final_output
    elif stat == ['Tm']:
        final_output = [all_teams]
        return final_output
    # now grab that col of the dataset for those players you want
    data_for_relevant_col = []
    for curr_col in col_to_use:
        curr_vec = []
        for curr_row in range(0, np.shape(curr_data)[0]):
            # dealing with empty data bc didn't attempt something so making 0 if empty
            if isfloat(curr_data[curr_row][curr_col]):
                curr_vec.append(curr_data[curr_row][curr_col])
            else:
                curr_vec.append(0)
        data_for_relevant_col.append(curr_vec)
    # to sub-select certain players or positions - right now can't do both
    if name != "all":
        final_output = []
        for curr_cat in range(0, np.shape(data_for_relevant_col)[0]):
            final_output.append(data_for_relevant_col[curr_cat][all_names.index(name)])
    elif pos != "all":
        # this is the index of all rows with this position
        idx = [curr_pos for curr_pos in range(0, len(all_pos)) if pos in all_pos[curr_pos]]
        final_output = []
        for curr_cat in range(0, np.shape(data_for_relevant_col)[0]):
            for curr_idx in idx:
                final_output.append(data_for_relevant_col[curr_cat][curr_idx])
    else:
        final_output = data_for_relevant_col
    return final_output


def build_dataframe(all_data, all_headers, curr_year):
    """
    Build one dataframe for each website scraped for each year. This function handles data at the
    level of either players or teams and scales up for many different sites scraped and for each stat included
    :param all_data: a list of dicts. Each dict['year'] contains a list for each player/ team on that year
    :param all_headers: relevant stats kept for that year (i.e. data columns)
    :param curr_year: year you want to analyze - pass this in to maintain symmetry with earlier functions
    :return: all_df: a list of dataFrames that corresponds to each incoming dict
    """
    all_df = []
    for i in range(0,len(all_data)):
        yearly_data = all_data[i]
        yearly_headers = all_headers[i]
        curr_dict = {}
        for curr_header in yearly_headers[str(curr_year)]:
            A = get_stats_player(yearly_data, yearly_headers, curr_year, stat=curr_header)
            curr_dict[curr_header] = A[0]
        curr_df = pd.DataFrame(data=curr_dict)
        all_df.append(curr_df)
    return all_df


def get_yearly_data_foxsports(curr_soup, wins_tag=False):
    """
    The foxsports website organizes their tables differently and have to format the input strings
    due to leading and lagging enters.
    Pass the soup an pull every single row and every single column

    :param curr_soup: Output from BeautifulSoup
    :param wins_tag: If pulling wins and losses the team name is annoyingly formatted differently
    so parse appropriately due to * for if they made playoffs or not, etc.
    :return: a, curr_headers
    a: the list with all data from the foxsports site
    curr_headers: the description of each column
    """
    (a,curr_headers) = get_data_table(curr_soup)
    # reformat header and output variable bc doing weird stuff with strings
    curr_headers = [i.rstrip().lstrip() for i in curr_headers]
    curr_headers.remove(curr_headers[0])  # this rank isn't usefule will recreate later
    curr_headers.insert(0, 'Tm')  # put in tm name
    for curr_row in range(0, np.shape(a)[0]):
        # remove formatting for the team name and remove the Scoring rank
        if wins_tag == False:
            nremove = [int(s) for s in re.findall(r'\d+', a[curr_row][0])]
            new_row = [a[curr_row][i].lstrip(("\n\n%f\n\n" % (nremove[0]))).rstrip()
                     for i in range(0,len(a[curr_row])) if (isinstance(a[curr_row][i], str)) & (curr_headers[i] == 'Tm')]
            new_row[0] = new_row[0].replace('\n', ' ')
            a[curr_row][0] = new_row[0]
        elif wins_tag == True:
            b = [a[curr_row][0].lstrip()]
            c = b[0].find("\n\n")
            d = [b[0][0:c]]
            a[curr_row][0] = d[0].replace('\n', ' ')
    return a, curr_headers


def reorder_team_df_and_combine(all_df, tags):
    """
    Combine team dataFrames but make sure do so correctly because each previous dataFrame
    was sorted in a different order, so reordering so all teams are in the right order
    pushing things to a dict, and then building a new dataFrame
    Will use the team order of the first dataFrame provided
    :param all_df: all dataFrames to combine
    :param tags: if two dataFrames have the same stat label then append a tag so that
    won't overwrite a given statistic
    :return: new_df: the new combined dataFrame
    """
    teams = all_df[0]['Tm'].tolist()
    for i in range(1,len(all_df)):
        teams_according_to_curr_df = all_df[i]['Tm'].tolist()
        A = all_df[i]
        idx = [teams_according_to_curr_df.index(curr_team) for curr_team in teams]
        A = A.reindex(idx)
        all_df[i] = A
    new_dict = {}
    new_dict['Tm'] = teams
    for i in range(0, len(all_df)):
        header_that_df = list(all_df[i])
        # now iterate through all columns for that df and take column with those players
        numeric_headers = [list(all_df[i])[j] for j
                   in range(0, len(header_that_df)) if isfloat(all_df[i].iloc[0, j])]
        for curr_header in numeric_headers:
            if curr_header in new_dict:
                new_dict[curr_header+tags[i]] = all_df[i][curr_header].tolist()
            else:
                new_dict[curr_header] = all_df[i][curr_header].tolist()
    new_df = pd.DataFrame.from_dict(new_dict)
    return new_df


def add_column_for_change_in_stat(df, stat_to_consider):
    """
    Add a new column to a dataframe that looks at the change in a statistic across 2 years
    the column label will be d + stat --> as in delta stat
    Currently ignoring if there was a position change
    :param df: the dataFrame to add the column to
    :param stat_to_consider: statistic to evaluate across years
    :return: df: dataFrame with column added
    """
    # lets get the years to consider
    all_years = [int(i) for i in list(df)[::-1]]
    stats_include = ['Player', stat_to_consider]
    for curr_year in all_years[0:-1]:
        # current year stat
        curr_stat = df[str(curr_year)][stats_include]
        # for previous year don't care about what position was
        previous_yr_stat = df[str(curr_year-1)][stats_include]
        # now for each player iterate through and look the year before for that player
        stat_difference = []
        for curr_player in curr_stat['Player']:
            prev_stat = previous_yr_stat[stats_include][previous_yr_stat['Player'] == curr_player]
            if prev_stat.empty:
                stat_difference.append(np.nan)
            else:
                stat_player_curr_year = curr_stat[stat_to_consider][curr_stat['Player'] == curr_player].tolist()
                stat_player_last_year = prev_stat[stat_to_consider].tolist()
                stat_difference.append(stat_player_curr_year[0]-stat_player_last_year[0])
        # now define this as a dict so that can append into dataframe
        df[str(curr_year)]['d'+stat_to_consider] = stat_difference
    return df


def apply_stat_between_df(criterion1, df1, df2, stat1, stat2):
    """
    Apply some criterion from dataFrame1 to any statistic from dataFrames 1 and 2
    while accounting for differences in player order. Output will be sorted appropriately
    :param criterion1:
    :param df1: dataFrame #1
    :param df2: dataFrame #2
    :param stat1: statistic #1
    :param stat2: statistic #2
    :return: output_df1, output_df1
    output_df1: single column df with stat you care about
    output_df2: single column df with stat you care about sorted in df1 order
    """
    players_keep = df1['Player'][criterion1]
    output_df1 = df1[stat1][criterion1]
    idx2 = df2['Player'].isin(players_keep.tolist())
    output_df2 = df2[stat2][idx2]
    return output_df1, output_df2


def abline(slope, intercept):
    """
    plot a line on a plot given the slope and an intercept
    :param slope:
    :param intercept:
    :return: line on the plot
    """
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, '-')


def combine_df(all_df, tags):
    """
    Combining dataFrames into one master df assuming that the first df has all the players neccessary
    :param all_df:
    :param tags:
    :return:
    """
    players = all_df[0]['Player']
    # building giant dict
    new_dict = {}
    new_dict['Player'] = players.tolist()
    new_dict['Tm'] = all_df[0]['Tm'][all_df[0]['Player'].isin(players.tolist()) ]
    new_dict['Pos'] = all_df[0]['Pos'][all_df[0]['Player'].isin(players.tolist()) ]
    for i in range(0,len(all_df)):
        # these are the indices of those players in that df
        idx_player = all_df[i]['Player'].isin(players.tolist())
        header_that_df = list(all_df[i])
        # now iterate through all columns for that df and take column with those players
        numeric_headers = [list(all_df[i])[j] for j
                   in range(0,len(header_that_df)) if isfloat(all_df[i].iloc[0,j])]
        for curr_header in numeric_headers:
            if curr_header in new_dict:
                new_dict[curr_header+tags[i]] = all_df[i][curr_header][idx_player].tolist()
            else:
                new_dict[curr_header] = all_df[i][curr_header][idx_player].tolist()
    new_df = pd.DataFrame.from_dict(new_dict)
    return new_df


def keep_duplicate_player_most_minutes(input_df):
    """
    Because players can be traded mid season, you don't want to count that player twice
    Will search through the dataFrame for all players and only keep the entry with the most minutes
    :param input_df: dataFrame
    :return: output_df: dataFrame with each player only once
    """
    idx_to_keep = []
    idx_to_drop = []
    for curr_player in input_df['Player']:
        check_dup = input_df[['Player','MP']][input_df['Player'] == curr_player]
        idx_to_keep.append(check_dup['MP'].idxmax())
        all_idx = check_dup['MP'].index.tolist()
        idx_to_drop.extend([i for i in all_idx if i != check_dup['MP'].idxmax()])
    idx_to_keep = np.unique(idx_to_keep)
    idx_to_drop = np.unique(idx_to_drop)
    output_df = input_df.drop(input_df.index[idx_to_drop])
    return output_df