#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
.. module:: Visualization
   :synopsis: Visualize results and data analysis

.. moduleauthor:: Jose Angel Velasco - (C) Tessella Spain by Capgemini Engineering - 2021
"""
# Import dependencies
import configparser
from sys import path
import numpy
import matplotlib.pyplot

path.append("../../")

# Import custom libraries
import src.utils.utils
import src.data.data_processing

FONTSIZE = 11


def plot_barplot(df, var_x, var_y, path, logger):
    """

        **Plot Barplot**

        Plot a barplot to compare the incidence of crimes according to characteristics

        :param df: Dataframe to plot
        :param var_x: x-axis variable
        :param var_y: y-axis variable
        :param path: path where it is saved plot in png format
        :param logger: logger to record exception
    """
    import matplotlib.pyplot

    try:
        matplotlib.pyplot.figure(figsize=(14, 14))
        matplotlib.pyplot.barh(y=df[var_x].values,
                               width=df[var_y].values,
                               align='center',
                               )
        matplotlib.pyplot.grid(True)
        matplotlib.pyplot.xlabel(var_y, fontsize=FONTSIZE)
        matplotlib.pyplot.ylabel(var_x, fontsize=FONTSIZE)
        matplotlib.pyplot.xticks(rotation=45)
        matplotlib.pyplot.yticks(rotation=45)
        matplotlib.pyplot.tick_params(axis='both', labelsize=FONTSIZE)
        matplotlib.pyplot.savefig(path, bbox_inches='tight',
                                  pad_inches=0)
        matplotlib.pyplot.show()
        matplotlib.pyplot.close()
    except Exception as exception_msg:
        logger.error('(!) Error in plot_barplot:{}'.format(str(exception_msg)))


def plot_scatterplot(df, var_x, var_y, scale, color, path, logger):
    """

        **Plot scatter**

        plot with the coordinates of the crimes

        :param df: Dataframe to plot
        :param var_x: x-axis variable
        :param var_y: y-axis variable
        :param scale: scale of the point
        :param path: path where it is saved plot in png format
    """
    import matplotlib.pyplot

    try:
        matplotlib.pyplot.figure(figsize=(14, 10))
        matplotlib.pyplot.scatter(x=df[var_x],
                                  y=df[var_y],
                                  s=df[scale],
                                  c=df[color],
                                  alpha=0.7,
                                  edgecolors='blue')
        matplotlib.pyplot.grid(True)
        # matplotlib.pyplot.legend()
        matplotlib.pyplot.xlabel(var_x, fontsize=FONTSIZE)
        matplotlib.pyplot.ylabel(var_y, fontsize=FONTSIZE)
        matplotlib.pyplot.savefig(path, bbox_inches='tight',
                                  pad_inches=0)
        matplotlib.pyplot.show()
        matplotlib.pyplot.close()
    except Exception as exception_msg:
        logger.error('(!) Error in plot_scatterplot:{}'.format(str(exception_msg)))


# Load configuration info
try:
    config_object = configparser.ConfigParser()
    config_object.read('..\\config.ini')
    data_path = config_object._sections['paths']['data_path']
    logs_path = config_object._sections['paths']['logs_path']
    figures_path = config_object._sections['paths']['figures_path']

    processed_file = config_object._sections['dataset']['processed']
except Exception as exception_msg:
    print('(!) Error in visualize.load_configuration:{}'.format(str(exception_msg)))
    raise

# Initialize reports
try:
    log_file_name = 'visualize'
    logger = src.utils.utils.set_up_logger(path=logs_path + log_file_name)
    logger.info('::: Start visualize :::')
except Exception as exception_msg:
    print('(!) Error in visualize.initialize_logging:{}'.format(str(exception_msg)))
    raise

# Load data
try:
    logger.info('Loading processed data')
    df = src.data.data_processing.load_data(path=data_path + 'processed\\' + processed_file, logger=logger)
    df.reset_index(inplace=True)
except Exception as exception_msg:
    logger.error('(!) Error in visualize.loading_data:{}'.format(str(exception_msg)))
    raise

# Plot scatterplot of crimes
try:
    logger.info('scatterplots')

    # Determine frequency of crimes per location
    df_freq_loc_crimes = src.utils.utils.get_frequencies(df=df,
                                                         column='Location',
                                                         logger=logger)

    # Extract X and Y coordinates
    df_freq_loc_crimes['Location_X'] = df_freq_loc_crimes['Location'].apply(
        lambda x: float(x.strip('()').split(',')[0]))
    df_freq_loc_crimes['Location_Y'] = df_freq_loc_crimes['Location'].apply(
        lambda x: float(x.strip('()').split(',')[1]))

    # Plot map of crimes
    plot_scatterplot(df=df_freq_loc_crimes,
                     var_x='Location_X',
                     var_y='Location_Y',
                     scale='FREQ_CRIMES_PERCENTAGE',
                     color='COUNT_CRIMES',
                     path=figures_path + 'crimes_map0.png',
                     logger=logger)

    # Crimes per location
    df_aux0 = df.groupby(['Location'])['index'].count().reset_index().rename(columns={'index': 'COUNT_CRIMES'})

    # Plot boxplot
    fig5, ax5 = matplotlib.pyplot.subplots()
    red_square = dict(markerfacecolor='r', marker='s')
    ax5.boxplot(df_aux0[df_aux0['COUNT_CRIMES'] < 100]['COUNT_CRIMES'],
                vert=False,
                flierprops=red_square)
    matplotlib.pyplot.grid(True)
    matplotlib.pyplot.xlabel('Yearly Crimes per location')
    matplotlib.pyplot.ylabel(' ')
    matplotlib.pyplot.savefig(figures_path + 'boxplot.png', bbox_inches='tight',
                              pad_inches=0)
    matplotlib.pyplot.show()

    # Plot histogram
    matplotlib.pyplot.figure()
    matplotlib.pyplot.hist(x=df_aux0[df_aux0['COUNT_CRIMES'] < 100]['COUNT_CRIMES'],
                           bins=50,
                           density=False,
                           facecolor='b',
                           alpha=0.75)
    matplotlib.pyplot.grid(True)
    matplotlib.pyplot.ylabel('Crimes Count')
    matplotlib.pyplot.xlabel('# Avg. Yearly Crimes per location')
    matplotlib.pyplot.savefig(figures_path + 'histogram.png', bbox_inches='tight',
                              pad_inches=0)
    matplotlib.pyplot.show()

    # Group crimes by year and location
    df_aux = df.groupby(['Location', 'YEAR'])['index'].count().reset_index().rename(columns={'index': 'COUNT_CRIMES'})
    # Extract X and Y coordinates
    df_aux['Location_X'] = df_aux['Location'].apply(lambda x: float(x.strip('()').split(',')[0]))
    df_aux['Location_Y'] = df_aux['Location'].apply(lambda x: float(x.strip('()').split(',')[1]))
    # Define a color y year
    df_aux['COLOR'] = df_aux['YEAR'].map({2016: 'blue', 2017: 'red'})
    df_aux['UNIT_SCALE'] = 1
    df_aux['UNIT_COLOR'] = 'blue'

    # Plot map of crimes with color by year and size by crimes
    plot_scatterplot(df=df_aux,
                     var_x='Location_X',
                     var_y='Location_Y',
                     scale='COUNT_CRIMES',
                     color='UNIT_COLOR',
                     path=figures_path + 'crimes_map1.png',
                     logger=logger)

    # Plot map of crimes with color by year and size by crimes
    plot_scatterplot(df=df_aux,
                     var_x='Location_X',
                     var_y='Location_Y',
                     scale='UNIT_SCALE',
                     color='COUNT_CRIMES',
                     path=figures_path + 'crimes_map2.png',
                     logger=logger)

    # Plot map of crimes with color by year and size by crimes (only 2016)
    plot_scatterplot(df=df_aux[df_aux['YEAR'] == 2016],
                     var_x='Location_X',
                     var_y='Location_Y',
                     scale='COUNT_CRIMES',
                     color='YEAR',
                     path=figures_path + 'crimes_map_by_year_2016.png',
                     logger=logger)

    # Plot map of crimes with color by year and size by crimes (only 2017)
    plot_scatterplot(df=df_aux[df_aux['YEAR'] == 2017],
                     var_x='Location_X',
                     var_y='Location_Y',
                     scale='COUNT_CRIMES',
                     color='YEAR',
                     path=figures_path + 'crimes_map_by_year_2017.png',
                     logger=logger)

    # Add color column by level of crimes
    df_aux['COLOR_COUNT_CRIMES'] = df_aux['COUNT_CRIMES'].apply(
        lambda x: 'red' if x > 100 else 'yellow' if x > 50 else 'orange' if x > 25 else 'lime' if x > 15 else 'blue')
    df_aux['COUNT_CRIMES_NORM'] = 1000 * df_aux['COUNT_CRIMES'] / df_aux['COUNT_CRIMES'].shape[0]

    # Plot map of crimes with color by year and size by crimes (only 2017)
    plot_scatterplot(df=df_aux,
                     var_x='Location_X',
                     var_y='Location_Y',
                     scale='COUNT_CRIMES',
                     color='COLOR_COUNT_CRIMES',
                     path=figures_path + 'crimes_map_by_crimes.png',
                     logger=logger)

    # Plot map of crimes with color by year and size by crimes (only 2017)
    plot_scatterplot(df=df_aux[df_aux['COLOR_COUNT_CRIMES'].isin(['red', 'yellow'])],
                     var_x='Location_X',
                     var_y='Location_Y',
                     scale='COUNT_CRIMES',
                     color='COLOR_COUNT_CRIMES',
                     path=figures_path + 'crimes_map_by_crimes_only_100.png',
                     logger=logger)

    # Determine the kind of crimes in hot locations
    severe_crimes_locations = df_aux[df_aux['COLOR_COUNT_CRIMES'].isin(['red', 'yellow'])]['Location']
    df_severe_crimes = df[df['Location'].isin(list(severe_crimes_locations))]
    # Group by crime and count
    df_aux2 = df_severe_crimes.groupby(['OFFENSE_CODE_GROUP'])['index'].count().reset_index() \
        .rename(columns={'index': 'COUNT_CRIMES'}).sort_values(by=['COUNT_CRIMES'], ascending=False)
    # express count in percentage
    df_aux2['COUNT_CRIMES_PERC'] = 100 * df_aux2['COUNT_CRIMES'] / df_aux2['COUNT_CRIMES'].sum()
    df_aux2['COUNT_CRIMES_PERC_CUM'] = df_aux2['COUNT_CRIMES_PERC'].cumsum()
    # plot
    plot_barplot(df=df_aux2[df_aux2['COUNT_CRIMES_PERC'] > 2],
                 var_x='OFFENSE_CODE_GROUP',
                 var_y='COUNT_CRIMES_PERC',
                 path=figures_path + 'barplot_hot_crimes.png',
                 logger=logger)

    # plot
    plot_barplot(df=df_aux2[df_aux2['COUNT_CRIMES_PERC_CUM'] < 80],
                 var_x='OFFENSE_CODE_GROUP',
                 var_y='COUNT_CRIMES_PERC_CUM',
                 path=figures_path + 'barplot_hot_crimes2.png',
                 logger=logger)

    # determine crimes per hour in general
    df_aux3 = df.groupby(['HOUR'])['index'].count().reset_index() \
        .rename(columns={'index': 'COUNT_CRIMES'}).sort_values(by=['HOUR'], ascending=True)
    df_aux3['COUNT_CRIMES_PERC'] = 100 * df_aux3['COUNT_CRIMES'] / df_aux3['COUNT_CRIMES'].sum()

    # determine crimes per hour from only severe crime zones
    zone1 = df_aux[df_aux['COLOR_COUNT_CRIMES'].isin(['red'])]['Location']
    df_zone1 = df[df['Location'].isin(list(zone1))]
    df_zone1_aux = df_zone1.groupby(['HOUR'])['index'].count().reset_index() \
        .rename(columns={'index': 'COUNT_CRIMES'}).sort_values(by=['HOUR'], ascending=True)
    df_zone1_aux['COUNT_CRIMES_PERC'] = 100 * df_zone1_aux['COUNT_CRIMES'] / df_zone1_aux['COUNT_CRIMES'].sum()

    # determine crimes per hour from only severe crime zones
    zone2 = df_aux[df_aux['COLOR_COUNT_CRIMES'].isin(['yellow'])]['Location']
    df_zone2 = df[df['Location'].isin(list(zone2))]
    df_zone2_aux = df_zone2.groupby(['HOUR'])['index'].count().reset_index() \
        .rename(columns={'index': 'COUNT_CRIMES'}).sort_values(by=['HOUR'], ascending=True)
    df_zone2_aux['COUNT_CRIMES_PERC'] = 100 * df_zone2_aux['COUNT_CRIMES'] / df_zone2_aux['COUNT_CRIMES'].sum()

    # plot
    matplotlib.pyplot.figure(figsize=(12, 8))
    matplotlib.pyplot.plot(df_aux3['HOUR'],
                           df_aux3['COUNT_CRIMES_PERC'],
                           linewidth=2,
                           color='k',
                           label='Average')
    matplotlib.pyplot.plot(df_zone1_aux['HOUR'],
                           df_zone1_aux['COUNT_CRIMES_PERC'],
                           linewidth=4,
                           color='red',
                           label='Hot Zones (>100 crimes/loc)')
    matplotlib.pyplot.plot(df_zone2_aux['HOUR'],
                           df_zone2_aux['COUNT_CRIMES_PERC'],
                           linewidth=4,
                           color='yellow',
                           label='Yellow Zones (>50 crimes/loc)')

    matplotlib.pyplot.grid()
    matplotlib.pyplot.legend()
    matplotlib.pyplot.xlabel('HOUR', fontsize=FONTSIZE)
    matplotlib.pyplot.ylabel('HOURLY CRIMES (%)', fontsize=FONTSIZE)
    matplotlib.pyplot.xticks(rotation=45)
    matplotlib.pyplot.yticks(rotation=45)
    matplotlib.pyplot.tick_params(axis='both', labelsize=FONTSIZE)
    matplotlib.pyplot.xticks(numpy.arange(0, 24, 1))
    matplotlib.pyplot.yticks(numpy.arange(0, 12, 1))
    matplotlib.pyplot.savefig(figures_path + 'hourly_crimes.png')
    matplotlib.pyplot.show()
    matplotlib.pyplot.close()

    # determine crimes per hour in general
    df['DAY'] = df['DAY_OF_WEEK'].map({'Monday': 1, 'Tuesday': 2, 'Wednesday': 3,
                                         'Thursday': 4, 'Friday': 5, 'Saturday': 6, 'Sunday': 7})
    df_aux_day = df.groupby(['DAY'])['index'].count().reset_index() \
        .rename(columns={'index': 'COUNT_CRIMES'}).sort_values(by=['DAY'], ascending=True)
    df_aux_day['COUNT_CRIMES_PERC'] = 100 * df_aux_day['COUNT_CRIMES'] / df_aux_day['COUNT_CRIMES'].sum()

    # determine crimes per hour from only severe crime zones
    zone1 = df_aux[df_aux['COLOR_COUNT_CRIMES'].isin(['red'])]['Location']
    df_zone1 = df[df['Location'].isin(list(zone1))]
    df_zone1_aux = df_zone1.groupby(['DAY'])['index'].count().reset_index() \
        .rename(columns={'index': 'COUNT_CRIMES'}).sort_values(by=['DAY'], ascending=True)
    df_zone1_aux['COUNT_CRIMES_PERC'] = 100 * df_zone1_aux['COUNT_CRIMES'] / df_zone1_aux['COUNT_CRIMES'].sum()

    # determine crimes per hour from only severe crime zones
    zone2 = df_aux[df_aux['COLOR_COUNT_CRIMES'].isin(['yellow'])]['Location']
    df_zone2 = df[df['Location'].isin(list(zone2))]
    df_zone2_aux = df_zone2.groupby(['DAY'])['index'].count().reset_index() \
        .rename(columns={'index': 'COUNT_CRIMES'}).sort_values(by=['DAY'], ascending=True)
    df_zone2_aux['COUNT_CRIMES_PERC'] = 100 * df_zone2_aux['COUNT_CRIMES'] / df_zone2_aux['COUNT_CRIMES'].sum()

    # plot
    matplotlib.pyplot.figure(figsize=(12, 8))
    matplotlib.pyplot.plot(df_aux_day['DAY'],
                           df_aux_day['COUNT_CRIMES_PERC'],
                           linewidth=2,
                           color='k',
                           label='Average')

    matplotlib.pyplot.plot(df_zone1_aux['DAY'],
                           df_zone1_aux['COUNT_CRIMES_PERC'],
                           linewidth=4,
                           color='red',
                           label='Hot Zones (>100 crimes/loc)')
    matplotlib.pyplot.plot(df_zone2_aux['DAY'],
                           df_zone2_aux['COUNT_CRIMES_PERC'],
                           linewidth=4,
                           color='yellow',
                           label='Yellow Zones (>50 crimes/loc)')

    matplotlib.pyplot.grid()
    matplotlib.pyplot.legend()
    matplotlib.pyplot.xlabel('DAY', fontsize=FONTSIZE)
    matplotlib.pyplot.ylabel('DAILY CRIMES (%)', fontsize=FONTSIZE)
    matplotlib.pyplot.xticks(rotation=45)
    matplotlib.pyplot.yticks(rotation=45)
    matplotlib.pyplot.tick_params(axis='both', labelsize=FONTSIZE)
    #matplotlib.pyplot.xticks(numpy.arange(0,7,1))
    #matplotlib.pyplot.xticks(['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'])
    # matplotlib.pyplot.yticks(numpy.arange(0, 12, 1))
    matplotlib.pyplot.savefig(figures_path + 'daily_crimes.png', bbox_inches='tight',
                              pad_inches=0)
    matplotlib.pyplot.show()
    matplotlib.pyplot.close()


    # determine crimes per hour in general
    df_aux_mes = df.groupby(['MONTH'])['index'].count().reset_index() \
        .rename(columns={'index': 'COUNT_CRIMES'}).sort_values(by=['MONTH'], ascending=True)
    df_aux_mes['COUNT_CRIMES_PERC'] = 100 * df_aux_mes['COUNT_CRIMES'] / df_aux_mes['COUNT_CRIMES'].sum()

    # determine crimes per hour from only severe crime zones
    zone1 = df_aux[df_aux['COLOR_COUNT_CRIMES'].isin(['red'])]['Location']
    df_zone1 = df[df['Location'].isin(list(zone1))]
    df_zone1_aux = df_zone1.groupby(['MONTH'])['index'].count().reset_index() \
        .rename(columns={'index': 'COUNT_CRIMES'}).sort_values(by=['MONTH'], ascending=True)
    df_zone1_aux['COUNT_CRIMES_PERC'] = 100 * df_zone1_aux['COUNT_CRIMES'] / df_zone1_aux['COUNT_CRIMES'].sum()

    # determine crimes per hour from only severe crime zones
    zone2 = df_aux[df_aux['COLOR_COUNT_CRIMES'].isin(['yellow'])]['Location']
    df_zone2 = df[df['Location'].isin(list(zone2))]
    df_zone2_aux = df_zone2.groupby(['MONTH'])['index'].count().reset_index() \
        .rename(columns={'index': 'COUNT_CRIMES'}).sort_values(by=['MONTH'], ascending=True)
    df_zone2_aux['COUNT_CRIMES_PERC'] = 100 * df_zone2_aux['COUNT_CRIMES'] / df_zone2_aux['COUNT_CRIMES'].sum()

    # plot
    matplotlib.pyplot.figure(figsize=(12, 8))
    matplotlib.pyplot.plot(df_aux_mes['MONTH'],
                           df_aux_mes['COUNT_CRIMES_PERC'],
                           linewidth=2,
                           color='k',
                           label='Average')

    matplotlib.pyplot.plot(df_zone1_aux['MONTH'],
                           df_zone1_aux['COUNT_CRIMES_PERC'],
                           linewidth=4,
                           color='red',
                           label='Hot Zones (>100 crimes/loc)')
    matplotlib.pyplot.plot(df_zone2_aux['MONTH'],
                           df_zone2_aux['COUNT_CRIMES_PERC'],
                           linewidth=4,
                           color='yellow',
                           label='Yellow Zones (>50 crimes/loc)')

    matplotlib.pyplot.grid()
    matplotlib.pyplot.legend()
    matplotlib.pyplot.xlabel('MONTH', fontsize=FONTSIZE)
    matplotlib.pyplot.ylabel('MONTHLY CRIMES (%)', fontsize=FONTSIZE)
    matplotlib.pyplot.xticks(rotation=45)
    matplotlib.pyplot.yticks(rotation=45)
    matplotlib.pyplot.tick_params(axis='both', labelsize=FONTSIZE)
    #matplotlib.pyplot.xticks(numpy.arange(0,7,1))
    matplotlib.pyplot.xticks(numpy.arange(1,13,1))
    #matplotlib.pyplot.yticks(numpy.arange(0, 12, 1))
    matplotlib.pyplot.savefig(figures_path + 'monthly_crimes.png', bbox_inches='tight',
                              pad_inches=0)
    matplotlib.pyplot.show()
    matplotlib.pyplot.close()



except Exception as exception_msg:
    logger.error('(!) Error in visualize.plot_scatter:{}'.format(str(exception_msg)))
    raise

# Bar plot distributions
try:
    logger.info('Bar plot distributions')
    # Obtain some distribution of frequency of occurrence to known how is distributed the crimes
    # Add results to the logging file
    logger.info('Obtain distribution of frequency of crimes occurence')

    # Define relevant columns
    columns = ['OFFENSE_CODE_GROUP', 'REPORTING_AREA', 'DAY_OF_WEEK', 'HOUR', 'MONTH']

    # Iter per columns ang ger the distribution of crimes ocurrence
    logger.info('Iter per columns ang ger the distribution of crimes ocurrence')
    logger.info('save figures in {}'.format(figures_path))

    for column in columns:
        logger.info(column)

        # Get crimes frequencies
        df_freq = src.utils.utils.get_frequencies(df=df,
                                                  column=column,
                                                  logger=logger)

        # Plot crimes frequencies
        plot_barplot(df=df_freq[0:20],
                     var_x=column,
                     var_y='FREQ_CRIMES_PERCENTAGE',
                     path=figures_path + 'crimes_distribution_per_{}.png'.format(column),
                     logger=logger)

        # report the first 5 values in the log
        for obs in range(0, 5):
            logger.info('column value:{} frequency of crimes:{}'.format(df_freq.loc[obs][column],
                                                                        df_freq.loc[obs][
                                                                            'FREQ_CRIMES_PERCENTAGE']))
except Exception as exception_msg:
    logger.error('(!) Error in visualize.bar_plot_distributions:{}'.format(str(exception_msg)))
    raise

logger.info('::: Finish :::')
