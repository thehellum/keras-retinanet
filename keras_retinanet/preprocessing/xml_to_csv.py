import xml.etree.ElementTree as ElementTree
import sys
import os
import csv
import numpy as np
import pandas as pd
import argparse

'''
    Code is copied from code.activestate.com/recipes/410469-xml-as-dictionary/  19.06.2019
    Added functionality of argument parser as described by Ph03n1x: https://stackoverflow.com/questions/14360389/getting-file-path-from-command-line-argument-in-python

    USAGE: python3 xml_to_csv.py /path/to/xml --outputDirectory /path/to/place/csv

    Notice that --outputDirectory is an optional argument. If no output directory is specified, then a directory at is created at inputDirectory/../data
'''

class XmlDictConfig(dict):
    def __init__(self, parent_element):
        childrenNames = []
        for child in parent_element.getchildren():
            childrenNames.append(child.tag)

        if parent_element.items(): #attributes
            self.update(dict(parent_element.items()))
        for element in parent_element:
            if element:
                # treat like dict - we assume that if the first two tags
                # in a series are different, then they are all different.
                #print len(element), element[0].tag, element[1].tag
                if len(element) == 1 or element[0].tag != element[1].tag:
                    aDict = XmlDictConfig(element)
                # treat like list - we assume that if the first two tags
                # in a series are the same, then the rest are the same.
                else:
                    # here, we put the list in dictionary; the key is the
                    # tag name the list elements all share in common, and
                    # the value is the list itself
                    aDict = {element[0].tag: XmlListConfig(element)}
                # if the tag has attributes, add those to the dict
                if element.items():
                    aDict.update(dict(element.items()))

                if childrenNames.count(element.tag) > 1:
                    try:
                        currentValue = self[element.tag]
                        currentValue.append(aDict)
                        self.update({element.tag: currentValue})
                    except: #the first of its kind, an empty list must be created
                        self.update({element.tag: [aDict]}) #aDict is written in [], i.e. it will be a list

                else:
                     self.update({element.tag: aDict})
            # this assumes that if you've got an attribute in a tag,
            # you won't be having any text. This may or may not be a
            # good idea -- time will tell. It works for the way we are
            # currently doing XML configuration files...
            elif element.items():
                self.update({element.tag: dict(element.items())})
            # finally, if there are no child tags and no attributes, extract
            # the text
            else:
                self.update({element.tag: element.text})


# def write_to_csv(data_dirs):
#     index = 0
#     df = pd.DataFrame(columns=['path', 'xmin', 'ymin', 'xmax', 'ymax', 'class'])
#     for dictionary in data_dirs:
#         for file_name in os.listdir(sys.path[0] + dictionary):
#             if '.xml' in file_name:
#                 tree = ElementTree.parse(sys.path[0] + dictionary + '\\' + file_name)
#                 root = tree.getroot()
#                 xmldict = XmlDictConfig(root)
#                 df.loc[index] = [xmldict['path']] + [xmldict['object']['bndbox']['xmin']] + [xmldict['object']['bndbox']['ymin']] + [xmldict['object']['bndbox']['xmax']] + [xmldict['object']['bndbox']['ymax']] + [xmldict['object']['name']]
#                 index += 1
#     print(df)
#     df.to_csv(path_or_buf=sys.path[0] + '\\test.csv', header=False, index=False, index_label=False)


def write_all_to_csv(inputDirectory, outputDirectory, label_file_name='dnv_dataset.csv', class_file_name='class_id.csv'):
    index = 0
    classes = []
    df = pd.DataFrame(columns=['path', 'xmin', 'ymin', 'xmax', 'ymax', 'class']) # Data frame

    for path, dirnames, filenames in os.walk(inputDirectory):
        for name in filenames:
            if '.xml' in name and (name[:-4] + '.jpg') in filenames:
                tree = ElementTree.parse(os.path.join(path, name))
                root = tree.getroot()
                xmldict = XmlDictConfig(root)
                try:
                    if type(xmldict['object']) == type([]):
                        for elem in xmldict['object']:
                            df.loc[index] =  [os.path.join(path, (name[:-4] + '.jpg')),
                                             elem['bndbox']['xmin'],
                                             elem['bndbox']['ymin'],
                                             elem['bndbox']['xmax'],
                                             elem['bndbox']['ymax'],
                                             elem['name']]
                            index += 1
                            if not any(elem['name'] in i for i in classes):
                                classes.append((elem['name'], len(classes)))
                    else:
                        df.loc[index] =  [os.path.join(path, (name[:-4] + '.jpg')),
                                         xmldict['object']['bndbox']['xmin'],
                                         xmldict['object']['bndbox']['ymin'],
                                         xmldict['object']['bndbox']['xmax'],
                                         xmldict['object']['bndbox']['ymax'],
                                         xmldict['object']['name']]
                        index += 1
                        if not any(xmldict['object']['name'] in i for i in classes):
                            classes.append((xmldict['object']['name'], len(classes)))

                except KeyError as arg:
                    if arg == 'path':
                        print('No object in the image')
                        df.loc[index] = [xmldict['path'],'','','','','']
                        index += 1
                    else:
                        print('Invalid xml-format')

    cdf = pd.DataFrame(data=classes, columns=['class', 'id'])
    # Creates label file
    df.to_csv(path_or_buf=os.path.join(outputDirectory, label_file_name), header=False, index=False, index_label=False)
    # Creates class file
    cdf.to_csv(path_or_buf=os.path.join(outputDirectory, class_file_name), header=False, index=False, index_label=False)

    print(df)
    print(cdf)


def makedirs(path):
    # Intended behavior: try to create the directory,
    # pass if the directory exists already, fails otherwise.
    # Meant for Python 2.7/3.n compatibility.
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise


def create_arg_parser():
    """"Creates and returns the ArgumentParser object."""

    parser = argparse.ArgumentParser(description='XML2CSV')
    parser.add_argument('inputDirectory',
                    help='Path to the input directory.')
    parser.add_argument('--outputDirectory',
                    help='Path to the output directory.')
    return parser



def main(args=None):
    # Parse args
    arg_parser = create_arg_parser()
    parsed_args = arg_parser.parse_args(sys.argv[1:])

    # Check if spesified input exist
    if os.path.exists(parsed_args.inputDirectory):
        print("Input directory exist at: ", parsed_args.inputDirectory)
    else:
        print("Input directory does not exist at specified path")


    # Check if spesified optional path
    if parsed_args.outputDirectory is not None:
        save_dir = parsed_args.outputDirectory
        if os.path.exists(save_dir):
            print("Output directory exist at: ", save_dir)
        else:
            makedirs(parsed_args.outputDirectory)
            print("Created directory at: ", save_dir)

    # Create directory at inputDirectory/../data
    else:
        save_dir = os.path.abspath(os.path.join(parsed_args.inputDirectory, os.pardir, 'data'))
        if not os.path.exists(save_dir):
            makedirs(save_dir)
            print("Created directory at: ", save_dir)
        else:
            print("Output directory exist at: ", save_dir)


    # Convert xml to csv to spesified path
    write_all_to_csv(parsed_args.inputDirectory, save_dir)


if __name__ == '__main__':
    main()
