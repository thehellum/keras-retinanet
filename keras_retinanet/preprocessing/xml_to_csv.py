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


def write_all_to_csv(inputDirectory, outputDirectory, label_file_name='dnv_dataset.csv', class_file_name='class_id.csv'):
    index_val, index_train = 0,0
    classes = []
    df_train = pd.DataFrame(columns=['path', 'xmin', 'ymin', 'xmax', 'ymax', 'class']) # Data frame
    df_val = pd.DataFrame(columns=['path', 'xmin', 'ymin', 'xmax', 'ymax', 'class'])
    for path, dirnames, filenames in os.walk(inputDirectory):
        for name in filenames:
            # if ('.xml' in name and (name[:-4] + '.jpg') in filenames) or ('.xml' in name and (name[:-4] + '.jpeg') in filenames) or ('.xml' in name and (name[:-4] + '.png') in filenames):
            # if '.xml' in name and ( (name[:-4] + '.jpg') or (name[:-4] + '.jpeg') or (name[:-4] + '.png') ) in filenames:
            if '.xml' in name:
                if name[:-4] + '.jpg' in filenames:
                    image_name = name[:-4] + '.jpg'
                elif name[:-4] + '.jpeg' in filenames:
                    image_name = name[:-4] + '.jpeg'
                elif name[:-4] + '.png' in filenames:
                    image_name = name[:-4] + '.png'
                else:
                    continue

                tree = ElementTree.parse(os.path.join(path, name))
                root = tree.getroot()
                xmldict = XmlDictConfig(root)
                try:
                    split_prob = np.random.uniform(low=0,high=1)
                    if type(xmldict['object']) == type([]):
                        for elem in xmldict['object']:
                            if elem['name'] == 'motor_vessel':
                                label_name = 'semi_open_pleasure_craft'
                            else:
                                label_name = elem['name']
                            if split_prob <= 0.8:
                                df_train.loc[index_train] =  [os.path.join(path, image_name),
                                                 elem['bndbox']['xmin'],
                                                 elem['bndbox']['ymin'],
                                                 elem['bndbox']['xmax'],
                                                 elem['bndbox']['ymax'],
                                                 label_name]
                                index_train += 1
                            else:
                                df_val.loc[index_val] =  [os.path.join(path, image_name),
                                                 elem['bndbox']['xmin'],
                                                 elem['bndbox']['ymin'],
                                                 elem['bndbox']['xmax'],
                                                 elem['bndbox']['ymax'],
                                                 label_name]
                                index_val += 1

                            if not any(label_name in i for i in classes):
                                classes.append((label_name, len(classes)))
                    else:
                        if xmldict['object']['name'] == 'motor_vessel':
                            label_name = 'semi_open_pleasure_craft'
                        else:
                            label_name = xmldict['object']['name']
                        if split_prob <= 0.8:
                            df_train.loc[index_train] = [os.path.join(os.path.abspath(path), image_name),
                                             xmldict['object']['bndbox']['xmin'],
                                             xmldict['object']['bndbox']['ymin'],
                                             xmldict['object']['bndbox']['xmax'],
                                             xmldict['object']['bndbox']['ymax'],
                                             label_name]
                        else:
                            df_val.loc[index_val] = [os.path.join(os.path.abspath(path), image_name),
                                             xmldict['object']['bndbox']['xmin'],
                                             xmldict['object']['bndbox']['ymin'],
                                             xmldict['object']['bndbox']['xmax'],
                                             xmldict['object']['bndbox']['ymax'],
                                             label_name]
                            index_val += 1
                        if not any(label_name in i for i in classes):
                            classes.append((label_name, len(classes)))

                except KeyError as arg:
                    if arg == 'path':
                        print('No object in the image')
                        df.loc[index] = [xmldict['path'],'','','','','']
                        index += 1
                    else:
                        print('Error: ',arg)

    cdf = pd.DataFrame(data=classes, columns=['class', 'id'])
    # Creates label file
    df_full = pd.concat([df_train,df_val])

    df_val.to_csv(path_or_buf=os.path.join(outputDirectory, 'validation.csv'), header=False, index=False, index_label=False)
    df_train.to_csv(path_or_buf=os.path.join(outputDirectory, 'train.csv'), header=False, index=False, index_label=False)
    df_full.to_csv(path_or_buf=os.path.join(os.path.abspath(outputDirectory), label_file_name), header=False, index=False, index_label=False)
    # Creates class file
    cdf.to_csv(path_or_buf=os.path.join(os.path.abspath(outputDirectory), class_file_name), header=False, index=False, index_label=False)
    return label_file_name

def train_val_split(split, path_from, path_to, label_file_name='dnv_dataset.csv'):
    # Load data
    df = pd.read_csv(filepath_or_buffer=os.path.join(path_from, label_file_name))
    row_count, _ = df.shape
    split_index = int(0.8*row_count)
    while True:
        if df.iloc[split_index][0] == df.iloc[split_index+1][0]:
            break
        else:
            split_index += 1
    train_df = df.iloc[:split_index,:]
    val_df = df.iloc[split_index:,:]
    train_df.to_csv(os.path.join(path_to, 'train.csv'), header=False, index=False, index_label=False)
    val_df.to_csv(os.path.join(path_to, 'validation.csv'), header=False, index=False, index_label=False)


def filter_by_class(path_to_csv, save_name, class_list):
    with open(path_to_csv, 'rt') as input, open(save_name, 'wt',  newline='') as output:
        writer = csv.writer(output)
        for row in csv.reader(input):
            if np.intersect1d(row,class_list).size == 1:
                writer.writerow(row)

                
def filter_class_id(path_to_csv, save_name, class_list):
    index = 0
    with open(path_to_csv, 'rt') as input, open(save_name, 'wt', newline='') as output:
        writer = csv.writer(output)
        for class_name, id in csv.reader(input):
            if class_name in class_list:
                writer.writerow([class_name, str(index)])
                index += 1


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


def count_number_of_classes(filename):
    classes = {}
    with open(filename,'r') as f:
        reader = csv.reader(f)
        for row in reader:
            key = row[5]
            if key in classes:
                classes[key] = classes[key] +1
            else:
                classes[key] = 1
    return classes


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
        save_dir = os.path.abspath(os.path.join(parsed_args.inputDirectory, os.pardir, 'csv'))
        if not os.path.exists(save_dir):
            makedirs(save_dir)
            print("Created directory at: ", save_dir)
        else:
            print("Output directory exist at: ", save_dir)


    # Convert xml to csv to spesified path
    label_file_name = write_all_to_csv(parsed_args.inputDirectory, save_dir)
    class_count = count_number_of_classes(os.path.join(save_dir, 'dnv_dataset.csv'))
    class_list = []
    for key,val in class_count.items():
        if val > 100:
            class_list.append(key)

    filter_by_class(path_to_csv=os.path.join(save_dir, 'train.csv'), save_name=os.path.join(save_dir, 'train_filtered.csv'), class_list=class_list)
    filter_by_class(path_to_csv=os.path.join(save_dir, 'validation.csv'), save_name=os.path.join(save_dir, 'validation_filtered.csv'), class_list=class_list)
    filter_class_id(path_to_csv=os.path.join(save_dir,'class_id.csv'), save_name=os.path.join(save_dir,'class_id_filtered.csv'), class_list=class_list)


if __name__ == '__main__':


    #filter_class_id(path_to_csv=os.path.join(save_dir,'class_id.csv'), class_list=class_list)
    main()
