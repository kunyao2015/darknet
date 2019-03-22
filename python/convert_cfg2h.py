#! /usr/bin/env python

import os
import io
import argparse
from collections import defaultdict


parser = argparse.ArgumentParser(description='cfg file converter')
parser.add_argument('config_path', help='Path to Darknet cfg file.')
parser.add_argument('output_path', help='Paht to output file')



def _main(args):
    #config_path = os.path.expanduser(args.config_path)
    config_path = args.config_path
    assert config_path.endswith('.cfg'), '{} is not a .cfg file'.format(
        config_path)

    output_path = args.output_path
    assert output_path.endswith('.h'), '{} is not a .h file'.format(
        output_path)

    print('Parsing Darknet config.')
    str = ''
    with open(config_path, 'r', encoding='utf-8') as file:
        for eachline in file:
            eachline = eachline.strip()
            if len(eachline) == 0:
                continue
            str += eachline + '$'
    #print(str)

    if os.path.exists(output_path):
        os.remove(output_path)
        print('delete old file:%s' % output_path)

    index = output_path.rfind('.')
    output_path_name = output_path[:index]
    header = '#ifndef ' + output_path_name.upper() + '_H\n'
    header += '#define ' + output_path_name.upper() + '_H\n\n'
    
    body = 'const char* config_str = "' + str + '";\n\n'

    footer = '#endif\n'
    with open(output_path,'w') as fw:
       fw.writelines(header + body + footer)

        


if __name__ == '__main__':
    _main(parser.parse_args())
