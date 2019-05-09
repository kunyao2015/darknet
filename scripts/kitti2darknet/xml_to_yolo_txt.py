# �˴����VOC_KITTI�ļ���ͬĿ¼
import glob
import xml.etree.ElementTree as ET
# ���������Ϊ����xml�����������˳�����ڲ���Ҫ����
class_names = ['Car', 'Cyclist', 'Pedestrian']
# xml�ļ�·��
path = './Annotations/' 
# ת��һ��xml�ļ�Ϊtxt
def single_xml_to_txt(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    # �����txt�ļ�·��
    txt_file = xml_file.split('.')[0]+'.txt'
    with open(txt_file, 'w') as txt_file:
        for member in root.findall('object'):
            #filename = root.find('filename').text
            picture_width = int(root.find('size')[0].text)
            picture_height = int(root.find('size')[1].text)
            class_name = member[0].text
            # ������Ӧ��index
            class_num = class_names.index(class_name)

            box_x_min = int(member[1][0].text) # ���ϽǺ�����
            box_y_min = int(member[1][1].text) # ���Ͻ�������
            box_x_max = int(member[1][2].text) # ���½Ǻ�����
            box_y_max = int(member[1][3].text) # ���½�������
            # ת�����λ�úͿ��
            x_center = (box_x_min + box_x_max) / (2 * picture_width)
            y_center = (box_y_min + box_y_max) / (2 * picture_height)
            width = (box_x_max - box_x_min) / picture_width
            height = (box_y_max - box_y_min) / picture_height
            print(class_num, x_center, y_center, width, height)
            txt_file.write(str(class_num) + ' ' + str(x_center) + ' ' + str(y_center) + ' ' + str(width) + ' ' + str(height) + '\n')
# ת���ļ����µ�����xml�ļ�Ϊtxt
def dir_xml_to_txt(path):
    for xml_file in glob.glob(path + '*.xml'):
        single_xml_to_txt(xml_file)
dir_xml_to_txt(path)