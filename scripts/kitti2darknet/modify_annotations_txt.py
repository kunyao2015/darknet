import glob
import string
txt_list = glob.glob('./Labels/*.txt') # �洢Labels�ļ�������txt�ļ�·��
def show_category(txt_list):
    category_list= []
    for item in txt_list:
        try:
            with open(item) as tdf:
                for each_line in tdf:
                    labeldata = each_line.strip().split(' ') # ȥ��ǰ�������ַ�������ֿ�
                    category_list.append(labeldata[0]) # ֻҪ��һ���ֶΣ������
        except IOError as ioerr:
            print('File error:'+str(ioerr))
    print(set(category_list)) # �������
def merge(line):
    each_line=''
    for i in range(len(line)):
        if i!= (len(line)-1):
            each_line=each_line+line[i]+' '
        else:
            each_line=each_line+line[i] # ���һ���ֶκ��治�ӿո�
    each_line=each_line+'\n'
    return (each_line)
print('before modify categories are:\n')
show_category(txt_list)
for item in txt_list:
    new_txt=[]
    try:
        with open(item, 'r') as r_tdf:
            for each_line in r_tdf:
                labeldata = each_line.strip().split(' ')
                if labeldata[0] in ['Truck','Van','Tram']: # �ϲ�������
                    labeldata[0] = labeldata[0].replace(labeldata[0],'Car')
                if labeldata[0] == 'Person_sitting': # �ϲ�������
                    labeldata[0] = labeldata[0].replace(labeldata[0],'Pedestrian')
                if labeldata[0] == 'DontCare': # ����Dontcare��
                    continue
                if labeldata[0] == 'Misc': # ����Misc��
                    continue
                new_txt.append(merge(labeldata)) # ����д���µ�txt�ļ�
        with open(item,'w+') as w_tdf: # w+�Ǵ�ԭ�ļ�������ɾ������д�����ݽ�ȥ
            for temp in new_txt:
                w_tdf.write(temp)
    except IOError as ioerr:
        print('File error:'+str(ioerr))
print('\nafter modify categories are:\n')
show_category(txt_list) 