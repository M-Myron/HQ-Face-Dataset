import json
import pandas as pd
import argparse

def mainpart(input_path, excel_label_path, result_path):
    period = {}
    # 读取待添加标签的json文件
    with open(input_path, 'r+') as f:
        data = json.load(f)
        f.close()
    # 读取带标签的excel
    df = pd.read_excel(excel_label_path)

    #找到json文件中，每个数据在表格中的对应数据（根据uid查找对应），把表格中需要添加的label加到json中
    # data_added就是json中某组数据在表格中的对应数据，根据uid进行匹配
    for i in range(len(data)):
        data_added = df[lambda df: df['UID'] == data[i]['uid']]
        # 此处为添加标签，如某个标签不需要添加，删除对应代码即可
        if not data_added.empty:
            data[i].update({'channel_id': data_added['channel_id'].tolist()})
            data[i].update({'author': data_added['author'].tolist()})
            data[i].update({'person_id': list(map(int,data_added['person_id'].tolist()))})
            data[i].update({'scenarios': data_added['scenarios'].tolist()})
            data[i].update({'resolution': data_added['resolution'].tolist()})
            data[i].update({'speech/non-speech': data_added['speech/non-speech'].tolist()})
            data[i].update({'subtitle (auto/self/none)': data_added['subtitle (auto/self/none)'].tolist()})
            data[i].update({'background': data_added['background'].tolist()})
            data[i].update({'indoor/outdoor': data_added['indoor/outdoor'].tolist()})
            data[i].update({'defocus':list(map(int,data_added['defocus'].tolist()))})

    # 写入新json文件
    with open(result_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str)
    parser.add_argument('--label_dir', type=str)
    parser.add_argument('--output_dir', type=str)
    args = parser.parse_args()

    input_path = args.input_dir
    excel_label_path = args.label_dir
    result_path = args.output_dir

    mainpart(input_path, excel_label_path, result_path)      
        
        
        
        
        
        
        
      
