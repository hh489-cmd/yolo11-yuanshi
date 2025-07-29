from ultralytics import YOLO


if __name__ == '__main__':
    #model = YOLO("yolo11-APConv.yaml")  # build a new model from YAML
    model = YOLO("yolo11-cls.yaml")  # build a new model from YAML

    # model = YOLO("yolo11n-cls.pt")  # load a pretrained model (recommended for training)
    # model = YOLO("yolo11n-cls.yaml").load("yolo11n-cls.pt")  # build a new model from YAML


    #model.train(data="E:\yo\yolo11-yuanshi\shujuji",  # 数据集配置文
    #model.train(data=r"E:\yo\yolo11-yuanshi\shujuji",  # 数据集配置文件
    #model.train(data=r"E:\yo\yolo11-yuanshi\plant disease_split",  # 数据集配置文件
    model.train(data=r"E:\yo\yolo11-yuanshi\plant disease_split_paddy doctor",

                epochs=50,
                imgsz=512,
                batch=64,
                device='0',# 使用GPU训练
                # weights="yolo11n.pt"

    )