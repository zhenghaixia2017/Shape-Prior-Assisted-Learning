from ultralytics.models import YOLO, RTDETR
import warnings
warnings.filterwarnings('ignore')

# Load a model
if __name__ == '__main__':


    model = YOLO(r'runs/detect/v26s_poseprior_perbox_1_0+repretrained_100+200epochs+fall2/weights/best.pt')  
    metrics = model.val(data=r'falling_datasets/falling_datasets/test-data.yaml' , batch=1, end2end=False)  
    print(f"mAP50: {metrics.box.map50}")         # mAP50
    print(f"Precision: {metrics.box.p}")         # 精确率
    print(f"Recall: {metrics.box.r}")



