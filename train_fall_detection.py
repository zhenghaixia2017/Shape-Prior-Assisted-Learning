from train_pose_prior import PosePriorTrainer  
from ultralytics.utils import DEFAULT_CFG

if __name__ == '__main__':
    overrides = {
        'model': 'yolo26_fall_poseprior.yaml',          
        'data': r'falling_datasets/falling_datasets/data.yaml',
        'imgsz': 640,
        'epochs': 200,
        'batch': 16,
        'workers': 8,
        'device': '0',
        'single_cls': True,                             
        'pretrained': False,                            
        'name': 'v26s_poseprior_perbox_1_0+repretrained_100+200epochs+fall',
        'resume': False,
        'weights': 'runs/detect/v26s+poseprior+pretrained+200epochs+coco2/weights/epoch100.pt',
        'shape_reg_weight': 0.5,
        'center_reg_weight': 0.0,
    }

    trainer = PosePriorTrainer(cfg=DEFAULT_CFG, overrides=overrides)
    trainer.train()