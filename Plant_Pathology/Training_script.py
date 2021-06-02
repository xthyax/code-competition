import os
from module_script import Main_session
from utils.utils import set_GPU

class CFG:
    root = r"D:\KLA\Jhyn\Stuff\PP_2021\plant-pathology-2021-fgvc8"

    dup_csv_path = os.path.join( root, "duplicates.csv")
    classes = [
        'complex', 
        'frog_eye_leaf_spot', 
        'powdery_mildew', 
        'rust', 
        'scab',
        'healthy'
    ]

    pretrained_path = r"D:\KLA\Jhyn\_Model\Pre_trained\ENet\startingmodel.pth"

    models_path = "model_files"

    model_name = 'efficientnet-b4' 

    lr_rate = 1e-5
    seed = 42
    num_gpus = 2
    num_workers = min(8,num_gpus * 2) if num_gpus > 1 else 2
    batch_size = 16 * num_gpus
    img_size = 256
    folds = 5
    transform = True
    epochs = 100
    patient = 10

if __name__ == '__main__':

    os.makedirs(CFG.models_path, exist_ok=True)

    if os.path.exists(CFG.root):
        GET_CV = True
        
    else:
        CFG.root = r"../input/shopee-product-matching/"
        GET_CV = False
    
    set_GPU(CFG.num_gpus, memory_restraint=1500)
    
    session = Main_session(GET_CV, CFG)
    
    session.run_training(continue_training=True)

    sys.exit()
    
    _, pdata = session.get_predict_results()

    if GET_CV:
        pdata['labels'] = pdata.apply(combine_predictions, axis = 1)
        pdata['f1'] = f1_score(pdata['origin_labels'], pdata['labels'])
        score = pdata['f1'].mean()
        print(f"Our final f1 cv score is {score}")
        pdata[['image', 'labels']].to_csv('submission.csv', index = False)

    else:
        pdata['labels'] = pdata.apply(combine_predictions, axis = 1)
        pdata[['image', 'labels']].to_csv('submission.csv', index = False)






