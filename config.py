gpu_id = [0,3,4]
DATA ='/home/ubuntu/recognition/data/masks_clean/HAIR/'
#DATA = '/home/dev/Documents/HAIR/'
CHECKPOINT_DIR = 'checkpoints'
USE_MATPLOTLIB_VIS = False
RESIZE_TO = 128
VAL_EPOCH_DICE = 'val_epoch_dice'
VAL_EPOCH_BCE = 'val_epoch_bce'
PER_ITER_LOSS = 'per_iter_loss'
PER_EPOCH_LOSS = 'per_epoch_loss'
BASE_BATCH_SIZE = 4
BATCH_SIZE = len(gpu_id) * BASE_BATCH_SIZE
BATCH_SIZE=64
EPOCH_NUM = 200
PER_ITER_DICE = 'per_iter_dice'
PER_ITER_IOU = 'per_iter_iou'
