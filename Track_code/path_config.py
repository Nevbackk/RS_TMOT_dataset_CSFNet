#路径配置及训练参数设置

# ==================== 数据集路径配置 ====================
# 数据集根目录
DATA_ROOT = '/root/autodl-tmp/VISO_data'

# 训练集（每个序列的前 train_ratio 帧用于训练）
TRAIN_SEQS = ['002', '006', '029','025','038','041','043']   

# 验证集（与 TRAIN_SEQS 相同，但取每个序列的后 1-train_ratio 帧）
VAL_SEQS   = ['002', '006', '029','025','038','041','043']   

# 测试集
TEST_SEQS  = ['033','024','010','030']                
TEST_SEQ   = TEST_SEQS[0] if TEST_SEQS else ''

# 划分比例（默认0.8，即每个序列前80%帧用于训练，后20%用于验证）
TRAIN_RATIO = 0.8

# COCO 格式标注文件的存放目录（转换脚本生成）
ANNOTATIONS_DIR = f'{DATA_ROOT}/annotations'

# 训练/验证/测试对应的 JSON 文件名（固定，由转换脚本生成）
TRAIN_JSON = f'{ANNOTATIONS_DIR}/instances_train.json'
VAL_JSON   = f'{ANNOTATIONS_DIR}/instances_val.json'
TEST_JSON  = f'{ANNOTATIONS_DIR}/instances_test.json'


# ==================== 模型与训练超参数 ====================
# 类别数（根据 JSON 中 categories 设置）
NUM_CLASSES = 1

# 训练参数
BATCH_SIZE = 1          # 根据 GPU 显存调整
NUM_EPOCHS = 50         # 训练总轮数
GPUS = '0'              # 使用的 GPU 编号
NUM_WORKERS = 4         # 数据加载线程数
LR = 1.25e-4            # 初始学习率
SEQ_LEN = 3             # 时序长度（与训练时保持一致）
K = 450                 # 最大检测目标数

# 学习率下降步长（字符串，逗号分隔）
LR_STEP = '15'          