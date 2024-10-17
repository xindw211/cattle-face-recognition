import os
import random
import shutil
# Divide the training set and testing set
root_dir = 'datasets/cow'

# 训练集和验证集保存目录
train_dir = 'datasets/cow/train'
test_dir = 'datasets/cow/test'

categories = os.listdir(root_dir)
num_classes = len(categories)

num_test_classes = int(num_classes * 0.1)
test_categories = random.sample(categories, num_test_classes)

train_count = 0
test_count = 0

for c in categories:
    src_dir = os.path.join(root_dir, c)
    if c in test_categories:
        dst_dir = os.path.join(test_dir, c)
        test_count += len(os.listdir(src_dir))
    else:
        dst_dir = os.path.join(train_dir, c)
        train_count += len(os.listdir(src_dir))

    os.makedirs(dst_dir, exist_ok=True)

    for f in os.listdir(src_dir):
        src_file = os.path.join(src_dir, f)
        shutil.copy2(src_file, dst_dir)

print("Train set image count:", train_count)
print("Test set image count:", test_count)
