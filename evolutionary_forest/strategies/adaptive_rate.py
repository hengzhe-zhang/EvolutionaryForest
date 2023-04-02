import math
import matplotlib.pyplot as plt


class CosineAnnealingLR:
    def __init__(self, max_epochs, base_lr=0.001, min_lr=0.0001):
        self.max_epochs = max_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr

    def __call__(self, epoch):
        cos = (1 + math.cos(math.pi * epoch / self.max_epochs)) / 2
        lr = self.min_lr + (self.base_lr - self.min_lr) * cos
        return lr


if __name__ == '__main__':
    # 设置最大轮数和初始学习率
    max_epochs = 50
    base_lr = 0.1
    min_lr = 0.001

    # 创建CosineAnnealingLR对象和一个空列表，用于存储每个epoch的学习率
    lr_scheduler = CosineAnnealingLR(max_epochs, base_lr, min_lr)
    lr_history = []

    # 计算每个epoch的学习率并将其添加到列表中
    for epoch in range(max_epochs):
        lr = lr_scheduler(epoch)
        lr_history.append(lr)

    # 绘制学习率随epoch变化的曲线
    plt.plot(lr_history)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Cosine Annealing Learning Rate Schedule')
    plt.show()
