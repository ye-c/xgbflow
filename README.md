# xgbflow

模型的训练及验证过程中有很多通用的代码环节，xgbflow提供了一些日常用到的数据分析和模型训练（基于xgboost）的封装方法，大大减少了python代码量。

## 目录结构

```
xgbflow
├── __init__.py
├── api             # 训练模型相关api封装
│   ├── __init__.py
│   ├── draw.py         # 绘图
│   ├── mod.py          # 预测
│   └── xflow.py        # 常用机器学习
├── sample          # 样本处理
│   ├── __init__.py
│   ├── analyse.py      # 数据分析
│   └── process.py      # 数据处理
└── utils           # 工具
    ├── __init__.py
    └── markitdown.py   # markdown工具类
```

## Quick start

```
import sys
sys.path.append('/Users/yec/Codes')

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

from xgbflow.api import xflow, draw, mod
from xgbflow.utils.markitdown import MarkitDown
from xgbflow.feature import analyse
```