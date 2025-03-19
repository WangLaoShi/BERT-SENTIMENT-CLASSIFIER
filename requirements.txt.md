这里是 `requirements.txt` 文件，你可以将其保存到你的项目目录中，并使用 `pip install -r requirements.txt` 进行安装：

```
torch
torchvision
torchaudio
transformers
tqdm
pandas
numpy
scikit-learn
```

如果你的环境需要指定 PyTorch 的版本（如 GPU 版本），请参考 [PyTorch 官网](https://pytorch.org/get-started/locally/) 选择适合你的 CUDA 版本，例如：
```
torch==2.1.0+cu118
torchvision==0.16.0+cu118
torchaudio==2.1.0+cu118
```

如果你使用 CPU，可以简单安装：
```
torch
torchvision
torchaudio
transformers
tqdm
pandas
numpy
scikit-learn
```

你可以试试这个 `requirements.txt`，看看是否满足你的需求！🚀
