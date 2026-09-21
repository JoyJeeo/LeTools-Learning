# 安环境,下权重
```
cd kuavo_model/external_models/GalaxeaVLA
export UV_DEFAULT_INDEX=https://mirrors.aliyun.com/pypi/simple/
export UV_PYTHON_INSTALL_MIRROR=https://gh-proxy.com/https://github.com/astral-sh/python-build-standalone/releases/download
uv sync --index-strategy unsafe-best-match
```
权重路径https://huggingface.co/OpenGalaxea/G05/tree/main
只需要 g05-base, qwen3_5_2b_base_processpr, action_tokenizer.pt 就行

# 配置文件
LeTools-Learning/kuavo_model/external_models/GalaxeaVLA/configs/data/kuavo.yaml  这里修改最下面的数据路径
LeTools-Learning/kuavo_model/external_models/GalaxeaVLA/configs/task/kuavo.yaml  这里修改一些路径和配置
LeTools-Learning/kuavo_model/external_models/GalaxeaVLA/configs/tokenizer/actioncodec.yaml 这里修改action tokenizer的路径
LeTools-Learning/kuavo_model/external_models/GalaxeaVLA/configs/train.yaml 这里可以修改多少步保存一次

# 训练
CUDA_VISIBLE_DEVICES=2,3 uv run bash scripts/run/finetune.sh 2 kuavo  ，几张卡 中间那个数字就改成几

这里默认用的是flash attn4,实测在5880用不了，H100应该是可以的，如果遇到flash attn的报措，重新装一个flash attn2就好了，可以用letools网站里放的那个groot的

拷贝权重的时候， 除了checkpoints，.hydra, dataset_stats.json, action_tokenizer.pt也要拷