---
title: 这份文档文档能让你成为Megatron-LM安装与卸载大师
date: 2026-01-19 19:50:49
tags: 经验
---

本文档着重于`Megatron-LM`的安装指导与一些排错的技巧，并对该软件给出一个当前安装成功率比较高的组合。

Python不时会出现新的版本管理软件，安装上述软件的方式多数基本也大同小异；当然，随着megatron-core的依赖项的变迁，会调起本地编译的包也在增多，实在没必要把所有情况都面面俱到；但只要把步骤方法要点及原理讲明白，其实这些都是触类旁通的。掌握了文档中这些技巧与方法，以后面对新依赖报错也是可以快速上手的，如有疑难欢迎提交相关issues给AI。

此文档针对的受众是追求快速搞好Megatron-LM环境的相关人群等，不过也因本人知识水平有限，同时也避免重复的“造轮子”，对具体报错的理解和处理等方面需要参考AI回答或阅读网上的一些相关文章，因此允许演绎该作及共享也想不到谁要商用。

由于需要复现一些论文中的相关代码与结果等等；仅此，配置Megatron-LM只是为研究需要，以下是免责声明：

* 本文档不保证内容的**实效性**，有问题欢迎找AI深入探讨
* 这个前言其实一点有用的东西都没有

此文档献给被Megatron-LM环境的配置折磨的死去活来的人们，以上...

# 核心问题

安装Megatron-LM会面临的核心问题，其实是安装过程中会卡在

```text
Building wheels for collected packages: causal-conv1d, mamba-ssm, nv-grouped-gemm, transformer_engine_torch
```

这一步骤上。而这个问题究其本质，其实就是连不上在海外的github的分发服务器（CDN）。一而言之，不使用[官方的Docker容器](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/PyTorch)，直接在服务器上配置环境，面临的核心问题实际上**网络**的问题。

为了解决这个问题，一般而言就是挂个代理，但是这在服务器上是不现实的。对此有一个替代方案，就是手动去下载wheel预编译文件，然后本地安装，进而把问题转化为找到上述卡住的依赖库提前编译好的wheel文件，它们依赖的目标版本都恰好都满足的版本组合。同时，因为这些预编译文件都放在github上，可以使用一些代理网站在服务器上直接下载，而不是本地上传给服务器，这样可以有效地利用服务器的下载带宽。

TL;DR的话，在文档撰写的时间点，笔者大致检查过的几个最新的版本的可行组合是：（包括但不限于）

* python=3.11,3.12
* torch==2.6.0,2.7.0

本文会介绍~~实际安装过的~~python3.11+torch2.6.0的过程和中途可能遇到的问题，接下来是一些常见的手动代理手段介绍。

# 手动代理

为了本文档的完整性，这里附上一份手动代理的方法介绍。所有代理网站和镜像站都以`www.abc.xyz`代替，具体的代理网站就有请读者自行搜索了，后文给出的下载命令也不会直接添加代理。

## Python

虽然称不上手动代理，但是确实有一些不太走心的云服务提供商不修改为国内镜像源的情况，这里一并介绍了。

对于**单次下载**，pip提供了选项`-i` `--index-url`，用法如下：

```shell
pip install <package_name> -i https://www.abc.xyz/simple
```

如果想一次配置，多次使用，python也提供了相关命令：

```shell
pip config set global.index-url https://www.abc.xyz/simple
```

该命令的效果等同于修改`~/.pip/pip.conf`：

```yaml
[global]
index-url = https://www.abc.xyz/simple
```

另外提一句如果使用了[uv](https://docs.astral.sh/uv/)作为版本控制软件，它的pip兼容命令在单次下载时可以继续使用`-i`选项；但是由于不存在config子命令，并且目前uv会忽略pip.config配置文件，一次配置多次使用的方法略有改变，而且没有配置全局的方法，只能在项目文件中添加如下项目：

```toml
[[tool.uv.index]]
url = "htts://www.abc.xyz/simple"
default = true
```

## Github

对于Github，本文主要涉及的需求有两个：下载仓库和下载该项目的release。主流的解决方案是用互联网大善人&慈善家Cloudflare家的workers，当然我们直接用别人现成配置好的即可。至于使用方法，只需要在原始链接前面加上你想使用代理网站：

```shell
#下载仓库
git clone https://www.abc.xyz/https://github.com/xxx/xxx.git
#下载release
wget "https://www.abc.xyz/https://github.com/xxx/xxx/release/x.x.x/xxx.whl"
```

当然，还有一种方案就是，去找这个仓库有没有有人搬到Gitee上了，不过一般只能下载这个仓库，release的话Gitee是不会镜像过来的。

## Hugginface

对于Hugginface则分为两种情况，一是通过Python调用transformer库来下载模型和其它文件（使用他们家的`hugginface-cli`也属于这种情况），另一种情况是~~听信AI谗言~~直接通过wget等工具直接下载相关文件。

对于前者，官方提供一个非常简单环境变量HF_ENDPOINT来使用镜像站（所以实际链接也不是经典的代理格式），此时具体操作为为：

```shell
export HF_ENDPOINT=https://www.abc.xyz/
```

如果是想通过直链下载文件，则需要把域名替换掉：

```text
https://huggingface.co/xxx/xxx -> https://www.abc.xyz/xxx/xxx
```

# 在安装Megatron-LM之前

## 前置工作

首先确认自己的python版本是否是需要的版本，若不是则要切换版本。以使用conda安装python 3.12为例：

```shell
conda install python=3.12
```

然后还有一部分是为了安装megatron-core所需要的依赖：

```shell
pip install pybind11 "setuptools<80.0.0,>=77.0.0" "packaging>=24.2"
```

## 安装PyTorch

有一点是在安装megatron-core时，它的安装脚本需要确定环境中PyTorch的版本（即Megatron-LM安装选项`--no-build-isolation`的作用），因此需要提前安装好PyTorch。首先使用命令`nvidia-smi`检查服务器的显卡驱动，CUDA支持的版本是否大于等于PyTorch需要的版本，对于举例的2.6.0版本，可以通过一下命令指定版本安装：

```shell
pip install torch==2.6.0
```

上述命令安装的PyTorch的CUDA版本为应该为12.4。如果发现服务器的驱动CUDA小于该版本则比较麻烦，但推荐最好还是更新一下驱动，根据Linux发行版不同方法有所不同，本文档就略过了。

<details>
<summary>如果读者选择了torch版本不为2.6.0，或者想了解进一步了解CUDA版本的兼容逻辑</summary>
这里涉及四个CUDA版本：

1. PyTorch支持的版本
2. 系统中CUDA工具链的版本
3. 随着PyTorch下载下来的CUDA pip wheel版本
4. GPU驱动支持的版本

**PyTorch支持的版本**，本质上是指的是这个wheel包在预编译时链接的CUDA工具链的版本，可以通过调用`torch.version.cuda`获取该信息：

```shell
python -c "import torch; print(torch.version.cuda)" #torch==2.6.0时默认为12.4
```

**CUDA工具链**包括编译器`nvcc`和对应的运行库（`.so`动态链接库+`.h`头文件）。PyTorch在安装时同时携带一份和它编译版本一致的动态链接库，这些文件是直接打包进PyTorch的wheel文件之中的。而在PyTorch 2.0之后，为了支持图编译和自动算子融合，进而引入了OpenAI Triton后端，triton则需要CUDA pip wheel。

**CUDA Pip Wheel**则是NVIDIA官方提供的一个方案，它解决就是Python程序需要在运行时现写好一份编译好的GPU机器指令时，但是系统不一定有CUDA运行库的需求。它实际上就是通过pip wheel包来分发CUDA运行库。通过`pip list | grep nvidia`可以查看它们的版本号~~虽然没看出来有什么意义~~。

**GPU驱动支持的版本**最后决定了具体调用的CUDA运行库是否兼容。总所周知CUDA的地基是C++语言，而众所周知C++动态链接库的兼容性是一个非常严肃的玄学问题，最终取决于CUDA工具链和GPU驱动跨版本的兼容性。对此NVIDIA[官方的结论](https://docs.nvidia.com/deploy/cuda-compatibility/conclusion.html)是CUDA驱动程序保持向后兼容性。

这也就说，<ins>**需要满足 PyTorch支持的版本 <= (CUDA工具链版本，后文的Apex需要) <= GPU驱动支持的版本 **</ins>

---

</details>

## 安装 ninja-build

还有一个比较重要的东西是[ninja-build](https://ninja-build.org/)编译后端（作用相当于`make`）。它的意义在于让涉及C/C++的编译过程尽可能的快，最显著的一点就是它能自动执行多线程编译。安装ninja不仅可以在可以在安装Megatron-LM和Apex极大的加快编译过程，而且在PyTorch的运行过程中也可以加速一些编译过程（类似JIT）。

包名根据Linux发行版不同也有所不同，但可以用系统的包管理器试一试安装`ninja-build`或者`ninja`：

```shell
#例如对于Debian/Ubuntu发行版
apt install ninja-build
```

具体的安装说明可以参见[官方在Github上的安装文档](https://github.com/ninja-build/ninja/wiki/Pre-built-Ninja-packages)。

## 其它

其余是一些可以提前安装好的依赖：

```shell
pip install tensorboard #如果需要使用集成的tensorboard功能则需要手动下载
```

然后就可以正式进入Megatron-LM的安装了。

# 安装 Megatron-LM

需要解释的是Megatron-LM实际上指的是megatron-core的额外依赖选项mlm，同时NVIDIA提供了dev和lts两种依赖策略。其中lts+NGC可以看作是NVIDIA为了追求结果可复现性的努力。总结的话，有5种可行组合：

* （不带额外依赖项）：只有megatron-core和torch
* [dev]：追随最新上游依赖项
* [lts]：NGC（官方Docker容器）PyTorch 24.01的长期支持版本
* [mlm,dev]：（在dev的基础上），能够运行Megatron-LM程序 ~~笔者也不知道这个指的是什么~~
* [mlm,lts]：同上

本着利在千秋的思想，一般会选择的是更新、支持的更多的版本：

```shell
pip install --no-build-isolation megatron-core[mlm,dev]
```

当然此时直接执行安装命令，实际上还是会卡在文章开头的所述的环节上的。对此，我们可以去手动下载恰好满足`python=3.11`&`torch=2.6.0+cu12`的wheel预编译包：

```shell
# causal-conv1d
wget https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.5.3.post1/causal_conv1d-1.5.3.post1+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl

# mamba-ssm
wget https://github.com/state-spaces/mamba/releases/download/v2.3.0/mamba_ssm-2.3.0+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl

# nv_grouped_gemm
wget https://github.com/fanshiqing/grouped_gemm/releases/download/v1.1.4.post8/nv_grouped_gemm-1.1.4.post8+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
```

对于`transformer_engine_torch`则没有对应的预编译包，但该包的编译一般也不会卡住。在下载好之后就可以直接`pip install 本地文件`完成安装。之后再执行megatron-core的安装大概率就不会出现问题了。

以及，如果没有选择举例Python和Torch版本的情况下，预编译包的直链显然也是会不同的。同时为了文档面向未来的兼容性，这里大致讲解一下两个方法。第一种方法就是，直接尝试安装megatron-core，然后等待安装命令超时，之后可以在报错信息中看到具体的下载链接。第二种则是自行前往github发布页，根据本地的实际情况选择依赖包，如果看见包名中的`abi`默认选`FALSE`的即可。

不过如果本地选择的版本确实太新的话，就会导致没有对应的预编译包下载。此时pip安装程序会尝试拉取git仓库进行本地编译，然而这个动作多半也会卡住。解决方法也是自行通过代理`git clone`然后本地编译了。这种方式过于复杂，也确实没有带来明显的好处，如果读者确实需要则只能请教AI了。

# 安装 Apex

在安装Apex之前首先需要保证本地有CUDA和NCCL编译工具链。需要额外提醒的是NCCL编译工具链是需要单独安装的，若没有会报错找不到`nccl.h`。具体安装操作根据选取的Linux发行版有所不同，但网上可以搜索到大量的教程，本文档就也略过了。

首先需要说明的一点，直接`pip install apex`下载下来的是[另一个包](https://pypi.org/project/apex/)。实际所需的apex包只有拉取git仓库并编译的一个方法，不过安装过程是简单的，前往它的[Github主页](https://github.com/NVIDIA/apex)根据指引可以轻松完成安装。

```shell
git clone https://github.com/NVIDIA/apex
cd apex

# 多线程编译，并且安装所有扩展。
# APEX_PARALLEL_BUILD=8 
# NVCC_APPEND_FLAGS="--threads 4" 
# 具体扩展的选择方法见Github上的readme.md
NVCC_APPEND_FLAGS="--threads 4" \
    APEX_PARALLEL_BUILD=8 \
    APEX_CPP_EXT=1 APEX_CUDA_EXT=1 APEX_ALL_CONTRIB_EXT=1 \
    pip install -v --no-build-isolation .
```

这里可能会出一个问题，那就是在PyTorch的CUDA版本不等于CUDA工具链的版本时，安装apex的脚本应该会发出一个警告并退出。如果CUDA工具链的版本更高+低于驱动支持的CUDA版本的话，临时的解决方案是打开本地Apex的安装脚本`setup.py`，找到`check_cuda_...()`函数的调用并注释该行。同样的，不保证实际运行时不会出错。



# 安装 flash_attn

这是针对训练脚本中，启用了**Flash Attention**的情况(`--attention-backend`)的额外章节，之所以会有该章节是为了提醒<ins>**目前**</ins>在`python=3.11`&`torch=2.6.0+cu12`的情况下：

```shell
pip install flash_attn
```

自动选取的`flash_attn==2.8.3`预编译包有bug，具体表现为[符号链接出错](https://github.com/Dao-AILab/flash-attention/issues/2010)，可以考虑手动选取`2.7.4.post1`等较旧的版本。

# 如何卸载

```shell
rm -rf .venv
```

# 参考文档
* Megatron-LM的[Github主页](https://github.com/NVIDIA/Megatron-LM)
* Apex的[Github主页](https://github.com/NVIDIA/apex)
* NVIDIA官方的Megatron-LM[用户手册](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/index.html)
* ninja-build的[官网](https://ninja-build.org/)
* 清华镜像源针对配置pip的[帮助](https://mirrors.tuna.tsinghua.edu.cn/help/pypi/)
* uv有关设置index的[文档](https://docs.astral.sh/uv/concepts/indexes/)
* PyTorch的[历史版本安装文档](https://pytorch.org/get-started/previous-versions/)和[API文档](https://docs.pytorch.org/docs/stable/index.html)
* PyTorch引入`torch.complie`的[Blog](https://pytorch.org/blog/pytorch-2.0-release/)
* CUDA Pip Wheels的[文档](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#pip-wheels)
* CUDA的[兼容性文档](https://docs.nvidia.com/deploy/cuda-compatibility/index.html)
* flash_attn在Github上关于[`symbol not defined linker errors`的issue](https://github.com/Dao-AILab/flash-attention/issues/2010)
* 早知道，还得是[官方的Docker容器](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/PyTorch)