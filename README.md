# LLM-MatGen

材料结构生成工具，支持 10 类生成器、轻量结构检查和来源记录，提供 CLI、Python 与 MCP 接口。

## MP 优先获取已有结构

已有材料先通过 Materials Project 搜索，确定材料编号和晶型后下载母结构，再生成并查看球棍模型。MCP search、download、generate 已接通；MP_API_KEY 通过环境变量提供。多个晶型匹配时先确定目标，不默认选第一条。

## 生成后查看结构

十类生成流程默认输出离线 viewer.html 球棍模型，支持候选切换、旋转缩放、晶胞、原子编号、点选和 XY/XZ/YZ 视角。生成命令加 --open 可直接打开浏览器；MCP 返回查看页面路径。

## 下载与安装

[下载 wheel 安装包](https://github.com/wyz-eceshi/LLM-MatGen/releases/download/v0.2.1/llm_matgen-0.2.1-py3-none-any.whl)

[下载完整源码包](https://github.com/wyz-eceshi/LLM-MatGen/releases/download/v0.2.1/llm_matgen-0.2.1.tar.gz)

需要 Python 3.10 或以上。在安装包目录执行：

```powershell
python -m pip install ./llm_matgen-0.2.1-py3-none-any.whl
llm-matgen --help
```

安装时联网下载依赖。项目名称已恢复为 LLM-MatGen，保留 dft-structure-building 命令作为兼容入口。完整源码与文档位于发布附件 tar.gz 中。
