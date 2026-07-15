# Stock Agent

一个基于大模型和机器学习的智能股票分析助手。

## 在另一台电脑安装

如果你想在另一台电脑上安装此工具，请按照以下步骤操作：

### 1. 克隆仓库
首先，将代码克隆到本地：
```bash
git clone https://github.com/YoungW9527/stock_agent_by_Young.git
cd stock-agent
```

### 2. 安装
在项目根目录下运行安装命令：
```bash
pip install .
```
*提示：建议在虚拟环境中安装。*

### 3. 配置
由于 `config.json` 包含私密信息，它不会被上传到 GitHub。你需要手动创建它：
1.  将 `config_example.json` 复制并重命名为 `config.json`。
2.  打开 `config.json`，填入你的 `llm_api_key` 以及其他必要的配置（如 Telegram Token）。

### 4. 使用
安装完成后，你可以直接在命令行运行：

*   **立即执行一次分析**：
    ```bash
    stock-agent --now
    ```
*   **作为后台进程运行**：
    ```bash
    stock-agent
    ```

## 配置说明
请确保在运行目录下存在 `config.json` 文件。
