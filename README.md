# EIS 阻抗谱自动化测量系统

用于 HP/Agilent 4284A 的阻抗谱测量工具，包含命令行脚本、Windows 桌面界面、仿真器、开路/短路校准与等效电路拟合。

## 功能

- 通过 VISA/GPIB 控制 HP 4284A，扫描频率并采集复阻抗。
- 实时显示 Nyquist、Bode 阻抗幅值与相位曲线。
- 支持单次扫描、自动偏压序列，以及可配置协议的串口继电器板。
- 支持每通道开路/短路校准、CSV 导出与等效电路拟合。
- 提供无硬件仿真模式，用于熟悉操作与验证流程。

## Windows 封装软件

发行包可放在本仓库的 **Releases** 页面。下载 `EISScanGUI-时间戳-win64.zip` 后，解压整个文件夹并运行 `EISScanGUI.exe`。请保留 `_internal` 目录及同级的 `eis_config.yaml`。

真实测量需要安装 VISA 实现及 GPIB 接口驱动；继电器板还需要串口驱动和匹配的协议设置。仿真模式不连接仪器。

详细操作参见 [EXE 使用说明](EXE使用说明.md)。

## 从源码运行

推荐使用 Windows 64 位 Python 3.10 或更新版本。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe eis_gui.py
```

在界面中选择“仿真器”可进行无硬件扫描。实际测量前，检查 `eis_config.yaml` 中的 GPIB 地址、交流幅值、直流偏压、扫频范围及继电器配置。

命令行仿真（执行校准与扫描）：

```powershell
.\.venv\Scripts\python.exe eis_main.py --simulator --run
```

真实仪器测量：

```powershell
.\.venv\Scripts\python.exe eis_main.py --config eis_config.yaml --run
```

仅执行校准：

```powershell
.\.venv\Scripts\python.exe eis_main.py --config eis_config.yaml --calibrate
```

源码默认将测量结果写入配置文件旁的 `data` 目录。运行真实测量命令前，请确认仪器连接和样品允许的电压范围。

## 打包 Windows 程序

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\build_exe.ps1 -PythonExe '.\.venv\Scripts\python.exe'
```

脚本自动安装 `requirements-exe.txt` 中的打包依赖，并在 `dist` 下生成完整程序目录及 ZIP。打包目录与 ZIP 不纳入 Git 历史，可将需要发布的版本上传为 Release 附件。

## 项目结构

```text
eis_gui.py             Windows 桌面界面
eis_main.py            命令行入口与测量调度
eis_config.yaml        仪器、扫描、校准及仿真配置
drivers/              HP 4284A、继电器板与仿真器驱动
measurements/          频率扫描与多通道测量
cal/                   开路/短路校准
analysis/              等效电路拟合与报告工具
utils/                 数据保存与绘图
assets/                程序图标
build_exe.ps1          Windows 打包脚本
```

其他扫描和分析脚本面向特定实验流程，使用前需检查其中的输入文件和输出目录。`analysis/bias_report.py` 的 PDF 报告功能还需要 XeLaTeX，并按本机安装位置调整该文件中的 `XELATEX`。

实验数据、校准结果、实验报告、虚拟环境和历史打包文件默认由 `.gitignore` 排除，原文件仍保留在本地。
