# Windows GUI 版使用说明

## 打包

在 Windows PowerShell 中进入本目录后运行：

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\build_exe.ps1
```

脚本使用 `requirements-exe.txt` 安装打包依赖，首次打包需要联网。GUI 版发行包会生成在 `dist\EISScanGUI-时间戳\`，压缩包为 `dist\EISScanGUI-时间戳-win64.zip`。请解压或复制整个文件夹，不要只拷贝 exe。

如 Python 不在默认位置，可指定解释器：

```powershell
.\build_exe.ps1 -PythonExe 'C:\Path\To\python.exe'
```

## 测量

运行 `EISScanGUI.exe`，选择 HP 4284A 或仿真器，填写扫描参数，然后点“开始完整扫描”。测量时 Nyquist、Bode 阻抗幅值和相位三个标签页会随频点实时更新。校准选项开启时，界面会在每个通道的开路和短路测量前弹窗提示调整探针。

配置文件 `eis_config.yaml` 与 exe 放在一起。界面参数可用“保存配置”写回该文件；扫描结果默认保存到 exe 目录旁的 `data` 文件夹。

使用仿真器时不会连接硬件。真实 HP 4284A 测量仍需要目标电脑安装可用的 VISA 实现和对应 GPIB 接口驱动。启用继电器板时还需配置串口及其协议命令。

## 偏压序列扫描

在“偏压序列扫描”中输入 `0:1:0.1`（从 0 V 到 1 V，步长 0.1 V）或 `-0.5,0,0.5`（指定列表），再点击“扫描偏压序列”。下降序列可输入 `1:-1:0.2`。范围写法总会包含终点；偏压值须在 -40 V 到 +40 V 之间。

程序对每个偏压执行一次完整频率扫描，在输出目录下新建 `bias_sequence_时间戳` 文件夹。每个偏压的数据和图像位于 `bias_<电压>V` 子文件夹；`bias_sequence.csv` 记录已完成的偏压。界面的三张实时曲线会叠加不同偏压的数据。开路/短路校准在序列开始前执行一次。扫描期间可点击“停止扫描”，程序将在当前频点结束后关闭直流偏压和设备连接。
