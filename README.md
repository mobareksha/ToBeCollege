# ToBeCollege：开局托儿所求和 10 辅助工具

这是一个用于微信小游戏“开局托儿所”的 Windows 本地辅助脚本：它截取 16 × 10 数字棋盘，识别数字，搜索总和为 10 的矩形区域，并可用鼠标拖框执行已计算的路线。

## 来源与改动

本项目基于 [xiaosongshuyy1/ToBeCollege](https://github.com/xiaosongshuyy1/ToBeCollege) 的原始代码继续开发。原作者完成了截图、数字模板识别、矩形消除与鼠标拖框的基础功能。

本版本在保留这些基础功能的前提下，加入了：

- 可保存并复用棋盘框选位置；
- 基于多个候选路线的搜索与残局精确求解；
- `--min-score` 阈值模式：预测清除数量未达到目标时不移动鼠标；
- 识别保护：置信度过低或同一数字异常集中时，不把错误识别产生的虚高结果当作可执行方案；
- 保存最近一次识别结果，便于排查。

分数会受每局随机数字分布影响。程序输出的是基于识别棋盘的预测清除格数，不保证能达到某个固定分数。

## 环境

- Windows
- Python 3.10 或更高版本
- 微信小游戏窗口保持在屏幕可见位置

安装依赖：

```powershell
python -m pip install -r requirements.txt
```

## 使用

在项目目录运行：

```powershell
python -X utf8 .\sum10_grid_bot.py --min-score 150
```

第一次运行时框选完整棋盘；之后会自动复用位置。程序会先显示识别矩阵与检查结果，再计算路线。预测清除数量低于 150 时会直接退出，不会控制鼠标；达到目标时仍需输入 `y` 才会执行。

如果微信窗口位置、缩放或分辨率改变，重新框选：

```powershell
python -X utf8 .\sum10_grid_bot.py --min-score 150 --select-region
```

仅检查识别结果，不进行鼠标操作：

```powershell
python -X utf8 .\sum10_grid_bot.py --preview
```

将鼠标移到屏幕左上角可触发 PyAutoGUI 的紧急停止。

## 说明

该项目仅供个人学习和娱乐使用。请遵循原作者“请勿商用”的说明，并遵守游戏平台规则。
