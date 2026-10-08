"""生成应用图标 assets/app_icon.ico：抽象 Nyquist 散点图。

造型取阻抗谱最典型的呈现方式：偏白的坐标纸底 + 淡网格 + xy 轴，
数据点用红色 "*" 号沿半圆弧排布（高频在左、低频在右），不画连线。

改配色或造型后重跑本脚本即可：  python make_icon.py
"""
import math
from pathlib import Path

from PIL import Image, ImageDraw

# 底板：偏白的"坐标纸"，配一圈淡边框，保证在浅色/深色任务栏上都能看清
CANVAS_BG = (249, 250, 253)
CANVAS_BORDER = (196, 205, 218)
GRID = (223, 230, 240)          # 淡网格
AXIS = (86, 96, 112)            # 坐标轴
MARK = (214, 63, 48)            # 数据点：红

BASE = 512                      # 绘制尺寸（再缩小到各档，天然抗锯齿）
SIZES = (16, 20, 24, 32, 40, 48, 64, 96, 128, 256)


def rounded_background(size):
    """圆角方底 + 淡边框。"""
    img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    inset = size * 0.012
    radius = int(size * 0.20)
    box = [inset, inset, size - inset, size - inset]
    draw.rounded_rectangle(box, radius=radius, fill=CANVAS_BG + (255,))
    draw.rounded_rectangle(box, radius=radius, outline=CANVAS_BORDER + (255,),
                           width=max(2, round(size * 0.022)))
    return img


def draw_asterisk(draw, cx, cy, radius, width, color):
    """画一个 "*" 号：三条过中心的线 = 六个角。"""
    for deg in (90, 30, 150):
        rad = math.radians(deg)
        dx = radius * math.cos(rad)
        dy = -radius * math.sin(rad)          # 屏幕 y 向下，取负才是"向上"
        draw.line([(cx - dx, cy - dy), (cx + dx, cy + dy)],
                  fill=color, width=width, joint='curve')


def arc_length_angles(rx, ry, count, steps=720):
    """把半椭圆弧（180°→360°，经过 270° 即在轴上方）按弧长等分成 count 段。

    直接等角度取点的话，弧被拉长后顶端会挤成一堆、两端却很稀，看起来不像
    一组弧上均匀采集的频点。
    """
    def point(deg):
        rad = math.radians(deg)
        return rx * math.cos(rad), ry * math.sin(rad)

    # 累积弧长表
    cumulative = [0.0]
    prev = point(180)
    for k in range(1, steps + 1):
        cur = point(180 + 180 * k / steps)
        cumulative.append(cumulative[-1] + math.dist(prev, cur))
        prev = cur

    total = cumulative[-1]
    angles = []
    for i in range(count):
        target = total * i / (count - 1)
        # 在累积表中定位 target 落在哪一小段，再做线性插值
        lo = 0
        while lo < steps and cumulative[lo + 1] < target:
            lo += 1
        span = cumulative[lo + 1] - cumulative[lo]
        frac = (target - cumulative[lo]) / span if span else 0.0
        angles.append(180 + 180 * (lo + frac) / steps)
    return angles


def plot_overlay(size):
    """在透明层上画坐标系、网格与数据点。"""
    layer = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)

    origin_x, origin_y = size * 0.20, size * 0.78
    axis_right, axis_top = size * 0.94, size * 0.12
    thin = max(1, round(size * 0.010))
    axis_w = max(2, round(size * 0.026))

    # 淡网格：4 条竖线 + 3 条横线，只画在坐标轴围出的区域内
    for i in range(1, 4):
        x = origin_x + (axis_right - origin_x) * i / 4
        draw.line([(x, origin_y), (x, axis_top)], fill=GRID + (255,), width=thin)
    for i in range(1, 4):
        y = origin_y - (origin_y - axis_top) * i / 4
        draw.line([(origin_x, y), (axis_right, y)], fill=GRID + (255,), width=thin)

    # xy 轴
    draw.line([(origin_x, origin_y), (axis_right, origin_y)],
              fill=AXIS + (255,), width=axis_w)
    draw.line([(origin_x, origin_y), (origin_x, axis_top)],
              fill=AXIS + (255,), width=axis_w)
    # 刻度
    tick = size * 0.035
    for i in range(1, 4):
        x = origin_x + (axis_right - origin_x) * i / 4
        draw.line([(x, origin_y), (x, origin_y + tick)], fill=AXIS + (255,), width=axis_w)
        y = origin_y - (origin_y - axis_top) * i / 4
        draw.line([(origin_x, y), (origin_x - tick, y)], fill=AXIS + (255,), width=axis_w)

    # 半圆弧上的散点：左端=高频 Rs，右端=低频 Rs+Rp
    # 按弧长均匀取点（等角度在拉长的弧上会两头稀、顶上挤），星号也别太大，
    # 否则缩到 16–32px 会糊成一片红
    # 拱形要"顶天立地"才一眼看出是 Nyquist 弧：横向铺满坐标区、纵向也拉高
    center_x = size * 0.50
    rx, ry = size * 0.30, size * 0.60
    count = 11
    # 星号笔画别太细：缩到 16–24px 时细线会断成点，认不出是 "*"
    radius = size * 0.043
    width = max(3, round(size * 0.024))
    for deg in arc_length_angles(rx, ry, count):
        rad = math.radians(deg)
        x = center_x + rx * math.cos(rad)
        y = origin_y + ry * math.sin(rad)
        draw_asterisk(draw, x, y, radius, width, MARK + (255,))
    return layer


def main():
    icon = rounded_background(BASE)
    icon = Image.alpha_composite(icon, plot_overlay(BASE))

    out_dir = Path(__file__).resolve().parent / 'assets'
    out_dir.mkdir(exist_ok=True)

    # exe 图标：多尺寸 ICO（资源管理器、任务栏、Alt+Tab 各取所需）
    ico_path = out_dir / 'app_icon.ico'
    icon.save(ico_path, format='ICO', sizes=[(s, s) for s in SIZES])

    # 运行期备用 PNG（DearPyGui 的 load_image 不支持 ICO）
    png_path = out_dir / 'app_icon_ui.png'
    icon.resize((64, 64), Image.LANCZOS).save(png_path)

    # 以及一张 256×256 预览，方便直接看效果
    icon.resize((256, 256), Image.LANCZOS).save(out_dir / 'app_icon_preview.png')

    print(f'icon:    {ico_path} ({ico_path.stat().st_size} B)')
    print(f'ui png:  {png_path} ({png_path.stat().st_size} B)')
    print(f'preview: {out_dir / "app_icon_preview.png"}')


if __name__ == '__main__':
    main()
