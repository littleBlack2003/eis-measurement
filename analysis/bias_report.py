"""单个偏压点的 EIS 报告生成: 拟合 + 绘图 + LaTeX PDF."""
import os
import subprocess
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

REPORT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          '试验方案', '偏压报告')


def _format_sci(x: float, nd: int = 3) -> str:
    return f"{x:.{nd}e}"


def generate_bias_report(csv_path: str, report_dir: str = None) -> str:
    """
    对一个偏压点的 EIS 数据生成报告 (拟合 + 绘图 + PDF).

    Returns
    -------
    str: 生成的 PDF 文件路径
    """
    from analysis.fitting import fit_equivalent_circuit

    if report_dir is None:
        report_dir = REPORT_DIR
    os.makedirs(report_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    freq = df['freq [Hz]'].values
    z = df['Z_real [ohm]'].values + 1j * df['Z_imag [ohm]'].values

    # 提取偏压标签: bias_n1p0V.csv -> -1.0
    base = os.path.basename(csv_path).replace('bias_', '').replace('.csv', '')
    vdc = _parse_bias_label(base)

    # 等效电路拟合
    fit = None
    try:
        fit = fit_equivalent_circuit(freq, z, use_cpe=False, freq_min=0)
    except Exception as e:
        logger.warning(f"Bias {vdc} V fit failed: {e}")

    # 绘图
    plot_path = os.path.join(report_dir, f'bias_{base}_plot.png')
    _make_plot(freq, z, fit, plot_path, vdc)

    # 生成 LaTeX
    tex_path = os.path.join(report_dir, f'Bias_{base}_报告.tex')
    _write_latex(tex_path, plot_path, vdc, freq, z, fit)

    # 编译
    pdf_path = tex_path.replace('.tex', '.pdf')
    _compile(tex_path)
    logger.info(f"Bias {vdc} V report: {pdf_path}")
    return pdf_path


def _parse_bias_label(label: str) -> float:
    """bias_n1p0V -> -1.0; bias_p0p5V -> +0.5; bias_0p0V -> 0.0"""
    label = label.rstrip('V')  # 去掉末尾 V
    sign = -1.0 if label.startswith('n') else 1.0
    num = label.lstrip('np')
    num = num.replace('p', '.')
    return sign * float(num)


def _make_plot(freq, z, fit, path, vdc):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    # Nyquist
    ax = axes[0]
    mask = (z.real > 0) & (-z.imag > 0)
    ax.plot(z.real[mask], -z.imag[mask], 'o', ms=4, label='data')
    if fit is not None:
        ax.plot(fit['z_fit'].real, -fit['z_fit'].imag, '-', lw=2,
                label=f"fit {fit['model']}")
    ax.set_xlabel("Z' [Ω]")
    ax.set_ylabel("-Z'' [Ω]")
    ax.set_title(f"Nyquist, Vdc={vdc:+.1f} V")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Bode |Z| (横轴取对数, 纵轴线性)
    ax = axes[1]
    ax.semilogx(freq, np.abs(z), 'o', ms=3, label='data')
    if fit is not None:
        ax.semilogx(fit['freq_fit'], np.abs(fit['z_fit']), '-', lw=2, label='fit')
    ax.set_xlabel("f [Hz]")
    ax.set_ylabel("|Z| [Ω]")
    ax.set_title(f"Bode |Z|, Vdc={vdc:+.1f} V")
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _write_latex(tex_path, plot_path, vdc, freq, z, fit):
    plot_rel = os.path.basename(plot_path)
    if fit is not None:
        rs, rp, c = fit['Rs'], fit['Rp'], fit['C']
        err = fit['norm_error']
        tau = fit.get('tau', np.nan)
        f_apex = fit.get('f_apex', np.nan)
    else:
        rs = rp = c = tau = f_apex = err = np.nan

    # 高频段统计
    hi = freq >= 1e4
    c_hi = -1.0 / (2 * np.pi * freq[hi] * z[hi].imag)
    c_hi = c_hi[c_hi > 0]
    c_med = np.median(c_hi) if len(c_hi) else np.nan
    z_med = np.median(np.abs(z))

    latex = f"""% !TEX program = xelatex
\\documentclass[12pt,a4paper]{{ctexart}}
\\usepackage{{amsmath,amssymb}}
\\usepackage{{graphicx}}
\\usepackage{{booktabs}}
\\usepackage{{geometry}}
\\geometry{{left=2.5cm,right=2.5cm,top=2.5cm,bottom=2.5cm}}
\\title{{\\textbf{{EIS 偏压点 V$_{{dc}}$ = {vdc:+.1f} V 测试报告}}}}
\\author{{}}
\\date{{{pd.Timestamp.now():%Y年%m月%d日 %H:%M}}}
\\begin{{document}}
\\maketitle

\\section{{测量条件}}
\\begin{{tabular}}{{ll}}
\\toprule
偏压 $V_{{dc}}$ & {vdc:+.1f} V \\\\
交流幅值 & 10 mVrms \\\\
频率范围 & {freq[0]:.0f} Hz -- {freq[-1]:.0f} Hz \\\\
频点数 & {len(freq)} \\\\
平均次数 & 3 \\\\
\\bottomrule
\\end{{tabular}}

\\section{{阻抗谱}}
\\includegraphics[width=\\textwidth]{{{plot_rel}}}

\\section{{等效电路拟合 ($R_s(R_pC)$ 模型)}}
\\begin{{tabular}}{{lcc}}
\\toprule
参数 & 数值 & 单位 \\\\
\\midrule
$R_s$ & {_fmt(rs)} & $\\Omega$ \\\\
$R_p$ & {_fmt(rp)} & $\\Omega$ \\\\
$C$ & {_fmt(c)} & F \\\\
$\\tau = R_p C$ & {_fmt(tau)} & s \\\\
弧顶频率 & {_fmt(f_apex)} & Hz \\\\
拟合误差 & {_fmt(err*100)} & \\% \\\\
\\bottomrule
\\end{{tabular}}

\\section{{关键指标}}
\\begin{{tabular}}{{lcc}}
\\toprule
指标 & 数值 & 单位 \\\\
\\midrule
$|Z|$ 中位数 & {_fmt(z_med)} & $\\Omega$ \\\\
高频结电容 (10k--1MHz 中位) & {_fmt(c_med)} & F \\\\
\\bottomrule
\\end{{tabular}}

\\end{{document}}
"""
    with open(tex_path, 'w', encoding='utf-8') as f:
        f.write(latex)


def _fmt(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    return f"{x:.4g}"


XELATEX = r'C:\texlive\2026\bin\windows\xelatex.exe'


def _compile(tex_path):
    """用 xelatex 编译 LaTeX (使用绝对路径, 避免 PATH 解析问题)."""
    try:
        subprocess.run(
            [XELATEX, '-interaction=nonstopmode', os.path.basename(tex_path)],
            cwd=os.path.dirname(tex_path),
            capture_output=True, timeout=120,
        )
    except Exception as e:
        logger.warning(f"LaTeX compile failed: {e}")
