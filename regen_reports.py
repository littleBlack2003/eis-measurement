"""为所有已有偏压数据重新生成报告 (不重新测量)."""
import os
import sys
import glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analysis.bias_report import generate_bias_report

files = sorted(glob.glob('data/bias_*.csv'))
print(f"发现 {len(files)} 个偏压数据文件")
ok, fail = 0, 0
for f in files:
    try:
        pdf = generate_bias_report(f)
        ok += 1
        print(f"  OK  {os.path.basename(f)} -> {os.path.basename(pdf)}")
    except Exception as e:
        fail += 1
        print(f"  FAIL {os.path.basename(f)}: {e}")

print(f"\n完成: {ok} 成功, {fail} 失败")
