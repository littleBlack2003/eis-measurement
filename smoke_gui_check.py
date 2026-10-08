"""临时冒烟测试：构建 GUI、渲染若干帧、触发主要回调，验证无异常。"""
import sys
import time

import dearpygui.dearpygui as dpg

import eis_gui

app = eis_gui.EISGui()          # 内部会 create_context + show_viewport
try:
    for _ in range(3):
        dpg.render_dearpygui_frame()
    print('[ok] UI 构建并渲染 3 帧')

    required = ['main_window', 'progress', 'fit_table', 'log_text', 'start_button',
                'bias_start_button', 'stop_button', 'status', 'fit_summary',
                'settle_time', 'output_dir', 'connect_button', 'list_resources_button',
                'conn_status']
    missing = [t for t in required if not dpg.does_item_exist(t)]
    print('[check] 缺失控件:', missing or '无')
    assert not missing

    # 1) 参数校验：故意填入非法值，应给出中文提示且不抛异常
    dpg.set_value('timeout', 'abc')
    try:
        app._settings()
        print('[FAIL] 非法 timeout 未报错')
    except ValueError as exc:
        print('[ok] 非法 timeout 提示:', exc)

    dpg.set_value('timeout', '30')
    dpg.set_value('freq_start', '1e6')
    dpg.set_value('freq_stop', '20')
    try:
        app._settings()
        print('[FAIL] 起止频率倒置未报错')
    except ValueError as exc:
        print('[ok] 频率校验:', exc)
    dpg.set_value('freq_start', '20')
    dpg.set_value('freq_stop', '1e6')

    # 1b) 仪器连接测试（仿真器模式，无需真实硬件）
    dpg.set_value('mode', '仿真器')
    app._test_connection()
    for _ in range(60):
        dpg.render_dearpygui_frame()
        app._poll_events()
        if app.conn_worker is None or not app.conn_worker.is_alive():
            break
        time.sleep(0.05)
    app._poll_events()
    state = dpg.get_value('conn_status')
    print('[ok] 连接测试状态:', state)
    assert '已连接' in state, state
    # 真实仪器模式：没有硬件时应走失败分支并给出可读提示，而不是崩掉
    dpg.set_value('mode', 'HP 4284A')
    app._test_connection()
    for _ in range(60):
        dpg.render_dearpygui_frame()
        app._poll_events()
        if app.conn_worker is None or not app.conn_worker.is_alive():
            break
        time.sleep(0.05)
    app._poll_events()
    print('[ok] 无硬件时连接测试状态:', dpg.get_value('conn_status'))
    # 无论成功失败，按钮都要恢复可用，否则一次失败就再也点不动
    assert dpg.get_item_configuration('connect_button')['enabled'], '测试连接按钮未恢复可用'

    # 2) 实时曲线与进度条
    app._reset_live_curves()
    for i in range(1, 11):
        app._record_point(None, 0, 10 ** (1 + i / 5), 100.0 + i, -20.0 - i,
                          120.0, -15.0, i, 10)
        dpg.render_dearpygui_frame()
    print('[ok] 实时曲线 10 点，进度条 overlay =', dpg.get_item_configuration('progress')['overlay'])
    # 多通道/多偏压应各占一种配色
    app._record_point(0.5, 1, 100.0, 50.0, -8.0, 51.0, -9.0, 1, 10)
    dpg.render_dearpygui_frame()
    print('[ok] 曲线组数:', len(app.live_series))
    app._reset_live_curves()
    dpg.render_dearpygui_frame()
    print('[ok] 重置后曲线组数:', len(app.live_series))

    # 3) 真实数据拟合 + 表格刷新 + 导出
    from analysis.auto_fit import find_latest_csv, fit_csv
    csvs = find_latest_csv('./data')
    print('[info] 找到 CSV:', len(csvs))
    if csvs:
        results = [fit_csv(p) for p in csvs[:2]]
        app._show_fit_results(results)
        for _ in range(3):
            dpg.render_dearpygui_frame()
        print('[ok] 拟合曲线条数:', len(app.fit_series))
        rows = dpg.get_item_children('fit_table', slot=1)
        print('[ok] 拟合表格行数:', len(rows))
        print('[ok] 摘要首行:', dpg.get_value('fit_summary').splitlines()[0])
        app._export_fit_report()
        dpg.render_dearpygui_frame()
        print('[ok] 导出提示:', dpg.get_value('fit_export_hint'))

    # 4) 清空日志 / 打开目录按钮回调存在
    app._clear_log()
    print('[ok] 清空日志，当前日志长度:', len(dpg.get_value('log_text')))

    # 5) 校准弹窗取消路径（close_prompt 事件）
    app.events.put(('close_prompt',))
    app._poll_events()
    dpg.render_dearpygui_frame()
    print('[ok] close_prompt 处理完成')
finally:
    dpg.destroy_context()
print('[done] 冒烟测试结束，无未捕获异常')
