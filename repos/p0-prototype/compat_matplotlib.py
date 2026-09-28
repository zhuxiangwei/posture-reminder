"""Windows 智能应用控制（SAC）拦截 matplotlib 时的兼容桩。

================================================================ 背景

本机 Windows 的「智能应用控制」（Smart App Control）处于**强制**模式：

    HKLM\\SYSTEM\\CurrentControlSet\\Control\\CI\\Policy
        VerifiedAndReputablePolicyState = 1     # 1 = Enforced

它按**文件内容/签名**放行程序（实测：把被拦文件改名、换目录都没用），
会拦下未签名的第三方扩展 DLL。实测只拦到 matplotlib 的：

    site-packages/matplotlib/_image.cp313-win_amd64.pyd
    -> ImportError: DLL load failed while importing _image:
       应用程序控制策略已阻止此文件。

该 pyd 本身完好（MZ 头正确、体积/md5 正常），**不是病毒也不是损坏**，
纯粹是执行策略拦截。同目录的 _path / ft2font / _c_internal_utils，
以及 numpy / cv2 / PySide6 全部正常 —— 仅这一个文件被拦。

================================================================ 为什么必须处理

mediapipe 的视觉得分链是**硬导入**：

    mediapipe/tasks/python/__init__.py
        -> from . import vision
    mediapipe/tasks/python/vision/__init__.py:19
        -> import ...vision.drawing_utils      # 无条件
    .../vision/drawing_utils.py:21
        -> import matplotlib.pyplot as plt      # 无条件，就是这里炸

于是 `mediapipe.tasks.python.vision` 整包导入失败，
`posture_monitor.py` 里 `vision` 退化成 None，最终报出
`AttributeError: 'NoneType' object has no attribute 'PoseLandmarker'`
—— 一个和真实病因毫无关系的报错。

================================================================ 为什么可以安全打桩

`matplotlib` 在 mediapipe 里**只被 used 于一个 3D 调试可视化函数**
（`drawing_utils.plot_landmarks`，用到 plt.figure / plt.axes / plt.show）。
**推理管线完全不碰 matplotlib**，本项目代码也 0 处引用它（全仓 grep 确认）。

实测：打好桩后 lite / full / heavy 三档模型 × VIDEO 模式全部跑通
（create_from_options + detect_for_video 均正常）。

================================================================ 用法

在**导入 mediapipe 之前**调用 install()：

    import compat_matplotlib
    compat_matplotlib.install()

    import mediapipe            # 此后才能成功

install() 是幂等的，重复调用无副作用。真实 matplotlib 能正常导入时
（即未被 SAC 拦），本模块**什么都不做**，绝不干扰原环境。
"""

import sys
import types


def _is_importable(name):
    """目标模块当前能否正常导入。

    注意：成功的导入**会**把模块留在 `sys.modules` 里（就是普通 import 的语义）。
    这里只关心"能不能导入"，不刻意清理 —— 真实可用的库本就该留在缓存里；
    之后 install() 若要打桩，会直接覆盖 `sys.modules` 的对应条目。
    """
    import importlib
    try:
        importlib.import_module(name)
        return True
    except Exception:
        # 任何异常都算「不可用」：SAC 拦截抛的是 ImportError(DLL load failed)，
        # 但残缺安装可能抛别的，一并视为不可用，交给桩兜底。
        return False


def _make_pyplot_stub():
    """造一个 matplotlib.pyplot 替身。

    只提供 drawing_utils 顶层 import 时立刻需要的导入成功，
    以及 plot_landmarks 里用到的三个名字。**任何实际绘图调用都会抛错**，
    以免「看起来能画但画不出来」这种更难查的故障。
    """

    def _unavailable(feature):
        def _raise(*_args, **_kwargs):
            raise NotImplementedError(
                f"matplotlib 已被 compat_matplotlib 打桩（SAC 拦截 _image.pyd），"
                f"{feature}() 不可用。\n"
                f"  仅 mediapipe 的 3D 调试可视化 plot_landmarks() 会用到它；\n"
                f"  坐姿判定的推理管线不需要 matplotlib。\n"
                f"  若确实需要 3D 可视化：关闭 Windows 智能应用控制后重启程序。"
            )
        return _raise

    pyplot = types.ModuleType("matplotlib.pyplot")
    pyplot.__doc__ = "compat_matplotlib 桩（真实 matplotlib 不可用时）。"
    for _name in ("figure", "axes", "show", "plot", "scatter",
                  "subplot", "subplots", "imshow", "savefig", "close"):
        setattr(pyplot, _name, _unavailable(_name))
    return pyplot


def install(verbose=False):
    """在 mediapipe 之前装好桩。返回 True 表示真的打了桩。

    - 真实 matplotlib 可用   -> 返回 False，不做任何改动
    - 真实 matplotlib 不可用 -> 用桩顶掉，返回 True
    """
    if _is_importable("matplotlib.pyplot"):
        if verbose:
            print("[compat_matplotlib] 真实 matplotlib 可用，不介入。")
        return False

    stub = types.ModuleType("matplotlib")
    stub.__doc__ = "compat_matplotlib 桩（真实 matplotlib 不可用时）。"
    stub.__version__ = "0.0.0+compat-stub"
    stub.__file__ = __file__
    # 真实包是目录包；不设 __path__ 会让 `import matplotlib.xxx` 直接失败，
    # 这里给个空列表，配合下面显式注册的子模块即可满足 mediapipe 的用法。
    stub.__path__ = []

    pyplot = _make_pyplot_stub()
    stub.pyplot = pyplot

    sys.modules["matplotlib"] = stub
    sys.modules["matplotlib.pyplot"] = pyplot

    if verbose:
        print("[compat_matplotlib] 已用桩顶掉 matplotlib（SAC 拦截 _image.pyd）。")
    return True


def stub_active():
    """当前 sys.modules 里躺的是不是我们的桩（供自检/测试用）。"""
    m = sys.modules.get("matplotlib")
    return bool(m is not None and getattr(m, "__version__", "") == "0.0.0+compat-stub")


# 顺手兜住 `import compat_matplotlib` 之后直接 `from matplotlib import pyplot`
# 这种写法：不装桩，保持惰性，由调用方显式 install()。
__all__ = ["install", "stub_active"]
