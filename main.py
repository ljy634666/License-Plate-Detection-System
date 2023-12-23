import sys
import PyQt5.QtWidgets


# 从自定义的gui模块中导入LicensePlateDetector类
from gui import LicensePlateDetector

if __name__ == "__main__":
    # 创建QApplication类的实例。sys.argv是一个从命令行解析的参数列表。
    app = PyQt5.QtWidgets.QApplication(sys.argv)

    # 创建LicensePlateDetector类的实例，这会触发界面的构建
    ex = LicensePlateDetector()

    sys.exit(app.exec_())


