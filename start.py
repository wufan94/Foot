import sys
import logging
from PyQt5.QtWidgets import QApplication
from gui import MainWindow

def setup_logging():
    """设置日志系统"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('virtual_fitting.log', encoding='utf-8')
        ]
    )

def main():                              
    """程序入口"""
    try:
        # 设置日志
        setup_logging()
        logging.info("程序启动...")
        
        # 创建应用程序
        app = QApplication(sys.argv)
        
        # 设置应用程序信息                       
        app.setApplicationName("虚拟试穿分析系统")
        app.setApplicationVersion("1.0.0")
        
        # 创建并显示主窗口
        window = MainWindow()
        window.show()
        
        # 确保窗口前置显示
        window.raise_()
        window.activateWindow()
        
        logging.info("主窗口已显示")
        
        # 运行应用程序
        return app.exec_()
        
    except Exception as e:
        logging.error(f"程序启动失败: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # 显示错误对话框
        if QApplication.instance():
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.critical(None, "错误", f"程序启动失败:\n{str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())  