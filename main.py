import sys
from PyQt5.QtWidgets import QApplication
from gui import CustodianApp

def main():
    app = QApplication(sys.argv)
    ex = CustodianApp()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()