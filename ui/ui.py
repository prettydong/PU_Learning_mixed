# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'v0.01.ui'
##
## Created by: Qt User Interface Compiler version 6.0.4
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(522, 349)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.groupBox = QGroupBox(self.centralwidget)
        self.groupBox.setObjectName(u"groupBox")
        self.groupBox.setGeometry(QRect(10, 10, 171, 311))
        self.tableView = QTableView(self.groupBox)
        self.tableView.setObjectName(u"tableView")
        self.tableView.setGeometry(QRect(20, 20, 131, 251))
        self.tableView.setFrameShadow(QFrame.Plain)
        self.tableView.setLineWidth(2)
        self.tableView.setMidLineWidth(2)
        self.pushButton = QPushButton(self.groupBox)
        self.pushButton.setObjectName(u"pushButton")
        self.pushButton.setGeometry(QRect(20, 280, 21, 24))
        self.pushButton_2 = QPushButton(self.groupBox)
        self.pushButton_2.setObjectName(u"pushButton_2")
        self.pushButton_2.setGeometry(QRect(50, 280, 21, 24))
        self.pushButton_3 = QPushButton(self.groupBox)
        self.pushButton_3.setObjectName(u"pushButton_3")
        self.pushButton_3.setGeometry(QRect(100, 280, 51, 24))
        self.groupBox_2 = QGroupBox(self.centralwidget)
        self.groupBox_2.setObjectName(u"groupBox_2")
        self.groupBox_2.setGeometry(QRect(370, 10, 131, 161))
        self.checkBox = QCheckBox(self.groupBox_2)
        self.checkBox.setObjectName(u"checkBox")
        self.checkBox.setGeometry(QRect(20, 30, 77, 20))
        self.checkBox.setChecked(True)
        self.checkBox_2 = QCheckBox(self.groupBox_2)
        self.checkBox_2.setObjectName(u"checkBox_2")
        self.checkBox_2.setGeometry(QRect(20, 60, 77, 20))
        self.checkBox_2.setChecked(True)
        self.checkBox_3 = QCheckBox(self.groupBox_2)
        self.checkBox_3.setObjectName(u"checkBox_3")
        self.checkBox_3.setGeometry(QRect(20, 90, 77, 20))
        self.checkBox_3.setChecked(True)
        self.checkBox_4 = QCheckBox(self.groupBox_2)
        self.checkBox_4.setObjectName(u"checkBox_4")
        self.checkBox_4.setGeometry(QRect(20, 120, 77, 20))
        self.checkBox_4.setChecked(True)
        self.groupBox_3 = QGroupBox(self.centralwidget)
        self.groupBox_3.setObjectName(u"groupBox_3")
        self.groupBox_3.setGeometry(QRect(370, 180, 131, 141))
        self.pushButton_4 = QPushButton(self.groupBox_3)
        self.pushButton_4.setObjectName(u"pushButton_4")
        self.pushButton_4.setGeometry(QRect(10, 20, 51, 24))
        self.pushButton_5 = QPushButton(self.groupBox_3)
        self.pushButton_5.setObjectName(u"pushButton_5")
        self.pushButton_5.setGeometry(QRect(70, 20, 51, 24))
        self.pushButton_6 = QPushButton(self.groupBox_3)
        self.pushButton_6.setObjectName(u"pushButton_6")
        self.pushButton_6.setGeometry(QRect(10, 60, 51, 24))
        self.pushButton_7 = QPushButton(self.groupBox_3)
        self.pushButton_7.setObjectName(u"pushButton_7")
        self.pushButton_7.setGeometry(QRect(70, 60, 51, 24))
        self.pushButton_14 = QPushButton(self.groupBox_3)
        self.pushButton_14.setObjectName(u"pushButton_14")
        self.pushButton_14.setGeometry(QRect(10, 100, 51, 24))
        self.pushButton_15 = QPushButton(self.groupBox_3)
        self.pushButton_15.setObjectName(u"pushButton_15")
        self.pushButton_15.setGeometry(QRect(70, 100, 51, 24))
        self.groupBox_4 = QGroupBox(self.centralwidget)
        self.groupBox_4.setObjectName(u"groupBox_4")
        self.groupBox_4.setGeometry(QRect(190, 10, 171, 311))
        self.tableView_3 = QTableView(self.groupBox_4)
        self.tableView_3.setObjectName(u"tableView_3")
        self.tableView_3.setGeometry(QRect(20, 20, 131, 251))
        self.tableView_3.setFrameShadow(QFrame.Plain)
        self.tableView_3.setLineWidth(2)
        self.tableView_3.setMidLineWidth(2)
        self.pushButton_11 = QPushButton(self.groupBox_4)
        self.pushButton_11.setObjectName(u"pushButton_11")
        self.pushButton_11.setGeometry(QRect(20, 280, 21, 24))
        self.pushButton_12 = QPushButton(self.groupBox_4)
        self.pushButton_12.setObjectName(u"pushButton_12")
        self.pushButton_12.setGeometry(QRect(50, 280, 21, 24))
        self.pushButton_13 = QPushButton(self.groupBox_4)
        self.pushButton_13.setObjectName(u"pushButton_13")
        self.pushButton_13.setGeometry(QRect(100, 280, 51, 24))
        MainWindow.setCentralWidget(self.centralwidget)
        self.groupBox_3.raise_()
        self.groupBox.raise_()
        self.groupBox_2.raise_()
        self.groupBox_4.raise_()
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"PU_Learning_tool_box 0.01 A", None))
        self.groupBox.setTitle(QCoreApplication.translate("MainWindow", u"Algorithm", None))
        self.pushButton.setText(QCoreApplication.translate("MainWindow", u"+", None))
        self.pushButton_2.setText(QCoreApplication.translate("MainWindow", u"-", None))
        self.pushButton_3.setText(QCoreApplication.translate("MainWindow", u"Clear", None))
        self.groupBox_2.setTitle(QCoreApplication.translate("MainWindow", u"Metrics", None))
        self.checkBox.setText(QCoreApplication.translate("MainWindow", u"Accuracy", None))
        self.checkBox_2.setText(QCoreApplication.translate("MainWindow", u"F1", None))
        self.checkBox_3.setText(QCoreApplication.translate("MainWindow", u"Recall", None))
        self.checkBox_4.setText(QCoreApplication.translate("MainWindow", u"Precision", None))
        self.groupBox_3.setTitle(QCoreApplication.translate("MainWindow", u"Other", None))
        self.pushButton_4.setText(QCoreApplication.translate("MainWindow", u"Run", None))
        self.pushButton_5.setText(QCoreApplication.translate("MainWindow", u"Help", None))
        self.pushButton_6.setText(QCoreApplication.translate("MainWindow", u"Show", None))
        self.pushButton_7.setText(QCoreApplication.translate("MainWindow", u"Save", None))
        self.pushButton_14.setText(QCoreApplication.translate("MainWindow", u"Opts", None))
        self.pushButton_15.setText(QCoreApplication.translate("MainWindow", u"About", None))
        self.groupBox_4.setTitle(QCoreApplication.translate("MainWindow", u"Dataset", None))
        self.pushButton_11.setText(QCoreApplication.translate("MainWindow", u"+", None))
        self.pushButton_12.setText(QCoreApplication.translate("MainWindow", u"-", None))
        self.pushButton_13.setText(QCoreApplication.translate("MainWindow", u"Clear", None))
    # retranslateUi
