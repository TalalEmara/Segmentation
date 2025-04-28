import cv2
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QLabel, QPushButton, \
    QGroupBox, QSpinBox, QDoubleSpinBox, QComboBox, QBoxLayout, QCheckBox
from Core.OptimalThreshold import iterative_threshold
from Core.RegionGrowing import region_growing
from GUI.styles import GroupBoxStyle, button_style, second_button_style, label_style
from Core.canny import canny
from Core.imageMode import rgb_to_grayscale
from GUI.ImageViewer import ImageViewer
import time


class FetchFeature(QMainWindow):
    def __init__(self):
        super().__init__()  # Initialize QMainWindow
        self.setWindowTitle("Raqib")
        self.setFixedSize(1200, 800)

        self.initializeUI()
        self.createOptimalParameters()
        self.setupLayout()
        self.styleUI()
        self.connectUI()

    def initializeUI(self):

        self.processingImage = None
        self.currentMode = "Optimal Threshold"
        self.logo = QLabel("Raqip")

        def createModePanel():
            self.optimalButton = QPushButton("Optimal Threshold")
            self.regionButton = QPushButton("Region Growing")


            self.optimalButton.clicked.connect(lambda: self.changeMode("Optimal Threshold"))
            self.regionButton.clicked.connect(lambda: self.changeMode("Region Growing"))


        createModePanel()

        self.inputViewer = ImageViewer("Input Image")
        self.outputViewer = ImageViewer("Output Image")
        self.outputViewer.setReadOnly(True)
        self.secondInputViewer = ImageViewer("Input Image")


        self.processButton = QPushButton("Process")


    def createOptimalParameters(self):
        self.parametersGroupBox = QGroupBox("Optimal Threshold")
        self.parametersGroupBox.setStyleSheet(GroupBoxStyle)

        self.optimalThresholdLabel = QLabel("Optimal Threshold:")
        self.optimalThresholdLabel.setAlignment(Qt.AlignCenter)


        self.maxIterationsLabel = QLabel("Max iteratons")
        self.maxIterationsLabel.setAlignment(Qt.AlignCenter)
        self.maxIterations = QSpinBox()
        self.maxIterations.setSingleStep(1)
        self.maxIterations.setRange(1,1000)
        self.maxIterations.setValue(200)
        # … after windowSize setup …


        # Lambda-style (float %) threshold
        self.minimumChangeLabel = QLabel("Minimum change")
        self.minimumChangeLabel.setAlignment(Qt.AlignCenter)
        self.minimumChange = QDoubleSpinBox()
        self.minimumChange.setRange(0.1, 10.0)
        self.minimumChange.setSingleStep(0.1)
        self.minimumChange.setValue(0.01)

        # Layout
        layout = QHBoxLayout()
        layout.addWidget(self.maxIterationsLabel)
        layout.addWidget(self.maxIterations)
        layout.addWidget(self.minimumChangeLabel)
        layout.addWidget(self.minimumChange)
        layout.addWidget(self.optimalThresholdLabel)

        self.parametersGroupBox.setLayout(layout)


    def createRegionParameters(self):
        self.parametersGroupBox = QGroupBox("Region Growing Parameters")
        self.parametersGroupBox.setStyleSheet(GroupBoxStyle)

        self.regionThresholdLabel = QLabel("threshold")
        self.regionThresholdLabel.setAlignment(Qt.AlignCenter)
        self.regionThreshold = QSpinBox()
        self.regionThreshold.setRange(0, 255)
        self.regionThreshold.setValue(20)

        self.coloredImageLabel = QLabel("Colored:")
        self.coloredImageLabel.setAlignment(Qt.AlignCenter)
        self.coloredImageCheck = QCheckBox()

        layout = QHBoxLayout()

        layout.addWidget(self.regionThresholdLabel)
        layout.addWidget(self.regionThreshold)
        layout.addWidget(self.coloredImageLabel)
        layout.addWidget(self.coloredImageCheck)

        self.parametersGroupBox.setLayout(layout)




    def setupLayout(self):
        mainWidget = QWidget(self)
        self.setCentralWidget(mainWidget)

        mainLayout = QHBoxLayout()
        modesLayout = QVBoxLayout()
        workspace = QVBoxLayout()
        self.imagesLayout = QHBoxLayout()
        imagesLayoutH = QHBoxLayout()
        self.parametersLayout = QHBoxLayout()

        self.parametersLayout.addWidget(self.parametersGroupBox)
        self.parametersLayout.addWidget(self.processButton)

        # Add widgets to layout
        modesLayout.addWidget(self.logo, alignment=Qt.AlignCenter)
        modesLayout.addWidget(self.regionButton)
        modesLayout.addWidget(self.optimalButton)
        # modesLayout.addWidget(self.houghCirclesButton)
        # modesLayout.addWidget(self.houghEllipseButton)
        # modesLayout.addWidget(self.snakeButton)
        modesLayout.addStretch()



        self.imagesLayout.addWidget(self.inputViewer)
        self.imagesLayout.addWidget(self.outputViewer)
        # Nest layouts
        mainLayout.addLayout(modesLayout,20)
        mainLayout.addLayout(workspace,80)

        workspace.addLayout(self.imagesLayout)
        workspace.addLayout(self.parametersLayout)


        mainWidget.setLayout(mainLayout)

    def changeMode(self, mode):
        """Change the current mode and update the UI accordingly."""
        self.currentMode = mode

        # Remove existing parametersGroupBox if it exists
        if hasattr(self, "parametersGroupBox"):
            self.parametersLayout.removeWidget(self.parametersGroupBox)
            self.parametersGroupBox.deleteLater()  # Properly delete the widget

        # Create the corresponding parameter panel
        if mode == "Optimal Threshold":
            self.createOptimalParameters()

        elif mode == "Region Growing":
            self.createRegionParameters()


        self.parametersLayout.insertWidget(0, self.parametersGroupBox)

    def styleUI(self):
        self.logo.setStyleSheet("font-family: 'Franklin Gothic';"
                                " font-size: 32px;"
                                " font-weight:600;"
                                " padding:30px;")


        self.processButton.setFixedWidth(250)
        self.processButton.setFixedHeight(40)
        # self.processButton.setStyleSheet(second_button_style)
        self.optimalButton.setStyleSheet(button_style)
        self.regionButton.setStyleSheet(button_style)



    def connectUI(self):
        self.processButton.clicked.connect(self.processImage)
        self.inputViewer.selectionMade.connect(self.on_selection_made)

    def on_selection_made(self,coords):
        self.selected = coords

    def processImage(self):
        self.processingImage = self.inputViewer.image.copy()
        if self.currentMode == "Optimal Threshold":
            finalThreshold,self.processingImage = iterative_threshold(self.processingImage,self.minimumChange.value(),self.maxIterations.value())
            self.optimalThresholdLabel.setText(f"Optimal Threshold: {finalThreshold}")

        elif self.currentMode == "Region Growing":

            self.processingImage = region_growing(self.processingImage, self.selected, self.regionThreshold.value(),not(self.coloredImageCheck.isChecked()))

        self.outputViewer.displayImage(self.processingImage)






if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    window = FetchFeature()
    window.show()
    sys.exit(app.exec_())
