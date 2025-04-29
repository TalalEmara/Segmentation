import cv2
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QLabel, QPushButton, \
    QGroupBox, QSpinBox, QDoubleSpinBox, QComboBox, QBoxLayout, QCheckBox
from Core.OptimalThreshold import iterative_threshold
from Core.otsu_thresholding import otsu_threshold
from Core.RegionGrowing import region_growing
from Core.spectral import multi_otsu
from Core.LocalThresholding import local_optimal_thresholding
from Core.kmean_clustering import segment_image
from GUI.styles import GroupBoxStyle, button_style, second_button_style, label_style
from Core.canny import canny
from Core.imageMode import rgb_to_grayscale
from GUI.ImageViewer import ImageViewer

import time
from PIL import Image  


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
            self.otsuButton = QPushButton("Otsu Threshold")
            self.kmeansButton = QPushButton("K-means Clustering")
            self.spectralButton = QPushButton("Spectral Threshold") 
            self.localButton = QPushButton("Local Threshold")  

            self.optimalButton.clicked.connect(lambda: self.changeMode("Optimal Threshold"))
            self.otsuButton.clicked.connect(lambda: self.changeMode("Otsu Threshold")) 
            self.regionButton.clicked.connect(lambda: self.changeMode("Region Growing"))
            self.kmeansButton.clicked.connect(lambda: self.changeMode("K-means Clustering"))
            self.spectralButton.clicked.connect(lambda: self.changeMode("Spectral Threshold"))  
            self.localButton.clicked.connect(lambda: self.changeMode("Local Threshold"))


        createModePanel()

        self.inputViewer = ImageViewer("Input Image")
        self.outputViewer = ImageViewer("Output Image")
        self.outputViewer.setReadOnly(True)
        self.secondInputViewer = ImageViewer("Input Image")


        self.processButton = QPushButton("Process")


    def createSpectralParameters(self):
        self.parametersGroupBox = QGroupBox("Spectral Parameters")
        self.parametersGroupBox.setStyleSheet(GroupBoxStyle)
        
        self.classesLabel = QLabel("Number of classes:")
        self.classesSpinBox = QSpinBox()
        self.classesSpinBox.setRange(2, 10)
        self.classesSpinBox.setValue(3)
        
        self.spectralResultLabel = QLabel("Thresholds:")
        self.spectralResultLabel.setAlignment(Qt.AlignCenter)
        
        layout = QHBoxLayout()
        layout.addWidget(self.classesLabel)
        layout.addWidget(self.classesSpinBox)
        layout.addWidget(self.spectralResultLabel)
        layout.addWidget(self.processButton)
        
        self.parametersGroupBox.setLayout(layout)

    def createLocalParameters(self):
        self.parametersGroupBox = QGroupBox("Local Parameters")
        self.parametersGroupBox.setStyleSheet(GroupBoxStyle)
        
        self.patchLabel = QLabel("Patch size:")
        self.patchSpinBox = QSpinBox()
        self.patchSpinBox.setRange(16, 256)
        self.patchSpinBox.setValue(64)
        
        self.methodLabel = QLabel("Method:")
        self.methodCombo = QComboBox()
        self.methodCombo.addItems(["optimal", "otsu", "spectral"])
        
        layout = QHBoxLayout()
        layout.addWidget(self.patchLabel)
        layout.addWidget(self.patchSpinBox)
        layout.addWidget(self.methodLabel)
        layout.addWidget(self.methodCombo)
        layout.addWidget(self.processButton)
        
        self.parametersGroupBox.setLayout(layout)
        
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


    def createKmeansParameters(self):
        self.parametersGroupBox = QGroupBox("K-means Parameters")
        self.parametersGroupBox.setStyleSheet(GroupBoxStyle)
        
        self.kLabel = QLabel("Number of clusters:")
        self.kLabel.setAlignment(Qt.AlignCenter)
        self.kSpinBox = QSpinBox()
        self.kSpinBox.setRange(2, 20)
        self.kSpinBox.setValue(5)
        
      
        placeholder = QWidget()
        placeholder.setFixedWidth(self.minimumChange.width())
        
        layout = QHBoxLayout()
        layout.addWidget(self.kLabel)
        layout.addWidget(self.kSpinBox)
        layout.addWidget(placeholder)  
 
        
        self.parametersGroupBox.setLayout(layout)

        

    def createOtsuParameters(self):
        self.parametersGroupBox = QGroupBox("Otsu Threshold")
        self.parametersGroupBox.setStyleSheet(GroupBoxStyle)

        self.otsuThresholdLabel = QLabel("Calculated Threshold:")
        self.otsuThresholdLabel.setAlignment(Qt.AlignCenter)

        layout = QHBoxLayout()
        layout.addWidget(self.otsuThresholdLabel)
     

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
        modesLayout.addWidget(self.otsuButton)
        modesLayout.addWidget(self.kmeansButton)
        modesLayout.addWidget(self.spectralButton)  
        modesLayout.addWidget(self.localButton)
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

        elif mode == "Otsu Threshold":
            self.createOtsuParameters()

        elif mode == "Region Growing":
            self.createRegionParameters()

        elif mode == "K-means Clustering": 
            self.createKmeansParameters()

        elif mode == "Spectral Threshold": 
            self.createSpectralParameters()
        elif mode == "Local Threshold":  
            self.createLocalParameters()

        elif self.currentMode == "Spectral Threshold":
            classes = self.classesSpinBox.value()
            if len(self.processingImage.shape) == 3:
                self.processingImage = cv2.cvtColor(self.processingImage, cv2.COLOR_BGR2GRAY)
            regions, thresholds = multi_otsu(self.processingImage, classes=classes)
            self.processingImage = (regions * (255//(classes-1))).astype(np.uint8)
            self.spectralResultLabel.setText(f"Thresholds: {thresholds}")
        elif self.currentMode == "Local Threshold":
            patch_size = self.patchSpinBox.value()
            method = self.methodCombo.currentText()
            if len(self.processingImage.shape) == 3:
                self.processingImage = cv2.cvtColor(self.processingImage, cv2.COLOR_BGR2GRAY)
            self.processingImage = local_optimal_thresholding(
                self.processingImage,
                threshold_type=method,
                patch_size=patch_size
            )


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
        self.otsuButton.setStyleSheet(button_style) 
        self.kmeansButton.setStyleSheet(button_style)
        self.spectralButton.setStyleSheet(button_style)
        self.localButton.setStyleSheet(button_style)



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

        elif self.currentMode == "Otsu Threshold":
            finalThreshold, self.processingImage = otsu_threshold(self.processingImage)
            self.otsuThresholdLabel.setText(f"Otsu Threshold: {finalThreshold}")

        elif self.currentMode == "K-means Clustering":
            K =self.kSpinBox.value()

            if self.processingImage.ndim == 2:  # Grayscale image
                self.processingImage = np.stack([self.processingImage]*3, axis=-1)

            self.processingImage, _ = segment_image(self.processingImage, K)

        self.outputViewer.displayImage(self.processingImage)






if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    window = FetchFeature()
    window.show()
    sys.exit(app.exec_())
