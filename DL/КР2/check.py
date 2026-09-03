import sys
import shutil
from pathlib import Path
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QVBoxLayout, 
                             QWidget, QProgressBar, QShortcut)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QKeySequence
from PyQt5.QtGui import QPixmap, QPalette, QColor, QFont


class ImageSorter(QMainWindow):
    def __init__(self):
        super().__init__()
        self.src_dir = Path("dataset_dumplings")
        self.good_dir = Path("dataset_filtered")
        self.bad_dir = Path("dataset_wrong")
        
        self.images = list(self.src_dir.rglob("*.jpg")) + list(self.src_dir.rglob("*.png"))
        self.total = len(self.images)
        self.current_idx = 0

        self.init_ui()
        self.load_image()

    def init_ui(self):
        self.setWindowTitle("Dumpling Sorter 3000")
        self.resize(1000, 800)
        self.setStyleSheet("background-color: #2b2b2b; color: #ffffff;")

        # Layouts
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # Info Label
        self.info_lbl = QLabel()
        self.info_lbl.setFont(QFont("Arial", 14, QFont.Bold))
        self.info_lbl.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.info_lbl)

        # Image Viewer
        self.img_lbl = QLabel("No Images Found")
        self.img_lbl.setAlignment(Qt.AlignCenter)
        self.img_lbl.setStyleSheet("border: 2px solid #3d3d3d; background: #1e1e1e;")
        layout.addWidget(self.img_lbl, 1)

        # Help / Status
        self.help_lbl = QLabel("⬅ LEFT: Wrong | RIGHT: Keep ➡")
        self.help_lbl.setAlignment(Qt.AlignCenter)
        self.help_lbl.setFont(QFont("Arial", 12))
        self.help_lbl.setStyleSheet("color: #aaaaaa; padding: 10px;")
        layout.addWidget(self.help_lbl)

        # Progress
        self.progress = QProgressBar()
        self.progress.setStyleSheet("""
            QProgressBar {border: 0px; background: #3d3d3d; height: 10px; border-radius: 5px;}
            QProgressBar::chunk {background: #00ADB5; border-radius: 5px;}
        """)
        layout.addWidget(self.progress)

    def keyPressEvent(self, event):
        if self.current_idx >= self.total: return
        
        key = event.key()
        if key == Qt.Key_Right:
            self.move_image(is_good=True)
        elif key == Qt.Key_Left:
            self.move_image(is_good=False)

    def move_image(self, is_good):
        src = self.images[self.current_idx]
        root = self.good_dir if is_good else self.bad_dir
        
        # Replicate folder structure (e.g., dataset_filtered/pelmeni/img.jpg)
        rel_path = src.relative_to(self.src_dir)
        dst = root / rel_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            shutil.move(src, dst)
        except Exception as e:
            print(f"Error moving {src.name}: {e}")

        self.current_idx += 1
        self.load_image()

    def load_image(self):
        if self.current_idx >= self.total:
            self.img_lbl.setText("🎉 ALL DONE! 🎉")
            self.info_lbl.setText("")
            self.progress.setValue(100)
            return

        img_path = str(self.images[self.current_idx])
        pixmap = QPixmap(img_path)
        
        # Scale nicely
        w, h = self.img_lbl.width(), self.img_lbl.height()
        self.img_lbl.setPixmap(pixmap.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        
        # Update Info
        folder = self.images[self.current_idx].parent.name.upper()
        self.info_lbl.setText(f"[{folder}]  {self.current_idx + 1} / {self.total}")
        self.progress.setValue(int((self.current_idx / self.total) * 100))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageSorter()
    window.show()
    sys.exit(app.exec_())