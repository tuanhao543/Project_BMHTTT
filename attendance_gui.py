import sys
import cv2
import psycopg2
from psycopg2 import OperationalError, Error
import os
from datetime import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import preprocess_input as preprocess_mask
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.metrics.pairwise import cosine_similarity
import time
import traceback
import re # For validating names

from PySide6.QtWidgets import (QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout,
                               QSizePolicy, QPushButton, QMessageBox, QDialog, QLineEdit)
from PySide6.QtCore import Qt, QTimer, Slot, Signal, QDateTime
from PySide6.QtGui import QImage, QPixmap, QFont

# --- (BẮT BUỘC) Định nghĩa Lớp Tùy chỉnh NẾU bạn đã dùng nó khi huấn luyện embedding model ---
class L2NormalizationLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(L2NormalizationLayer, self).__init__(**kwargs)
    def call(self, inputs):
        import tensorflow as tf # Đảm bảo tf có sẵn trong scope này
        return tf.math.l2_normalize(inputs, axis=1)
    def get_config(self):
        config = super(L2NormalizationLayer, self).get_config()
        return config
    def compute_output_shape(self, input_shape):
        return input_shape
# -------------------------------------------------------------------------------------

DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "chamcong"
DB_USER = "NTH"
DB_PASSWORD = "NTHao543@"

FACE_DETECTOR_PROTOTXT = "deploy.prototxt"
FACE_DETECTOR_CAFFEMODEL = "res10_300x300_ssd_iter_140000.caffemodel"
MASK_MODEL_PATH = 'mask_detection.keras'
RECOGNITION_MODEL_PATH = 'face_embedding_model.keras'
KNOWN_FACES_DIR = 'data_anh' # Dùng để tạo known embeddings

MASK_INPUT_SIZE = (224, 224)
RECOGNITION_INPUT_SIZE = (160, 160)
FACE_CONFIDENCE_THRESHOLD = 0.6
RECOGNITION_THRESHOLD = 0.55
MASK_LABELS = {0: 'Co Khau Trang', 1: 'Khong Khau Trang', 2: 'Co Khau Trang'}

# --- Hàm tiền xử lý ---
def preprocess_facenet(image):
    import tensorflow as tf
    image = tf.cast(image, tf.float32)
    image = (image - 127.5) / 128.0
    return image

# ============================================
# Lớp Cửa sổ Chấm công (Hiển thị Webcam)
# ============================================
class AttendanceWindow(QDialog):
    attendance_recorded = Signal(tuple) # Signal: (name, status: "Check-in"/"Check-out")
    def __init__(self, parent=None, db_conn=None, db_cursor=None, face_detector=None,
                 mask_model=None, recognition_model=None, known_embeddings=None, known_names=None):
        super().__init__(parent)
        self.setWindowTitle("Chấm Công - Vui lòng nhìn vào Camera")
        self.setMinimumSize(700, 550)

        self.conn = db_conn
        self.cursor = db_cursor
        self.face_detector = face_detector
        self.mask_model = mask_model
        self.recognition_model = recognition_model
        self.known_embeddings = known_embeddings
        self.known_names = known_names

        if not all([self.conn, self.cursor, self.face_detector, self.mask_model, self.recognition_model, self.known_embeddings is not None, self.known_names is not None]):
            QMessageBox.critical(self, "Lỗi Khởi tạo", "Thiếu các thành phần cần thiết (CSDL, Models hoặc Embeddings). Cửa sổ sẽ đóng.")
            QTimer.singleShot(0, self.reject)
            return

        self.detection_start_time = None
        self.detected_name = None
        self.required_stable_time = 2.0
        self.is_processing_attendance = False

        layout = QVBoxLayout(self)
        self.video_label = QLabel("Đang căn chỉnh...")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        info_font = QFont()
        info_font.setPointSize(16)
        self.video_label.setFont(info_font)
        layout.addWidget(self.video_label)

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.video_label.setText("LỖI: Không thể mở webcam!")
            QMessageBox.critical(self, "Lỗi Webcam", "Không thể mở webcam!")
            self.timer = None
            QTimer.singleShot(0, self.reject)
            return

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)
        print("[AttendanceWindow] Initialized and webcam running.")

    def update_frame(self):
        if self.timer is None or not self.cap.isOpened() or self.is_processing_attendance: return
        ret, frame = self.cap.read()
        if not ret: return

        # frame = cv2.flip(frame, 1)
        draw_frame = frame.copy()
        (h, w) = frame.shape[:2]
        found_unmasked_face = False
        masked_face_detected_in_frame = False
        best_unmasked_face_info = None
        max_area = 0

        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.face_detector.setInput(blob)
        detections = self.face_detector.forward()

        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > FACE_CONFIDENCE_THRESHOLD:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                startX, startY = max(0, startX), max(0, startY)
                endX, endY = min(w - 1, endX), min(h - 1, endY)
                if startX >= endX or startY >= endY: continue
                face = frame[startY:endY, startX:endX]
                if face.size == 0: continue

                try:
                    face_mask_input = cv2.resize(face, MASK_INPUT_SIZE)
                    face_mask_input = img_to_array(face_mask_input)
                    face_mask_input = np.expand_dims(face_mask_input, axis=0)
                    face_mask_input = preprocess_mask(face_mask_input)  # Dùng hàm preprocess của model mask
                    mask_pred = self.mask_model.predict(face_mask_input, verbose=0)
                    mask_label_idx = np.argmax(mask_pred, axis=1)[0]

                    if mask_label_idx == 1:
                        found_unmasked_face = True
                        current_area = (endX - startX) * (endY - startY)


                        face_rec_input = cv2.resize(face, RECOGNITION_INPUT_SIZE)

                        face_rec_input = img_to_array(face_rec_input)
                        face_rec_input = np.expand_dims(face_rec_input, axis=0)
                        face_rec_input = preprocess_facenet(face_rec_input)
                        current_embedding = self.recognition_model.predict(face_rec_input, verbose=0)[0]

                        if not self.known_embeddings: # Xử lý trường hợp list embedding rỗng
                            recognized_name = "Unknown (No DB)"
                            best_similarity = 0.0
                        else:
                            similarities = cosine_similarity([current_embedding], self.known_embeddings)[0]
                            best_match_idx = np.argmax(similarities)
                            best_similarity = similarities[best_match_idx]
                            if best_similarity >= RECOGNITION_THRESHOLD and best_match_idx < len(self.known_names):
                                recognized_name = self.known_names[best_match_idx]
                            else:
                                recognized_name = "Unknown"

                        if current_area > max_area:
                            max_area = current_area
                            best_unmasked_face_info = (box.astype("int"), recognized_name, best_similarity)

                    elif mask_label_idx == 0 or mask_label_idx == 2:
                        masked_face_detected_in_frame = True
                        mask_warning_color = (0, 255, 255)
                        cv2.rectangle(draw_frame, (startX, startY), (endX, endY), mask_warning_color, 2)
                        text_y_mask = startY - 10 if startY - 10 > 10 else startY + 10

                        warning_text = "Dang deo khau trang!"
                        cv2.putText(draw_frame, warning_text, (startX, text_y_mask),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, mask_warning_color, 2, lineType=cv2.LINE_AA)


                except Exception as e_inner:
                    print(f"[Error] Inner loop processing error: {e_inner}\n{traceback.format_exc()}")

        if best_unmasked_face_info:
            current_box, current_name, current_similarity = best_unmasked_face_info
            (startX, startY, endX, endY) = current_box

            if current_name != "Unknown" and "No DB" not in current_name:
                if current_name == self.detected_name:
                    if self.detection_start_time is not None:
                        elapsed_time = time.time() - self.detection_start_time
                        if elapsed_time >= self.required_stable_time:
                            # --- Đủ thời gian -> Chấm công và đóng ---
                            # ... (Dừng timer, gọi record_attendance, emit signal, accept()) ...
                            print(f"Stable detection: {current_name} for {elapsed_time:.2f}s")
                            self.is_processing_attendance = True;
                            self.timer.stop()
                            attendance_status = self.record_attendance(current_name)
                            if "Error" not in attendance_status:
                                self.attendance_recorded.emit((current_name, attendance_status)); self.accept()
                            else:
                                QMessageBox.critical(self, "Lỗi Ghi CSDL",
                                                     f"Lỗi: {attendance_status}"); self.is_processing_attendance = False; self.timer.start()
                            return
                        else:  # Chưa đủ thời gian
                            remaining_time = self.required_stable_time - elapsed_time
                            self.video_label.setText(
                                f"Nhận diện: {current_name}\nGiữ yên trong {remaining_time:.1f} giây...")
                    else:  # Lần đầu thấy
                        self.detection_start_time = time.time()
                        self.video_label.setText(f"Nhận diện: {current_name}\nĐang xác nhận...")
                else:  # Người mới hoặc khác người cũ
                    self.detected_name = current_name
                    self.detection_start_time = time.time()
                    self.video_label.setText(f"Nhận diện: {current_name}\nĐang xác nhận...")

                # Vẽ hộp cho mặt không khẩu trang đang được theo dõi
                box_color = (0, 255, 0)  # Xanh lá
                label_display = current_name
            else:  # Là "Unknown" (không khẩu trang)
                self.detected_name = "Unknown"
                self.detection_start_time = None
                self.video_label.setText("Không nhận diện được")
                box_color = (0, 0, 255)  # Đỏ
                label_display = "Unknown"

                # Vẽ đè hộp của mặt không khẩu trang đang ưu tiên (nếu có)
            cv2.rectangle(draw_frame, (startX, startY), (endX, endY), box_color, 2)
            text_y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.putText(draw_frame, label_display, (startX, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2, lineType=cv2.LINE_AA)

        elif masked_face_detected_in_frame:  # Không có mặt không khẩu trang nào được ưu tiên, nhưng có mặt đeo khẩu trang
            self.detected_name = None  # Reset theo dõi
            self.detection_start_time = None
            # *** HIỂN THỊ THÔNG BÁO YÊU CẦU CỞI KHẨU TRANG ***
            self.video_label.setText("Đang đeo khẩu trang.\nVui lòng cởi ra và thử lại.")
            # Hộp cảnh báo màu vàng đã được vẽ trong vòng lặp

        else:  # Không tìm thấy mặt nào cả
            self.detected_name = None
            self.detection_start_time = None
            self.video_label.setText("Vui lòng nhìn thẳng vào camera")

            # --- Hiển thị frame cuối cùng lên GUI ---
        try:
            # ... (Logic chuyển đổi và hiển thị ảnh như cũ) ...
            rgb_image = cv2.cvtColor(draw_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape;
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            qt_pixmap = QPixmap.fromImage(qt_image)
            label_w = self.video_label.width();
            label_h = self.video_label.height()
            scaled_pixmap = qt_pixmap.scaled(label_w, label_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.video_label.setPixmap(scaled_pixmap)
        except Exception as e_display:
            pass

    def record_attendance(self, name):
        # ... (Logic ghi CSDL vào/ra như cũ) ...
        if not self.conn or not self.cursor: return "Error: No DB Connection"; status = "Unknown Status"
        try:
            now = datetime.now(); now_str = now.strftime('%Y-%m-%d %H:%M:%S')
            find_sql = "SELECT log_id FROM DiemDanh WHERE ten = %s AND thoigianra IS NULL ORDER BY thoigianvao DESC LIMIT 1 "
            self.cursor.execute(find_sql, (name,)); result = self.cursor.fetchone()
            if result:
                log_id_to_update = result[0]; update_sql = "UPDATE DiemDanh SET thoigianra = %s WHERE log_id = %s"
                self.cursor.execute(update_sql, (now_str, log_id_to_update)); self.conn.commit(); status = "Check-out"
                print(f"[{now_str}] Recorded {status} for: {name} (Log ID: {log_id_to_update})")
            else:
                insert_sql = "INSERT INTO DiemDanh (ten, thoigianvao) VALUES (%s, %s)"
                self.cursor.execute(insert_sql, (name, now_str)); self.conn.commit(); status = "Check-in"
                print(f"[{now_str}] Recorded {status} for: {name}")
            return status
        except Error as e:
            error_msg = f"DB Error ({type(e).__name__}): {e}"
            print(f"[ERROR] DB recording error for {name}: {error_msg}")
            try: self.conn.rollback()
            except Error as rb_e: print(f"[ERROR] Rollback error: {rb_e}")
            return f"Error: DB Operation Failed"
        except Exception as ex:
             error_msg = f"Unexpected Error: {ex}"
             print(f"[ERROR] Unexpected recording error: {error_msg}\n{traceback.format_exc()}")
             return f"Error: Unexpected"

    def closeEvent(self, event):
        print("[AttendanceWindow] Closing...")
        if hasattr(self, 'timer') and self.timer is not None: self.timer.stop()
        if hasattr(self, 'cap') and self.cap.isOpened(): self.cap.release()
        print("[AttendanceWindow] Webcam released.")
        event.accept()

# ============================================
# Lớp Cửa sổ Thêm Nhân Viên
# ============================================
class AddEmployeeWindow(QDialog):
    def __init__(self, parent=None, db_conn=None, db_cursor=None, known_faces_dir=None, face_detector_instance=None, wearing_glasses=False): # <<< Thêm wearing_glasses
        super().__init__(parent)
        self.setWindowTitle("Thêm Nhân Viên Mới - Tự động chụp")
        self.setMinimumSize(680, 650) # Có thể điều chỉnh

        self.parent_window = parent
        self.conn = db_conn
        self.cursor = db_cursor
        self.known_faces_dir = known_faces_dir
        self.face_detector = face_detector_instance
        self.wearing_glasses = wearing_glasses # <<< Lưu lựa chọn

        # Kiểm tra các thành phần cần thiết
        if not all([self.conn, self.cursor, self.known_faces_dir, self.face_detector]):
             QMessageBox.critical(self, "Lỗi Khởi tạo", "Thiếu kết nối CSDL, đường dẫn dataset hoặc face detector.")
             QTimer.singleShot(0, self.reject); return # Đóng dialog nếu lỗi
        if not os.path.exists(self.known_faces_dir):
             try:
                 os.makedirs(self.known_faces_dir)
                 print(f"Đã tạo thư mục dataset: {self.known_faces_dir}")
             except OSError as e:
                  QMessageBox.critical(self, "Lỗi Thư mục", f"Không thể tạo thư mục dataset:\n{self.known_faces_dir}\nLỗi: {e}")
                  QTimer.singleShot(0, self.reject); return

        # --- Cấu hình Chụp Tự động ---
        self.stage1_duration = 5.0 # Giây - Chụp có kính (hoặc không kính nếu ban đầu chọn No)
        self.stage2_duration = 5.0 # Giây - Chụp không kính (chỉ áp dụng nếu ban đầu có kính)
        self.capture_interval = int(1000 / 7) # Milliseconds - Khoảng 7 ảnh/giây
        self.capture_duration = self.stage1_duration + (self.stage2_duration if self.wearing_glasses else 0) # Tổng thời gian tối đa

        # Biến trạng thái
        self.captured_faces = [] # Lưu ảnh mặt đã crop
        self.is_capturing = False
        self.capture_stage = 0 # 0: chuẩn bị, 1: đang chụp stage 1, 2: đang chờ xác nhận bỏ kính, 3: đang chụp stage 2
        self.capture_start_time = None # Thời điểm bắt đầu stage 1
        self.stage2_start_time = None # Thời điểm bắt đầu chụp stage 2
        self.remove_glasses_prompted = False # Cờ đã yêu cầu bỏ kính chưa

        # --- Layout ---
        layout = QVBoxLayout(self)

        # --- Label hiển thị Video và Hướng dẫn ---
        self.video_label = QLabel("Chuẩn bị...")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label.setMinimumSize(600, 450)
        info_font = QFont(); info_font.setPointSize(14) # Font cho hướng dẫn
        self.video_label.setFont(info_font)
        layout.addWidget(self.video_label)

        # --- Layout điều khiển ---
        control_layout = QHBoxLayout()

        # --- Nút Xác nhận Bỏ Kính ---
        self.btn_confirm_no_glasses = QPushButton("Xác nhận Đã Bỏ Kính")
        self.btn_confirm_no_glasses.setVisible(False)
        self.btn_confirm_no_glasses.setEnabled(False)
        self.btn_confirm_no_glasses.clicked.connect(self.confirm_no_glasses)
        control_layout.addWidget(self.btn_confirm_no_glasses)

        # --- Ô nhập tên ---
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Nhập tên (không dấu, không cách)...")
        self.name_input.setEnabled(False)
        control_layout.addWidget(self.name_input)

        # --- Nút Lưu ---
        self.btn_save = QPushButton("Lưu")
        self.btn_save.setEnabled(False)
        self.btn_save.clicked.connect(self.save_employee)
        control_layout.addWidget(self.btn_save)

        layout.addLayout(control_layout)

        # --- Webcam ---
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.video_label.setText("LỖI: Không thể mở webcam!")
            QMessageBox.critical(self, "Lỗi Webcam", "Không thể mở webcam!")
            self.timer_display = None
            self.timer_capture = None
            QTimer.singleShot(0, self.reject)
            return

        # --- Timers ---
        self.timer_display = QTimer(self)
        self.timer_display.timeout.connect(self.update_display_frame)
        self.timer_display.start(50)

        self.timer_capture = QTimer(self)
        self.timer_capture.timeout.connect(self.auto_capture_image)

        print("[AddEmployeeWindow] Initialized. Starting auto capture sequence soon...")
        QTimer.singleShot(2000, self.start_auto_capture_stage1) # Bắt đầu sau 2s

    def start_auto_capture_stage1(self):
        """Bắt đầu giai đoạn chụp ảnh đầu tiên."""
        if self.timer_capture is None: return
        self.is_capturing = True
        self.capture_stage = 1
        self.capture_start_time = time.time()
        self.captured_faces = []
        self.timer_capture.start(self.capture_interval)
        duration = self.stage1_duration
        print(f"[AddEmployeeWindow] Auto capture stage 1 started (Duration: {duration}s)")
        # Cập nhật hiển thị ngay lập tức
        self.update_display_frame()

    def detect_and_crop_face(self, frame):
        """Phát hiện khuôn mặt lớn nhất và crop."""
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.face_detector.setInput(blob)
        detections = self.face_detector.forward()
        best_confidence = 0; best_box = None
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > FACE_CONFIDENCE_THRESHOLD and confidence > best_confidence:
                best_confidence = confidence; box = detections[0, 0, i, 3:7] * np.array([w, h, w, h]); best_box = box.astype("int")
        if best_box is not None:
            (startX, startY, endX, endY) = best_box; startX, startY = max(0, startX), max(0, startY); endX, endY = min(w - 1, endX), min(h - 1, endY)
            if startX < endX and startY < endY: return frame[startY:endY, startX:endX]
        return None

    def update_display_frame(self):
        """Chỉ cập nhật hiển thị frame và thông báo."""
        if self.timer_display is None or not self.cap.isOpened(): return
        ret, frame = self.cap.read()
        if ret:
            # frame = cv2.flip(frame, 1)
            draw_frame = frame.copy()
            h, w = draw_frame.shape[:2]
            cv2.rectangle(draw_frame, (w//4, h//4), (3*w//4, 3*h//4), (0, 255, 0), 1)

            status_text = ""
            current_time = time.time()
            capture_count = len(self.captured_faces)

            if self.capture_stage == 1:
                elapsed_time = current_time - self.capture_start_time if self.capture_start_time else 0
                duration = self.stage1_duration
                remaining_time = max(0, duration - elapsed_time)
                glasses_status = "(CO KINH)" if self.wearing_glasses else "(KHONG KINH)"
                status_text = f"Dang chup {glasses_status}... ({capture_count} anh)\nDi chuyen khuon mat cham rai\nCon lai: {remaining_time:.1f} s"
            elif self.capture_stage == 2:
                 status_text = "!!! VUI LONG BO KINH RA !!!\nNhan 'Xac nhan' khi san sang chup tiep"
            elif self.capture_stage == 3:
                 elapsed_time = current_time - self.stage2_start_time if self.stage2_start_time else 0
                 duration = self.stage2_duration
                 remaining_time = max(0, duration - elapsed_time)
            elif not self.btn_save.isEnabled() and self.capture_stage == 0 : # Chưa bắt đầu hoặc lỗi
                 status_text = "Chuan bi chup tu dong..."
            elif self.btn_save.isEnabled(): # Đã chụp xong, chờ lưu
                 status_text = f"Da chup xong {len(self.captured_faces)} anh mat.\nVui long nhap ten."

            # Vẽ text lên frame nếu có nội dung
            if status_text:
                font = cv2.FONT_HERSHEY_SIMPLEX; scale = 0.7; color = (0, 255, 255); thickness = 2
                y0, dy = 30, 25
                for i, line in enumerate(status_text.split('\n')):
                    cv2.putText(draw_frame, line, (15, y0 + i*dy), font, scale, color, thickness, cv2.LINE_AA)

            # Hiển thị frame
            try:
                rgb_image = cv2.cvtColor(draw_frame, cv2.COLOR_BGR2RGB)
                h_img, w_img, ch = rgb_image.shape; bytes_per_line = ch * w_img
                qt_image = QImage(rgb_image.data, w_img, h_img, bytes_per_line, QImage.Format_RGB888)
                qt_pixmap = QPixmap.fromImage(qt_image)
                label_w, label_h = self.video_label.width(), self.video_label.height()
                scaled_pixmap = qt_pixmap.scaled(label_w, label_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.video_label.setPixmap(scaled_pixmap)
            except Exception as e: pass

    @Slot()
    def auto_capture_image(self):
        """Chụp ảnh tự động theo giai đoạn."""
        if not self.is_capturing or self.timer_capture is None or not self.cap.isOpened(): return

        current_time = time.time()
        if self.capture_stage == 1:
            elapsed_time = current_time - self.capture_start_time
            if elapsed_time >= self.stage1_duration:
                if self.wearing_glasses: self.pause_for_glasses_removal()
                else: self.stop_capture_finalize()
                return
        elif self.capture_stage == 3:
            elapsed_time = current_time - self.stage2_start_time
            if elapsed_time >= self.stage2_duration:
                self.stop_capture_finalize()
                return
        else: return # Không chụp ở stage 0 hoặc 2

        ret, frame = self.cap.read()
        if ret:
            face_crop = self.detect_and_crop_face(frame)
            if face_crop is not None and face_crop.size > 100:
                self.captured_faces.append(face_crop)

    def pause_for_glasses_removal(self):
        """Tạm dừng chụp, yêu cầu bỏ kính."""
        print("[AddEmployeeWindow] Pausing for glasses removal.")
        self.is_capturing = False
        self.timer_capture.stop()
        self.capture_stage = 2
        self.btn_confirm_no_glasses.setVisible(True)
        self.btn_confirm_no_glasses.setEnabled(True)
        # update_display_frame sẽ hiển thị thông báo

    @Slot()
    def confirm_no_glasses(self):
        """Tiếp tục chụp giai đoạn 2."""
        print("[AddEmployeeWindow] User confirmed glasses removed. Starting stage 2.")
        self.is_capturing = True
        self.capture_stage = 3
        self.stage2_start_time = time.time()
        self.btn_confirm_no_glasses.setVisible(False)
        self.btn_confirm_no_glasses.setEnabled(False)
        self.timer_capture.start(self.capture_interval)
        # update_display_frame sẽ hiển thị thông báo

    def stop_capture_finalize(self):
        """Kết thúc chụp, xử lý ảnh và bật input."""
        print("[AddEmployeeWindow] Auto capture finished.")
        self.is_capturing = False
        self.capture_stage = 0
        if self.timer_capture: self.timer_capture.stop()

        if not self.captured_faces:
             QMessageBox.warning(self, "Không có ảnh", "Không chụp được khuôn mặt nào."); self.name_input.setEnabled(False); self.btn_save.setEnabled(False); self.video_label.setText("Chụp ảnh thất bại."); return

        print(f"Total captured faces: {len(self.captured_faces)}")
        self.name_input.setEnabled(True)
        self.btn_save.setEnabled(True)

        # Hiển thị ảnh mặt cuối cùng
        last_face = self.captured_faces[-1]
        try:
            rgb_image = cv2.cvtColor(last_face, cv2.COLOR_BGR2RGB); h, w, ch = rgb_image.shape; bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888); qt_pixmap = QPixmap.fromImage(qt_image)
            label_w, label_h = self.video_label.width(), self.video_label.height()
            scaled_pixmap = qt_pixmap.scaled(label_w // 2, label_h // 2, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.video_label.setPixmap(scaled_pixmap); self.video_label.setAlignment(Qt.AlignCenter)
        except Exception as e: print(f"Lỗi hiển thị ảnh mặt cuối: {e}")
        QTimer.singleShot(100, lambda: self.video_label.setText(f"Đã chụp xong {len(self.captured_faces)} ảnh mặt.\nVui lòng nhập tên."))

    @Slot()
    def save_employee(self):
        """Lưu thông tin và ảnh nhân viên mới."""
        name_raw = self.name_input.text().strip()
        if not name_raw: QMessageBox.warning(self, "Thiếu thông tin", "Vui lòng nhập tên."); return
        if not re.match("^[a-zA-Z0-9_]+$", name_raw): QMessageBox.warning(self, "Tên không hợp lệ", "Tên chỉ chứa chữ cái không dấu, số, gạch dưới (_)."); return
        name = name_raw
        if not self.captured_faces: QMessageBox.warning(self, "Thiếu ảnh", "Không có ảnh mặt nào được chụp."); return

        employee_dir = os.path.join(self.known_faces_dir, name)
        try:
            # Kiểm tra tên trong CSDL Persons
            try:
                 self.cursor.execute("SELECT 1 FROM Persons WHERE name = %s", (name,))
                 if self.cursor.fetchone(): QMessageBox.warning(self, "Tên trùng lặp", f"Tên '{name}' đã tồn tại trong CSDL Persons."); return
            except psycopg2.errors.UndefinedTable: print("[Warning] Bảng Persons không tồn tại, bỏ qua kiểm tra tên trùng DB.")

            # Xử lý thư mục
            if os.path.exists(employee_dir):
                 reply = QMessageBox.question(self,"Thư mục tồn tại", f"Thư mục '{name}' đã tồn tại. Ghi đè ảnh?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                 if reply == QMessageBox.No: return
            else: os.makedirs(employee_dir); print(f"Đã tạo thư mục mới: {employee_dir}")

            # Lưu ảnh mặt đã crop
            print(f"Đang lưu {len(self.captured_faces)} ảnh mặt cho {name}...")
            saved_count = 0
            for i, face_img in enumerate(self.captured_faces):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                file_name = f"{name}_{timestamp}_{i+1}.jpg"
                file_path = os.path.join(employee_dir, file_name)
                if cv2.imwrite(file_path, face_img): saved_count += 1
                else: print(f" ! Lỗi khi lưu: {file_path}")
            print(f" -> Đã lưu thành công {saved_count}/{len(self.captured_faces)} ảnh.")

            # Thêm vào bảng Persons
            try:
                reg_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                self.cursor.execute("INSERT INTO Persons (name, registration_date) VALUES (%s, %s)", (name, reg_date))
                self.conn.commit()
                print(f"Đã thêm '{name}' vào CSDL Persons.")
            except Error as e_person:
                print(f"[ERROR] Lỗi khi thêm vào bảng Persons: {e_person}"); self.conn.rollback(); QMessageBox.critical(self,"Lỗi CSDL", f"Lỗi khi ghi vào bảng Persons:\n{e_person}"); return

            # Thông báo cho MainWindow cập nhật embeddings
            if self.parent_window and hasattr(self.parent_window, 'reload_models_and_embeddings'):
                 print("Yêu cầu MainWindow tải lại embeddings...")
                 self.parent_window.reload_models_and_embeddings()

            QMessageBox.information(self, "Thành công", f"Đã thêm nhân viên '{name}' và lưu {saved_count} ảnh thành công!")
            self.accept() # Đóng cửa sổ


        except Error as e:

            print(f"[ERROR] Lỗi CSDL khi kiểm tra/thêm nhân viên {name}: {e}\n{traceback.format_exc()}")

            QMessageBox.critical(self, "Lỗi CSDL", f"Lỗi CSDL:\n{e}")

            try:

                if self.conn:
                    print("[AddEmployeeWindow] Rolling back DB transaction...")

                    self.conn.rollback()


            except Error as rb_e:

                print(f"[ERROR] Lỗi khi rollback CSDL: {rb_e}")

    def closeEvent(self, event):
        """Đóng webcam và dừng timers khi cửa sổ này đóng."""
        print("[AddEmployeeWindow] Closing...")
        if hasattr(self, 'timer_display') and self.timer_display is not None: self.timer_display.stop()
        if hasattr(self, 'timer_capture') and self.timer_capture is not None: self.timer_capture.stop()
        if hasattr(self, 'cap') and self.cap.isOpened(): self.cap.release()
        print("[AddEmployeeWindow] Webcam released.")
        event.accept()


# ============================================
# Lớp Cửa sổ Chính (MainWindow)
# ============================================
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hệ Thống Chấm Công")
        self.setMinimumSize(400, 250)

        self.conn = None
        self.cursor = None
        self.face_detector = None
        self.mask_model = None
        self.recognition_model = None
        self.known_embeddings = []
        self.known_names = []
        self.attendance_window = None
        self.add_employee_window = None

        self.status_label = QLabel("Đang khởi tạo...")
        self.status_label.setAlignment(Qt.AlignCenter)
        status_font = QFont(); status_font.setPointSize(10)
        self.status_label.setFont(status_font)

        self.init_database()
        models_loaded_ok = False
        if self.conn:
            models_loaded_ok = self.load_models_and_embeddings()

        layout = QVBoxLayout(self)
        button_layout = QHBoxLayout()
        self.btn_attendance = QPushButton("Chấm Công")
        self.btn_attendance.setMinimumHeight(50)
        self.btn_attendance.clicked.connect(self.open_attendance_window)
        self.btn_add_employee = QPushButton("Thêm Nhân Viên")
        self.btn_add_employee.setMinimumHeight(50)
        self.btn_add_employee.clicked.connect(self.open_add_employee_window) # Đã sửa ở code trước

        buttons_enabled = self.conn is not None and models_loaded_ok
        self.btn_attendance.setEnabled(buttons_enabled)
        self.btn_add_employee.setEnabled(buttons_enabled)
        self.status_label.setText("Sẵn sàng" if buttons_enabled else "Lỗi: Không thể tải CSDL hoặc Model")
        if not buttons_enabled: print("[WARNING] Buttons disabled due to initialization error.")

        button_layout.addWidget(self.btn_attendance)
        button_layout.addWidget(self.btn_add_employee)
        layout.addLayout(button_layout)
        layout.addWidget(self.status_label)

    def init_database(self):
        """Kết nối CSDL PostgreSQL và tạo bảng Persons nếu chưa có."""
        try:
            print(f"[Main] Connecting to PostgreSQL: host={DB_HOST}, dbname={DB_NAME}")
            self.conn = psycopg2.connect(host=DB_HOST, port=DB_PORT, dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD)
            self.cursor = self.conn.cursor()
            print("[Main] PostgreSQL connected.")
            # Tạo bảng Persons nếu chưa có
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS Persons (
                    person_id SERIAL PRIMARY KEY,
                    name TEXT NOT NULL UNIQUE,
                    registration_date TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
            """)
            self.conn.commit()
            print("[Main] Checked/Created 'Persons' table.")
            # Tạo bảng DiemDanh nếu chưa có (để đảm bảo tồn tại)
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS DiemDanh (
                    log_id SERIAL PRIMARY KEY,
                    ten TEXT NOT NULL,
                    thoigianvao TIMESTAMP NOT NULL,
                    thoigianra TIMESTAMP NULL
                )
            """)
            self.conn.commit()
            print("[Main] Checked/Created 'DiemDanh' table.")

        except OperationalError as e: self.conn = None; print(f"[Main] DB Connection Error: {e}"); QMessageBox.critical(self, "DB Error", f"Connection Error:\n{e}")
        except Error as e: self.conn = None; print(f"[Main] DB Error: {e}"); QMessageBox.critical(self, "DB Error", f"DB Operation Error:\n{e}")
        except Exception as ex: self.conn = None; print(f"[Main] Unknown DB Init Error: {ex}"); QMessageBox.critical(self, "Unknown Error", f"DB Init Error:\n{ex}")

    def load_models_and_embeddings(self):
        """Tải các mô hình AI và tạo/tải CSDL embedding."""
        print("="*10 + " Loading AI Models & Embeddings " + "="*10)
        models_ok = True
        try:
            # Load face detector
            print("[Main] Loading Face Detector...")
            if not os.path.exists(FACE_DETECTOR_PROTOTXT) or not os.path.exists(FACE_DETECTOR_CAFFEMODEL): raise FileNotFoundError("Detector files missing")
            self.face_detector = cv2.dnn.readNetFromCaffe(FACE_DETECTOR_PROTOTXT, FACE_DETECTOR_CAFFEMODEL)
            print("[Main] Face Detector loaded.")

            # Load mask model
            print("[Main] Loading Mask Model...")
            if not os.path.exists(MASK_MODEL_PATH): raise FileNotFoundError(f"Mask Model missing: {MASK_MODEL_PATH}")
            self.mask_model = tf.keras.models.load_model(MASK_MODEL_PATH, compile=False)
            print("[Main] Mask Model loaded.")

            # Load recognition model
            print("[Main] Loading Recognition Model...")
            if not os.path.exists(RECOGNITION_MODEL_PATH): raise FileNotFoundError(f"Recognition Model missing: {RECOGNITION_MODEL_PATH}")
            custom_objects={'L2NormalizationLayer': L2NormalizationLayer}
            self.recognition_model = tf.keras.models.load_model(RECOGNITION_MODEL_PATH, custom_objects=custom_objects, compile=False)
            print("[Main] Recognition Model loaded.")

            # Create embeddings
            print("[Main] Creating known face embeddings database...")
            if os.path.exists(KNOWN_FACES_DIR) and os.path.isdir(KNOWN_FACES_DIR): # Kiểm tra thư mục tồn tại
                person_names = sorted([d for d in os.listdir(KNOWN_FACES_DIR) if os.path.isdir(os.path.join(KNOWN_FACES_DIR, d))])
                temp_embeddings = []; temp_names = []; total_images_processed = 0
                for person_name in person_names:
                     person_dir = os.path.join(KNOWN_FACES_DIR, person_name)
                     embeddings_for_person = []; img_count_person = 0
                     for image_name in os.listdir(person_dir):
                          image_path = os.path.join(person_dir, image_name)
                          try:
                              img = cv2.imread(image_path)
                              if img is None: continue
                              face_crop = img # Giả sử ảnh đã crop khi lưu
                              if face_crop.size > 0:
                                   face_resized = cv2.resize(face_crop, RECOGNITION_INPUT_SIZE)
                                   face_array = img_to_array(face_resized)
                                   face_array = np.expand_dims(face_array, axis=0)
                                   face_preprocessed = preprocess_facenet(face_array)
                                   embedding = self.recognition_model.predict(face_preprocessed, verbose=0)[0]
                                   embeddings_for_person.append(embedding)
                                   img_count_person += 1
                                   total_images_processed += 1
                          except Exception as e_img: print(f"[Warning] Error processing image {image_path}: {e_img}"); pass
                     if embeddings_for_person:
                          avg_embedding = np.mean(embeddings_for_person, axis=0)
                          temp_embeddings.append(avg_embedding); temp_names.append(person_name)
                self.known_embeddings = temp_embeddings; self.known_names = temp_names
                if not self.known_embeddings: print("[Main][Warning] Known embeddings DB is empty.")
                else: print(f"[Main] Embeddings DB created/updated for {len(self.known_names)} people from {total_images_processed} images.")
            else: print(f"[Main][Warning] Known faces directory not found or is not a directory: {KNOWN_FACES_DIR}")

        except Exception as e:
             print(f"[Main][ERROR] Loading models/embeddings error: {e}\n{traceback.format_exc()}")
             QMessageBox.critical(self, "AI Model Error", f"Could not load AI models or create embeddings:\n{e}")
             self.face_detector = self.mask_model = self.recognition_model = None
             self.known_embeddings = []; self.known_names = []
             models_ok = False
        return models_ok

    @Slot()
    def reload_models_and_embeddings(self):
        """Tải lại CSDL embedding sau khi thêm người mới."""
        print("[Main] Reloading embeddings...")
        # Chỉ cần chạy lại phần tạo embedding, không cần load lại model
        models_loaded_ok = self.load_models_and_embeddings() # Gọi lại để load và tính lại embedding
        buttons_enabled = self.conn is not None and models_loaded_ok
        self.btn_attendance.setEnabled(buttons_enabled)
        self.btn_add_employee.setEnabled(buttons_enabled)
        self.status_label.setText("Sẵn sàng" if buttons_enabled else "Lỗi: Không thể tải CSDL hoặc Model")
        if buttons_enabled: print("[Main] Buttons re-enabled after reload.")
        else: print("[Main][Warning] Buttons remain disabled after reload.")

    @Slot()
    def open_attendance_window(self):
        """Mở cửa sổ chấm công và kết nối tín hiệu."""
        if not all([self.conn, self.cursor, self.face_detector, self.mask_model, self.recognition_model]):
             QMessageBox.warning(self, "Chưa sẵn sàng", "Chưa thể chấm công do lỗi CSDL hoặc Model."); return
        if not self.known_embeddings: # Kiểm tra thêm có embedding không
              QMessageBox.warning(self, "Chưa sẵn sàng", "Chưa có dữ liệu người quen để chấm công. Vui lòng thêm nhân viên trước.")
              return

        print("Opening Attendance Window...")
        self.status_label.setText("Opening attendance camera...")
        # Tạo mới mỗi lần để đảm bảo trạng thái sạch
        self.attendance_window = AttendanceWindow(
            parent=self, db_conn=self.conn, db_cursor=self.cursor,
            face_detector=self.face_detector, mask_model=self.mask_model,
            recognition_model=self.recognition_model,
            known_embeddings=self.known_embeddings, known_names=self.known_names
        )
        self.attendance_window.attendance_recorded.connect(self.handle_attendance_success)
        self.attendance_window.rejected.connect(self.handle_attendance_cancel)
        self.attendance_window.show()

    @Slot()
    def open_add_employee_window(self):
        """Hỏi người dùng về kính và mở cửa sổ thêm nhân viên."""
        if not self.conn or not self.cursor or not self.face_detector:
             QMessageBox.warning(self, "Chưa sẵn sàng", "Không thể thêm nhân viên do lỗi CSDL hoặc Face Detector."); return

        reply = QMessageBox.question(self, "Xác nhận Đeo Kính", "Bạn có đang đeo kính không?", QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel, QMessageBox.Cancel)
        if reply == QMessageBox.Cancel: return
        wearing_glasses_choice = (reply == QMessageBox.Yes)
        print(f"[Main] User selected: Wearing glasses = {wearing_glasses_choice}.")

        # Tạo mới mỗi lần
        print("Opening Add Employee Window...")
        self.add_employee_window = AddEmployeeWindow(
            parent=self, db_conn=self.conn, db_cursor=self.cursor,
            known_faces_dir=KNOWN_FACES_DIR, face_detector_instance=self.face_detector,
            wearing_glasses=wearing_glasses_choice
        )
        self.add_employee_window.show()

    @Slot(tuple)
    def handle_attendance_success(self, result_tuple):
        """Xử lý khi nhận được tín hiệu chấm công thành công."""
        try:
            recognized_name, status = result_tuple  # status sẽ là "Check-in" hoặc "Check-out"
            print(f"[Main] Attendance success signal received: {status} for {recognized_name}")

            # --- DỊCH TRẠNG THÁI ---
            if status == "Check-in":
                status_viet = "Chấm công vào"
            elif status == "Check-out":
                status_viet = "Chấm công ra"
            else:
                status_viet = status  # Giữ nguyên nếu là trạng thái khác (ví dụ: lỗi)

            # --- TẠO THÔNG BÁO TIẾNG VIỆT ---
            success_message = f"Đã ghi nhận {status_viet} thành công cho: {recognized_name}"  # <<< SỬA ĐỔI Ở ĐÂY

            self.status_label.setText(success_message)  # Cập nhật label trạng thái
            # Hiển thị hộp thoại thông báo
            QMessageBox.information(self, f"{status_viet} Thành Công", success_message)  # <<< SỬA TIÊU ĐỀ HỘP THOẠI

        except Exception as e:
            print(f"[ERROR] Handling success signal error: {e}")
            self.status_label.setText("Lỗi xử lý kết quả chấm công")

    @Slot()
    def handle_attendance_cancel(self):
        """Xử lý khi cửa sổ chấm công bị đóng (không thành công hoặc do người dùng)."""
        print("[Main] Attendance window closed/cancelled.")
        self.status_label.setText("Ready")

    def close_db_connection(self):
        """Đóng kết nối CSDL."""
        if self.conn:
            try: self.conn.close(); print("[Main] PostgreSQL connection closed."); self.conn = None; self.cursor = None
            except Error as e: print(f"[Main] Error closing PostgreSQL connection: {e}")

    def closeEvent(self, event):
        """Đóng CSDL khi cửa sổ chính đóng."""
        print("[Main] Closing main application...")
        # Đóng các cửa sổ con nếu đang mở
        if self.attendance_window and self.attendance_window.isVisible(): self.attendance_window.close()
        if self.add_employee_window and self.add_employee_window.isVisible(): self.add_employee_window.close()
        self.close_db_connection()
        event.accept()

# --- Chạy ứng dụng ---
if __name__ == '__main__':
    # Kiểm tra file model
    required_files = [FACE_DETECTOR_PROTOTXT, FACE_DETECTOR_CAFFEMODEL, MASK_MODEL_PATH, RECOGNITION_MODEL_PATH]
    files_ok = True
    for f in required_files:
        if not os.path.exists(f): print(f"[CRITICAL ERROR] Required file missing: {f}"); files_ok = False
    if not files_ok:
         # Hiển thị lỗi và thoát nếu thiếu file model
         app_temp = QApplication.instance() or QApplication(sys.argv)
         msg_box = QMessageBox()
         msg_box.setIcon(QMessageBox.Critical); msg_box.setText("Missing required model files. Cannot start.");
         msg_box.setWindowTitle("Initialization Error"); msg_box.setStandardButtons(QMessageBox.Ok); msg_box.exec()
         sys.exit(1)

    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec())