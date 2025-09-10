
import pyttsx3
import os
import cv2
import numpy as np
import pickle
import mysql.connector
from datetime import datetime, time, timedelta, date
import time as time_module
import logging
import hashlib
import threading
import queue
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests
import pandas as pd
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash, send_file, Response
from flask_socketio import SocketIO, emit
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from functools import wraps
import base64
from io import BytesIO
from PIL import Image
import uuid
import schedule
import atexit
import plotly
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# --- EARLY LOGGING SETUP (injected) ---
import logging as _logging_for_patch
if not _logging_for_patch.getLogger().handlers:
    _logging_for_patch.basicConfig(
        level=_logging_for_patch.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
logger = _logging_for_patch.getLogger(__name__)
# --- end early logging setup ---

# --- Helper utilities injected ---
def close_cursor_safe(cursor):
    try:
        if cursor is not None:
            cursor.close()
    except Exception:
        pass

def close_conn_safe(conn):
    try:
        if conn is not None:
            close_conn_safe(conn)
    except Exception:
        pass


def normalize_status(status):
    if not status:
        return status
    s = str(status).strip().lower()
    if s == "present":
        return "Present"
    elif s == "absent":
        return "Absent"
    elif s == "leave":
        return "Leave"
    else:
        return status

MIN_IMAGES_PER_PERSON = 3
RECOGNITION_THRESHOLD = 0.8
MODEL_TRAINED = False

def safe_index(seq, idx):
    try:
        if seq is None:
            return None
        if idx < 0:
            if abs(idx) > len(seq):
                return None
        return seq[idx]
    except Exception:
        return None

# Register safe_index filter in Jinja if app exists
try:
    app
except NameError:
    app = None

if app is not None:
    try:
        app.jinja_env.filters['safe_index'] = safe_index
    except Exception:
        pass

# Make MODEL_TRAINED visible to templates
try:
    if app is not None:
        @app.context_processor
        def inject_model_status():
            try:
                return dict(MODEL_TRAINED=MODEL_TRAINED, MIN_IMAGES_PER_PERSON=MIN_IMAGES_PER_PERSON)
            except Exception:
                return dict(MODEL_TRAINED=False, MIN_IMAGES_PER_PERSON=MIN_IMAGES_PER_PERSON)
except Exception:
    pass

# --- end helpers ---


# Try to import AI models, fallback to basic if not available
try:
    from mtcnn import MTCNN
    from keras_facenet import FaceNet
    from sklearn.svm import SVC
    from sklearn.preprocessing import LabelEncoder
    AI_AVAILABLE = True
    print("✅ AI models loaded successfully!")
except ImportError:
    AI_AVAILABLE = False
    print("⚠️ AI models not available. Using basic face detection.")

# Flask App Configuration
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret-for-local')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB max file size

# Initialize Flask extensions
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Initialize text-to-speech engine
try:
    tts_engine = pyttsx3.init()
    tts_engine.setProperty('rate', 150)  # Speed of speech
    tts_engine.setProperty('volume', 0.9)  # Volume level
    # Get available voices and set a preferred one if available
    voices = tts_engine.getProperty('voices')
    if voices:
        tts_engine.setProperty('voice', voices[0].id)  # Use the first available voice
except Exception as e:
    logger.error(f"Failed to initialize TTS engine: {e}")
    tts_engine = None
    

# Attendance verification tracking
checked_in_employees = set()
last_checkin_reset_date = date.today()

# Database Configuration
import os

# Database configuration via environment variables (do NOT hard-code in production)
db_config = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'user': os.getenv('DB_USER', 'root'),
    'password': os.getenv('DB_PASSWORD', '0727'),
    'database': os.getenv('DB_NAME', 'attendance_system04')
}

# Company timing configuration
OPENING_TIME = time(11, 0)  # 11:00 AM
LATE_CUTOFF = time(13, 0)   # 1:00 PM
LUNCH_START = time(13, 0)   # 1:00 PM
LUNCH_END = time(14, 0)     # 2:00 PM
HALF_DAY_CUTOFF = time(17, 0)  # 5:00 PM
CLOSING_TIME = time(18, 0)  # 6:00 PM
OVERTIME_START = time(18, 0)  # 6:00 PM

# CCTV Configuration - Multiple camera support
# CCTV Configuration - Multiple camera support
CCTV_CAMERAS = {
    'laptop_camera': {
        'name': 'Laptop Camera',
        'rtsp_url': 'rtsp://Lobby:Techno%4012345@192.168.1.3:554/onvif2?tcp',  # 0 represents the default laptop/webcam
        'location': 'Laptop Webcam',
        'status': 'active'
    }
}

# Recognition system state
recognition_active = False
recognition_thread =  None
recognition_start_time = time(10, 0)  # 10 AM
recognition_end_time = time(21, 0)    # 9 PM
control_queue = queue.Queue()
live_stream_queue = queue.Queue(maxsize=1)
recognition_stats = {
    'faces_detected': 0,
    'employees_recognized': 0,
    'unknown_faces': 0,
    'last_recognition': None
}

# Initialize AI models if available
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/premium_app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create necessary directories
os.makedirs('uploads', exist_ok=True)
os.makedirs('uploads/faces', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('logs', exist_ok=True)
os.makedirs('backups', exist_ok=True)
os.makedirs('exports', exist_ok=True)

def get_db_connection():
    """Create and return a database connection"""
    try:
        conn = mysql.connector.connect(**db_config)
        return conn
    except mysql.connector.Error as err:
        logger.error(f"Database connection error: {err}")
        return None
    
def announce_attendance(employee_name, status="present"):
    """Announce employee attendance using text-to-speech based on time rules"""
    if not tts_engine:
        logger.warning("TTS engine not available for announcement")
        return
    
    try:
        # Reset daily check-ins if it's a new day
        reset_daily_checkins()
        
        current_time = datetime.now().time()
        checkin_start = time(9, 0)   # 9:00 AM
        checkin_end = time(16, 0)     # 4:00 PM
        checkout_end = time(22, 0)    # 10:00 PM
        
        # Check if we're within attendance hours (9 AM - 10 PM)
        if not (checkin_start <= current_time <= checkout_end):
            logger.info(f"Outside attendance hours: {current_time}")
            return
        
        # Determine if this is check-in or check-out period
        is_checkin_period = checkin_start <= current_time < checkin_end
        is_checkout_period = checkin_end <= current_time <= checkout_end
        
        announcement = None
        
        if is_checkin_period:
            # Check-in logic: announce only once per day
            if employee_name not in checked_in_employees:
                announcement = f"{employee_name} Check-In"
                checked_in_employees.add(employee_name)
                logger.info(f"Check-in announced for {employee_name}")
            else:
                logger.info(f"{employee_name} already checked in today, no announcement")
        
        elif is_checkout_period:
            # Check-out logic: announce every time
            announcement = f"{employee_name} Check-Out"
            logger.info(f"Check-out announced for {employee_name}")
        
        # Make the announcement if needed
        if announcement:
            # Run announcement in a separate thread to avoid blocking
            def run_announcement():
                try:
                    tts_engine.say(announcement)
                    tts_engine.runAndWait()
                    logger.info(f"Announced: {announcement}")
                except Exception as e:
                    logger.error(f"Error in TTS announcement: {e}")
            
            threading.Thread(target=run_announcement, daemon=True).start()
            
    except Exception as e:
        logger.error(f"Error in attendance announcement: {e}")

if AI_AVAILABLE:
    try:
        detector = MTCNN(
            min_face_size=60,
            steps_threshold=[0.6, 0.7, 0.8],
            scale_factor=0.8
        )
        embedder = FaceNet()
        svm_model = SVC(kernel='linear', probability=True)
        label_encoder = LabelEncoder()
        print("✅ AI models loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading AI models: {e}")
        AI_AVAILABLE = False
        # Auto-train model if there are embeddings but no model
if AI_AVAILABLE:
    conn = get_db_connection()
    if conn:
        try:
            cursor = conn.cursor(dictionary=True)
            cursor.execute("SELECT COUNT(*) as count FROM face_embeddings")
            embedding_count = cursor.fetchone()['count']
            
            # Check if we have embeddings but no valid model
            if embedding_count > 0 and (not hasattr(svm_model, 'classes_') or len(svm_model.classes_) <= 1):
                logger.info("Auto-training model on startup")
                success, message = train_model()
                logger.info(f"Auto-training result: {message}")
                
        except Exception as e:
            logger.error(f"Error checking for auto-training: {e}")
        finally:
            if conn.is_connected():
                close_cursor_safe(cursor)
                close_conn_safe(conn)


def init_database():
    """Initialize database structure"""
    conn = None
    try:
        conn = mysql.connector.connect(
            host=db_config['host'],
            user=db_config['user'],
            password=db_config['password']
        )
        cursor = conn.cursor()
        
        # Create database if not exists
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {db_config['database']}")
        cursor.execute(f"USE {db_config['database']}")
        
        # Enhanced tables with additional fields
        tables = [
            """
            CREATE TABLE IF NOT EXISTS employees (
                id INT AUTO_INCREMENT PRIMARY KEY,
                employee_id VARCHAR(20) UNIQUE NOT NULL,
                name VARCHAR(100) NOT NULL,
                position VARCHAR(100),
                department VARCHAR(100),
                email VARCHAR(100) UNIQUE,
                phone VARCHAR(20),
                join_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                status ENUM('active', 'inactive', 'suspended') DEFAULT 'active',
                password_hash VARCHAR(255) NOT NULL,
                profile_image VARCHAR(255),
                emergency_contact VARCHAR(100),
                emergency_phone VARCHAR(20),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS face_embeddings (
                id INT AUTO_INCREMENT PRIMARY KEY,
                employee_id INT NOT NULL,
                embedding BLOB NOT NULL,
                quality_score FLOAT DEFAULT 0.0,
                capture_location VARCHAR(100),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (employee_id) REFERENCES employees(id) ON DELETE CASCADE
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS attendance (
                id INT AUTO_INCREMENT PRIMARY KEY,
                employee_id INT NOT NULL,
                check_in DATETIME NOT NULL,
                check_out DATETIME NULL,
                confidence FLOAT NOT NULL,
                status ENUM('present', 'late', 'half_day', 'overtime', 'leave', 'absent') DEFAULT 'present',
                location VARCHAR(100),
                device_type ENUM('cctv', 'mobile', 'web') DEFAULT 'cctv',
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (employee_id) REFERENCES employees(id) ON DELETE CASCADE
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS system_logs (
                id INT AUTO_INCREMENT PRIMARY KEY,
                level VARCHAR(20) NOT NULL,
                message TEXT NOT NULL,
                user_id INT NULL,
                ip_address VARCHAR(45),
                user_agent TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS admin_users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                username VARCHAR(50) NOT NULL UNIQUE,
                password_hash VARCHAR(255) NOT NULL,
                role ENUM('super_admin', 'admin', 'manager') DEFAULT 'admin',
                permissions JSON,
                last_login DATETIME,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS leave_requests (
                id INT AUTO_INCREMENT PRIMARY KEY,
                employee_id INT NOT NULL,
                leave_type ENUM('sick', 'casual', 'annual', 'maternity', 'paternity', 'other') NOT NULL,
                start_date DATE NOT NULL,
                end_date DATE NOT NULL,
                reason TEXT NOT NULL,
                status ENUM('pending', 'approved', 'rejected') DEFAULT 'pending',
                approved_by INT NULL,
                approved_at DATETIME NULL,
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                FOREIGN KEY (employee_id) REFERENCES employees(id) ON DELETE CASCADE,
                FOREIGN KEY (approved_by) REFERENCES admin_users(id) ON DELETE SET NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS non_working_days (
                id INT AUTO_INCREMENT PRIMARY KEY,
                date DATE NOT NULL UNIQUE,
                description VARCHAR(255) NOT NULL,
                type ENUM('weekend', 'holiday', 'company_event') NOT NULL,
                is_paid BOOLEAN DEFAULT TRUE
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS cctv_cameras (
                id INT AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(100) NOT NULL,
                location VARCHAR(100) NOT NULL,
                rtsp_url VARCHAR(500) NOT NULL,
                status ENUM('active', 'inactive', 'maintenance') DEFAULT 'active',
                last_maintenance DATE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS attendance_alerts (
                id INT AUTO_INCREMENT PRIMARY KEY,
                employee_id INT NOT NULL,
                alert_type ENUM('late', 'absent', 'overtime', 'irregular') NOT NULL,
                message TEXT NOT NULL,
                status ENUM('active', 'acknowledged', 'resolved') DEFAULT 'active',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                resolved_at DATETIME NULL,
                FOREIGN KEY (employee_id) REFERENCES employees(id) ON DELETE CASCADE
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS system_settings (
                id INT AUTO_INCREMENT PRIMARY KEY,
                setting_key VARCHAR(100) UNIQUE NOT NULL,
                setting_value TEXT NOT NULL,
                description TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
            )
            """,
                        """
            CREATE TABLE IF NOT EXISTS checkin_requests (
                id INT AUTO_INCREMENT PRIMARY KEY,
                employee_id INT NOT NULL,
                request_date DATE NOT NULL,
                reason TEXT,
                status ENUM('pending', 'approved', 'rejected') DEFAULT 'pending',
                reviewed_by INT NULL,
                reviewed_at DATETIME NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (employee_id) REFERENCES employees(id) ON DELETE CASCADE,
                FOREIGN KEY (reviewed_by) REFERENCES admin_users(id) ON DELETE SET NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS model_training_history (
                id INT AUTO_INCREMENT PRIMARY KEY,
                training_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                employees_count INT DEFAULT 0,
                embeddings_count INT DEFAULT 0,
                status ENUM('success', 'failed') DEFAULT 'success',
                message TEXT,
                duration_seconds FLOAT DEFAULT 0
            )
            """
        ]
        
        for table_sql in tables:
            cursor.execute(table_sql)
        
        # Insert default admin user
        cursor.execute("SELECT COUNT(*) FROM admin_users WHERE username = 'admin'")
        if cursor.fetchone()[0] == 0:
            default_password = "Admin@123"
            password_hash = generate_password_hash(default_password)
            cursor.execute(
                "INSERT INTO admin_users (username, password_hash, role, permissions) VALUES (%s, %s, %s, %s)",
                ("admin", password_hash, "super_admin", json.dumps({"all": True}))
            )
            print(f"\nCreated default admin user: username=admin, password={default_password}")
            print("Please change this password immediately in the admin settings")
        
        # Insert default system settings
        default_settings = [
            ("company_name", "Skyhighes Technologies", "Company name for the system"),
            ("opening_time", "11:00", "Company opening time"),
            ("closing_time", "18:00", "Company closing time"),
            ("late_threshold", "13:00", "Late arrival threshold"),
            ("recognition_start_time", "10:00", "Face recognition start time"),
            ("recognition_end_time", "21:00", "Face recognition end time"),
            ("max_face_images", "15", "Maximum face images per employee"),
            ("min_confidence", "0.7", "Minimum confidence for face recognition"),
            ("backup_frequency", "daily", "System backup frequency"),
            ("email_notifications", "true", "Enable email notifications"),
            ("sms_notifications", "false", "Enable SMS notifications")
        ]
        
        for key, value, description in default_settings:
            cursor.execute(
                "INSERT IGNORE INTO system_settings (setting_key, setting_value, description) VALUES (%s, %s, %s)",
                (key, value, description)
            )
        
        # Insert CCTV cameras
        for camera_id, camera_info in CCTV_CAMERAS.items():
            cursor.execute(
                "INSERT IGNORE INTO cctv_cameras (name, location, rtsp_url, status) VALUES (%s, %s, %s, %s)",
                (camera_info['name'], camera_info['location'], camera_info['rtsp_url'], camera_info['status'])
            )
        
        conn.commit()
        logger.info("Database initialized successfully")
        print("✅ Database initialized successfully")
        
    except mysql.connector.Error as err:
        logger.error(f"Database initialization error: {err}")
        print(f"❌ Database error: {err}")
    except Exception as e:
        logger.error(f"Unexpected error during DB init: {e}")
        print(f"❌ Unexpected error: {e}")
    finally:
        if conn and conn.is_connected():
            close_cursor_safe(cursor)
            close_conn_safe(conn)

def load_model():
    """Load or initialize the face recognition model"""
    global svm_model, label_encoder
    
    try:
        if os.path.exists('models/svm_model.pkl') and os.path.exists('models/label_encoder.pkl'):
            with open('models/svm_model.pkl', 'rb') as f:
                svm_model = pickle.load(f)
            with open('models/label_encoder.pkl', 'rb') as f:
                label_encoder = pickle.load(f)
            
            # Verify the model is valid and not a dummy model
            if hasattr(svm_model, 'classes_') and len(svm_model.classes_) > 0 and not any('dummy' in str(cls) for cls in svm_model.classes_):
                logger.info(f"Model loaded successfully with {len(svm_model.classes_)} classes")
                return True
            else:
                logger.warning("Model files contain dummy data. Initializing new model.")
                return initialize_new_model()
        else:
            logger.warning("Model files not found. Initializing new model.")
            return initialize_new_model()
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return initialize_new_model()

def reset_daily_checkins():
    """Reset the daily check-in tracking at midnight"""
    global checked_in_employees, last_checkin_reset_date
    current_date = date.today()
    
    if current_date != last_checkin_reset_date:
        checked_in_employees.clear()
        last_checkin_reset_date = current_date
        logger.info("Daily check-in tracking reset")
        
def initialize_new_model():
    """Initialize a new model without dummy data"""
    global svm_model, label_encoder
    try:
        logger.info("Initializing new model without dummy data")
        # Create empty model structures
        label_encoder = LabelEncoder()
        svm_model = SVC(kernel='linear', probability=True)
        logger.info("Initialized empty model")
        return True
    except Exception as e:
        logger.error(f"Error initializing model: {e}")
        return False

def save_model():
    """Save the current model to disk"""
    try:
        os.makedirs('models', exist_ok=True)
        with open('models/svm_model.pkl', 'wb') as f:
            pickle.dump(svm_model, f)
        with open('models/label_encoder.pkl', 'wb') as f:
            pickle.dump(label_encoder, f)
        logger.info("Model saved successfully")
        return True
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        return False

def get_min_confidence():
    """Get min_confidence from system_settings or use default"""
    conn = get_db_connection()
    if not conn:
        logger.warning("Database connection failed, using default min_confidence: 0.75")
        return 0.75
    
    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT setting_value FROM system_settings WHERE setting_key = 'min_confidence'")
        result = cursor.fetchone()
        
        if result and result['setting_value']:
            try:
                return float(result['setting_value'])
            except ValueError:
                logger.error(f"Invalid min_confidence value in database: {result['setting_value']}, using default: 0.75")
                return 0.75
        else:
            logger.info("min_confidence not found in database, using default: 0.75")
            return 0.75
            
    except mysql.connector.Error as err:
        logger.error(f"Error getting min_confidence: {err}")
        return 0.75
    except Exception as e:
        logger.error(f"Unexpected error getting min_confidence: {e}")
        return 0.75
    finally:
        if conn and conn.is_connected():
            close_cursor_safe(cursor)
            close_conn_safe(conn)
            
def get_embedding(face_img):
    """Generate FaceNet embedding for a face image"""
    try:
        if not AI_AVAILABLE:
            return None
        face_img = cv2.resize(face_img, (160, 160))
        embedding = embedder.embeddings(np.expand_dims(face_img, axis=0))[0]
        return embedding
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        return None

def is_non_working_day(check_date):
    """Check if a date is a non-working day"""
    conn = get_db_connection()
    if not conn:
        return False
    
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT COUNT(*) FROM non_working_days WHERE date = %s",
            (check_date,)
        )
        count = cursor.fetchone()[0]
        return count > 0
    except mysql.connector.Error as err:
        logger.error(f"Error checking non-working day: {err}")
        return False
    finally:
        if conn.is_connected():
            close_cursor_safe(cursor)
            close_conn_safe(conn)

def determine_attendance_status(check_in_time):
    """Determine attendance status based on company timing rules"""
    check_in_time = check_in_time.time()
    
    # Lunch break - no attendance recording
    if LUNCH_START <= check_in_time < LUNCH_END:
        return None
    
    # Present (on time)
    if OPENING_TIME <= check_in_time < LATE_CUTOFF:
        return 'present'
    
    # Late
    if LATE_CUTOFF <= check_in_time < LUNCH_START:
        return 'late'
    
    # Half day
    if LUNCH_END <= check_in_time < HALF_DAY_CUTOFF:
        return 'half_day'
    
    # Overtime
    if check_in_time >= OVERTIME_START:
        return 'overtime'
    
    # Before opening or after closing
    return 'irregular'

def process_cctv_frame(frame, camera_location):
    """Improved process function:
    - Resizes frame for speed
    - Handles multiple faces per frame
    - Prevents duplicate attendance entries per employee per day
    - Uses safe DB handling and logging
    """
    if not AI_AVAILABLE:
        return frame

    min_confidence = get_min_confidence()
    try:
        # Work on a smaller copy for detection to improve speed
        h, w = frame.shape[:2]
        scale = 0.5 if max(h, w) > 800 else 1.0
        small_frame = cv2.resize(frame, (int(w*scale), int(h*scale)))
        rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Detect faces on the small frame
        try:
            faces = detector.detect_faces(rgb_small)
        except Exception as e:
            logger.error(f"Face detection error: {e}")
            return frame

        if not faces:
            return frame

        # Prepare mapping of employee_id -> db id
        conn = get_db_connection()
        employee_map = {}
        if conn:
            try:
                cursor = conn.cursor(dictionary=True)
                cursor.execute("SELECT id, employee_id FROM employees WHERE status = 'active'")
                rows = cursor.fetchall()
                for r in rows:
                    employee_map[str(r['employee_id'])] = r['id']
            except Exception as e:
                logger.error(f"Error loading employee map: {e}")
            finally:
                if conn and conn.is_connected():
                    close_cursor_safe(cursor)
                    close_conn_safe(conn)

        display_frame = frame.copy()
        now = datetime.now()

        # Process each detected face (scale coordinates back to original frame)
        for face in faces:
            try:
                x, y, w_box, h_box = face['box']
                # scale coordinates to original frame size
                x = int(x / scale); y = int(y / scale)
                w_box = int(w_box / scale); h_box = int(h_box / scale)
                x, y = max(0, x), max(0, y)
                w_box = min(w_box, frame.shape[1] - x - 1)
                h_box = min(h_box, frame.shape[0] - y - 1)
                if w_box < 20 or h_box < 20:
                    continue
                face_img = cv2.cvtColor(frame[y:y+h_box, x:x+w_box], cv2.COLOR_BGR2RGB)
                embedding = get_embedding(face_img)
                if embedding is None:
                    continue

                # Predict
                try:
                    proba = svm_model.predict_proba([embedding])[0]
                    max_proba = float(np.max(proba))
                    pred_id_enc = svm_model.predict([embedding])[0]
                    predicted_label = label_encoder.inverse_transform([pred_id_enc])[0]
                except Exception as e:
                    logger.error(f"Prediction error: {e}")
                    continue

                # Check confidence threshold
                if max_proba < (min_confidence if 'min_confidence' in globals() else RECOGNITION_THRESHOLD):
                    # Unknown face
                    recognition_stats['unknown_faces'] += 1
                    label = "Unknown"
                    color = (0, 0, 255)
                    cv2.rectangle(display_frame, (x, y), (x+w_box, y+h_box), color, 2)
                    cv2.putText(display_frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    continue

                # Map predicted label to DB employee id
                emp_db_id = employee_map.get(str(predicted_label))
                if not emp_db_id:
                    logger.warning(f"No DB mapping for label {predicted_label}")
                    continue

                # Insert attendance if not already present for today
                conn2 = get_db_connection()
                if conn2:
                    try:
                        cursor2 = conn2.cursor(dictionary=True)
                        cursor2.execute("SELECT id FROM attendance WHERE employee_id = %s AND DATE(check_in) = %s", (emp_db_id, now.date()))
                        if cursor2.fetchone():
                            # Already checked in today: update check_out if desired (optional)
                            cv2.rectangle(display_frame, (x, y), (x+w_box, y+h_box), (0, 200, 0), 2)
                            cv2.putText(display_frame, f"{predicted_label} (seen)", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,200,0), 2)
                        else:
                            status = determine_attendance_status(now) or 'present'
                            cursor2.execute(
                                "INSERT INTO attendance (employee_id, check_in, confidence, status, location, device_type) VALUES (%s, %s, %s, %s, %s, %s)",
                                (emp_db_id, now, max_proba, status, camera_location, 'cctv')
                            )
                            conn2.commit()
                            recognition_stats['employees_recognized'] += 1

                            # Announcement (non-blocking)
                            try:
                                announce_conn = get_db_connection()
                                if announce_conn:
                                    try:
                                        c = announce_conn.cursor(dictionary=True)
                                        c.execute("SELECT name FROM employees WHERE id = %s", (emp_db_id,))
                                        emp = c.fetchone()
                                        if emp:
                                            threading.Thread(target=announce_attendance, args=(emp['name'], status), daemon=True).start()
                                    except Exception as e:
                                        logger.error(f"Announcement DB error: {e}")
                                    finally:
                                        if announce_conn and announce_conn.is_connected():
                                            close_cursor_safe(c)
                                            close_conn_safe(announce_conn)
                            except Exception as e:
                                logger.error(f"Announcement error: {e}")

                            cv2.rectangle(display_frame, (x, y), (x+w_box, y+h_box), (0, 255, 0), 2)
                            cv2.putText(display_frame, f"{predicted_label} ({status})", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                    except Exception as e:
                        logger.error(f"DB attendance error: {e}")
                    finally:
                        if conn2 and conn2.is_connected():
                            close_cursor_safe(cursor2)
                            close_conn_safe(conn2)
            except Exception as e:
                logger.error(f"Error processing detected face: {e}")
                continue

        # emit stats and return display frame
        socketio.emit('recognition_stats', recognition_stats)
        return display_frame

    except Exception as e:
        logger.error(f"Frame processing error: {e}")
        return frame

def validate_rtsp_url(rtsp_url):
    """Validate RTSP URL format and test connection (or handle webcam index)"""
    if rtsp_url == 0:  # Webcam
        return True, "Webcam is valid"
    
    if not rtsp_url.startswith('rtsp://'):
        return False, "RTSP URL must start with rtsp://"
    
    # Test the connection for RTSP
    cap = None
    try:
        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            return False, "Failed to open RTSP stream"
        
        # Try to read a frame
        ret, frame = cap.read()
        if not ret:
            return False, "Failed to read frame from RTSP stream"
            
        return True, "RTSP URL is valid"
    except Exception as e:
        return False, f"Error testing RTSP URL: {str(e)}"
    finally:
        if cap:
            cap.release()



def start_recognition_service():
    """Low-latency background service for CCTV face recognition."""
    global recognition_active, recognition_stats, recognition_thread

    logger.info("CCTV recognition service starting (low-latency)")
    recognition_active = True

    # Determine RTSP source
    rtsp = CCTV_CAMERAS.get('laptop_camera', {}).get('rtsp_url', 0)

    cap = None
    try:
        # Open with FFMPEG backend for better RTSP support and low-latency options
        try:
            cap = cv2.VideoCapture(rtsp, cv2.CAP_FFMPEG) if isinstance(rtsp, str) else cv2.VideoCapture(rtsp)
        except Exception:
            cap = cv2.VideoCapture(rtsp)

        # Basic capture settings to reduce latency
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        try:
            cap.set(cv2.CAP_PROP_FPS, 25)
        except Exception:
            pass

        # Warm-up / quick validation
        start_time = time_module.time()
        opened = False
        while time_module.time() - start_time < 5:
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    opened = True
                    break
            time_module.sleep(0.1)

        if not opened:
            logger.error("Failed to open RTSP stream for CCTV camera")
            recognition_active = False
            return

    except Exception as e:
        logger.error(f"Error opening RTSP stream: {e}")
        recognition_active = False
        return

    # Placeholder frame shown when stream not available
    placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(placeholder, "Connecting to CCTV...", (150, 220),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(placeholder, "Skyhighes Technologies", (180, 260),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    frame_counter = 0
    last_stats_emit = time_module.time()
    last_reconnect_attempt = time_module.time()
    frame_interval = 1.0 / 15.0  # target ~15 FPS to reduce CPU and latency

    try:
        while recognition_active:
            # Reset daily tracking if needed
            reset_daily_checkins()

            current_time = time_module.time()

            # Reconnect logic (attempt quickly if capture closed)
            if current_time - last_reconnect_attempt > 5:
                if not cap.isOpened():
                    logger.info("Attempting to reconnect to CCTV RTSP stream")
                    try:
                        cap.release()
                    except Exception:
                        pass
                    try:
                        cap = cv2.VideoCapture(rtsp, cv2.CAP_FFMPEG) if isinstance(rtsp, str) else cv2.VideoCapture(rtsp)
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                        try:
                            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        except Exception:
                            pass
                        try:
                            cap.set(cv2.CAP_PROP_FPS, 25)
                        except Exception:
                            pass
                    except Exception as e:
                        logger.error(f"Reconnect failed: {e}")
                last_reconnect_attempt = current_time

            # Read a frame
            ret, frame = cap.read()
            if not ret or frame is None:
                # If no frame, send placeholder and continue quickly
                _, buffer = cv2.imencode('.jpg', placeholder, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
                frame_data = {
                    'camera_id': 'laptop_camera',
                    'frame': buffer.tobytes(),
                    'timestamp': time_module.time(),
                    'datetime': datetime.now()
                }
                update_live_stream_queue(frame_data)
                time_module.sleep(0.05)
                continue

            frame_counter += 1
            current_datetime = datetime.now()

            # Annotate frame for display
            display_frame = frame.copy()
            timestamp_str = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
            camera_name = CCTV_CAMERAS.get('laptop_camera', {}).get('name', 'CCTV Camera')
            location = CCTV_CAMERAS.get('laptop_camera', {}).get('location', 'Location')

            cv2.putText(display_frame, "Live CCTV Stream", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, f"{camera_name} - {location}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(display_frame, timestamp_str, (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Process recognition on alternate frames to save CPU (adjust as needed)
            try:
                if AI_AVAILABLE and (frame_counter % 2 == 0):
                    processed_frame = process_cctv_frame(display_frame, location)
                else:
                    processed_frame = display_frame
            except Exception as e:
                logger.error(f"Error processing frame: {e}")
                processed_frame = display_frame

            # Encode and push to live stream queue (reduce quality to save bandwidth)
            try:
                _, buffer = cv2.imencode('.jpg', processed_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
            except Exception:
                _, buffer = cv2.imencode('.jpg', processed_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])

            frame_data = {
                'camera_id': 'laptop_camera',
                'frame': buffer.tobytes(),
                'timestamp': time_module.time(),
                'datetime': current_datetime
            }
            update_live_stream_queue(frame_data)

            # Emit stats periodically
            if current_time - last_stats_emit > 2:
                socketio.emit('recognition_stats', recognition_stats)
                last_stats_emit = current_time

            # Control frame rate precisely
            elapsed = time_module.time() - current_time
            sleep_time = max(0, frame_interval - elapsed)
            time_module.sleep(sleep_time)

    except Exception as e:
        logger.error(f"Unexpected error in recognition service: {e}")

    finally:
        try:
            if cap:
                cap.release()
        except Exception:
            pass
        logger.info("CCTV recognition service stopped")


def update_live_stream_queue(frame_data):
    """Always keep only the latest frame (drop old ones)."""
    try:
        # Drop anything stale
        while not live_stream_queue.empty():
            try:
                live_stream_queue.get_nowait()
            except queue.Empty:
                break
        # Push newest frame
        try:
            live_stream_queue.put_nowait(frame_data)
        except queue.Full:
            try:
                live_stream_queue.get_nowait()
            except Exception:
                pass
            try:
                live_stream_queue.put_nowait(frame_data)
            except Exception:
                pass
    except Exception as e:
        logger.error(f"Error updating live stream queue: {e}")

def stop_recognition_service():
    """Stop the background recognition service"""
    global recognition_active
    
    if not recognition_active:
        return False, "Recognition service is not running"
    
    try:
        recognition_active = False
        if recognition_thread and recognition_thread.is_alive():
            recognition_thread.join(timeout=5)
        
        logger.info("Recognition service stopped")
        return True, "Recognition service stopped successfully"
    except Exception as e:
        logger.error(f"Error stopping recognition service: {e}")
        return False, f"Error stopping service: {str(e)}"


def allowed_file(filename):
    """Check if file has allowed extension"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

def train_model():
    # --- injected: require minimum images per person ---
    try:
        # If embeddings variable exists as a list of dicts with 'person_id'
        if 'embeddings' in globals() and embeddings is not None:
            from collections import Counter
            counts = Counter([e.get('person_id') for e in embeddings if e.get('person_id') is not None])
            to_keep = {pid for pid, c in counts.items() if c >= MIN_IMAGES_PER_PERSON}
            if not to_keep:
                logger.error(f"No person has >= {MIN_IMAGES_PER_PERSON} images. Training aborted.")
                return False
            embeddings[:] = [e for e in embeddings if e.get('person_id') in to_keep]
            logger.info(f"Training with persons: {sorted(list(to_keep))}")
    except Exception as _e:
        logger.warning(f"Could not apply MIN_IMAGES_PER_PERSON filtering: {_e}")
    # --- end injected ---


    """Train the face recognition model with proper employee_id mapping"""
    global svm_model, label_encoder
    
    if not AI_AVAILABLE:
        return False, "AI models not available"
    
    conn = get_db_connection()
    if not conn:
        return False, "Database connection error"
    
    start_time = time_module.time()
    
    try:
        cursor = conn.cursor(dictionary=True)
        
        # Get all embeddings with proper employee_id mapping
        cursor.execute("""
            SELECT e.employee_id, f.embedding 
            FROM face_embeddings f
            JOIN employees e ON f.employee_id = e.id
            WHERE e.status = 'active' AND f.embedding IS NOT NULL
        """)
        embeddings_data = cursor.fetchall()
        
        if not embeddings_data or len(embeddings_data) < 2:
            # Don't initialize with dummy data - just return an error
            return False, "Not enough face embeddings available (need at least 2)"
            
        embeddings = []
        labels = []
        
        for row in embeddings_data:
            try:
                embedding = pickle.loads(row['embedding'])
                # Skip zero embeddings
                if np.all(embedding == 0):
                    continue
                embeddings.append(embedding)
                labels.append(row['employee_id'])  # Use actual employee_id
            except Exception as e:
                logger.error(f"Error loading embedding: {e}")
                continue
        
        if len(embeddings) < 2:
            return False, f"Not enough training data. Only {len(embeddings)} valid embeddings available."
            
        # Convert to numpy arrays
        X = np.array(embeddings)
        y = np.array(labels)
        
        # Check if we have enough unique labels
        unique_labels = np.unique(y)
        if len(unique_labels) < 2:
            return False, f"Not enough unique employees for training. Only {len(unique_labels)} employee(s) available."
            
        # Train label encoder
        label_encoder = LabelEncoder()  # Reset encoder
        label_encoder.fit(y)
        y_encoded = label_encoder.transform(y)
        
        # Train SVM with better parameters
        svm_model = SVC(kernel='linear', probability=True, class_weight='balanced')
        svm_model.fit(X, y_encoded)
        
        # Save model
        save_model()
        
        # Calculate training duration
        duration = time_module.time() - start_time
        
        # Log training in database
        cursor.execute("""
            INSERT INTO model_training_history 
            (employees_count, embeddings_count, status, message, duration_seconds)
            VALUES (%s, %s, %s, %s, %s)
        """, (len(unique_labels), len(embeddings), 'success', 
              f"Model trained with {len(embeddings)} embeddings from {len(unique_labels)} employees", 
              duration))
        conn.commit()
        
        # Update the last model training time
        cursor.execute(
            "INSERT INTO system_settings (setting_key, setting_value, description) VALUES (%s, %s, %s) ON DUPLICATE KEY UPDATE setting_value = %s, updated_at = NOW()",
            ('last_model_training', datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'Last model training time', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        )
        conn.commit()
        
        return True, f"Model trained with {len(X)} face embeddings from {len(unique_labels)} employees in {duration:.2f} seconds"
        
    except Exception as e:
        # Log failed training
        duration = time_module.time() - start_time
        if conn:
            cursor.execute("""
                INSERT INTO model_training_history 
                (employees_count, embeddings_count, status, message, duration_seconds)
                VALUES (%s, %s, %s, %s, %s)
            """, (0, 0, 'failed', f"Training error: {str(e)}", duration))
            conn.commit()
        
        logger.error(f"Training error: {e}")
        return False, f"Training error: {str(e)}"
    finally:
        if conn and conn.is_connected():
            close_cursor_safe(cursor)
            close_conn_safe(conn)
            
def auto_train_model():
    """Automatically train the model when new faces are registered"""
    if not AI_AVAILABLE:
        return False, "AI models not available"
    
    conn = get_db_connection()
    if not conn:
        return False, "Database connection error"
    
    try:
        cursor = conn.cursor(dictionary=True)
        
        # Get count of embeddings that haven't been used in training
        cursor.execute("""
            SELECT COUNT(*) as new_embeddings 
            FROM face_embeddings 
            WHERE created_at > COALESCE(
                (SELECT MAX(training_date) FROM model_training_history WHERE status = 'success'), 
                '2000-01-01'
            )
        """)
        new_embeddings = cursor.fetchone()['new_embeddings']
        
        # If there are new embeddings, train the model
        if new_embeddings > 0:
            logger.info(f"Auto-training model with {new_embeddings} new embeddings")
            success, message = train_model()
            
            # Log training result
            cursor.execute("""
                INSERT INTO model_training_history (employees_count, embeddings_count, status, message)
                VALUES (%s, %s, %s, %s)
            """, (len(label_encoder.classes_) if success else 0, 
                  new_embeddings, 
                  'success' if success else 'failed', 
                  message))
            conn.commit()
            
            return success, message
        else:
            return True, "No new embeddings to train"
            
    except Exception as e:
        logger.error(f"Auto-training error: {e}")
        return False, f"Auto-training error: {str(e)}"
    finally:
        if conn and conn.is_connected():
            close_cursor_safe(cursor)
            close_conn_safe(conn)
        
# Test database connection
def test_db_connection():
    """Test database connection"""
    try:
        conn = mysql.connector.connect(**db_config)
        if conn.is_connected():
            print("✅ Database connection successful")
            cursor = conn.cursor()
            cursor.execute("SELECT DATABASE()")
            db_name = cursor.fetchone()
            print(f"✅ Connected to database: {db_name[0]}")
            close_cursor_safe(cursor)
            close_conn_safe(conn)
            return True
        else:
            print("❌ Database connection failed")
            return False
    except mysql.connector.Error as err:
        print(f"❌ Database error: {err}")
        return False

# Call this function after initializing the database

# Authentication decorators
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session or session.get('role') not in ['admin', 'super_admin']:
            flash('Admin access required', 'error')
            return redirect(url_for('dashboard'))
        return f(*args, **kwargs)
    return decorated_function

# Routes
@app.route('/')
def index():
    """Landing page"""
    return render_template('index.html')

@app.route('/admin/debug')
@admin_required
def debug_system():
    """Debug page to check recognition system status"""
    conn = get_db_connection()
    debug_info = {
        'ai_available': AI_AVAILABLE,
        'model_loaded': False,
        'total_employees': 0,
        'employees_with_embeddings': 0,
        'employee_details': []
    }
    
    if conn:
        try:
            cursor = conn.cursor(dictionary=True)
            
            # Check if model is loaded (updated check)
            debug_info['model_loaded'] = AI_AVAILABLE and hasattr(svm_model, 'classes_') and len(svm_model.classes_) > 0 and not all(label.startswith('dummy') for label in svm_model.classes_)
            
            # Get employee stats
            cursor.execute("SELECT COUNT(*) as count FROM employees WHERE status = 'active'")
            debug_info['total_employees'] = cursor.fetchone()['count']
            
            cursor.execute("""
                SELECT COUNT(DISTINCT fe.employee_id) as count 
                FROM face_embeddings fe
                JOIN employees e ON fe.employee_id = e.id
                WHERE e.status = 'active'
            """)
            debug_info['employees_with_embeddings'] = cursor.fetchone()['count']
            
            # Get details about employees with embeddings
            cursor.execute("""
                SELECT e.employee_id, e.name, COUNT(fe.id) as embedding_count
                FROM employees e
                LEFT JOIN face_embeddings fe ON e.id = fe.employee_id
                WHERE e.status = 'active'
                GROUP BY e.id
            """)
            debug_info['employee_details'] = cursor.fetchall()
            
        except Exception as e:
            logger.error(f"Error getting debug info: {e}")
        finally:
            if conn.is_connected():
                close_cursor_safe(cursor)
                close_conn_safe(conn)
    
    return render_template('debug.html', debug_info=debug_info)

@app.route('/admin/model/train', methods=['POST'])
@admin_required
def admin_train_model():
    """Admin endpoint to force model training"""
    try:
        success, message = train_model()
        if success:
            # Update the system status after successful training
            socketio.emit('model_update', {'status': 'trained', 'message': message})
            return jsonify({'success': True, 'message': message})
        else:
            return jsonify({'success': False, 'message': message})
    except Exception as e:
        logger.error(f"Model training error: {e}")
        return jsonify({'success': False, 'message': f'Training error: {str(e)}'})



@app.route('/login', methods=['GET', 'POST'])
def login():
    """User login"""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user_type = request.form.get('user_type')
        
        if not username or not password:
            flash('Please provide both username and password', 'error')
            return render_template('login.html')
        
        conn = get_db_connection()
        if not conn:
            flash('Database connection error', 'error')
            return render_template('login.html')
        
        try:
            cursor = conn.cursor(dictionary=True)
            
            if user_type == 'admin':
                cursor.execute(
                    "SELECT id, username, role, permissions, password_hash FROM admin_users WHERE username = %s",
                    (username,)
                )
                user = cursor.fetchone()
                
                if user and check_password_hash(user['password_hash'], password):
                    session['user_id'] = user['id']
                    session['username'] = user['username']
                    session['role'] = user['role']
                    session['permissions'] = json.loads(user['permissions'])
                    session['user_type'] = 'admin'
                    
                    logger.info(f"Admin login: {username}")
                    return redirect(url_for('admin_dashboard'))
                else:
                    flash('Invalid admin credentials', 'error')
            else:
                # Employee login
                cursor.execute(
                    "SELECT id, employee_id, name, department, password_hash, profile_image FROM employees WHERE employee_id = %s AND status = 'active'",
                    (username,)
                )
                user = cursor.fetchone()
                
                if user and check_password_hash(user['password_hash'], password):
                    session['user_id'] = user['id']
                    session['username'] = user['name']
                    session['employee_id'] = user['employee_id']
                    session['department'] = user['department']
                    session['profile_image'] = user['profile_image']
                    session['user_type'] = 'employee'
                    
                    logger.info(f"Employee login: {user['name']}")
                    return redirect(url_for('employee_dashboard'))
                else:
                    flash('Invalid employee credentials', 'error')
        
        except mysql.connector.Error as err:
            flash(f'Database error: {err}', 'error')
        except Exception as e:
            flash(f'Unexpected error: {str(e)}', 'error')
        finally:
            if conn.is_connected():
                close_cursor_safe(cursor)
                close_conn_safe(conn)
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    """User logout"""
    session.clear()
    flash('Logged out successfully', 'success')
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    """Main dashboard"""
    if session.get('user_type') == 'admin':
        return redirect(url_for('admin_dashboard'))
    else:
        return redirect(url_for('employee_dashboard'))

# Update the admin_dashboard function to pass pending requests count to session
@app.route('/admin/dashboard')
@admin_required
def admin_dashboard():
    """Admin dashboard"""
    conn = get_db_connection()
    if not conn:
        flash('Database connection error', 'error')
        return render_template('admin_dashboard.html', stats={})
    
    try:
        cursor = conn.cursor(dictionary=True)
        
        # Get statistics
        cursor.execute("SELECT COUNT(*) as total FROM employees WHERE status = 'active'")
        total_employees = cursor.fetchone()['total']
        
        cursor.execute("SELECT COUNT(*) as present FROM attendance WHERE DATE(check_in) = CURDATE()")
        present_today = cursor.fetchone()['present']
        
        cursor.execute("SELECT COUNT(*) as late FROM attendance WHERE DATE(check_in) = CURDATE() AND status = 'late'")
        late_today = cursor.fetchone()['late']
        
        cursor.execute("SELECT COUNT(*) as absent FROM employees WHERE status = 'active' AND id NOT IN (SELECT employee_id FROM attendance WHERE DATE(check_in) = CURDATE())")
        absent_today = cursor.fetchone()['absent']
        
        # Get pending check-in requests
        cursor.execute("SELECT COUNT(*) as pending_requests FROM checkin_requests WHERE status = 'pending'")
        pending_requests = cursor.fetchone()['pending_requests']
        
        # Store in session for the sidebar
        session['pending_requests_count'] = pending_requests
        
        # Get recent attendance
        cursor.execute("""
            SELECT e.name, e.department, a.check_in, a.status, a.confidence
            FROM attendance a
            JOIN employees e ON a.employee_id = e.id
            WHERE DATE(a.check_in) = CURDATE()
            ORDER BY a.check_in DESC
            LIMIT 10
        """)
        recent_attendance = cursor.fetchall()
        
        # Get system status
        system_status = {
            'recognition_active': recognition_active,
            'cameras_active': len([c for c in CCTV_CAMERAS.values() if c['status'] == 'active']),
            'total_cameras': len(CCTV_CAMERAS),
            'model_loaded': AI_AVAILABLE,
            'last_recognition': recognition_stats['last_recognition']
        }
        
        stats = {
            'total_employees': total_employees,
            'present_today': present_today,
            'late_today': late_today,
            'absent_today': absent_today,
            'pending_requests': pending_requests,
            'recent_attendance': recent_attendance,
            'system_status': system_status,
            'recognition_stats': recognition_stats
        }
        
    except mysql.connector.Error as err:
        flash(f'Database error: {err}', 'error')
        stats = {}
    except Exception as e:
        flash(f'Unexpected error: {str(e)}', 'error')
        stats = {}
    finally:
        if conn.is_connected():
            close_cursor_safe(cursor)
            close_conn_safe(conn)
    
    return render_template('admin_dashboard.html', stats=stats)

# Add API endpoint for pending requests count
@app.route('/api/pending_requests_count')
@admin_required
def pending_requests_count():
    """API endpoint to get pending requests count"""
    conn = get_db_connection()
    if not conn:
        return jsonify({'count': 0})
    
    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT COUNT(*) as count FROM checkin_requests WHERE status = 'pending'")
        result = cursor.fetchone()
        return jsonify({'count': result['count']})
    except Exception as e:
        return jsonify({'count': 0})
    finally:
        if conn and conn.is_connected():
            close_cursor_safe(cursor)
            close_conn_safe(conn)
    
    return render_template('admin_dashboard.html', stats=stats)

@app.route('/employee/dashboard')
@login_required
def employee_dashboard():
    """Employee dashboard"""
    if session.get('user_type') != 'employee':
        return redirect(url_for('dashboard'))
    
    conn = get_db_connection()
    if not conn:
        flash('Database connection error', 'error')
        return render_template('employee_dashboard.html', data={})
    
    try:
        cursor = conn.cursor(dictionary=True)
        
        # Get employee info
        cursor.execute("""
            SELECT name, department, position, join_date, profile_image
            FROM employees WHERE id = %s
        """, (session['user_id'],))
        employee = cursor.fetchone()
        
        # Get today's attendance
        today = date.today()
        cursor.execute("""
            SELECT check_in, check_out, status, confidence
            FROM attendance WHERE employee_id = %s AND DATE(check_in) = %s
        """, (session['user_id'], today))
        today_attendance = cursor.fetchone()
        
        # Get attendance history
        cursor.execute("""
            SELECT check_in, check_out, status, confidence
            FROM attendance WHERE employee_id = %s
            ORDER BY check_in DESC LIMIT 10
        """, (session['user_id'],))
        attendance_history = cursor.fetchall()
        
        # Check if today is non-working day
        is_non_working = is_non_working_day(today)
        
        # Check if manual check-in request exists for today
        cursor.execute("""
            SELECT id, status FROM checkin_requests 
            WHERE employee_id = %s AND request_date = %s
        """, (session['user_id'], today))
        checkin_request = cursor.fetchone()
        
        data = {
            'employee': employee,
            'today_attendance': today_attendance,
            'attendance_history': attendance_history,
            'is_non_working': is_non_working,
            'checkin_request': checkin_request
        }
        
    except mysql.connector.Error as err:
        flash(f'Database error: {err}', 'error')
        data = {}
    except Exception as e:
        flash(f'Unexpected error: {str(e)}', 'error')
        data = {}
    finally:
        if conn.is_connected():
            close_cursor_safe(cursor)
            close_conn_safe(conn)
    
    return render_template('employee_dashboard.html', data=data)

@app.route('/employee/request_checkin', methods=['POST'])
@login_required
def request_checkin():
    """Request manual check-in"""
    if session.get('user_type') != 'employee':
        flash('Employee access required', 'error')
        return redirect(url_for('dashboard'))
    
    reason = request.form.get('reason', '')
    today = date.today()
    
    conn = get_db_connection()
    if not conn:
        flash('Database connection error', 'error')
        return redirect(url_for('employee_dashboard'))
    
    try:
        cursor = conn.cursor()
        
        # Check if already have attendance for today
        cursor.execute("""
            SELECT id FROM attendance 
            WHERE employee_id = %s AND DATE(check_in) = %s
        """, (session['user_id'], today))
        if cursor.fetchone():
            flash('You already have attendance for today', 'warning')
            return redirect(url_for('employee_dashboard'))
        
        # Check if already have pending request for today
        cursor.execute("""
            SELECT id FROM checkin_requests 
            WHERE employee_id = %s AND request_date = %s AND status = 'pending'
        """, (session['user_id'], today))
        if cursor.fetchone():
            flash('You already have a pending request for today', 'warning')
            return redirect(url_for('employee_dashboard'))
        
        # Create new request
        cursor.execute("""
            INSERT INTO checkin_requests (employee_id, request_date, reason)
            VALUES (%s, %s, %s)
        """, (session['user_id'], today, reason))
        conn.commit()
        
        flash('Manual check-in request submitted. Waiting for admin approval.', 'success')
        
    except mysql.connector.Error as err:
        flash(f'Database error: {err}', 'error')
    except Exception as e:
        flash(f'Unexpected error: {str(e)}', 'error')
    finally:
        if conn.is_connected():
            close_cursor_safe(cursor)
            close_conn_safe(conn)
    
    return redirect(url_for('employee_dashboard'))

@app.route('/api/attendance_stats')
@admin_required
def api_attendance_stats():
    """API endpoint for attendance statistics"""
    conn = get_db_connection()
    if not conn:
        return jsonify({'error': 'Database connection error'})
    
    try:
        cursor = conn.cursor(dictionary=True)
        
        # Get parameters from request
        start_date = request.args.get('start', date.today().strftime('%Y-%m-%d'))
        end_date = request.args.get('end', date.today().strftime('%Y-%m-%d'))
        department = request.args.get('dept', 'all')
        
        # Build query based on filters
        query = """
            SELECT 
                DATE(check_in) as date,
                status,
                COUNT(*) as count
            FROM attendance a
            JOIN employees e ON a.employee_id = e.id
            WHERE DATE(check_in) BETWEEN %s AND %s
        """
        
        params = [start_date, end_date]
        
        if department != 'all':
            query += " AND e.department = %s"
            params.append(department)
            
        query += " GROUP BY DATE(check_in), status ORDER BY date"
        
        cursor.execute(query, params)
        results = cursor.fetchall()
        
        # Process data for charts
        # (Implement data processing based on your specific needs)
        
        return jsonify({
            'today_counts': [10, 5, 2],  # Replace with actual data
            'weekly_present': [15, 18, 20, 17, 22],
            'weekly_late': [3, 2, 4, 1, 3],
            'weekly_absent': [2, 0, 1, 2, 0],
            'weeks': ['Week 1', 'Week 2', 'Week 3', 'Week 4', 'Week 5']
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})
    finally:
        if conn and conn.is_connected():
            close_cursor_safe(cursor)
            close_conn_safe(conn)

@app.route('/api/department_stats')
@admin_required
def api_department_stats():
    """API endpoint for department statistics"""
    # Implement department statistics logic
    return jsonify({
        'dept_names': ['Engineering', 'Marketing', 'Sales', 'HR'],
        'dept_present': [25, 18, 22, 15],
        'dept_late': [3, 2, 4, 1],
        'dept_absent': [1, 0, 2, 1]
    })

@app.route('/employee/update_profile', methods=['GET', 'POST'])
@login_required
def update_profile():
    """Update employee profile"""
    if session.get('user_type') != 'employee':
        flash('Employee access required', 'error')
        return redirect(url_for('dashboard'))
    
    conn = get_db_connection()
    if not conn:
        flash('Database connection error', 'error')
        return redirect(url_for('employee_dashboard'))
    
    try:
        cursor = conn.cursor(dictionary=True)
        
        if request.method == 'POST':
            if 'profile_image' in request.files:
                file = request.files['profile_image']
                if file and allowed_file(file.filename):
                    # Save the uploaded file
                    filename = secure_filename(f"{session['employee_id']}_{uuid.uuid4().hex}.jpg")
                    save_path = os.path.join(app.config['UPLOAD_FOLDER'], 'profiles', filename)
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    file.save(save_path)
                    
                    # Update database
                    cursor.execute(
                        "UPDATE employees SET profile_image = %s WHERE id = %s",
                        (save_path, session['user_id'])
                    )
                    conn.commit()
                    
                    # Update session
                    session['profile_image'] = save_path
                    
                    flash('Profile image updated successfully', 'success')
                    return redirect(url_for('employee_dashboard'))
                else:
                    flash('Invalid file type. Only JPG/PNG allowed.', 'error')
        
        # Get employee data
        cursor.execute("""
            SELECT name, employee_id, department, position, email, phone, profile_image
            FROM employees WHERE id = %s
        """, (session['user_id'],))
        employee = cursor.fetchone()
        
        if not employee:
            flash('Employee not found', 'error')
            return redirect(url_for('employee_dashboard'))
            
        return render_template('update_profile.html', employee=employee)
        
    except mysql.connector.Error as err:
        flash(f'Database error: {err}', 'error')
    except Exception as e:
        flash(f'Unexpected error: {str(e)}', 'error')
    finally:
        if conn.is_connected():
            close_cursor_safe(cursor)
            close_conn_safe(conn)
    
    return redirect(url_for('employee_dashboard'))

@app.route('/admin/employees')
@admin_required
def manage_employees():
    """Manage employees"""
    conn = get_db_connection()
    if not conn:
        flash('Database connection error', 'error')
        return render_template('manage_employees.html', employees=[])
    
    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT id, employee_id, name, department, position, status, join_date, profile_image
            FROM employees ORDER BY name
        """)
        employees = cursor.fetchall()
        
    except mysql.connector.Error as err:
        flash(f'Database error: {err}', 'error')
        employees = []
    finally:
        if conn.is_connected():
            close_cursor_safe(cursor)
            close_conn_safe(conn)
    
    return render_template('manage_employees.html', employees=employees)

@app.route('/admin/employees/view/<int:employee_id>')
@admin_required
def view_employee(employee_id):
    """View employee details"""
    conn = get_db_connection()
    if not conn:
        flash('Database connection error', 'error')
        return redirect(url_for('manage_employees'))
    
    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT id, employee_id, name, position, department, email, phone, 
                   join_date, status, profile_image, emergency_contact, emergency_phone
            FROM employees WHERE id = %s
        """, (employee_id,))
        employee = cursor.fetchone()
        
        if not employee:
            flash('Employee not found', 'error')
            return redirect(url_for('manage_employees'))
            
        # Get attendance records
        cursor.execute("""
            SELECT check_in, check_out, status, confidence, location
            FROM attendance WHERE employee_id = %s
            ORDER BY check_in DESC LIMIT 10
        """, (employee_id,))
        attendance_records = cursor.fetchall()
        
        # Get check-in requests
        cursor.execute("""
            SELECT request_date, reason, status, reviewed_at
            FROM checkin_requests WHERE employee_id = %s
            ORDER BY request_date DESC LIMIT 5
        """, (employee_id,))
        checkin_requests = cursor.fetchall()
        
    except mysql.connector.Error as err:
        flash(f'Database error: {err}', 'error')
        return redirect(url_for('manage_employees'))
    except Exception as e:
        flash(f'Unexpected error: {str(e)}', 'error')
        return redirect(url_for('manage_employees'))
    finally:
        if conn.is_connected():
            close_cursor_safe(cursor)
            close_conn_safe(conn)
    
    return render_template('view_employee.html', 
                         employee=employee, 
                         attendance_records=attendance_records,
                         checkin_requests=checkin_requests)

@app.route('/admin/checkin_requests')
@admin_required
def admin_checkin_requests():
    """Manage check-in requests"""
    conn = get_db_connection()
    if not conn:
        flash('Database connection error', 'error')
        return render_template('admin_checkin_requests.html', requests=[])
    
    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT r.id, r.request_date, r.reason, r.status, r.created_at,
                   e.name AS employee_name, e.department, e.employee_id
            FROM checkin_requests r
            JOIN employees e ON r.employee_id = e.id
            WHERE r.status = 'pending'
            ORDER BY r.created_at DESC
        """)
        requests = cursor.fetchall()
        
    except mysql.connector.Error as err:
        flash(f'Database error: {err}', 'error')
        requests = []
    finally:
        if conn.is_connected():
            close_cursor_safe(cursor)
            close_conn_safe(conn)
    
    return render_template('admin_checkin_requests.html', requests=requests)

@app.route('/admin/approve_checkin/<int:request_id>')
@admin_required
def approve_checkin(request_id):
    """Approve check-in request"""
    conn = get_db_connection()
    if not conn:
        flash('Database connection error', 'error')
        return redirect(url_for('admin_checkin_requests'))
    
    try:
        cursor = conn.cursor(dictionary=True)
        
        # Get request details
        cursor.execute("""
            SELECT employee_id, request_date 
            FROM checkin_requests 
            WHERE id = %s AND status = 'pending'
        """, (request_id,))
        req = cursor.fetchone()
        
        if not req:
            flash('Request not found or already processed', 'error')
            return redirect(url_for('admin_checkin_requests'))
        
        # Check if already have attendance for that day
        cursor.execute("""
            SELECT id FROM attendance 
            WHERE employee_id = %s AND DATE(check_in) = %s
        """, (req['employee_id'], req['request_date']))
        if cursor.fetchone():
            flash('Employee already has attendance for this date', 'error')
            return redirect(url_for('admin_checkin_requests'))
        
        # Create attendance record
        now = datetime.now()
        status = determine_attendance_status(now) or 'present'
        cursor.execute("""
            INSERT INTO attendance (employee_id, check_in, confidence, status, device_type, notes)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (
            req['employee_id'], 
            now, 
            1.0, 
            status, 
            'manual', 
            'Approved manual check-in'
        ))
        
        # Update request status
        cursor.execute("""
            UPDATE checkin_requests 
            SET status = 'approved', reviewed_by = %s, reviewed_at = %s
            WHERE id = %s
        """, (session['user_id'], now, request_id))
        
        conn.commit()
        flash('Check-in request approved successfully', 'success')
        
    except mysql.connector.Error as err:
        flash(f'Database error: {err}', 'error')
    except Exception as e:
        flash(f'Unexpected error: {str(e)}', 'error')
    finally:
        if conn.is_connected():
            close_cursor_safe(cursor)
            close_conn_safe(conn)
    
    return redirect(url_for('admin_checkin_requests'))


@app.route('/admin/test_face_recognition', methods=['GET', 'POST'])
@admin_required
def test_face_recognition():
    """Test face recognition for a specific employee"""
    # Get the min_confidence threshold
    min_confidence = get_min_confidence()
    
    conn = get_db_connection()
    if not conn:
        flash('Database connection error', 'error')
        return redirect(url_for('admin_dashboard'))
    
    try:
        cursor = conn.cursor(dictionary=True)
        
        # Get all employees with face embeddings
        cursor.execute("""
            SELECT e.id, e.employee_id, e.name, e.department, 
                   COUNT(fe.id) as embedding_count
            FROM employees e
            LEFT JOIN face_embeddings fe ON e.id = fe.employee_id
            WHERE e.status = 'active'
            GROUP BY e.id
            HAVING embedding_count > 0
            ORDER BY e.name
        """)
        employees = cursor.fetchall()
        
        test_result = None
        selected_employee = None
        
        if request.method == 'POST':
            employee_id = request.form.get('employee_id')
            if employee_id:
                # Get the selected employee details
                cursor.execute("""
                    SELECT id, employee_id, name, department
                    FROM employees WHERE id = %s
                """, (employee_id,))
                selected_employee = cursor.fetchone()
                
                # Get the employee's embeddings
                cursor.execute("""
                    SELECT embedding FROM face_embeddings 
                    WHERE employee_id = %s
                """, (employee_id,))
                embeddings_data = cursor.fetchall()
                
                if embeddings_data:
                    embeddings = []
                    for row in embeddings_data:
                        try:
                            embedding = pickle.loads(row['embedding'])
                            embeddings.append(embedding)
                        except Exception as e:
                            logger.error(f"Error loading embedding: {e}")
                            continue
                    
                    # Process the test image
                    if 'test_image' in request.files:
                        file = request.files['test_image']
                        if file and allowed_file(file.filename):
                            # Read and process the image
                            file_bytes = file.read()
                            nparr = np.frombuffer(file_bytes, np.uint8)
                            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                            
                            if img is not None:
                                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                
                                # Detect faces
                                faces = detector.detect_faces(rgb_img)
                                if faces:
                                    # Get the best face (highest confidence)
                                    best_face = max(faces, key=lambda f: f['confidence'])
                                    x, y, w, h = best_face['box']
                                    
                                    # Extract face region
                                    face_img = rgb_img[y:y+h, x:x+w]
                                    if face_img.size > 0:
                                        # Generate embedding for the test face
                                        test_embedding = get_embedding(face_img)
                                        
                                        if test_embedding is not None:
                                            # Compare with stored embeddings
                                            similarities = []
                                            for stored_embedding in embeddings:
                                                similarity = np.dot(test_embedding, stored_embedding) / (
                                                    np.linalg.norm(test_embedding) * np.linalg.norm(stored_embedding)
                                                )
                                                similarities.append(similarity)
                                            
                                            # Calculate average similarity
                                            avg_similarity = np.mean(similarities) if similarities else 0
                                            max_similarity = np.max(similarities) if similarities else 0
                                            
                                            # Determine result using min_confidence from database
                                            is_match = max_similarity > min_confidence
                                            
                                            # Log if recognition is skipped due to low confidence
                                            if not is_match:
                                                logger.info(f"Test recognition skipped: confidence {max_similarity:.2f} < min_confidence {min_confidence:.2f}")
                                            
                                            test_result = {
                                                'employee': selected_employee,
                                                'similarities': similarities,
                                                'avg_similarity': avg_similarity,
                                                'max_similarity': max_similarity,
                                                'is_match': is_match,
                                                'faces_detected': len(faces),
                                                'best_confidence': best_face['confidence'],
                                                'min_confidence': min_confidence  # Include in result for display
                                            }
                                        else:
                                            flash('Failed to generate embedding from test image', 'error')
                                    else:
                                        flash('Invalid face region detected', 'error')
                                else:
                                    flash('No faces detected in the test image', 'error')
                            else:
                                flash('Invalid image file', 'error')
                        else:
                            flash('Please select a valid image file', 'error')
                    else:
                        flash('No test image provided', 'error')
                else:
                    flash('Selected employee has no face embeddings', 'error')
        
    except Exception as e:
        logger.error(f"Error in face recognition test: {e}")
        flash(f'Error during face recognition test: {str(e)}', 'error')
    finally:
        if conn and conn.is_connected():
            close_cursor_safe(cursor)
            close_conn_safe(conn)
    
    return render_template('test_face_recognition.html', 
                         employees=employees,
                         selected_employee=selected_employee,
                         test_result=test_result)



@app.route('/api/capture_test_image', methods=['POST'])
@admin_required
def capture_test_image():
    """Capture image from webcam for testing"""
    try:
        # Get image data from request
        image_data = request.json.get('image')
        if not image_data:
            return jsonify({'success': False, 'message': 'No image data provided'})
        
        # Remove data URL prefix
        if 'base64,' in image_data:
            image_data = image_data.split('base64,')[1]
        
        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Save temporary image
        temp_dir = 'uploads/temp'
        os.makedirs(temp_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        filename = f'test_{timestamp}.jpg'
        filepath = os.path.join(temp_dir, filename)
        
        cv2.imwrite(filepath, img)
        
        return jsonify({
            'success': True, 
            'filename': filename,
            'filepath': filepath
        })
        
    except Exception as e:
        logger.error(f"Error capturing test image: {e}")
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})
    

@app.route('/admin/reject_checkin/<int:request_id>')
@admin_required
def reject_checkin(request_id):
    """Reject check-in request"""
    conn = get_db_connection()
    if not conn:
        flash('Database connection error', 'error')
        return redirect(url_for('admin_checkin_requests'))
    
    try:
        cursor = conn.cursor()
        
        # Update request status
        cursor.execute("""
            UPDATE checkin_requests 
            SET status = 'rejected', reviewed_by = %s, reviewed_at = %s
            WHERE id = %s
        """, (session['user_id'], datetime.now(), request_id))
        
        conn.commit()
        flash('Check-in request rejected', 'success')
        
    except mysql.connector.Error as err:
        flash(f'Database error: {err}', 'error')
    finally:
        if conn.is_connected():
            close_cursor_safe(cursor)
            close_conn_safe(conn)
    
    return redirect(url_for('admin_checkin_requests'))

# ... (your existing imports and setup code remains the same)

@app.route('/admin/attendance_analysis')
@admin_required
def attendance_analysis():
    """Attendance analysis dashboard with modern charts"""
    conn = get_db_connection()
    if not conn:
        flash('Database connection error', 'error')
        return render_template('attendance_analysis.html')
    
    try:
        cursor = conn.cursor(dictionary=True)
        
        # Weekly attendance summary
        cursor.execute("""
            SELECT 
                YEARWEEK(check_in) AS week,
                COUNT(DISTINCT employee_id) AS total_employees,
                SUM(status = 'present') AS present_count,
                SUM(status = 'late') AS late_count,
                SUM(status = 'absent') AS absent_count
            FROM attendance
            WHERE check_in >= DATE_SUB(CURDATE(), INTERVAL 12 WEEK)
            GROUP BY YEARWEEK(check_in)
            ORDER BY week DESC
            LIMIT 12
        """)
        weekly_data = cursor.fetchall()
        
        # Monthly attendance summary
        cursor.execute("""
            SELECT 
                DATE_FORMAT(check_in, '%Y-%m') AS month,
                COUNT(DISTINCT employee_id) AS total_employees,
                SUM(status = 'present') AS present_count,
                SUM(status = 'late') AS late_count,
                SUM(status = 'absent') AS absent_count
            FROM attendance
            WHERE check_in >= DATE_SUB(CURDATE(), INTERVAL 12 MONTH)
            GROUP BY DATE_FORMAT(check_in, '%Y-%m')
            ORDER BY month DESC
            LIMIT 12
        """)
        monthly_data = cursor.fetchall()
        
        # Department-wise attendance
        cursor.execute("""
            SELECT 
                e.department,
                COUNT(DISTINCT a.employee_id) AS total_employees,
                SUM(a.status = 'present') AS present_count,
                SUM(a.status = 'late') AS late_count,
                SUM(a.status = 'absent') AS absent_count
            FROM attendance a
            JOIN employees e ON a.employee_id = e.id
            WHERE DATE(a.check_in) = CURDATE()
            GROUP BY e.department
        """)
        dept_data = cursor.fetchall()
        
        # Today's attendance distribution - NORMALIZE STATUS VALUES
        cursor.execute("""
            SELECT 
                status,
                COUNT(*) AS count
            FROM attendance
            WHERE DATE(check_in) = CURDATE()
            GROUP BY status
        """)
        today_data = cursor.fetchall()
        
        # Process data for charts with normalized status values
        weeks = [f"Week {d['week'] % 100}" for d in weekly_data][::-1]
        weekly_present = [d['present_count'] for d in weekly_data][::-1]
        weekly_late = [d['late_count'] for d in weekly_data][::-1]
        weekly_absent = [d['absent_count'] for d in weekly_data][::-1]
        
        months = [d['month'][5:7] + '/' + d['month'][2:4] for d in monthly_data][::-1]
        monthly_present = [d['present_count'] for d in monthly_data][::-1]
        monthly_late = [d['late_count'] for d in monthly_data][::-1]
        monthly_absent = [d['absent_count'] for d in monthly_data][::-1]
        
        dept_names = [d['department'] for d in dept_data]
        dept_present = [d['present_count'] for d in dept_data]
        dept_late = [d['late_count'] for d in dept_data]
        dept_absent = [d['absent_count'] for d in dept_data]
        
        # Normalize status values to Title Case and handle missing values
        status_map = {
            'present': 'Present',
            'late': 'Late',
            'absent': 'Absent',
            'half_day': 'Half Day',
            'overtime': 'Overtime'
        }
        
        # Initialize all status counts to zero
        status_counts = {v: 0 for v in status_map.values()}
        
        # Update with actual values from database
        for d in today_data:
            normalized_status = status_map.get(d['status'], 'Absent')  # Default to Absent if unknown
            status_counts[normalized_status] = d['count']
        
        # Create ordered lists for the chart
        statuses = list(status_map.values())
        counts = [status_counts[status] for status in statuses]
        
        # Prepare data for JSON serialization
        analysis_data = {
            'weeks': weeks,
            'weekly_present': weekly_present,
            'weekly_late': weekly_late,
            'weekly_absent': weekly_absent,
            'months': months,
            'monthly_present': monthly_present,
            'monthly_late': monthly_late,
            'monthly_absent': monthly_absent,
            'dept_names': dept_names,
            'dept_present': dept_present,
            'dept_late': dept_late,
            'dept_absent': dept_absent,
            'statuses': statuses,
            'counts': counts
        }
        
    except mysql.connector.Error as err:
        flash(f'Database error: {err}', 'error')
        analysis_data = {}
    except Exception as e:
        flash(f'Unexpected error: {str(e)}', 'error')
        analysis_data = {}
    finally:
        if conn.is_connected():
            close_cursor_safe(cursor)
            close_conn_safe(conn)
    
    return render_template('attendance_analysis.html', analysis_data=analysis_data)

# ... (the rest of your code remains the same)

@app.route('/api/recognition/status')
@app.route('/api/recognition/status')
@admin_required
def recognition_status():
    """Get recognition service status"""
    # Get last model training time
    conn = get_db_connection()
    last_training = "N/A"
    if conn:
        try:
            cursor = conn.cursor(dictionary=True)
            cursor.execute("SELECT setting_value FROM system_settings WHERE setting_key = 'last_model_training'")
            result = cursor.fetchone()
            if result:
                last_training = result['setting_value']
        except Exception as e:
            logger.error(f"Error getting last model training time: {e}")
        finally:
            if conn.is_connected():
                close_cursor_safe(cursor)
                close_conn_safe(conn)
    
    return jsonify({
        'active': recognition_active,
        'stats': recognition_stats,
        'cameras': CCTV_CAMERAS,
        'model_loaded': AI_AVAILABLE and hasattr(svm_model, 'classes_') and len(svm_model.classes_) > 0,
        'last_training': last_training
    })

@app.route('/admin/employees/add', methods=['GET', 'POST'])
@admin_required
def add_employee():
    """Add new employee"""
    if request.method == 'POST':
        employee_id = request.form.get('employee_id')
        name = request.form.get('name')
        position = request.form.get('position')
        department = request.form.get('department')
        email = request.form.get('email')
        phone = request.form.get('phone')
        password = request.form.get('password')
        
        if not employee_id or not name or not password:
            flash('Employee ID, name and password are required', 'error')
            return redirect(url_for('add_employee'))
        
        conn = get_db_connection()
        if not conn:
            flash('Database connection error', 'error')
            return redirect(url_for('add_employee'))
            
        try:
            cursor = conn.cursor()
            password_hash = generate_password_hash(password)
            
            cursor.execute(
                "INSERT INTO employees (employee_id, name, position, department, email, phone, password_hash) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s)",
                (employee_id, name, position, department, email, phone, password_hash)
            )
            conn.commit()
            flash('Employee added successfully', 'success')
            return redirect(url_for('manage_employees'))
            
        except mysql.connector.Error as err:
            flash(f'Database error: {err}', 'error')
        except Exception as e:
            flash(f'Unexpected error: {str(e)}', 'error')
        finally:
            if conn.is_connected():
                close_cursor_safe(cursor)
                close_conn_safe(conn)
    
    return render_template('add_employee.html')

@app.route('/admin/employees/edit/<int:employee_id>', methods=['GET', 'POST'])
@admin_required
def edit_employee(employee_id):
    """Edit employee"""
    conn = get_db_connection()
    if not conn:
        flash('Database connection error', 'error')
        return redirect(url_for('manage_employees'))
    
    try:
        cursor = conn.cursor(dictionary=True)
        
        if request.method == 'POST':
            name = request.form.get('name')
            position = request.form.get('position')
            department = request.form.get('department')
            email = request.form.get('email')
            phone = request.form.get('phone')
            status = request.form.get('status')
            
            cursor.execute(
                "UPDATE employees SET name = %s, position = %s, department = %s, "
                "email = %s, phone = %s, status = %s WHERE id = %s",
                (name, position, department, email, phone, status, employee_id)
            )
            conn.commit()
            flash('Employee updated successfully', 'success')
            return redirect(url_for('manage_employees'))
        
        # Get employee data
        cursor.execute("SELECT * FROM employees WHERE id = %s", (employee_id,))
        employee = cursor.fetchone()
        
        if not employee:
            flash('Employee not found', 'error')
            return redirect(url_for('manage_employees'))
            
        return render_template('edit_employee.html', employee=employee)
        
    except mysql.connector.Error as err:
        flash(f'Database error: {err}', 'error')
    except Exception as e:
        flash(f'Unexpected error: {str(e)}', 'error')
    finally:
        if conn.is_connected():
            close_cursor_safe(cursor)
            close_conn_safe(conn)
    
    return redirect(url_for('manage_employees'))

@app.route('/admin/employees/delete/<int:employee_id>', methods=['POST'])
@admin_required
def delete_employee(employee_id):
    """Delete employee"""
    conn = get_db_connection()
    if not conn:
        flash('Database connection error', 'error')
        return redirect(url_for('manage_employees'))
    
    try:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM employees WHERE id = %s", (employee_id,))
        conn.commit()
        flash('Employee deleted successfully', 'success')
    except mysql.connector.Error as err:
        flash(f'Database error: {err}', 'error')
    except Exception as e:
        flash(f'Unexpected error: {str(e)}', 'error')
    finally:
        if conn.is_connected():
            close_cursor_safe(cursor)
            close_conn_safe(conn)
    
    return redirect(url_for('manage_employees'))

@app.route('/admin/employees/register_face/<int:employee_id>', methods=['GET', 'POST'])
@admin_required
def register_face(employee_id):
    """Register face for an employee with enhanced validation"""
    conn = get_db_connection()
    if not conn:
        flash('Database connection error', 'error')
        return redirect(url_for('manage_employees'))
    
    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM employees WHERE id = %s", (employee_id,))
        employee = cursor.fetchone()
        
        if not employee:
            flash('Employee not found', 'error')
            return redirect(url_for('manage_employees'))
            
        if request.method == 'POST':
            if 'face_image' not in request.files:
                flash('No file selected', 'error')
                return redirect(request.url)
                
            file = request.files['face_image']
            if file.filename == '':
                flash('No selected file', 'error')
                return redirect(request.url)
                
            if file and allowed_file(file.filename):
                # Read image file
                file_bytes = file.read()
                nparr = np.frombuffer(file_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if img is None:
                    flash('Invalid image file', 'error')
                    return redirect(request.url)
                
                # Check if image is empty or too small
                if img.size == 0 or img.shape[0] < 50 or img.shape[1] < 50:
                    flash('Image is too small or empty', 'error')
                    return redirect(request.url)
                
                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                if not AI_AVAILABLE:
                    flash('AI models not available', 'error')
                    return redirect(request.url)
                    
                # Detect faces with enhanced parameters
                try:
                    faces = detector.detect_faces(rgb_img)
                    if not faces:
                        flash('No faces detected in image. Please ensure face is clearly visible.', 'error')
                        return redirect(request.url)
                    
                    # Get best quality face
                    best_face = max(faces, key=lambda f: f['confidence'] * (f['box'][2] * f['box'][3]))
                    
                    if best_face['confidence'] < 0.7:  # Lowered threshold from 0.8
                        flash(f'Low face detection confidence: {best_face["confidence"]:.2f}. Please use a clearer image.', 'warning')
                    
                    x, y, w, h = best_face['box']
                    # Ensure valid coordinates
                    x = max(0, x)
                    y = max(0, y)
                    w = min(w, img.shape[1] - x - 1)
                    h = min(h, img.shape[0] - y - 1)
                    
                    # Check if face region is valid
                    if w <= 10 or h <= 10:
                        flash('Invalid face region detected', 'error')
                        return redirect(request.url)
                    
                    face_img = rgb_img[y:y+h, x:x+w]
                    
                    if face_img.size == 0:
                        flash('Invalid face region detected', 'error')
                        return redirect(request.url)
                    
                    # Generate embedding
                    embedding = get_embedding(face_img)
                    if embedding is None:
                        flash('Error generating face embedding', 'error')
                        return redirect(request.url)
                    
                    # Check if embedding is valid (not all zeros)
                    if np.all(embedding == 0):
                        flash('Invalid face embedding generated', 'error')
                        return redirect(request.url)
                    
                    # Save to database
                    embedding_blob = pickle.dumps(embedding)
                    cursor.execute(
                        "INSERT INTO face_embeddings (employee_id, embedding, quality_score) "
                        "VALUES (%s, %s, %s)",
                        (employee_id, embedding_blob, best_face['confidence'])
                    )
                    conn.commit()
                    
                    # Retrain model automatically after adding new face
                    success, message = train_model()
                    if success:
                        flash(f'Face registered successfully! {message}', 'success')
                    else:
                        flash(f'Face registered but model training failed: {message}', 'warning')
                    
                    return redirect(url_for('manage_employees'))
                    
                except Exception as e:
                    logger.error(f"Face detection error: {e}")
                    flash(f'Error processing image: {str(e)}', 'error')
                    return redirect(request.url)
                
        return render_template('register_face.html', employee=employee)
        
    except mysql.connector.Error as err:
        flash(f'Database error: {err}', 'error')
    except Exception as e:
        flash(f'Unexpected error: {str(e)}', 'error')
    finally:
        if conn.is_connected():
            close_cursor_safe(cursor)
            close_conn_safe(conn)
    
    return redirect(url_for('manage_employees'))

    
@app.route('/admin/attendance')
@admin_required
def view_attendance():
    """View attendance records"""
    conn = get_db_connection()
    if not conn:
        flash('Database connection error', 'error')
        return render_template('view_attendance.html', attendance_records=[], start_date=date.today(), end_date=date.today())
    
    try:
        cursor = conn.cursor(dictionary=True)
        
        # Get date range from request
        start_date = request.args.get('start_date', date.today().strftime('%Y-%m-%d'))
        end_date = request.args.get('end_date', date.today().strftime('%Y-%m-%d'))
        
        cursor.execute("""
            SELECT e.name, e.department, a.check_in, a.check_out, a.status, a.confidence, a.location
            FROM attendance a
            JOIN employees e ON a.employee_id = e.id
            WHERE DATE(a.check_in) BETWEEN %s AND %s
            ORDER BY a.check_in DESC
        """, (start_date, end_date))
        attendance_records = cursor.fetchall()
        
    except mysql.connector.Error as err:
        flash(f'Database error: {err}', 'error')
        attendance_records = []
    finally:
        if conn.is_connected():
            close_cursor_safe(cursor)
            close_conn_safe(conn)
    
    return render_template('view_attendance.html', 
                         attendance_records=attendance_records,
                         start_date=start_date, end_date=end_date)

@app.route('/admin/recognition')
@admin_required
def recognition_control():
    """Recognition service control"""
    return render_template('recognition_control.html', 
                         recognition_active=recognition_active,
                         cameras=CCTV_CAMERAS)

@app.route('/api/recognition/start', methods=['POST'])
@admin_required
def start_recognition():
    """Start recognition service API"""
    global recognition_active, recognition_thread
    
    if recognition_active:
        return jsonify({'success': False, 'message': 'Recognition service is already running'})
    
    try:
        recognition_active = True
        recognition_thread = threading.Thread(target=start_recognition_service, daemon=True)
        recognition_thread.start()
        
        logger.info("Recognition service started")
        return jsonify({'success': True, 'message': 'Recognition service started successfully'})
    except Exception as e:
        recognition_active = False
        logger.error(f"Error starting recognition service: {e}")
        return jsonify({'success': False, 'message': f'Error starting service: {str(e)}'})


@app.route('/api/recognition/stop', methods=['POST'])
@admin_required
def stop_recognition():
    """Stop recognition service API"""
    success, message = stop_recognition_service()
    return jsonify({'success': success, 'message': message})

@app.route('/api/model/train', methods=['POST'])
@admin_required
def train_model_api():
    """Train face recognition model API"""
    success, message = train_model()
    return jsonify({'success': success, 'message': message})


@app.route('/live-stream')
@login_required
def live_stream():
    """Live CCTV stream"""
    return render_template('live_stream.html', cameras=CCTV_CAMERAS)

def generate_frames():
    """Generate video frames for streaming with stable connection and minimal latency"""
    # Create a static placeholder frame
    placeholder = None
    if placeholder is None:
        blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(blank_frame, "Skyhighes Technologies", (150, 220), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(blank_frame, "Live CCTV Stream", (180, 260), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        _, buffer = cv2.imencode('.jpg', blank_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
        placeholder = buffer.tobytes()
    
    last_valid_frame = placeholder
    frame_interval = 1/25  # Target 25 FPS
    last_frame_time = time_module.time()
    
    while True:
        try:
            current_time = time_module.time()
            elapsed = current_time - last_frame_time
            
            # Only process if it's time for a new frame
            if elapsed >= frame_interval:
                frame_data = None
                frame_bytes = last_valid_frame
                
                # Try to get the latest frame without blocking
                if not live_stream_queue.empty():
                    try:
                        # Get the most recent frame by emptying the queue
                        while not live_stream_queue.empty():
                            frame_data = live_stream_queue.get_nowait()
                        
                        if frame_data and 'frame' in frame_data:
                            frame_bytes = frame_data['frame']
                            last_valid_frame = frame_bytes  # Update last valid frame
                            last_frame_time = current_time
                    except queue.Empty:
                        # Use last valid frame if queue is empty
                        frame_bytes = last_valid_frame
                else:
                    # Use last valid frame if queue is empty
                    frame_bytes = last_valid_frame
                
                # Yield frame in MJPEG format
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            # Small sleep to prevent CPU overload
            time_module.sleep(0.0001)
            
        except Exception as e:
            # Log error but continue with placeholder
            logger.error(f"Error in frame generation: {e}")
            # Yield placeholder on error but continue
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + placeholder + b'\r\n')
            time_module.sleep(0.1)

def reconnect_camera(camera_id, camera_info):
    """Reconnect to a camera with enhanced error handling"""
    try:
        # Use default backend for webcam
        if camera_info['rtsp_url'] == 0:
            cap = cv2.VideoCapture(CCTV_CAMERAS['laptop_camera']['rtsp_url'])
        else:
            # Use FFMPEG backend for RTSP streams
            cap = cv2.VideoCapture(camera_info['rtsp_url'], cv2.CAP_FFMPEG)
        
        # Set parameters
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 25)
        
        # Try to open with timeout
        start_time = time_module.time()
        opened = False
        while time_module.time() - start_time < 5:  # 5 second timeout
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    opened = True
                    break
            time_module.sleep(0.1)
        
        if opened:
            logger.info(f"Reconnected to camera {camera_id}")
            return cap
        else:
            cap.release()
            logger.warning(f"Failed to reconnect to camera {camera_id}")
    except Exception as e:
        logger.error(f"Error reconnecting to camera {camera_id}: {e}")
    return None

@app.route('/video-feed')
@login_required
def video_feed():
    """Video feed endpoint with proper MJPEG headers"""
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame',
                   headers={'Cache-Control': 'no-store, no-cache, must-revalidate, max-age=0', 'Pragma': 'no-cache'})


# Socket.IO events
@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    emit('connected', {'message': 'Connected to Skyhighes Technologies Attendance System'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    pass

# Error handlers
@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500

# Cleanup function
def cleanup():
    """Cleanup function for graceful shutdown"""
    global recognition_active
    if recognition_active:
        stop_recognition_service()
    logger.info("Application shutting down")

# Register cleanup function
atexit.register(cleanup)

# Create templates directory
os.makedirs('templates', exist_ok=True)

@app.route('/admin/model_dashboard')
@admin_required
def model_dashboard():
    """Dashboard to show model training statistics"""
    conn = get_db_connection()
    if not conn:
        flash('Database connection error', 'error')
        return render_template('model_dashboard.html', stats={}, training_history=[])
    
    try:
        cursor = conn.cursor(dictionary=True)
        
        # Get total statistics
        cursor.execute("SELECT COUNT(*) as total FROM employees WHERE status = 'active'")
        total_employees = cursor.fetchone()['total']
        
        cursor.execute("""
            SELECT COUNT(DISTINCT fe.employee_id) as employees_with_faces,
                   COUNT(fe.id) as total_embeddings
            FROM face_embeddings fe
            JOIN employees e ON fe.employee_id = e.id
            WHERE e.status = 'active'
        """)
        face_stats = cursor.fetchone()
        
        # Get model info
        model_loaded = AI_AVAILABLE and hasattr(svm_model, 'classes_') and len(svm_model.classes_) > 0
        if model_loaded:
            trained_employees = len(svm_model.classes_)
        else:
            trained_employees = 0
        
        # Get training history
        cursor.execute("""
            SELECT * FROM model_training_history 
            ORDER BY training_date DESC 
            LIMIT 10
        """)
        training_history = cursor.fetchall()
        
        # Get last training time
        cursor.execute("""
            SELECT training_date, status 
            FROM model_training_history 
            WHERE status = 'success'
            ORDER BY training_date DESC 
            LIMIT 1
        """)
        last_training = cursor.fetchone()
        
        stats = {
            'total_employees': total_employees,
            'employees_with_faces': face_stats['employees_with_faces'],
            'total_embeddings': face_stats['total_embeddings'],
            'model_loaded': model_loaded,
            'trained_employees': trained_employees,
            'last_training': last_training
        }
        
    except mysql.connector.Error as err:
        flash(f'Database error: {err}', 'error')
        stats = {}
        training_history = []
    except Exception as e:
        flash(f'Unexpected error: {str(e)}', 'error')
        stats = {}
        training_history = []
    finally:
        if conn.is_connected():
            close_cursor_safe(cursor)
            close_conn_safe(conn)
    
    return render_template('model_dashboard.html', stats=stats, training_history=training_history)


@app.route('/admin/cctv_config', methods=['GET', 'POST'])
@admin_required
def cctv_config():
    """Configure CCTV cameras with improved validation"""
    if request.method == 'POST':
        camera_id = request.form.get('camera_id')
        name = request.form.get('name')
        location = request.form.get('location')
        rtsp_url = request.form.get('rtsp_url')
        status = request.form.get('status')
        
        if not camera_id or not name or not rtsp_url:
            flash('Camera ID, name and RTSP URL are required', 'error')
            return redirect(url_for('cctv_config'))
        
        # Validate RTSP URL
        is_valid, message = validate_rtsp_url(rtsp_url)
        if not is_valid:
            flash(f'Invalid RTSP URL: {message}', 'error')
            return redirect(url_for('cctv_config'))
        
        # Update configuration
        conn = get_db_connection()
        if conn:
            try:
                cursor = conn.cursor()
                
                # Update or insert camera configuration
                cursor.execute(
                    """INSERT INTO cctv_cameras (id, name, location, rtsp_url, status)
                    VALUES (%s, %s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE
                    name = VALUES(name),
                    location = VALUES(location),
                    rtsp_url = VALUES(rtsp_url),
                    status = VALUES(status),
                    updated_at = NOW()""",
                    (camera_id, name, location, rtsp_url, status)
                )
                conn.commit()
                
                # Update in-memory configuration immediately
                CCTV_CAMERAS[camera_id] = {
                    'name': name,
                    'location': location,
                    'rtsp_url': rtsp_url,
                    'status': status
                }
                
                # If recognition is active, restart it to apply changes
                if recognition_active:
                    stop_recognition_service()
                    time_module.sleep(1)
                    start_recognition_service()
                
                flash('CCTV configuration updated successfully', 'success')
            except mysql.connector.Error as err:
                flash(f'Database error: {err}', 'error')
            finally:
                if conn.is_connected():
                    close_cursor_safe(cursor)
                    close_conn_safe(conn)
        
        return redirect(url_for('cctv_config'))
    
    # Load current CCTV configuration
    conn = get_db_connection()
    if not conn:
        flash('Database connection error', 'error')
        return render_template('cctv_config.html', cameras={})
    
    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM cctv_cameras ORDER BY name")
        cameras = {str(row['id']): dict(row) for row in cursor.fetchall()}
        return render_template('cctv_config.html', cameras=cameras)
    except mysql.connector.Error as err:
        flash(f'Database error: {err}', 'error')
        return render_template('cctv_config.html', cameras={})
    finally:
        if conn and conn.is_connected():
            close_cursor_safe(cursor)
            close_conn_safe(conn)
            
# Create all required templates

# Add this CSS to your existing templates

# Update index.html template with premium CSS
with open('templates/index.html', 'w') as f:
    f.write("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Skyhighes Technologies - Premium Attendance System</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary: #1a2a6c;
            --secondary: #b21f1f;
            --accent: #ff8a00;
            --light: #f8f9fa;
            --dark: #212529;
            --gradient: linear-gradient(135deg, var(--primary), var(--secondary), var(--accent));
        }
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: 'Poppins', sans-serif;
        }
        
        body {
            background: linear-gradient(135deg, #1a2a6c, #b21f1f, #1a2a6c);
            color: white;
            min-height: 100vh;
            background-attachment: fixed;
            background-size: 400% 400%;
            animation: gradientBG 15s ease infinite;
        }
        
        @keyframes gradientBG {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        
        .navbar {
            background: rgba(0, 0, 0, 0.7) !important;
            backdrop-filter: blur(10px);
            padding: 15px 0;
            transition: all 0.3s ease;
        }
        
        .navbar-brand {
            font-weight: 700;
            font-size: 1.5rem;
            color: white !important;
        }
        
        .hero {
            padding: 120px 0;
            text-align: center;
        }
        
        .hero h1 {
            font-size: 3.5rem;
            font-weight: 800;
            margin-bottom: 1.5rem;
            text-shadow: 2px 2px 10px rgba(0,0,0,0.3);
        }
        
        .hero p {
            font-size: 1.2rem;
            max-width: 700px;
            margin: 0 auto 2.5rem;
            opacity: 0.9;
        }
        
        .features {
            padding: 100px 0;
            position: relative;
        }
        
        .features::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            z-index: -1;
        }
        
        .feature-card {
            background: rgba(255, 255, 255, 0.15);
            border-radius: 20px;
            padding: 40px 30px;
            margin-bottom: 30px;
            transition: all 0.3s ease;
            border: 1px solid rgba(255,255,255,0.1);
            backdrop-filter: blur(5px);
            height: 100%;
            position: relative;
            overflow: hidden;
        }
        
        .feature-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 5px;
            background: var(--gradient);
        }
        
        .feature-card:hover {
            transform: translateY(-10px);
            background: rgba(255, 255, 255, 0.2);
            box-shadow: 0 15px 30px rgba(0,0,0,0.2);
        }
        
        .feature-card .icon {
            font-size: 3rem;
            margin-bottom: 20px;
            background: var(--gradient);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .feature-card h3 {
            font-size: 1.5rem;
            margin-bottom: 15px;
            font-weight: 600;
        }
        
        .feature-card p {
            opacity: 0.9;
            line-height: 1.6;
        }
        
        .btn-premium {
            background: var(--gradient);
            border: none;
            padding: 15px 35px;
            font-size: 1.1rem;
            font-weight: 600;
            border-radius: 50px;
            transition: all 0.3s ease;
            color: white;
            position: relative;
            overflow: hidden;
            z-index: 1;
        }
        
        .btn-premium::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 0%;
            height: 100%;
            background: linear-gradient(to right, #ff8a00, #da1b60);
            transition: all 0.3s ease;
            z-index: -1;
        }
        
        .btn-premium:hover {
            transform: translateY(-3px);
            box-shadow: 0 10px 20px rgba(0,0,0,0.2);
        }
        
        .btn-premium:hover::before {
            width: 100%;
        }
        
        .footer {
            padding: 40px 0;
            text-align: center;
            background: rgba(0, 0, 0, 0.7);
            backdrop-filter: blur(10px);
            margin-top: 80px;
        }
        
        /* Animations */
        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(30px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        .animate-up {
            animation: fadeInUp 0.8s ease forwards;
        }
        
        .delay-1 { animation-delay: 0.2s; }
        .delay-2 { animation-delay: 0.4s; }
        .delay-3 { animation-delay: 0.6s; }
        
        /* Responsive */
        @media (max-width: 768px) {
            .hero h1 {
                font-size: 2.5rem;
            }
            
            .hero p {
                font-size: 1rem;
            }
            
            .feature-card {
                padding: 30px 20px;
            }
        }
    </style>
</head>
<body>
    <nav class="navbar navbar-expand-lg navbar-dark">
        <div class="container">
            <a class="navbar-brand" href="#">
                <i class="fas fa-camera-retro me-2"></i>
                Skyhighes Technologies
            </a>
            <div class="d-flex">
                <a href="/login" class="btn btn-outline-light btn-premium">Login</a>
            </div>
        </div>
    </nav>

    <div class="container">
        <div class="hero">
            <h1 class="animate-up">Premium Face Recognition Attendance System</h1>
            <p class="animate-up delay-1">Advanced AI-powered attendance management with premium interface and CCTV integration</p>
            <a href="/login" class="btn btn-premium animate-up delay-2">Get Started <i class="fas fa-arrow-right ms-2"></i></a>
        </div>

        <div class="features">
            <h2 class="text-center mb-5 display-4 animate-up">Advanced Features</h2>
            <div class="row">
                <div class="col-md-4 mb-4 animate-up delay-1">
                    <div class="feature-card">
                        <div class="text-center">
                            <i class="fas fa-robot icon"></i>
                        </div>
                        <h3 class="text-center">AI-Powered Recognition</h3>
                        <p class="text-center">State-of-the-art facial recognition with 99% accuracy using deep learning algorithms.</p>
                    </div>
                </div>
                <div class="col-md-4 mb-4 animate-up delay-2">
                    <div class="feature-card">
                        <div class="text-center">
                            <i class="fas fa-video icon"></i>
                        </div>
                        <h3 class="text-center">CCTV Integration</h3>
                        <p class="text-center">Seamless integration with existing CCTV systems for automated attendance.</p>
                    </div>
                </div>
                <div class="col-md-4 mb-4 animate-up delay-3">
                    <div class="feature-card">
                        <div class="text-center">
                            <i class="fas fa-chart-line icon"></i>
                        </div>
                        <h3 class="text-center">Real-Time Analytics</h3>
                        <p class="text-center">Comprehensive dashboards with real-time attendance analytics and reporting.</p>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <footer class="footer">
        <div class="container">
            <p>&copy; 2024 Skyhighes Technologies. Premium Attendance System.</p>
            <p>Enterprise Edition v3.0</p>
        </div>
    </footer>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        // Add intersection observer for animations
        document.addEventListener('DOMContentLoaded', function() {
            const animatedElements = document.querySelectorAll('.animate-up');
            
            const observer = new IntersectionObserver((entries) => {
                entries.forEach(entry => {
                    if (entry.isIntersecting) {
                        entry.target.style.visibility = 'visible';
                        observer.unobserve(entry.target);
                    }
                });
            }, { threshold: 0.1 });
            
            animatedElements.forEach(element => {
                element.style.visibility = 'hidden';
                observer.observe(element);
            });
        });
    </script>
</body>
</html>""")

# Update login.html with premium CSS
with open('templates/login.html', 'w') as f:
    f.write("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Login | Skyhighes Technologies</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary: #1a2a6c;
            --secondary: #b21f1f;
            --accent: #ff8a00;
            --light: #f8f9fa;
            --dark: #212529;
            --gradient: linear-gradient(135deg, var(--primary), var(--secondary), var(--accent));
        }
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: 'Poppins', sans-serif;
        }
        
        body {
            background: linear-gradient(135deg, #1a2a6c, #b21f1f, #1a2a6c);
            height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            background-size: 400% 400%;
            animation: gradientBG 15s ease infinite;
        }
        
        @keyframes gradientBG {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        
        .login-container {
            width: 100%;
            max-width: 450px;
            padding: 20px;
        }
        
        .login-card {
            background: rgba(255, 255, 255, 0.9);
            border-radius: 20px;
            box-shadow: 0 15px 30px rgba(0,0,0,0.2);
            overflow: hidden;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.2);
            transform: translateY(0);
            transition: all 0.3s ease;
        }
        
        .login-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 20px 40px rgba(0,0,0,0.3);
        }
        
        .login-header {
            background: var(--gradient);
            color: white;
            padding: 40px 30px;
            text-align: center;
            position: relative;
            overflow: hidden;
        }
        
        .login-header::before {
            content: '';
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: rgba(255,255,255,0.1);
            transform: rotate(45deg);
        }
        
        .login-header i {
            font-size: 3.5rem;
            margin-bottom: 15px;
            display: block;
        }
        
        .brand-text {
            font-weight: 800;
            font-size: 1.8rem;
            letter-spacing: 1px;
            margin-bottom: 5px;
            position: relative;
            z-index: 1;
        }
        
        .login-body {
            padding: 30px;
        }
        
        .form-group {
            margin-bottom: 20px;
            position: relative;
        }
        
        .form-label {
            font-weight: 500;
            margin-bottom: 8px;
            color: #495057;
        }
        
        .form-control {
            border: 2px solid #e9ecef;
            border-radius: 10px;
            padding: 12px 15px;
            font-size: 1rem;
            transition: all 0.3s ease;
            background: rgba(255,255,255,0.8);
        }
        
        .form-control:focus {
            border-color: var(--primary);
            box-shadow: 0 0 0 0.2rem rgba(26, 42, 108, 0.25);
        }
        
        .input-group-text {
            background: transparent;
            border: 2px solid #e9ecef;
            border-right: none;
            border-radius: 10px 0 0 10px;
        }
        
        .form-check-input:checked {
            background-color: var(--primary);
            border-color: var(--primary);
        }
        
        .btn-login {
            background: var(--gradient);
            border: none;
            padding: 12px;
            font-weight: 600;
            width: 100%;
            border-radius: 10px;
            color: white;
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }
        
        .btn-login::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: linear-gradient(to right, #ff8a00, #da1b60);
            opacity: 0;
            transition: all 0.3s ease;
        }
        
        .btn-login:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }
        
        .btn-login:hover::before {
            opacity: 1;
        }
        
        .btn-login span {
            position: relative;
            z-index: 1;
        }
        
        .alert {
            border-radius: 10px;
            border: none;
            padding: 12px 15px;
        }
        
        /* Animation */
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .login-card {
            animation: fadeIn 0.8s ease forwards;
        }
    </style>
</head>
<body>
    <div class="login-container">
        <div class="login-card">
            <div class="login-header">
                <i class="fas fa-camera-retro"></i>
                <h2 class="brand-text">SKYHIGHES TECHNOLOGIES</h2>
                <p>Premium Attendance System</p>
            </div>
            <div class="login-body">
                {% with messages = get_flashed_messages(with_categories=true) %}
                    {% if messages %}
                        {% for category, message in messages %}
                            <div class="alert alert-{{ category }} alert-dismissible fade show" role="alert">
                                {{ message }}
                                <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
                            </div>
                        {% endfor %}
                    {% endif %}
                {% endwith %}
                
                <form method="POST" action="/login">
                    <div class="form-group">
                        <label for="username" class="form-label">Username</label>
                        <div class="input-group">
                            <span class="input-group-text"><i class="fas fa-user"></i></span>
                            <input type="text" class="form-control" id="username" name="username" required>
                        </div>
                    </div>
                    <div class="form-group">
                        <label for="password" class="form-label">Password</label>
                        <div class="input-group">
                            <span class="input-group-text"><i class="fas fa-lock"></i></span>
                            <input type="password" class="form-control" id="password" name="password" required>
                        </div>
                    </div>
                    <div class="form-group">
                        <div class="form-check form-check-inline">
                            <input class="form-check-input" type="radio" name="user_type" id="admin" value="admin" checked>
                            <label class="form-check-label" for="admin">
                                Admin User
                            </label>
                        </div>
                        <div class="form-check form-check-inline">
                            <input class="form-check-input" type="radio" name="user_type" id="employee" value="employee">
                            <label class="form-check-label" for="employee">
                                Employee
                            </label>
                        </div>
                    </div>
                    <button type="submit" class="btn btn-login mt-3">
                        <span>Login <i class="fas fa-sign-in-alt ms-2"></i></span>
                    </button>
                </form>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>""")

# Update admin_dashboard.html with premium CSS
with open('templates/admin_dashboard.html', 'w') as f:
    f.write("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Admin Dashboard | Skyhighes Technologies</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {
            --primary: #1a2a6c;
            --secondary: #b21f1f;
            --accent: #ff8a00;
            --light: #f8f9fa;
            --dark: #212529;
            --gradient: linear-gradient(135deg, var(--primary), var(--secondary), var(--accent));
            --sidebar-width: 280px;
        }
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: 'Poppins', sans-serif;
        }
        
        body {
            background-color: #f8f9fa;
            color: #495057;
            overflow-x: hidden;
        }
        
        /* Sidebar Styles */
        .sidebar {
            background: var(--gradient);
            color: white;
            height: 100vh;
            position: fixed;
            width: var(--sidebar-width);
            padding-top: 20px;
            transition: all 0.3s ease;
            z-index: 1000;
            box-shadow: 5px 0 15px rgba(0,0,0,0.1);
        }
        
        .sidebar-brand {
            padding: 0 20px 20px;
            text-align: center;
            border-bottom: 1px solid rgba(255,255,255,0.1);
            margin-bottom: 20px;
        }
        
        .sidebar-brand i {
            font-size: 2rem;
            margin-bottom: 10px;
        }
        
        .sidebar-brand h4 {
            font-weight: 700;
            margin-bottom: 5px;
        }
        
        .sidebar .nav-link {
            color: rgba(255,255,255,0.9);
            padding: 12px 25px;
            margin: 8px 15px;
            border-radius: 10px;
            transition: all 0.3s ease;
            font-weight: 500;
            position: relative;
            overflow: hidden;
        }
        
        .sidebar .nav-link::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: rgba(255,255,255,0.1);
            transition: all 0.3s ease;
        }
        
        .sidebar .nav-link:hover::before,
        .sidebar .nav-link.active::before {
            left: 0;
        }
        
        .sidebar .nav-link:hover, 
        .sidebar .nav-link.active {
            color: white;
            background: rgba(255,255,255,0.1);
            transform: translateX(5px);
        }
        
        .sidebar .nav-link i {
            margin-right: 12px;
            width: 20px;
            text-align: center;
            font-size: 1.1rem;
        }
        
        .sidebar .badge {
            font-size: 0.7rem;
            padding: 4px 8px;
        }
        
        /* Main Content */
        .main-content {
            margin-left: var(--sidebar-width);
            padding: 20px;
            transition: all 0.3s ease;
        }
        
        .navbar {
            background: white;
            box-shadow: 0 2px 15px rgba(0,0,0,0.1);
            margin-left: var(--sidebar-width);
            padding: 15px 20px;
            transition: all 0.3s ease;
        }
        
        /* Stat Cards */
        .stat-card {
            border-radius: 15px;
            overflow: hidden;
            margin-bottom: 25px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.08);
            transition: all 0.3s ease;
            background: white;
        }
        
        .stat-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 25px rgba(0,0,0,0.15);
        }
        
        .stat-card-header {
            padding: 20px;
            color: white;
            font-weight: 600;
            font-size: 1.1rem;
        }
        
        .stat-card-body {
            padding: 25px;
            text-align: center;
        }
        
        .stat-number {
            font-size: 2.5rem;
            font-weight: 700;
            margin: 10px 0;
            background: var(--gradient);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .card-1 .stat-card-header { background: linear-gradient(to right, #1a2a6c, #3a5fc5); }
        .card-2 .stat-card-header { background: linear-gradient(to right, #00b09b, #96c93d); }
        .card-3 .stat-card-header { background: linear-gradient(to right, #ff8a00, #da1b60); }
        .card-4 .stat-card-header { background: linear-gradient(to right, #654ea3, #da22ff); }
        
        /* System Status */
        .system-status {
            background: white;
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.08);
            margin-bottom: 25px;
            transition: all 0.3s ease;
        }
        
        .system-status:hover {
            box-shadow: 0 10px 25px rgba(0,0,0,0.12);
        }
        
        .status-badge {
            padding: 6px 12px;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 600;
        }
        
        .status-active {
            background: #d4edda;
            color: #155724;
        }
        
        .status-inactive {
            background: #f8d7da;
            color: #721c24;
        }
        
        /* Recent Table */
        .recent-table {
            background: white;
            border-radius: 15px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.08);
            overflow: hidden;
            transition: all 0.3s ease;
        }
        
        .recent-table:hover {
            box-shadow: 0 10px 25px rgba(0,0,0,0.12);
        }
        
        .table-header {
            background: linear-gradient(to right, #f8f9fa, #e9ecef);
            padding: 20px;
            border-bottom: 1px solid #dee2e6;
        }
        
        .table-header h5 {
            margin: 0;
            font-weight: 600;
            color: #495057;
        }
        
        /* Buttons */
        .btn-success, .btn-danger {
            padding: 8px 16px;
            border-radius: 8px;
            font-weight: 500;
            transition: all 0.3s ease;
        }
        
        .btn-success:hover, .btn-danger:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }
        
        /* Animations */
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .animate-card {
            animation: fadeIn 0.6s ease forwards;
        }
        
        .delay-1 { animation-delay: 0.2s; }
        .delay-2 { animation-delay: 0.4s; }
        .delay-3 { animation-delay: 0.6s; }
        
        /* Responsive */
        @media (max-width: 992px) {
            .sidebar {
                width: 80px;
                transform: translateX(0);
            }
            
            .sidebar .nav-link span {
                display: none;
            }
            
            .sidebar .nav-link i {
                margin-right: 0;
                font-size: 1.3rem;
            }
            
            .sidebar-brand h4, .sidebar-brand p {
                display: none;
            }
            
            .main-content, .navbar {
                margin-left: 80px;
            }
        }
        
        @media (max-width: 768px) {
            .sidebar {
                width: 0;
                transform: translateX(-100%);
            }
            
            .main-content, .navbar {
                margin-left: 0;
            }
            
            .sidebar.show {
                width: 280px;
                transform: translateX(0);
            }
            
            .navbar-toggler {
                display: block;
            }
        }
    </style>
</head>
<body>
    <!-- Sidebar -->
    <div class="sidebar">
        <div class="sidebar-brand">
            <i class="fas fa-camera-retro"></i>
            <h4>Skyhighes Technologies</h4>
            <p>Premium Attendance</p>
        </div>
        <ul class="nav flex-column">
            <li class="nav-item">
                <a class="nav-link active" href="/admin/dashboard">
                    <i class="fas fa-tachometer-alt"></i> <span>Dashboard</span>
                </a>
            </li>
            <li class="nav-item">
                <a class="nav-link" href="/admin/employees">
                    <i class="fas fa-users"></i> <span>Manage Employees</span>
                </a>
            </li>
            <li class="nav-item">
                <a class="nav-link" href="/admin/attendance">
                    <i class="fas fa-clipboard-list"></i> <span>View Attendance</span>
                </a>
            </li>
            <li class="nav-item">
                <a class="nav-link" href="/admin/recognition">
                    <i class="fas fa-video"></i> <span>CCTV Recognition</span>
                </a>
            </li>
            <li class="nav-item">
                <a class="nav-link" href="/admin/checkin_requests">
                    <i class="fas fa-bell"></i> <span>Check-in Requests</span> 
                    <span class="badge bg-danger" id="request-badge">{% if stats.pending_requests > 0 %}{{ stats.pending_requests }}{% endif %}</span>
                </a>
            </li>
            <li class="nav-item">
                <a class="nav-link" href="/admin/attendance_analysis">
                    <i class="fas fa-chart-bar"></i> <span>Attendance Analysis</span>
                </a>
            </li>
            <li class="nav-item">
                <a class="nav-link" href="/admin/model_dashboard">
                    <i class="fas fa-brain"></i> <span>Model Dashboard</span>
                </a>
            </li>
            <li class="nav-item">
                <a class="nav-link" href="/admin/test_face_recognition">
                    <i class="fas fa-check-circle"></i> <span>Test Face Recognition</span>
                </a>
            </li>
            <li class="nav-item">
                <a class="nav-link" href="/live-stream">
                    <i class="fas fa-tv"></i> <span>Live Stream</span>
                </a>
            </li>
            <li class="nav-item mt-4">
                <a class="nav-link" href="/logout">
                    <i class="fas fa-sign-out-alt"></i> <span>Logout</span>
                </a>
            </li>
        </ul>
    </div>

    <!-- Navbar -->
    <nav class="navbar">
        <div class="container-fluid">
            <div class="d-flex align-items-center">
                <button class="btn btn-sm btn-outline-secondary d-lg-none me-2" id="sidebarToggle">
                    <i class="fas fa-bars"></i>
                </button>
                <span class="navbar-brand mb-0 h1">Admin Dashboard</span>
            </div>
            <div class="d-flex align-items-center">
                <span class="me-3"><i class="fas fa-user-circle me-2"></i> {{ session.username }}</span>
                <div class="dropdown">
                    <button class="btn btn-sm btn-outline-secondary dropdown-toggle" type="button" data-bs-toggle="dropdown">
                        <i class="fas fa-cog"></i>
                    </button>
                    <ul class="dropdown-menu dropdown-menu-end">
                        <li><a class="dropdown-item" href="#"><i class="fas fa-user me-2"></i> Profile</a></li>
                        <li><a class="dropdown-item" href="#"><i class="fas fa-cog me-2"></i> Settings</a></li>
                        <li><hr class="dropdown-divider"></li>
                        <li><a class="dropdown-item" href="/logout"><i class="fas fa-sign-out-alt me-2"></i> Logout</a></li>
                    </ul>
                </div>
            </div>
        </div>
    </nav>

    <!-- Main Content -->
    <div class="main-content">
        <div class="container-fluid">
            <h2 class="mb-4">Today's Overview</h2>
            
            <!-- Stats Cards -->
            <div class="row">
                <div class="col-md-3">
                    <div class="stat-card card-1 animate-card">
                        <div class="stat-card-header">
                            <i class="fas fa-users me-2"></i> Total Employees
                        </div>
                        <div class="stat-card-body">
                            <div class="stat-number">{{ stats.total_employees }}</div>
                            <div>Active Staff</div>
                        </div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="stat-card card-2 animate-card delay-1">
                        <div class="stat-card-header">
                            <i class="fas fa-check-circle me-2"></i> Present Today
                        </div>
                        <div class="stat-card-body">
                            <div class="stat-number">{{ stats.present_today }}</div>
                            <div>Employees Present</div>
                        </div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="stat-card card-3 animate-card delay-2">
                        <div class="stat-card-header">
                            <i class="fas fa-clock me-2"></i> Late Arrivals
                        </div>
                        <div class="stat-card-body">
                            <div class="stat-number">{{ stats.late_today }}</div>
                            <div>Late Today</div>
                        </div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="stat-card card-4 animate-card delay-3">
                        <div class="stat-card-header">
                            <i class="fas fa-user-slash me-2"></i> Absent Today
                        </div>
                        <div class="stat-card-body">
                            <div class="stat-number">{{ stats.absent_today }}</div>
                            <div>Employees Absent</div>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="row mt-4">
                <div class="col-md-6">
                    <!-- System Status -->
                    <div class="system-status animate-card">
                        <h4 class="mb-4">System Status</h4>
                        <div class="row">
                            <div class="col-6 mb-3">
                                <div>Recognition Status:</div>
                                <span id="recognition-status-badge" class="status-badge {% if stats.system_status.recognition_active %}status-active{% else %}status-inactive{% endif %}">
                                    {% if stats.system_status.recognition_active %}ACTIVE{% else %}INACTIVE{% endif %}
                                </span>
                            </div>
                            <div class="col-6 mb-3">
                                <div>Cameras Active:</div>
                                <span>{{ stats.system_status.cameras_active }}/{{ stats.system_status.total_cameras }}</span>
                            </div>
                            <div class="col-6 mb-3">
                                <div>AI Model:</div>
                                <span class="status-badge {% if stats.system_status.model_loaded %}status-active{% else %}status-inactive{% endif %}">
                                    {% if stats.system_status.model_loaded %}LOADED{% else %}NOT LOADED{% endif %}
                                </span>
                            </div>
                            <div class="col-6 mb-3">
                                <div>Last Recognition:</div>
                                <span>{% if stats.recognition_stats.last_recognition %}{{ stats.recognition_stats.last_recognition.strftime('%H:%M:%S') }}{% else %}N/A{% endif %}</span>
                            </div>
                        </div>
                        <div class="mt-3">
                            <button class="btn btn-success me-2" id="start-recognition">
                                <i class="fas fa-play me-1"></i> Start Recognition
                            </button>
                            <button class="btn btn-danger" id="stop-recognition">
                                <i class="fas fa-stop me-1"></i> Stop Recognition
                            </button>
                        </div>
                    </div>
                </div>
                
                <div class="col-md-6">
                    <!-- Recent Activity -->
                    <div class="recent-table animate-card">
                        <div class="table-header">
                            <h5 class="mb-0">Recent Attendance</h5>
                        </div>
                        <div class="table-responsive">
                            <table class="table table-hover">
                                <thead>
                                    <tr>
                                        <th>Name</th>
                                        <th>Department</th>
                                        <th>Time</th>
                                        <th>Status</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {% for record in stats.recent_attendance %}
                                    <tr>
                                        <td>{{ record.name }}</td>
                                        <td>{{ record.department }}</td>
                                        <td>{{ record.check_in.strftime('%H:%M') }}</td>
                                        <td>
                                            {% if record.status == 'present' %}
                                                <span class="badge bg-success">Present</span>
                                            {% elif record.status == 'late' %}
                                                <span class="badge bg-warning">Late</span>
                                            {% elif record.status == 'half_day' %}
                                                <span class="badge bg-info">Half Day</span>
                                            {% else %}
                                                <span class="badge bg-secondary">{{ record.status }}</span>
                                            {% endif %}
                                        </td>
                                    </tr>
                                    {% else %}
                                    <tr>
                                        <td colspan="4" class="text-center">No attendance records today</td>
                                    </tr>
                                    {% endfor %}
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        // Sidebar toggle for mobile
        document.getElementById('sidebarToggle').addEventListener('click', function() {
            document.querySelector('.sidebar').classList.toggle('show');
        });
        
        // Recognition control
        document.getElementById('start-recognition').addEventListener('click', function() {
            fetch('/api/recognition/start', { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        alert('Recognition service started');
                        // Update UI immediately
                        document.getElementById('recognition-status-badge').textContent = 'ACTIVE';
                        document.getElementById('recognition-status-badge').classList.remove('status-inactive');
                        document.getElementById('recognition-status-badge').classList.add('status-active');
                    } else {
                        alert('Error: ' + data.message);
                    }
                });
        });
        
        document.getElementById('stop-recognition').addEventListener('click', function() {
            fetch('/api/recognition/stop', { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        alert('Recognition service stopped');
                        // Update UI immediately
                        document.getElementById('recognition-status-badge').textContent = 'INACTIVE';
                        document.getElementById('recognition-status-badge').classList.remove('status-active');
                        document.getElementById('recognition-status-badge').classList.add('status-inactive');
                    } else {
                        alert('Error: ' + data.message);
                    }
                });
        });
        
        // Add intersection observer for animations
        document.addEventListener('DOMContentLoaded', function() {
            const animatedElements = document.querySelectorAll('.animate-card');
            
            const observer = new IntersectionObserver((entries) => {
                entries.forEach(entry => {
                    if (entry.isIntersecting) {
                        entry.target.style.visibility = 'visible';
                        observer.unobserve(entry.target);
                    }
                });
            }, { threshold: 0.1 });
            
            animatedElements.forEach(element => {
                element.style.visibility = 'hidden';
                observer.observe(element);
            });
        });
    </script>
</body>
</html>""")

# Add this to your templates (create templates/admin_base.html)
with open('templates/admin_base.html', 'w') as f:
    f.write("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% block title %}Skyhighes Technologies - Premium Attendance{% endblock %}</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    {% block extra_css %}{% endblock %}
    <style>
        :root {
            --primary: #1a2a6c;
            --secondary: #b21f1f;
            --accent: #ff8a00;
            --light: #f8f9fa;
            --dark: #212529;
            --gradient: linear-gradient(135deg, var(--primary), var(--secondary), var(--accent));
            --sidebar-width: 280px;
        }
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: 'Poppins', sans-serif;
        }
        
        body {
            background-color: #f8f9fa;
            color: #495057;
            overflow-x: hidden;
        }
        
        /* Sidebar Styles */
        .sidebar {
            background: var(--gradient);
            color: white;
            height: 100vh;
            position: fixed;
            width: var(--sidebar-width);
            padding-top: 20px;
            transition: all 0.3s ease;
            z-index: 1000;
            box-shadow: 5px 0 15px rgba(0,0,0,0.1);
        }
        
        .sidebar-brand {
            padding: 0 20px 20px;
            text-align: center;
            border-bottom: 1px solid rgba(255,255,255,0.1);
            margin-bottom: 20px;
        }
        
        .sidebar-brand i {
            font-size: 2rem;
            margin-bottom: 10px;
        }
        
        .sidebar-brand h4 {
            font-weight: 700;
            margin-bottom: 5px;
        }
        
        .sidebar .nav-link {
            color: rgba(255,255,255,0.9);
            padding: 12px 25px;
            margin: 8px 15px;
            border-radius: 10px;
            transition: all 0.3s ease;
            font-weight: 500;
            position: relative;
            overflow: hidden;
        }
        
        .sidebar .nav-link::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: rgba(255,255,255,0.1);
            transition: all 0.3s ease;
        }
        
        .sidebar .nav-link:hover::before,
        .sidebar .nav-link.active::before {
            left: 0;
        }
        
        .sidebar .nav-link:hover, 
        .sidebar .nav-link.active {
            color: white;
            background: rgba(255,255,255,0.1);
            transform: translateX(5px);
        }
        
        .sidebar .nav-link i {
            margin-right: 12px;
            width: 20px;
            text-align: center;
            font-size: 1.1rem;
        }
        
        .sidebar .badge {
            font-size: 0.7rem;
            padding: 4px 8px;
        }
        
        /* Main Content */
        .main-content {
            margin-left: var(--sidebar-width);
            padding: 20px;
            transition: all 0.3s ease;
        }
        
        .navbar {
            background: white;
            box-shadow: 0 2px 15px rgba(0,0,0,0.1);
            margin-left: var(--sidebar-width);
            padding: 15px 20px;
            transition: all 0.3s ease;
        }
        
        /* Responsive */
        @media (max-width: 992px) {
            .sidebar {
                width: 80px;
                transform: translateX(0);
            }
            
            .sidebar .nav-link span {
                display: none;
            }
            
            .sidebar .nav-link i {
                margin-right: 0;
                font-size: 1.3rem;
            }
            
            .sidebar-brand h4, .sidebar-brand p {
                display: none;
            }
            
            .main-content, .navbar {
                margin-left: 80px;
            }
        }
        
        @media (max-width: 768px) {
            .sidebar {
                width: 0;
                transform: translateX(-100%);
            }
            
            .main-content, .navbar {
                margin-left: 0;
            }
            
            .sidebar.show {
                width: 280px;
                transform: translateX(0);
            }
            
            .navbar-toggler {
                display: block;
            }
        }
    </style>
</head>
<body>
    <!-- Sidebar -->
    <div class="sidebar">
        <div class="sidebar-brand">
            <i class="fas fa-camera-retro"></i>
            <h4>Skyhighes Technologies</h4>
            <p>Premium Attendance</p>
        </div>
        <ul class="nav flex-column">
            <li class="nav-item">
                <a class="nav-link {% if request.endpoint == 'admin_dashboard' %}active{% endif %}" href="{{ url_for('admin_dashboard') }}">
                    <i class="fas fa-tachometer-alt"></i> <span>Dashboard</span>
                </a>
            </li>
            <li class="nav-item">
                <a class="nav-link {% if request.endpoint in ['manage_employees', 'add_employee', 'edit_employee', 'view_employee', 'register_face'] %}active{% endif %}" href="{{ url_for('manage_employees') }}">
                    <i class="fas fa-users"></i> <span>Manage Employees</span>
                </a>
            </li>
            <li class="nav-item">
                <a class="nav-link {% if request.endpoint == 'view_attendance' %}active{% endif %}" href="{{ url_for('view_attendance') }}">
                    <i class="fas fa-clipboard-list"></i> <span>View Attendance</span>
                </a>
            </li>
            <li class="nav-item">
                <a class="nav-link {% if request.endpoint in ['recognition_control', 'cctv_config'] %}active{% endif %}" href="{{ url_for('recognition_control') }}">
                    <i class="fas fa-video"></i> <span>CCTV Recognition</span>
                </a>
            </li>
            <li class="nav-item">
                <a class="nav-link {% if request.endpoint == 'admin_checkin_requests' %}active{% endif %}" href="{{ url_for('admin_checkin_requests') }}">
                    <i class="fas fa-bell"></i> <span>Check-in Requests</span> 
                    <span class="badge bg-danger" id="request-badge">
                        {% if session.get('pending_requests_count', 0) > 0 %}{{ session.pending_requests_count }}{% endif %}
                    </span>
                </a>
            </li>
            <li class="nav-item">
                <a class="nav-link {% if request.endpoint == 'attendance_analysis' %}active{% endif %}" href="{{ url_for('attendance_analysis') }}">
                    <i class="fas fa-chart-bar"></i> <span>Attendance Analysis</span>
                </a>
            </li>
            <li class="nav-item">
                <a class="nav-link {% if request.endpoint == 'model_dashboard' %}active{% endif %}" href="{{ url_for('model_dashboard') }}">
                    <i class="fas fa-brain"></i> <span>Model Dashboard</span>
                </a>
            </li>
            <li class="nav-item">
                <a class="nav-link {% if request.endpoint == 'test_face_recognition' %}active{% endif %}" href="{{ url_for('test_face_recognition') }}">
                    <i class="fas fa-check-circle"></i> <span>Test Face Recognition</span>
                </a>
            </li>
            <li class="nav-item">
                <a class="nav-link {% if request.endpoint == 'live_stream' %}active{% endif %}" href="{{ url_for('live_stream') }}">
                    <i class="fas fa-tv"></i> <span>Live Stream</span>
                </a>
            </li>
            <li class="nav-item mt-4">
                <a class="nav-link" href="{{ url_for('logout') }}">
                    <i class="fas fa-sign-out-alt"></i> <span>Logout</span>
                </a>
            </li>
        </ul>
    </div>

    <!-- Navbar -->
    <nav class="navbar">
        <div class="container-fluid">
            <div class="d-flex align-items-center">
                <button class="btn btn-sm btn-outline-secondary d-lg-none me-2" id="sidebarToggle">
                    <i class="fas fa-bars"></i>
                </button>
                <span class="navbar-brand mb-0 h1">{% block page_title %}Admin Panel{% endblock %}</span>
            </div>
            <div class="d-flex align-items-center">
                <span class="me-3"><i class="fas fa-user-circle me-2"></i> {{ session.username }}</span>
                <div class="dropdown">
                    <button class="btn btn-sm btn-outline-secondary dropdown-toggle" type="button" data-bs-toggle="dropdown">
                        <i class="fas fa-cog"></i>
                    </button>
                    <ul class="dropdown-menu dropdown-menu-end">
                        <li><a class="dropdown-item" href="#"><i class="fas fa-user me-2"></i> Profile</a></li>
                        <li><a class="dropdown-item" href="#"><i class="fas fa-cog me-2"></i> Settings</a></li>
                        <li><hr class="dropdown-divider"></li>
                        <li><a class="dropdown-item" href="{{ url_for('logout') }}"><i class="fas fa-sign-out-alt me-2"></i> Logout</a></li>
                    </ul>
                </div>
            </div>
        </div>
    </nav>

    <!-- Main Content -->
    <div class="main-content">
        {% with messages = get_flashed_messages(with_categories=true) %}
            {% if messages %}
                {% for category, message in messages %}
                    <div class="alert alert-{{ category }} alert-dismissible fade show" role="alert">
                        {{ message }}
                        <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
                    </div>
                {% endfor %}
            {% endif %}
        {% endwith %}
        
        {% block content %}{% endblock %}
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        // Sidebar toggle for mobile
        document.getElementById('sidebarToggle').addEventListener('click', function() {
            document.querySelector('.sidebar').classList.toggle('show');
        });
        
        // Update pending requests count
        function updatePendingRequests() {
            fetch('/api/pending_requests_count')
                .then(response => response.json())
                .then(data => {
                    const badge = document.getElementById('request-badge');
                    if (data.count > 0) {
                        badge.textContent = data.count;
                        badge.style.display = 'inline-block';
                    } else {
                        badge.textContent = '';
                        badge.style.display = 'none';
                    }
                });
        }
        
        // Update every 30 seconds
        setInterval(updatePendingRequests, 30000);
        updatePendingRequests(); // Initial update
    </script>
    {% block extra_js %}{% endblock %}
</body>
</html>""")

with open('templates/employee_dashboard.html', 'w') as f:
    f.write("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Employee Dashboard | Skyhighes Technologies</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        :root {
            --primary: #1a2a6c;
            --secondary: #b21f1f;
            --accent: #ff8a00;
        }
        body {
            background-color: #f8f9fa;
        }
        .employee-header {
            background: linear-gradient(to right, var(--primary), var(--secondary));
            color: white;
            padding: 30px 0;
            margin-bottom: 30px;
        }
        .profile-card {
            background: white;
            border-radius: 15px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        .profile-header {
            background: linear-gradient(to right, #3a5fc5, #1a2a6c);
            color: white;
            padding: 20px;
            text-align: center;
        }
        .profile-avatar {
            width: 100px;
            height: 100px;
            border-radius: 50%;
            border: 3px solid white;
            margin: 0 auto 15px;
            background: #eee;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 2.5rem;
            color: #666;
            overflow: hidden;
        }
        .profile-avatar img {
            width: 100%;
            height: 100%;
            object-fit: cover;
        }
        .attendance-card {
            background: white;
            border-radius: 15px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            padding: 20px;
            margin-bottom: 20px;
        }
        .attendance-status {
            font-size: 1.5rem;
            font-weight: bold;
            margin: 10px 0;
        }
        .btn-checkin {
            background: linear-gradient(to right, #00b09b, #96c93d);
            border: none;
            padding: 10px 20px;
            font-weight: bold;
            width: 100%;
            margin-top: 15px;
        }
        .btn-checkout {
            background: linear-gradient(to right, #ff8a00, #da1b60);
            border: none;
            padding: 10px 20px;
            font-weight: bold;
            width: 100%;
            margin-top: 15px;
        }
        .btn-request {
            background: linear-gradient(to right, #654ea3, #da22ff);
            border: none;
            padding: 10px 20px;
            font-weight: bold;
            width: 100%;
            margin-top: 15px;
        }
        .history-table {
            background: white;
            border-radius: 15px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        .badge-present {
            background: linear-gradient(to right, #00b09b, #96c93d);
        }
        .badge-late {
            background: linear-gradient(to right, #ff8a00, #da1b60);
        }
        .badge-halfday {
            background: linear-gradient(to right, #654ea3, #da22ff);
        }
        .badge-pending {
            background: linear-gradient(to right, #ffd700, #ffa500);
        }
        .badge-approved {
            background: linear-gradient(to right, #00b09b, #96c93d);
        }
        .badge-rejected {
            background: linear-gradient(to right, #ff416c, #ff4b2b);
        }
    </style>
</head>
<body>
    <!-- Header -->
    <div class="employee-header">
        <div class="container">
            <div class="d-flex justify-content-between align-items-center">
                <div>
                    <h1><i class="fas fa-camera-retro me-3"></i> Skyhighes Technologies</h1>
                    <p>Premium Attendance System</p>
                </div>
                <div>
                    <a href="/logout" class="btn btn-light">
                        <i class="fas fa-sign-out-alt me-2"></i> Logout
                    </a>
                </div>
            </div>
        </div>
    </div>

    <!-- Main Content -->
    <div class="container">
        <div class="row">
            <div class="col-md-4">
                <!-- Profile Card -->
                <div class="profile-card">
                    <div class="profile-header">
                        <div class="profile-avatar">
                            {% if data.employee.profile_image %}
                                <img src="{{ url_for('static', filename=data.employee.profile_image) }}" alt="Profile">
                            {% else %}
                                <i class="fas fa-user"></i>
                            {% endif %}
                        </div>
                        <h3>{{ data.employee.name }}</h3>
                        <p>{{ data.employee.department }} Department</p>
                    </div>
                    <div class="profile-body p-3">
                        <div class="row mb-2">
                            <div class="col-6">
                                <small>Employee ID</small>
                                <div><strong>{{ session.employee_id }}</strong></div>
                            </div>
                            <div class="col-6">
                                <small>Position</small>
                                <div><strong>{{ data.employee.position }}</strong></div>
                            </div>
                        </div>
                        <div class="row mb-2">
                            <div class="col-6">
                                <small>Join Date</small>
                                <div><strong>{{ data.employee.join_date.strftime('%d %b %Y') }}</strong></div>
                            </div>
                            <div class="col-6">
                                <small>Status</small>
                                <div><span class="badge bg-success">Active</span></div>
                            </div>
                        </div>
                        <hr>
                        <div class="text-center">
                            <a href="/employee/update_profile" class="btn btn-outline-primary btn-sm">
                                <i class="fas fa-user-edit me-1"></i> Update Profile
                            </a>
                        </div>
                    </div>
                </div>

                <!-- Today's Attendance -->
                <div class="attendance-card mt-4">
                    <h4 class="mb-4">Today's Attendance</h4>
                    
                    {% if data.is_non_working %}
                        <div class="alert alert-info">
                            <i class="fas fa-info-circle me-2"></i> Today is a non-working day
                        </div>
                    {% elif data.today_attendance %}
                        {% if data.today_attendance.check_out %}
                            <div class="alert alert-info">
                                <i class="fas fa-info-circle me-2"></i> You've completed your attendance today
                            </div>
                            <div class="row">
                                <div class="col-6">
                                    <small>Check In</small>
                                    <div><strong>{{ data.today_attendance.check_in.strftime('%H:%M') }}</strong></div>
                                </div>
                                <div class="col-6">
                                    <small>Check Out</small>
                                    <div><strong>{{ data.today_attendance.check_out.strftime('%H:%M') }}</strong></div>
                                </div>
                            </div>
                        {% else %}
                            <div class="alert alert-success">
                                <i class="fas fa-check-circle me-2"></i> You checked in at {{ data.today_attendance.check_in.strftime('%H:%M') }}
                            </div>
                        {% endif %}
                        
                        <div class="attendance-status mt-3">
                            Status: 
                            {% if data.today_attendance.status == 'present' %}
                                <span class="badge badge-present">Present</span>
                            {% elif data.today_attendance.status == 'late' %}
                                <span class="badge badge-late">Late</span>
                            {% elif data.today_attendance.status == 'half_day' %}
                                <span class="badge badge-halfday">Half Day</span>
                            {% else %}
                                <span class="badge bg-secondary">{{ data.today_attendance.status }}</span>
                            {% endif %}
                        </div>
                        
                        {% if not data.today_attendance.check_out %}
                            <button class="btn btn-checkout">
                                <i class="fas fa-sign-out-alt me-2"></i> Check Out
                            </button>
                        {% endif %}
                    {% elif data.checkin_request %}
                        {% if data.checkin_request.status == 'pending' %}
                            <div class="alert alert-warning">
                                <i class="fas fa-clock me-2"></i> Manual check-in request pending approval
                            </div>
                            <div class="attendance-status">
                                Status: <span class="badge badge-pending">Pending</span>
                            </div>
                        {% elif data.checkin_request.status == 'approved' %}
                            <div class="alert alert-success">
                                <i class="fas fa-check-circle me-2"></i> Manual check-in approved
                            </div>
                        {% else %}
                            <div class="alert alert-danger">
                                <i class="fas fa-times-circle me-2"></i> Manual check-in request rejected
                            </div>
                        {% endif %}
                    {% else %}
                        <div class="alert alert-warning">
                            <i class="fas fa-exclamation-triangle me-2"></i> You haven't checked in today
                        </div>
                        <button class="btn btn-checkin" data-bs-toggle="modal" data-bs-target="#checkinModal">
                            <i class="fas fa-fingerprint me-2"></i> Check In Now
                        </button>
                        <button class="btn btn-request" data-bs-toggle="modal" data-bs-target="#requestModal">
                            <i class="fas fa-hand-paper me-2"></i> Request Manual Check-in
                        </button>
                    {% endif %}
                </div>
            </div>
            
            <div class="col-md-8">
                <!-- Attendance History -->
                <div class="history-table">
                    <div class="table-header bg-light p-3">
                        <h4 class="mb-0">Attendance History</h4>
                    </div>
                    <div class="table-responsive">
                        <table class="table table-hover">
                            <thead class="table-light">
                                <tr>
                                    <th>Date</th>
                                    <th>Check In</th>
                                    <th>Check Out</th>
                                    <th>Status</th>
                                    <th>Confidence</th>
                                </tr>
                            </thead>
                            <tbody>
                                {% for record in data.attendance_history %}
                                <tr>
                                    <td>{{ record.check_in.strftime('%d %b %Y') }}</td>
                                    <td>{{ record.check_in.strftime('%H:%M') }}</td>
                                    <td>{{ record.check_out.strftime('%H:%M') if record.check_out else '-' }}</td>
                                    <td>
                                        {% if record.status == 'present' %}
                                            <span class="badge badge-present">Present</span>
                                        {% elif record.status == 'late' %}
                                            <span class="badge badge-late">Late</span>
                                        {% elif record.status == 'half_day' %}
                                            <span class="badge badge-halfday">Half Day</span>
                                        {% else %}
                                            <span class="badge bg-secondary">{{ record.status }}</span>
                                        {% endif %}
                                    </td>
                                    <td>{{ record.confidence|round(2) }}</td>
                                </tr>
                                {% else %}
                                <tr>
                                    <td colspan="5" class="text-center">No attendance records found</td>
                                </tr>
                                {% endfor %}
                            </tbody>
                        </table>
                    </div>
                </div>
                
                <!-- Leave Request -->
                <div class="attendance-card mt-4">
                    <h4 class="mb-4">Leave Request</h4>
                    <div class="alert alert-info">
                        <i class="fas fa-info-circle me-2"></i> You can submit leave requests here
                    </div>
                    <button class="btn btn-primary">
                        <i class="fas fa-plus me-2"></i> Request Leave
                    </button>
                </div>
            </div>
        </div>
    </div>

    <!-- Manual Check-in Request Modal -->
    <div class="modal fade" id="requestModal" tabindex="-1" aria-labelledby="requestModalLabel" aria-hidden="true">
        <div class="modal-dialog">
            <div class="modal-content">
                <div class="modal-header">
                    <h5 class="modal-title" id="requestModalLabel">Request Manual Check-in</h5>
                    <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                </div>
                <form method="POST" action="/employee/request_checkin">
                    <div class="modal-body">
                        <div class="mb-3">
                            <label class="form-label">Reason for manual check-in</label>
                            <textarea class="form-control" name="reason" rows="3" required></textarea>
                        </div>
                        <div class="alert alert-info">
                            <i class="fas fa-info-circle me-2"></i> Your request will be reviewed by admin
                        </div>
                    </div>
                    <div class="modal-footer">
                        <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Cancel</button>
                        <button type="submit" class="btn btn-primary">Submit Request</button>
                    </div>
                </form>
            </div>
        </div>
    </div>

    <footer class="bg-light py-4 mt-5">
        <div class="container">
            <div class="text-center">
                <p class="mb-0">&copy; 2024 Skyhighes Technologies. Premium Attendance System.</p>
                <p class="text-muted">Employee Portal v2.0</p>
            </div>
        </div>
    </footer>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>""")
    

with open('templates/test_face_recognition.html', 'w') as f:
    f.write("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Test Face Recognition | Skyhighes Technologies</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        .test-container {
            background: white;
            border-radius: 15px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            padding: 25px;
            margin-bottom: 30px;
        }
        .camera-container {
            position: relative;
            width: 100%;
            max-width: 640px;
            margin: 0 auto;
        }
        #video-element {
            width: 100%;
            border-radius: 10px;
            transform: scaleX(-1); /* Mirror for front camera */
        }
        .capture-btn {
            position: absolute;
            bottom: 20px;
            left: 50%;
            transform: translateX(-50%);
            width: 60px;
            height: 60px;
            border-radius: 50%;
            background: #dc3545;
            border: 4px solid white;
            cursor: pointer;
        }
        .result-card {
            background: #f8f9fa;
            border-radius: 10px;
            padding: 20px;
            margin-top: 20px;
        }
        .similarity-gauge {
            width: 200px;
            height: 200px;
            margin: 0 auto;
        }
    </style>
</head>
<body>
    {% extends "admin_base.html" %}

    {% block page_title %}Test Face Recognition{% endblock %}

    {% block content %}
        <div class="container-fluid">
            <div class="d-flex justify-content-between align-items-center mb-4">
                <h2>Test Face Recognition</h2>
                <a href="/admin/dashboard" class="btn btn-outline-secondary">
                    <i class="fas fa-arrow-left me-1"></i> Back to Dashboard
                </a>
            </div>
            
            {% with messages = get_flashed_messages(with_categories=true) %}
                {% if messages %}
                    {% for category, message in messages %}
                        <div class="alert alert-{{ category }} alert-dismissible fade show" role="alert">
                            {{ message }}
                            <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
                        </div>
                    {% endfor %}
                {% endif %}
            {% endwith %}
            
            <div class="test-container">
                <h4 class="mb-4">Test Configuration</h4>
                
                <form method="POST" id="testForm" enctype="multipart/form-data">
                    <div class="row">
                        <div class="col-md-6">
                            <div class="mb-3">
                                <label for="employee_id" class="form-label">Select Employee</label>
                                <select class="form-select" id="employee_id" name="employee_id" required>
                                    <option value="">-- Select Employee --</option>
                                    {% for employee in employees %}
                                        <option value="{{ employee.id }}" {% if selected_employee and selected_employee.id == employee.id %}selected{% endif %}>
                                            {{ employee.name }} ({{ employee.employee_id }}) - {{ employee.embedding_count }} embeddings
                                        </option>
                                    {% endfor %}
                                </select>
                            </div>
                            
                            <div class="mb-3">
                                <label class="form-label">Test Image Source</label>
                                <div class="form-check">
                                    <input class="form-check-input" type="radio" name="image_source" id="source_upload" value="upload" checked>
                                    <label class="form-check-label" for="source_upload">
                                        Upload Image
                                    </label>
                                </div>
                                <div class="form-check">
                                    <input class="form-check-input" type="radio" name="image_source" id="source_camera" value="camera">
                                    <label class="form-check-label" for="source_camera">
                                        Use Camera
                                    </label>
                                </div>
                            </div>
                            
                            <div id="uploadSection">
                                <div class="mb-3">
                                    <label for="test_image" class="form-label">Upload Test Image</label>
                                    <input type="file" class="form-control" id="test_image" name="test_image" accept="image/*">
                                </div>
                            </div>
                            
                            <div id="cameraSection" style="display: none;">
                                <div class="camera-container mb-3">
                                    <video id="video-element" autoplay playsinline></video>
                                    <div class="capture-btn" id="capture-btn"></div>
                                </div>
                                <input type="hidden" id="captured_image" name="captured_image">
                            </div>
                            
                            <button type="submit" class="btn btn-primary">
                                <i class="fas fa-play me-2"></i> Run Test
                            </button>
                        </div>
                        
                        {% if test_result %}
                        <div class="col-md-6">
                            <div class="result-card">
                                <h5>Test Results</h5>
                                
                                <div class="text-center mb-3">
                                    <h6>{{ test_result.employee.name }}</h6>
                                    <p class="text-muted">{{ test_result.employee.department }}</p>
                                </div>
                                
                                <div class="mb-3">
                                    <div class="fw-bold">Minimum Confidence Threshold:</div>
                                    <div class="h5">{{ (test_result.min_confidence * 100)|round(2) }}%</div>
                                </div>
                                
                                <div class="similarity-gauge mb-3">
                                    <canvas id="similarityChart"></canvas>
                                </div>
                                
                                <div class="row text-center">
                                    <div class="col-6">
                                        <div class="fw-bold">Max Similarity</div>
                                        <div class="h4 {% if test_result.is_match %}text-success{% else %}text-danger{% endif %}">
                                            {{ (test_result.max_similarity * 100)|round(2) }}%
                                        </div>
                                    </div>
                                    <div class="col-6">
                                        <div class="fw-bold">Average Similarity</div>
                                        <div class="h4">
                                            {{ (test_result.avg_similarity * 100)|round(2) }}%
                                        </div>
                                    </div>
                                </div>
                                
                                <div class="mt-3 text-center">
                                    {% if test_result.is_match %}
                                        <span class="badge bg-success">MATCH</span>
                                        <p class="text-success mt-2">Face recognized successfully!</p>
                                    {% else %}
                                        <span class="badge bg-danger">NO MATCH</span>
                                        <p class="text-danger mt-2">Face not recognized.</p>
                                    {% endif %}
                                </div>
                                
                                <div class="mt-3">
                                    <small class="text-muted">
                                        Faces detected: {{ test_result.faces_detected }} | 
                                        Best confidence: {{ (test_result.best_confidence * 100)|round(2) }}%
                                    </small>
                                </div>
                            </div>
                        </div>
                        {% endif %}
                    </div>
                </form>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        // Toggle between upload and camera
        document.querySelectorAll('input[name="image_source"]').forEach(radio => {
            radio.addEventListener('change', function() {
                if (this.value === 'upload') {
                    document.getElementById('uploadSection').style.display = 'block';
                    document.getElementById('cameraSection').style.display = 'none';
                } else {
                    document.getElementById('uploadSection').style.display = 'none';
                    document.getElementById('cameraSection').style.display = 'block';
                    initializeCamera();
                }
            });
        });
        
        // Camera initialization
        let stream = null;
        
        async function initializeCamera() {
            try {
                stream = await navigator.mediaDevices.getUserMedia({ 
                    video: { width: 640, height: 480 },
                    audio: false 
                });
                const videoElement = document.getElementById('video-element');
                videoElement.srcObject = stream;
            } catch (error) {
                console.error('Error accessing camera:', error);
                alert('Could not access camera. Please check permissions.');
            }
        }
        
        // Capture image from camera
        document.getElementById('capture-btn').addEventListener('click', function() {
            const videoElement = document.getElementById('video-element');
            const canvas = document.createElement('canvas');
            canvas.width = videoElement.videoWidth;
            canvas.height = videoElement.videoHeight;
            const ctx = canvas.getContext('2d');
            
            // Draw the video frame to the canvas
            ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height);
            
            // Convert to data URL
            const imageData = canvas.toDataURL('image/jpeg');
            document.getElementById('captured_image').value = imageData;
            
            // Show preview (optional)
            alert('Image captured! Click "Run Test" to proceed.');
        });
        
        // Clean up camera when leaving page
        window.addEventListener('beforeunload', function() {
            if (stream) {
                stream.getTracks().forEach(track => track.stop());
            }
        });
        
        // Similarity gauge chart
        {% if test_result %}
        const similarityCtx = document.getElementById('similarityChart').getContext('2d');
        const similarityGauge = new Chart(similarityCtx, {
            type: 'doughnut',
            data: {
                datasets: [{
                    data: [{{ (test_result.max_similarity * 100)|round(2) }}, {{ 100 - (test_result.max_similarity * 100)|round(2) }}],
                    backgroundColor: [
                        '{% if test_result.is_match %}#28a745{% else %}#dc3545{% endif %}',
                        '#e9ecef'
                    ],
                    borderWidth: 0
                }]
            },
            options: {
                cutout: '70%',
                rotation: -90,
                circumference: 180,
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        enabled: false
                    }
                }
            }
        });
        {% endif %}
    </script>
    {% endblock %}
</body>
</html>""")
            
            
with open('templates/model_dashboard.html', 'w') as f:
    f.write("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Model Dashboard | Skyhighes Technologies</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        .stat-card {
            border-radius: 10px;
            overflow: hidden;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            transition: transform 0.3s;
        }
        .stat-card:hover {
            transform: translateY(-5px);
        }
        .stat-card-header {
            padding: 15px;
            color: white;
            font-weight: bold;
        }
        .stat-card-body {
            padding: 20px;
            background: white;
            text-align: center;
        }
        .stat-number {
            font-size: 2.5rem;
            font-weight: bold;
            margin: 10px 0;
        }
        .card-1 .stat-card-header { background: linear-gradient(to right, #1a2a6c, #3a5fc5); }
        .card-2 .stat-card-header { background: linear-gradient(to right, #00b09b, #96c93d); }
        .card-3 .stat-card-header { background: linear-gradient(to right, #ff8a00, #da1b60); }
        .card-4 .stat-card-header { background: linear-gradient(to right, #654ea3, #da22ff); }
        
        .history-table {
            background: white;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            overflow: hidden;
            margin-top: 20px;
        }
    </style>
</head>
<body>
    {% extends "admin_base.html" %}

    {% block page_title %}Model Dashboard{% endblock %}

    {% block content %}
        <div class="container-fluid">
            <div class="d-flex justify-content-between align-items-center mb-4">
                <h2>Model Training Dashboard</h2>
                <div>
                    <button id="train-now-btn" class="btn btn-primary">
                        <i class="fas fa-brain me-2"></i> Train Now
                    </button>
                    <a href="/admin/dashboard" class="btn btn-outline-secondary ms-2">
                        <i class="fas fa-arrow-left me-2"></i> Back to Dashboard
                    </a>
                </div>
            </div>
            
            <div class="row">
                <div class="col-md-3">
                    <div class="stat-card card-1">
                        <div class="stat-card-header">
                            <i class="fas fa-users me-2"></i> Total Employees
                        </div>
                        <div class="stat-card-body">
                            <div class="stat-number">{{ stats.total_employees }}</div>
                            <div>Registered Employees</div>
                        </div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="stat-card card-2">
                        <div class="stat-card-header">
                            <i class="fas fa-camera me-2"></i> Employees with Faces
                        </div>
                        <div class="stat-card-body">
                            <div class="stat-number">{{ stats.employees_with_faces }}</div>
                            <div>Face Data Available</div>
                        </div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="stat-card card-3">
                        <div class="stat-card-header">
                            <i class="fas fa-database me-2"></i> Total Embeddings
                        </div>
                        <div class="stat-card-body">
                            <div class="stat-number">{{ stats.total_embeddings }}</div>
                            <div>Face Embeddings</div>
                        </div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="stat-card card-4">
                        <div class="stat-card-header">
                            <i class="fas fa-brain me-2"></i> Trained Employees
                        </div>
                        <div class="stat-card-body">
                            <div class="stat-number">{{ stats.trained_employees }}</div>
                            <div>In Current Model</div>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="row mt-4">
                <div class="col-md-6">
                    <div class="card">
                        <div class="card-header">
                            <h5 class="mb-0">Model Status</h5>
                        </div>
                        <div class="card-body">
                            <div class="mb-3">
                                <strong>Model Loaded:</strong> 
                                <span class="badge {% if stats.model_loaded %}bg-success{% else %}bg-danger{% endif %}">
                                    {% if stats.model_loaded %}Yes{% else %}No{% endif %}
                                </span>
                            </div>
                            <div class="mb-3">
                                <strong>Last Training:</strong> 
                                {% if stats.last_training %}
                                    {{ stats.last_training.training_date.strftime('%Y-%m-%d %H:%M:%S') }}
                                    <span class="badge bg-success">Success</span>
                                {% else %}
                                    Never
                                {% endif %}
                            </div>
                            <div class="alert alert-info">
                                <i class="fas fa-info-circle me-2"></i>
                                Model training is now automatic. The system will train whenever new face data is added.
                            </div>
                        </div>
                    </div>
                </div>
                <div class="col-md-6">
                    <div class="card">
                        <div class="card-header">
                            <h5 class="mb-0">Training Statistics</h5>
                        </div>
                        <div class="card-body">
                            <canvas id="trainingChart" height="200"></canvas>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="history-table">
                <div class="table-header bg-light p-3">
                    <h5 class="mb-0">Training History</h5>
                </div>
                <div class="table-responsive">
                    <table class="table table-hover">
                        <thead>
                            <tr>
                                <th>Training Date</th>
                                <th>Employees</th>
                                <th>Embeddings</th>
                                <th>Status</th>
                                <th>Duration</th>
                                <th>Message</th>
                            </tr>
                        </thead>
                        <tbody>
                            {% for training in training_history %}
                            <tr>
                                <td>{{ training.training_date.strftime('%Y-%m-%d %H:%M:%S') }}</td>
                                <td>{{ training.employees_count }}</td>
                                <td>{{ training.embeddings_count }}</td>
                                <td>
                                    {% if training.status == 'success' %}
                                        <span class="badge bg-success">Success</span>
                                    {% else %}
                                        <span class="badge bg-danger">Failed</span>
                                    {% endif %}
                                </td>
                                <td>{{ training.duration_seconds|round(2) }}s</td>
                                <td>{{ training.message|truncate(50) }}</td>
                            </tr>
                            {% else %}
                            <tr>
                                <td colspan="6" class="text-center">No training history available</td>
                            </tr>
                            {% endfor %}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        // Train now button
        document.getElementById('train-now-btn').addEventListener('click', function() {
            this.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i> Training...';
            this.disabled = true;
            
            fetch('/admin/model/train', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            })
            .then(response => response.json())
            .then(data => {
                alert(data.message);
                location.reload();
            })
            .catch(error => {
                alert('Error: ' + error);
                this.innerHTML = '<i class="fas fa-brain me-2"></i> Train Now';
                this.disabled = false;
            });
        });
        
        // Training chart
        const ctx = document.getElementById('trainingChart').getContext('2d');
        const trainingChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: [{% for training in training_history %}'{{ training.training_date.strftime('%m-%d %H:%M') }}'{% if not loop.last %}, {% endif %}{% endfor %}],
                datasets: [{
                    label: 'Embeddings',
                    data: [{% for training in training_history %}{{ training.embeddings_count }}{% if not loop.last %}, {% endif %}{% endfor %}],
                    backgroundColor: 'rgba(54, 162, 235, 0.5)',
                    borderColor: 'rgba(54, 162, 235, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        beginAtZero: true
                    }
                }
            }
        });
    </script>
    {% endblock %}
</body>
</html>
""")

with open('templates/update_profile.html', 'w') as f:
    f.write("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Update Profile | Skyhighes Technologies</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        .profile-container {
            background: white;
            border-radius: 15px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            padding: 25px;
            margin-top: 30px;
        }
        .profile-image {
            width: 150px;
            height: 150px;
            border-radius: 50%;
            border: 3px solid #eee;
            margin: 0 auto 20px;
            overflow: hidden;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .profile-image img {
            width: 100%;
            height: 100%;
            object-fit: cover;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="d-flex justify-content-between align-items-center mb-4">
            <h2>Update Profile</h2>
            <a href="/employee/dashboard" class="btn btn-outline-secondary">
                <i class="fas fa-arrow-left me-1"></i> Back to Dashboard
            </a>
        </div>
        
        <div class="profile-container">
            {% with messages = get_flashed_messages(with_categories=true) %}
                {% if messages %}
                    {% for category, message in messages %}
                        <div class="alert alert-{{ category }} alert-dismissible fade show" role="alert">
                            {{ message }}
                            <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
                        </div>
                    {% endfor %}
                {% endif %}
            {% endwith %}
            
            <div class="text-center mb-4">
                <div class="profile-image">
                    {% if employee.profile_image %}
                        <img src="{{ url_for('static', filename=employee.profile_image) }}" alt="Profile">
                    {% else %}
                        <i class="fas fa-user fa-5x text-secondary"></i>
                    {% endif %}
                </div>
                <h4>{{ employee.name }}</h4>
                <p>{{ employee.department }} Department</p>
            </div>
            
            <form method="POST" enctype="multipart/form-data">
                <div class="mb-3">
                    <label class="form-label">Update Profile Image</label>
                    <input class="form-control" type="file" name="profile_image" accept="image/*">
                </div>
                <div class="alert alert-info">
                    <i class="fas fa-info-circle me-2"></i> Only JPG/PNG images allowed. Max size 5MB.
                </div>
                <button type="submit" class="btn btn-primary w-100">
                    <i class="fas fa-save me-2"></i> Update Profile
                </button>
            </form>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>""")

with open('templates/admin_checkin_requests.html', 'w') as f:
    f.write("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Check-in Requests | Skyhighes Technologies</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        .requests-table {
            background: white;
            border-radius: 15px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        .table-header {
            background: linear-gradient(to right, #1a2a6c, #3a5fc5);
            color: white;
            padding: 15px 20px;
        }
        .badge-pending {
            background: linear-gradient(to right, #ffd700, #ffa500);
        }
        .badge-approved {
            background: linear-gradient(to right, #00b09b, #96c93d);
        }
        .badge-rejected {
            background: linear-gradient(to right, #ff416c, #ff4b2b);
        }
    </style>
</head>
<body>
    {% extends "admin_base.html" %}

    {% block page_title %}Check-in Requests{% endblock %}

    {% block content %}
        <div class="container-fluid">
            <div class="d-flex justify-content-between align-items-center mb-4">
                <h2>Manual Check-in Requests</h2>
            </div>
            
            {% with messages = get_flashed_messages(with_categories=true) %}
                {% if messages %}
                    {% for category, message in messages %}
                        <div class="alert alert-{{ category }} alert-dismissible fade show" role="alert">
                            {{ message }}
                            <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
                        </div>
                    {% endfor %}
                {% endif %}
            {% endwith %}
            
            {% if requests %}
            <div class="requests-table">
                <div class="table-header">
                    <h4 class="mb-0">Pending Requests</h4>
                </div>
                <div class="table-responsive">
                    <table class="table table-hover">
                        <thead>
                            <tr>
                                <th>Employee</th>
                                <th>ID</th>
                                <th>Department</th>
                                <th>Request Date</th>
                                <th>Reason</th>
                                <th>Status</th>
                                <th>Requested At</th>
                                <th>Actions</th>
                            </tr>
                        </thead>
                        <tbody>
                            {% for req in requests %}
                            <tr>
                                <td>{{ req.employee_name }}</td>
                                <td>{{ req.employee_id }}</td>
                                <td>{{ req.department }}</td>
                                <td>{{ req.request_date }}</td>
                                <td>{{ req.reason|truncate(50) }}</td>
                                <td>
                                    <span class="badge badge-pending">Pending</span>
                                </td>
                                <td>{{ req.created_at.strftime('%Y-%m-%d %H:%M') }}</td>
                                <td>
                                    <a href="/admin/approve_checkin/{{ req.id }}" class="btn btn-sm btn-success me-1">
                                        <i class="fas fa-check"></i> Approve
                                    </a>
                                    <a href="/admin/reject_checkin/{{ req.id }}" class="btn btn-sm btn-danger">
                                        <i class="fas fa-times"></i> Reject
                                    </a>
                                </td>
                            </tr>
                            {% endfor %}
                        </tbody>
                    </table>
                </div>
            </div>
            {% else %}
            <div class="alert alert-info">
                <i class="fas fa-info-circle me-2"></i> No pending check-in requests
            </div>
            {% endif %}
        </div>
    {% endblock %}

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>""")

with open('templates/attendance_analysis.html', 'w') as f:
    f.write("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Attendance Analysis | Skyhighes Technologies</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {
            --primary: #1a2a6c;
            --secondary: #b21f1f;
            --accent: #ff8a00;
            --light-bg: #f8f9fa;
            --card-shadow: 0 10px 20px rgba(0,0,0,0.12);
            --card-radius: 16px;
        }
        
        body {
            background-color: var(--light-bg);
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        
        .dashboard-card {
            background: white;
            border-radius: var(--card-radius);
            box-shadow: var(--card-shadow);
            padding: 25px;
            margin-bottom: 30px;
            transition: transform 0.3s;
            height: 100%;
        }
        
        .dashboard-card:hover {
            transform: translateY(-5px);
        }
        
        .chart-container {
            height: 400px;
            margin-bottom: 30px;
            position: relative;
        }
        
        .stats-card {
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            color: white;
            border-radius: var(--card-radius);
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: var(--card-shadow);
            text-align: center;
        }
        
        .stats-number {
            font-size: 2.5rem;
            font-weight: 700;
            margin: 10px 0;
        }
        
        .stats-label {
            font-size: 0.9rem;
            opacity: 0.9;
        }
        
        .filter-section {
            background: white;
            border-radius: var(--card-radius);
            box-shadow: var(--card-shadow);
            padding: 20px;
            margin-bottom: 30px;
        }
        
        .date-filter {
            display: flex;
            gap: 15px;
            align-items: center;
            flex-wrap: wrap;
        }
        
        .btn-premium {
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            color: white;
            border: none;
            padding: 10px 25px;
            border-radius: 30px;
            transition: all 0.3s;
        }
        
        .btn-premium:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }
        
        .tab-content {
            padding: 20px 0;
        }
        
        .nav-tabs .nav-link {
            border: none;
            color: #6c757d;
            font-weight: 500;
            padding: 12px 25px;
            border-radius: 8px 8px 0 0;
        }
        
        .nav-tabs .nav-link.active {
            color: var(--primary);
            background: white;
            border-bottom: 3px solid var(--primary);
        }
        
        .card-title {
            color: var(--primary);
            font-weight: 600;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid rgba(26, 42, 108, 0.1);
        }
        
        .data-card {
            background: white;
            border-radius: var(--card-radius);
            box-shadow: var(--card-shadow);
            padding: 20px;
            margin-bottom: 20px;
        }
        
        .data-value {
            font-size: 2rem;
            font-weight: 700;
            margin: 10px 0;
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        @media (max-width: 768px) {
            .chart-container {
                height: 300px;
            }
            
            .stats-number {
                font-size: 2rem;
            }
        }
    </style>
</head>
<body>
    {% extends "admin_base.html" %}

    {% block page_title %}Attendance Analysis{% endblock %}

    {% block content %}
        <div class="container-fluid">
            <div class="d-flex justify-content-between align-items-center mb-4">
                <h2 style="color: var(--primary);">Attendance Analytics Dashboard</h2>
                <div class="dropdown">
                    <button class="btn btn-premium dropdown-toggle" type="button" data-bs-toggle="dropdown">
                        <i class="fas fa-download me-2"></i> Export Reports
                    </button>
                    <ul class="dropdown-menu">
                        <li><a class="dropdown-item" href="#">PDF Report</a></li>
                        <li><a class="dropdown-item" href="#">Excel Data</a></li>
                        <li><a class="dropdown-item" href="#">CSV File</a></li>
                    </ul>
                </div>
            </div>
            
            <!-- Real-time Stats Overview -->
            <div class="row">
                <div class="col-md-3">
                    <div class="stats-card">
                        <i class="fas fa-users fa-2x mb-2"></i>
                        <div class="stats-number">
                            {% set total_today = analysis_data.counts | sum %}
                            {{ total_today }}
                        </div>
                        <div class="stats-label">Today's Attendance</div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="stats-card">
                        <i class="fas fa-check-circle fa-2x mb-2"></i>
                        <div class="stats-number">
                            {% if analysis_data.counts and analysis_data.statuses %}
                                {% set present_index = analysis_data.statuses.index('Present') %}
                                {{ analysis_data.counts[present_index] if present_index != -1 else 0 }}
                            {% else %}
                                0
                            {% endif %}
                        </div>
                        <div class="stats-label">Present Today</div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="stats-card">
                        <i class="fas fa-clock fa-2x mb-2"></i>
                        <div class="stats-number">
                            {% if analysis_data.counts and analysis_data.statuses %}
                                {% set late_index = analysis_data.statuses.index('Late') %}
                                {{ analysis_data.counts[late_index] if late_index != -1 else 0 }}
                            {% else %}
                                0
                            {% endif %}
                        </div>
                        <div class="stats-label">Late Today</div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="stats-card">
                        <i class="fas fa-chart-line fa-2x mb-2"></i>
                        <div class="stats-number">
                            {% if analysis_data.weekly_present and analysis_data.weekly_present|length > 1 %}
                                {{ ((analysis_data.weekly_present[-1] / analysis_data.weekly_present[-2]) * 100 - 100) | round(1) }}%
                            {% else %}
                                N/A
                            {% endif %}
                        </div>
                        <div class="stats-label">Weekly Change</div>
                    </div>
                </div>
            </div>
            
            <!-- Filter Section -->
            <div class="filter-section">
                <h4 class="card-title">Filter Data</h4>
                <div class="row">
                    <div class="col-md-8">
                        <div class="date-filter">
                            <div class="form-group">
                                <label class="form-label">From Date</label>
                                <input type="date" class="form-control" id="startDate">
                            </div>
                            <div class="form-group">
                                <label class="form-label">To Date</label>
                                <input type="date" class="form-control" id="endDate">
                            </div>
                            <div class="form-group" style="align-self: flex-end;">
                                <button class="btn btn-premium" id="applyFilter">
                                    <i class="fas fa-filter me-2"></i> Apply Filter
                                </button>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-4">
                        <div class="form-group">
                            <label class="form-label">Department</label>
                            <select class="form-select" id="departmentFilter">
                                <option value="all">All Departments</option>
                                {% for dept in analysis_data.dept_names %}
                                    <option value="{{ dept }}">{{ dept }}</option>
                                {% endfor %}
                            </select>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Tabs for different chart views -->
            <ul class="nav nav-tabs" id="analysisTabs" role="tablist">
                <li class="nav-item" role="presentation">
                    <button class="nav-link active" id="overview-tab" data-bs-toggle="tab" data-bs-target="#overview" type="button" role="tab">Overview</button>
                </li>
                <li class="nav-item" role="presentation">
                    <button class="nav-link" id="weekly-tab" data-bs-toggle="tab" data-bs-target="#weekly" type="button" role="tab">Weekly</button>
                </li>
                <li class="nav-item" role="presentation">
                    <button class="nav-link" id="monthly-tab" data-bs-toggle="tab" data-bs-target="#monthly" type="button" role="tab">Monthly</button>
                </li>
                <li class="nav-item" role="presentation">
                    <button class="nav-link" id="department-tab" data-bs-toggle="tab" data-bs-target="#department" type="button" role="tab">By Department</button>
                </li>
            </ul>
            
            <div class="tab-content" id="analysisTabsContent">
                <!-- Overview Tab -->
                <div class="tab-pane fade show active" id="overview" role="tabpanel">
                    <div class="row mt-4">
                        <div class="col-md-6">
                            <div class="dashboard-card">
                                <h4 class="card-title">Today's Attendance Distribution</h4>
                                <div class="chart-container">
                                    <canvas id="todayChart"></canvas>
                                </div>
                            </div>
                        </div>
                        <div class="col-md-6">
                            <div class="dashboard-card">
                                <h4 class="card-title">Attendance Trend (Last 7 Days)</h4>
                                <div class="chart-container">
                                    <canvas id="trendChart"></canvas>
                                </div>
                            </div>
                        </div>
                    </div>
                    <div class="row">
                        <div class="col-md-12">
                            <div class="dashboard-card">
                                <h4 class="card-title">Monthly Comparison</h4>
                                <div class="chart-container">
                                    <canvas id="monthlyComparisonChart"></canvas>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Weekly Tab -->
                <div class="tab-pane fade" id="weekly" role="tabpanel">
                    <div class="dashboard-card mt-4">
                        <h4 class="card-title">Weekly Attendance Trend</h4>
                        <div class="chart-container">
                            <canvas id="weeklyChart"></canvas>
                        </div>
                    </div>
                </div>
                
                <!-- Monthly Tab -->
                <div class="tab-pane fade" id="monthly" role="tabpanel">
                    <div class="dashboard-card mt-4">
                        <h4 class="card-title">Monthly Attendance Overview</h4>
                        <div class="chart-container">
                            <canvas id="monthlyChart"></canvas>
                        </div>
                    </div>
                </div>
                
                <!-- Department Tab -->
                <div class="tab-pane fade" id="department" role="tabpanel">
                    <div class="dashboard-card mt-4">
                        <h4 class="card-title">Department-wise Attendance</h4>
                        <div class="chart-container">
                            <canvas id="departmentChart"></canvas>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    {% endblock %}

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        // Today's Attendance Pie Chart
        const todayCtx = document.getElementById('todayChart').getContext('2d');
        const todayChart = new Chart(todayCtx, {
            type: 'doughnut',
            data: {
                labels: {{ analysis_data.statuses | tojson }},
                datasets: [{
                    data: {{ analysis_data.counts | tojson }},
                    backgroundColor: [
                        '#28a745', // Present - green
                        '#ffc107', // Late - yellow
                        '#dc3545', // Absent - red
                        '#17a2b8', // Half Day - teal
                        '#6f42c1'  // Overtime - purple
                    ],
                    borderWidth: 2,
                    borderColor: '#fff'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'right',
                        labels: {
                            font: {
                                size: 13,
                                family: "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif"
                            }
                        }
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                const label = context.label || '';
                                const value = context.raw || 0;
                                const total = context.dataset.data.reduce((a, b) => a + b, 0);
                                const percentage = Math.round((value / total) * 100);
                                return `${label}: ${value} (${percentage}%)`;
                            }
                        }
                    }
                },
                cutout: '60%',
                animation: {
                    animateScale: true,
                    animateRotate: true
                }
            }
        });
        
        // Weekly Attendance Chart
        const weeklyCtx = document.getElementById('weeklyChart').getContext('2d');
        const weeklyChart = new Chart(weeklyCtx, {
            type: 'line',
            data: {
                labels: {{ analysis_data.weeks | tojson }},
                datasets: [
                    {
                        label: 'Present',
                        data: {{ analysis_data.weekly_present | tojson }},
                        borderColor: '#28a745',
                        backgroundColor: 'rgba(40, 167, 69, 0.1)',
                        tension: 0.3,
                        fill: true,
                        borderWidth: 2
                    },
                    {
                        label: 'Late',
                        data: {{ analysis_data.weekly_late | tojson }},
                        borderColor: '#ffc107',
                        backgroundColor: 'rgba(255, 193, 7, 0.1)',
                        tension: 0.3,
                        fill: true,
                        borderWidth: 2
                    },
                    {
                        label: 'Absent',
                        data: {{ analysis_data.weekly_absent | tojson }},
                        borderColor: '#dc3545',
                        backgroundColor: 'rgba(220, 53, 69, 0.1)',
                        tension: 0.3,
                        fill: true,
                        borderWidth: 2
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'top',
                    },
                    tooltip: {
                        mode: 'index',
                        intersect: false
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        grid: {
                            drawBorder: false
                        }
                    },
                    x: {
                        grid: {
                            display: false
                        }
                    }
                }
            }
        });
        
        // Monthly Attendance Chart
        const monthlyCtx = document.getElementById('monthlyChart').getContext('2d');
        const monthlyChart = new Chart(monthlyCtx, {
            type: 'bar',
            data: {
                labels: {{ analysis_data.months | tojson }},
                datasets: [
                    {
                        label: 'Present',
                        data: {{ analysis_data.monthly_present | tojson }},
                        backgroundColor: '#28a745',
                        barPercentage: 0.6,
                    },
                    {
                        label: 'Late',
                        data: {{ analysis_data.monthly_late | tojson }},
                        backgroundColor: '#ffc107',
                        barPercentage: 0.6,
                    },
                    {
                        label: 'Absent',
                        data: {{ analysis_data.monthly_absent | tojson }},
                        backgroundColor: '#dc3545',
                        barPercentage: 0.6,
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'top',
                    },
                },
                scales: {
                    x: {
                        stacked: true,
                        grid: {
                            display: false
                        }
                    },
                    y: {
                        stacked: true,
                        grid: {
                            drawBorder: false
                        }
                    }
                }
            }
        });
        
        // Department Chart
        const deptCtx = document.getElementById('departmentChart').getContext('2d');
        const departmentChart = new Chart(deptCtx, {
            type: 'bar',
            data: {
                labels: {{ analysis_data.dept_names | tojson }},
                datasets: [
                    {
                        label: 'Present',
                        data: {{ analysis_data.dept_present | tojson }},
                        backgroundColor: '#28a745',
                    },
                    {
                        label: 'Late',
                        data: {{ analysis_data.dept_late | tojson }},
                        backgroundColor: '#ffc107',
                    },
                    {
                        label: 'Absent',
                        data: {{ analysis_data.dept_absent | tojson }},
                        backgroundColor: '#dc3545',
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'top',
                    },
                },
                scales: {
                    x: {
                        grid: {
                            display: false
                        }
                    },
                    y: {
                        grid: {
                            drawBorder: false
                        }
                    }
                }
            }
        });
        
        // 7-Day Trend Chart
        const trendCtx = document.getElementById('trendChart').getContext('2d');
        const trendChart = new Chart(trendCtx, {
            type: 'line',
            data: {
                labels: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
                datasets: [{
                    label: 'Attendance Rate',
                    data: [85, 92, 78, 95, 88, 65, 70],
                    borderColor: '#1a2a6c',
                    backgroundColor: 'rgba(26, 42, 108, 0.1)',
                    tension: 0.4,
                    fill: true,
                    borderWidth: 3
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    y: {
                        beginAtZero: false,
                        min: 50,
                        max: 100,
                        grid: {
                            drawBorder: false
                        }
                    },
                    x: {
                        grid: {
                            display: false
                        }
                    }
                }
            }
        });
        
        // Monthly Comparison Chart
        const monthlyCompCtx = document.getElementById('monthlyComparisonChart').getContext('2d');
        const monthlyComparisonChart = new Chart(monthlyCompCtx, {
            type: 'bar',
            data: {
                labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
                datasets: [
                    {
                        label: '2023',
                        data: [85, 78, 90, 82, 88, 92],
                        backgroundColor: 'rgba(26, 42, 108, 0.7)',
                        barPercentage: 0.6,
                    },
                    {
                        label: '2024',
                        data: [88, 82, 95, 90, 92, 96],
                        backgroundColor: 'rgba(255, 138, 0, 0.7)',
                        barPercentage: 0.6,
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'top',
                    },
                },
                scales: {
                    x: {
                        grid: {
                            display: false
                        }
                    },
                    y: {
                        grid: {
                            drawBorder: false
                        },
                        ticks: {
                            callback: function(value) {
                                return value + '%';
                            }
                        }
                    }
                }
            }
        });
        
        // Real-time updates with AJAX
        function updateCharts() {
            fetch('/api/attendance_stats')
                .then(response => response.json())
                .then(data => {
                    // Update your charts with new data
                    todayChart.data.datasets[0].data = data.today_counts;
                    todayChart.update();
                    
                    weeklyChart.data.datasets[0].data = data.weekly_present;
                    weeklyChart.data.datasets[1].data = data.weekly_late;
                    weeklyChart.data.datasets[2].data = data.weekly_absent;
                    weeklyChart.update();
                    
                    // Update other charts similarly...
                });
        }
        
        // Update charts every 5 minutes
        setInterval(updateCharts, 300000);
        
        // Filter functionality
        document.getElementById('applyFilter').addEventListener('click', function() {
            const startDate = document.getElementById('startDate').value;
            const endDate = document.getElementById('endDate').value;
            const department = document.getElementById('departmentFilter').value;
            
            // Send request to server to get filtered data
            fetch(`/api/attendance_stats?start=${startDate}&end=${endDate}&dept=${department}`)
                .then(response => response.json())
                .then(data => {
                    // Update charts with filtered data
                    todayChart.data.datasets[0].data = data.today_counts;
                    todayChart.update();
                    
                    weeklyChart.data.labels = data.weeks;
                    weeklyChart.data.datasets[0].data = data.weekly_present;
                    weeklyChart.data.datasets[1].data = data.weekly_late;
                    weeklyChart.data.datasets[2].data = data.weekly_absent;
                    weeklyChart.update();
                    
                    // Update other charts similarly...
                });
        });
    </script>
</body>
</html>""")
# ... (your existing code continues above)

# Create all required templates
with open('templates/manage_employees.html', 'w') as f:
    f.write("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Manage Employees | Skyhighes Technologies</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        .employee-table {
            background: white;
            border-radius: 15px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        .table-header {
            background: linear-gradient(to right, #1a2a6c, #3a5fc5);
            color: white;
            padding: 15px 20px;
        }
        .status-active {
            color: #28a745;
        }
        .status-inactive {
            color: #dc3545;
        }
        .status-suspended {
            color: #ffc107;
        }
        .action-buttons .btn {
            margin-right: 5px;
        }
    </style>
</head>
<body>
    {% extends "admin_base.html" %}

    {% block page_title %}Manage Employees{% endblock %}

    {% block content %}
    
    
        <div class="container-fluid">
            <div class="d-flex justify-content-between align-items-center mb-4">
                <h2>Manage Employees</h2>
                <a href="/admin/employees/add" class="btn btn-primary">
                    <i class="fas fa-plus me-2"></i> Add Employee
                </a>
            </div>
            
            {% with messages = get_flashed_messages(with_categories=true) %}
                {% if messages %}
                    {% for category, message in messages %}
                        <div class="alert alert-{{ category }} alert-dismissible fade show" role="alert">
                            {{ message }}
                            <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
                        </div>
                    {% endfor %}
                {% endif %}
            {% endwith %}
            
            <div class="employee-table">
                <div class="table-header">
                    <h4 class="mb-0">Employee List</h4>
                </div>
                <div class="table-responsive">
                    <table class="table table-hover">
                        <thead>
                            <tr>
                                <th>ID</th>
                                <th>Name</th>
                                <th>Department</th>
                                <th>Position</th>
                                <th>Status</th>
                                <th>Join Date</th>
                                <th>Actions</th>
                            </tr>
                        </thead>
                        <tbody>
                            {% for employee in employees %}
                            <tr>
                                <td>{{ employee.employee_id }}</td>
                                <td>
                                    {% if employee.profile_image %}
                                    <img src="{{ url_for('static', filename=employee.profile_image) }}" alt="Profile" class="rounded-circle me-2" width="30" height="30">
                                    {% else %}
                                    <i class="fas fa-user-circle me-2"></i>
                                    {% endif %}
                                    {{ employee.name }}
                                </td>
                                <td>{{ employee.department }}</td>
                                <td>{{ employee.position }}</td>
                                <td>
                                    {% if employee.status == 'active' %}
                                        <span class="status-active"><i class="fas fa-circle"></i> Active</span>
                                    {% elif employee.status == 'inactive' %}
                                        <span class="status-inactive"><i class="fas fa-circle"></i> Inactive</span>
                                    {% else %}
                                        <span class="status-suspended"><i class="fas fa-circle"></i> Suspended</span>
                                    {% endif %}
                                </td>
                                <td>{{ employee.join_date.strftime('%Y-%m-%d') }}</td>
                                <td class="action-buttons">
                                    <a href="/admin/employees/view/{{ employee.id }}" class="btn btn-sm btn-info">
                                        <i class="fas fa-eye"></i>
                                    </a>
                                    <a href="/admin/employees/edit/{{ employee.id }}" class="btn btn-sm btn-warning">
                                        <i class="fas fa-edit"></i>
                                    </a>
                                    <a href="/admin/employees/register_face/{{ employee.id }}" class="btn btn-sm btn-primary">
                                        <i class="fas fa-camera"></i>
                                    </a>
                                    <form action="/admin/employees/delete/{{ employee.id }}" method="POST" class="d-inline">
                                        <button type="submit" class="btn btn-sm btn-danger" onclick="return confirm('Are you sure you want to delete this employee?')">
                                            <i class="fas fa-trash"></i>
                                        </button>
                                    </form>
                                </td>
                            </tr>
                            {% else %}
                            <tr>
                                <td colspan="7" class="text-center">No employees found</td>
                            </tr>
                            {% endfor %}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>

    {% endblock %}
</body>
</html>""")

with open('templates/view_employee.html', 'w') as f:
    f.write("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>View Employee | Skyhighes Technologies</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        .profile-card {
            background: white;
            border-radius: 15px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            padding: 25px;
            margin-bottom: 30px;
        }
        .profile-header {
            display: flex;
            align-items: center;
            margin-bottom: 20px;
        }
        .profile-image {
            width: 100px;
            height: 100px;
            border-radius: 50%;
            overflow: hidden;
            margin-right: 20px;
        }
        .profile-image img {
            width: 100%;
            height: 100%;
            object-fit: cover;
        }
        .attendance-table {
            background: white;
            border-radius: 15px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        .badge-present {
            background-color: #28a745;
        }
        .badge-late {
            background-color: #ffc107;
            color: #212529;
        }
        .badge-half_day {
            background-color: #17a2b8;
        }
        .badge-absent {
            background-color: #dc3545;
        }
        .badge-overtime {
            background-color: #6f42c1;
        }
        .nav-tabs .nav-link.active {
            font-weight: bold;
            border-bottom: 3px solid #1a2a6c;
        }
    </style>
</head>
<body>
    <div class="main-content">
        <div class="container-fluid">
            <div class="d-flex justify-content-between align-items-center mb-4">
                <h2>Employee Details</h2>
                <a href="/admin/employees" class="btn btn-outline-secondary">
                    <i class="fas fa-arrow-left me-1"></i> Back to Employees
                </a>
            </div>
            
            <div class="profile-card">
                <div class="profile-header">
                    <div class="profile-image">
                        {% if employee.profile_image %}
                            <img src="{{ url_for('static', filename=employee.profile_image) }}" alt="Profile">
                        {% else %}
                            <div class="d-flex align-items-center justify-content-center h-100 bg-light">
                                <i class="fas fa-user fa-2x text-secondary"></i>
                            </div>
                        {% endif %}
                    </div>
                    <div>
                        <h3>{{ employee.name }}</h3>
                        <p class="text-muted mb-0">{{ employee.position }} | {{ employee.department }}</p>
                        <p class="text-muted">Employee ID: {{ employee.employee_id }}</p>
                    </div>
                </div>
                
                <div class="row">
                    <div class="col-md-6">
                        <h5>Personal Information</h5>
                        <table class="table table-sm">
                            <tr>
                                <th>Email:</th>
                                <td>{{ employee.email or 'N/A' }}</td>
                            </tr>
                            <tr>
                                <th>Phone:</th>
                                <td>{{ employee.phone or 'N/A' }}</td>
                            </tr>
                            <tr>
                                <th>Status:</th>
                                <td>
                                    {% if employee.status == 'active' %}
                                        <span class="badge bg-success">Active</span>
                                    {% elif employee.status == 'inactive' %}
                                        <span class="badge bg-danger">Inactive</span>
                                    {% else %}
                                        <span class="badge bg-warning">Suspended</span>
                                    {% endif %}
                                </td>
                            </tr>
                            <tr>
                                <th>Join Date:</th>
                                <td>{{ employee.join_date.strftime('%Y-%m-%d') }}</td>
                            </tr>
                        </table>
                    </div>
                    <div class="col-md-6">
                        <h5>Emergency Contact</h5>
                        <table class="table table-sm">
                            <tr>
                                <th>Contact Person:</th>
                                <td>{{ employee.emergency_contact or 'N/A' }}</td>
                            </tr>
                            <tr>
                                <th>Emergency Phone:</th>
                                <td>{{ employee.emergency_phone or 'N/A' }}</td>
                            </tr>
                        </table>
                    </div>
                </div>
            </div>
            
            <ul class="nav nav-tabs" id="employeeTabs" role="tablist">
                <li class="nav-item" role="presentation">
                    <button class="nav-link active" id="attendance-tab" data-bs-toggle="tab" data-bs-target="#attendance" type="button" role="tab">Attendance Records</button>
                </li>
                <li class="nav-item" role="presentation">
                    <button class="nav-link" id="requests-tab" data-bs-toggle="tab" data-bs-target="#requests" type="button" role="tab">Check-in Requests</button>
                </li>
            </ul>
            
            <div class="tab-content" id="employeeTabsContent">
                <div class="tab-pane fade show active" id="attendance" role="tabpanel">
                    <div class="attendance-table mt-3">
                        <div class="table-header bg-light p-3">
                            <h5 class="mb-0">Attendance History</h5>
                        </div>
                        <div class="table-responsive">
                            <table class="table table-hover">
                                <thead>
                                    <tr>
                                        <th>Date</th>
                                        <th>Check In</th>
                                        <th>Check Out</th>
                                        <th>Status</th>
                                        <th>Confidence</th>
                                        <th>Location</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {% for record in attendance_records %}
                                    <tr>
                                        <td>{{ record.check_in.strftime('%Y-%m-%d') }}</td>
                                        <td>{{ record.check_in.strftime('%H:%M:%S') }}</td>
                                        <td>{{ record.check_out.strftime('%H:%M:%S') if record.check_out else 'N/A' }}</td>
                                        <td>
                                            {% if record.status == 'present' %}
                                                <span class="badge badge-present">Present</span>
                                            {% elif record.status == 'late' %}
                                                <span class="badge badge-late">Late</span>
                                            {% elif record.status == 'half_day' %}
                                                <span class="badge badge-half_day">Half Day</span>
                                            {% elif record.status == 'overtime' %}
                                                <span class="badge badge-overtime">Overtime</span>
                                            {% else %}
                                                <span class="badge badge-absent">{{ record.status|title }}</span>
                                            {% endif %}
                                        </td>
                                        <td>{{ record.confidence|round(2) }}</td>
                                        <td>{{ record.location or 'N/A' }}</td>
                                    </tr>
                                    {% else %}
                                    <tr>
                                        <td colspan="6" class="text-center">No attendance records found</td>
                                    </tr>
                                    {% endfor %}
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
                
                <div class="tab-pane fade" id="requests" role="tabpanel">
                    <div class="attendance-table mt-3">
                        <div class="table-header bg-light p-3">
                            <h5 class="mb-0">Check-in Requests</h5>
                        </div>
                        <div class="table-responsive">
                            <table class="table table-hover">
                                <thead>
                                    <tr>
                                        <th>Request Date</th>
                                        <th>Reason</th>
                                        <th>Status</th>
                                        <th>Reviewed At</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {% for request in checkin_requests %}
                                    <tr>
                                        <td>{{ request.request_date.strftime('%Y-%m-%d') }}</td>
                                        <td>{{ request.reason or 'No reason provided' }}</td>
                                        <td>
                                            {% if request.status == 'pending' %}
                                                <span class="badge bg-warning">Pending</span>
                                            {% elif request.status == 'approved' %}
                                                <span class="badge bg-success">Approved</span>
                                            {% else %}
                                                <span class="badge bg-danger">Rejected</span>
                                            {% endif %}
                                        </td>
                                        <td>{{ request.reviewed_at.strftime('%Y-%m-%d %H:%M') if request.reviewed_at else 'N/A' }}</td>
                                    </tr>
                                    {% else %}
                                    <tr>
                                        <td colspan="4" class="text-center">No check-in requests found</td>
                                    </tr>
                                    {% endfor %}
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>""")

with open('templates/add_employee.html', 'w') as f:
    f.write("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Add Employee | Skyhighes Technologies</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        .form-container {
            background: white;
            border-radius: 15px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            padding: 25px;
        }
    </style>
</head>
<body>
    <div class="main-content">
        <div class="container-fluid">
            <div class="d-flex justify-content-between align-items-center mb-4">
                <h2>Add New Employee</h2>
                <a href="/admin/employees" class="btn btn-outline-secondary">
                    <i class="fas fa-arrow-left me-1"></i> Back to Employees
                </a>
            </div>
            
            {% with messages = get_flashed_messages(with_categories=true) %}
                {% if messages %}
                    {% for category, message in messages %}
                        <div class="alert alert-{{ category }} alert-dismissible fade show" role="alert">
                            {{ message }}
                            <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
                        </div>
                    {% endfor %}
                {% endif %}
            {% endwith %}
            
            <div class="form-container">
                <form method="POST">
                    <div class="row">
                        <div class="col-md-6">
                            <div class="mb-3">
                                <label for="employee_id" class="form-label">Employee ID *</label>
                                <input type="text" class="form-control" id="employee_id" name="employee_id" required>
                            </div>
                            <div class="mb-3">
                                <label for="name" class="form-label">Full Name *</label>
                                <input type="text" class="form-control" id="name" name="name" required>
                            </div>
                            <div class="mb-3">
                                <label for="position" class="form-label">Position</label>
                                <input type="text" class="form-control" id="position" name="position">
                            </div>
                            <div class="mb-3">
                                <label for="department" class="form-label">Department</label>
                                <input type="text" class="form-control" id="department" name="department">
                            </div>
                        </div>
                        <div class="col-md-6">
                            <div class="mb-3">
                                <label for="email" class="form-label">Email</label>
                                <input type="email" class="form-control" id="email" name="email">
                            </div>
                            <div class="mb-3">
                                <label for="phone" class="form-label">Phone</label>
                                <input type="tel" class="form-control" id="phone" name="phone">
                            </div>
                            <div class="mb-3">
                                <label for="password" class="form-label">Password *</label>
                                <input type="password" class="form-control" id="password" name="password" required>
                                <div class="form-text">Minimum 8 characters with letters and numbers</div>
                            </div>
                        </div>
                    </div>
                    <button type="submit" class="btn btn-primary">
                        <i class="fas fa-save me-2"></i> Add Employee
                    </button>
                </form>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>""")

with open('templates/edit_employee.html', 'w') as f:
    f.write("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Edit Employee | Skyhighes Technologies</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        .form-container {
            background: white;
            border-radius: 15px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            padding: 25px;
        }
    </style>
</head>
<body>
    <div class="main-content">
        <div class="container-fluid">
            <div class="d-flex justify-content-between align-items-center mb-4">
                <h2>Edit Employee</h2>
                <a href="/admin/employees" class="btn btn-outline-secondary">
                    <i class="fas fa-arrow-left me-1"></i> Back to Employees
                </a>
            </div>
            
            {% with messages = get_flashed_messages(with_categories=true) %}
                {% if messages %}
                    {% for category, message in messages %}
                        <div class="alert alert-{{ category }} alert-dismissible fade show" role="alert">
                            {{ message }}
                            <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
                        </div>
                    {% endfor %}
                {% endif %}
            {% endwith %}
            
            <div class="form-container">
                <form method="POST">
                    <div class="row">
                        <div class="col-md-6">
                            <div class="mb-3">
                                <label for="employee_id" class="form-label">Employee ID</label>
                                <input type="text" class="form-control" id="employee_id" value="{{ employee.employee_id }}" disabled>
                                <input type="hidden" name="employee_id" value="{{ employee.employee_id }}">
                            </div>
                            <div class="mb-3">
                                <label for="name" class="form-label">Full Name *</label>
                                <input type="text" class="form-control" id="name" name="name" value="{{ employee.name }}" required>
                            </div>
                            <div class="mb-3">
                                <label for="position" class="form-label">Position</label>
                                <input type="text" class="form-control" id="position" name="position" value="{{ employee.position or '' }}">
                            </div>
                            <div class="mb-3">
                                <label for="department" class="form-label">Department</label>
                                <input type="text" class="form-control" id="department" name="department" value="{{ employee.department or '' }}">
                            </div>
                        </div>
                        <div class="col-md-6">
                            <div class="mb-3">
                                <label for="email" class="form-label">Email</label>
                                <input type="email" class="form-control" id="email" name="email" value="{{ employee.email or '' }}">
                            </div>
                            <div class="mb-3">
                                <label for="phone" class="form-label">Phone</label>
                                <input type="tel" class="form-control" id="phone" name="phone" value="{{ employee.phone or '' }}">
                            </div>
                            <div class="mb-3">
                                <label for="status" class="form-label">Status</label>
                                <select class="form-select" id="status" name="status">
                                    <option value="active" {% if employee.status == 'active' %}selected{% endif %}>Active</option>
                                    <option value="inactive" {% if employee.status == 'inactive' %}selected{% endif %}>Inactive</option>
                                    <option value="suspended" {% if employee.status == 'suspended' %}selected{% endif %}>Suspended</option>
                                </select>
                            </div>
                        </div>
                    </div>
                    <button type="submit" class="btn btn-primary">
                        <i class="fas fa-save me-2"></i> Update Employee
                    </button>
                </form>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>""")

with open('templates/register_face.html', 'w') as f:
    f.write("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Register Face | Skyhighes Technologies</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        .upload-container {
            background: white;
            border-radius: 15px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            padding: 25px;
            text-align: center;
        }
        .upload-area {
            border: 2px dashed #dee2e6;
            border-radius: 10px;
            padding: 40px;
            margin: 20px 0;
            cursor: pointer;
            transition: all 0.3s;
        }
        .upload-area:hover {
            border-color: #1a2a6c;
            background-color: #f8f9fa;
        }
        .upload-icon {
            font-size: 3rem;
            color: #6c757d;
            margin-bottom: 15px;
        }
        #image-preview {
            max-width: 100%;
            max-height: 300px;
            margin-top: 20px;
            display: none;
            border-radius: 10px;
        }
    </style>
</head>
<body>
    <div class="main-content">
        <div class="container-fluid">
            <div class="d-flex justify-content-between align-items-center mb-4">
                <h2>Register Face for {{ employee.name }}</h2>
                <a href="/admin/employees" class="btn btn-outline-secondary">
                    <i class="fas fa-arrow-left me-1"></i> Back to Employees
                </a>
            </div>
            
            {% with messages = get_flashed_messages(with_categories=true) %}
                {% if messages %}
                    {% for category, message in messages %}
                        <div class="alert alert-{{ category }} alert-dismissible fade show" role="alert">
                            {{ message }}
                            <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
                        </div>
                    {% endfor %}
                {% endif %}
            {% endwith %}
            
            <div class="upload-container">
                <h4>Upload Face Image</h4>
                <p class="text-muted">Please upload a clear frontal face image for facial recognition</p>
                
                <form method="POST" enctype="multipart/form-data">
                    <div class="upload-area" id="upload-area">
                        <div class="upload-icon">
                            <i class="fas fa-cloud-upload-alt"></i>
                        </div>
                        <h5>Drag & Drop or Click to Upload</h5>
                        <p class="text-muted">Supported formats: JPG, PNG, JPEG</p>
                        <input type="file" id="face_image" name="face_image" accept="image/*" class="d-none" required>
                    </div>
                    
                    <img id="image-preview" alt="Image preview">
                    
                    <div class="alert alert-info mt-3">
                        <i class="fas fa-info-circle me-2"></i>
                        For best results: 
                        <ul class="mb-0 mt-2">
                            <li>Use a well-lit environment</li>
                            <li>Face should be clearly visible without obstructions</li>
                            <li>Look directly at the camera</li>
                            <li>Avoid sunglasses, hats, or face coverings</li>
                        </ul>
                    </div>
                    
                    <button type="submit" class="btn btn-primary mt-3" id="submit-btn" disabled>
                        <i class="fas fa-camera me-2"></i> Register Face
                    </button>
                </form>
            </div>
        </div>
    </div>

    <script>
        const uploadArea = document.getElementById('upload-area');
        const fileInput = document.getElementById('face_image');
        const imagePreview = document.getElementById('image-preview');
        const submitBtn = document.getElementById('submit-btn');
        
        uploadArea.addEventListener('click', () => {
            fileInput.click();
        });
        
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.style.borderColor = '#1a2a6c';
            uploadArea.style.backgroundColor = '#f8f9fa';
        });
        
        uploadArea.addEventListener('dragleave', () => {
            uploadArea.style.borderColor = '#dee2e6';
            uploadArea.style.backgroundColor = 'transparent';
        });
        
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            if (e.dataTransfer.files.length) {
                fileInput.files = e.dataTransfer.files;
                handleFile(fileInput.files[0]);
            }
        });
        
        fileInput.addEventListener('change', () => {
            if (fileInput.files.length) {
                handleFile(fileInput.files[0]);
            }
        });
        
        function handleFile(file) {
            if (file.type.match('image.*')) {
                const reader = new FileReader();
                
                reader.onload = function(e) {
                    imagePreview.src = e.target.result;
                    imagePreview.style.display = 'block';
                    submitBtn.disabled = false;
                    uploadArea.style.display = 'none';
                };
                
                reader.readAsDataURL(file);
            } else {
                alert('Please select an image file');
            }
        }
    </script>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>""")

with open('templates/view_attendance.html', 'w') as f:
    f.write("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>View Attendance | Skyhighes Technologies</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        .filter-card {
            background: white;
            border-radius: 15px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            padding: 20px;
            margin-bottom: 30px;
        }
        .attendance-table {
            background: white;
            border-radius: 15px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        .table-header {
            background: linear-gradient(to right, #1a2a6c, #3a5fc5);
            color: white;
            padding: 15px 20px;
        }
        .badge-present {
            background-color: #28a745;
        }
        .badge-late {
            background-color: #ffc107;
            color: #212529;
        }
        .badge-half_day {
            background-color: #17a2b8;
        }
        .badge-absent {
            background-color: #dc3545;
        }
        .badge-overtime {
            background-color: #6f42c1;
        }
    </style>
</head>
<body>
            {% extends "admin_base.html" %}

            {% block page_title %}View Attendance{% endblock %}

            {% block content %}
        <div class="container-fluid">
            <div class="d-flex justify-content-between align-items-center mb-4">
                <h2>Attendance Records</h2>
            </div>
            
            <div class="filter-card">
                <h4 class="mb-4">Filter Records</h4>
                <form method="GET" class="row g-3">
                    <div class="col-md-5">
                        <label for="start_date" class="form-label">Start Date</label>
                        <input type="date" class="form-control" id="start_date" name="start_date" value="{{ start_date }}">
                    </div>
                    <div class="col-md-5">
                        <label for="end_date" class="form-label">End Date</label>
                        <input type="date" class="form-control" id="end_date" name="end_date" value="{{ end_date }}">
                    </div>
                    <div class="col-md-2 d-flex align-items-end">
                        <button type="submit" class="btn btn-primary w-100">
                            <i class="fas fa-filter me-2"></i> Filter
                        </button>
                    </div>
                </form>
            </div>
            
            <div class="attendance-table">
                <div class="table-header">
                    <h4 class="mb-0">Attendance Records</h4>
                </div>
                <div class="table-responsive">
                    <table class="table table-hover">
                        <thead>
                            <tr>
                                <th>Name</th>
                                <th>Department</th>
                                <th>Check In</th>
                                <th>Check Out</th>
                                <th>Status</th>
                                <th>Confidence</th>
                                <th>Location</th>
                            </tr>
                        </thead>
                        <tbody>
                            {% for record in attendance_records %}
                            <tr>
                                <td>{{ record.name }}</td>
                                <td>{{ record.department }}</td>
                                <td>{{ record.check_in.strftime('%Y-%m-%d %H:%M:%S') }}</td>
                                <td>{{ record.check_out.strftime('%Y-%m-%d %H:%M:%S') if record.check_out else 'N/A' }}</td>
                                <td>
                                    {% if record.status == 'present' %}
                                        <span class="badge badge-present">Present</span>
                                    {% elif record.status == 'late' %}
                                        <span class="badge badge-late">Late</span>
                                    {% elif record.status == 'half_day' %}
                                        <span class="badge badge-half_day">Half Day</span>
                                    {% elif record.status == 'overtime' %}
                                        <span class="badge badge-overtime">Overtime</span>
                                    {% else %}
                                        <span class="badge badge-absent">{{ record.status|title }}</span>
                                    {% endif %}
                                </td>
                                <td>{{ record.confidence|round(2) }}</td>
                                <td>{{ record.location or 'N/A' }}</td>
                            </tr>
                            {% else %}
                            <tr>
                                <td colspan="7" class="text-center">No attendance records found for the selected date range</td>
                            </tr>
                            {% endfor %}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    </div>
            {% endblock %}

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>""")

with open('templates/recognition_control.html', 'w') as f:
    f.write("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Recognition Control | Skyhighes Technologies</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        .control-card {
            background: white;
            border-radius: 15px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            padding: 25px;
            margin-bottom: 30px;
        }
        .status-active {
            color: #28a745;
        }
        .status-inactive {
            color: #dc3545;
        }
        .stats-card {
            background: white;
            border-radius: 15px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            padding: 20px;
            margin-bottom: 20px;
        }
        .stat-value {
            font-size: 2rem;
            font-weight: bold;
            margin: 10px 0;
        }
        .camera-card {
            background: white;
            border-radius: 15px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            padding: 20px;
            margin-bottom: 20px;
        }
        .camera-status {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            display: inline-block;
            margin-right: 5px;
        }
        .status-online {
            background-color: #28a745;
        }
        .status-offline {
            background-color: #dc3545;
        }
    </style>
</head>
<body>
    {% extends "admin_base.html" %}

    {% block page_title %}CCTV Recognition{% endblock %}

    {% block content %}
        <div class="container-fluid">
            <div class="d-flex justify-content-between align-items-center mb-4">
                <h2>Recognition System Control</h2>
            </div>
            
            <div class="row">
                <div class="col-md-8">
                    <div class="control-card">
                        <h4 class="mb-4">System Control</h4>
                        <div class="row mb-4">
                            <div class="col-md-6">
                                <div class="d-flex align-items-center">
                                    <h5 class="mb-0 me-3">Recognition Status:</h5>
                                    <span id="recognition-status" class="{% if recognition_active %}status-active{% else %}status-inactive{% endif %}">
                                        <i class="fas fa-circle"></i> {% if recognition_active %}ACTIVE{% else %}INACTIVE{% endif %}
                                    </span>
                                </div>
                            </div>
                            <div class="col-md-6 text-end">
                                <button id="start-btn" class="btn btn-success me-2" {% if recognition_active %}disabled{% endif %}>
                                    <i class="fas fa-play me-2"></i> Start Recognition
                                </button>
                                <button id="stop-btn" class="btn btn-danger" {% if not recognition_active %}disabled{% endif %}>
                                    <i class="fas fa-stop me-2"></i> Stop Recognition
                                </button>
                            </div>
                        </div>
                        
                       <!-- In the recognition_control.html template, update the train button section -->
<div class="row mb-4">
    <div class="col-md-12">
        <button id="train-btn" class="btn btn-primary">
            <i class="fas fa-brain me-2"></i> Train Model
        </button>
        <div id="training-status" class="alert alert-info mt-2" style="display: none;">
            <i class="fas fa-spinner fa-spin me-2"></i> Training in progress...
        </div>
        <div id="training-result" class="mt-2"></div>
        <div class="form-text mt-2">Retrain the facial recognition model with current data</div>
    </div>
</div>

<script>
    // Add this script to the recognition_control.html template
    document.getElementById('train-btn').addEventListener('click', function() {
        const trainingStatus = document.getElementById('training-status');
        const trainingResult = document.getElementById('training-result');
        
        trainingStatus.style.display = 'block';
        trainingResult.innerHTML = '';
        
        fetch('/admin/model/train', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        })
        .then(response => response.json())
        .then(data => {
            trainingStatus.style.display = 'none';
            
            if (data.success) {
                trainingResult.innerHTML = `
                    <div class="alert alert-success">
                        <i class="fas fa-check-circle me-2"></i> ${data.message}
                    </div>
                `;
                // Update AI status in system information
                document.getElementById('ai-status').textContent = 'Loaded';
            } else {
                trainingResult.innerHTML = `
                    <div class="alert alert-danger">
                        <i class="fas fa-exclamation-circle me-2"></i> ${data.message}
                    </div>
                `;
            }
        })
        .catch(error => {
            trainingStatus.style.display = 'none';
            trainingResult.innerHTML = `
                <div class="alert alert-danger">
                    <i class="fas fa-exclamation-circle me-2"></i> Error: ${error}
                </div>
            `;
        });
    });
</script>
                    
                    <div class="control-card">
                        <h4 class="mb-4">Cameras</h4>
                        <div class="row">
                            {% for camera_id, camera in cameras.items() %}
                            <div class="col-md-6 mb-3">
                                <div class="camera-card">
                                    <h5>{{ camera.name }}</h5>
                                    <p class="text-muted mb-2">{{ camera.location }}</p>
                                    <div class="mb-2">
                                        Status: 
                                        <span class="camera-status {% if camera.status == 'active' %}status-online{% else %}status-offline{% endif %}"></span>
                                        {{ camera.status|title }}
                                    </div>
                                    <div class="text-truncate small text-muted" title="{{ camera.rtsp_url }}">
                                        {{ camera.rtsp_url }}
                                    </div>
                                </div>
                            </div>
                            {% endfor %}
                        </div>
                    </div>
                </div>
                
                <div class="col-md-4">
                    <div class="stats-card">
                        <h5 class="mb-4">Recognition Statistics</h5>
                        <div class="row text-center">
                            <div class="col-6 mb-3">
                                <div class="stat-value" id="faces-detected">0</div>
                                <div>Faces Detected</div>
                            </div>
                            <div class="col-6 mb-3">
                                <div class="stat-value" id="employees-recognized">0</div>
                                <div>Employees Recognized</div>
                            </div>
                            <div class="col-6">
                                <div class="stat-value" id="unknown-faces">0</div>
                                <div>Unknown Faces</div>
                            </div>
                            <div class="col-6">
                                <div class="stat-value" id="last-recognition">N/A</div>
                                <div>Last Recognition</div>
                            </div>
                        </div>
                    </div>
                    
                    <div class="stats-card">
                        <h5 class="mb-4">System Information</h5>
                        <div class="mb-2">
                            <strong>AI Model:</strong> 
                            <span id="ai-status">{% if AI_AVAILABLE %}Loaded{% else %}Not Available{% endif %}</span>
                        </div>
                        <div class="mb-2">
                            <strong>Recognition Hours:</strong> 10:00 - 21:00
                        </div>
                        <div class="mb-2">
                            <strong>Active Cameras:</strong> {{ cameras|length }}
                        </div>
                        <div>
                            <strong>Last Model Training:</strong> <span id="model-last-trained">N/A</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        document.getElementById('start-btn').addEventListener('click', function() {
            fetch('/api/recognition/start', { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        alert('Recognition service started');
                        updateUI(true);
                    } else {
                        alert('Error: ' + data.message);
                    }
                });
        });
        
        document.getElementById('stop-btn').addEventListener('click', function() {
            fetch('/api/recognition/stop', { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        alert('Recognition service stopped');
                        updateUI(false);
                    } else {
                        alert('Error: ' + data.message);
                    }
                });
        });
        
        document.getElementById('train-btn').addEventListener('click', function() {
            if (confirm('Are you sure you want to train the model? This may take several minutes.')) {
                fetch('/api/model/train', { method: 'POST' })
                    .then(response => response.json())
                    .then(data => {
                        alert(data.message);
                    });
            }
        });
        
        function updateUI(isActive) {
            const statusElem = document.getElementById('recognition-status');
            const startBtn = document.getElementById('start-btn');
            const stopBtn = document.getElementById('stop-btn');
            
            if (isActive) {
                statusElem.innerHTML = '<i class="fas fa-circle"></i> ACTIVE';
                statusElem.className = 'status-active';
                startBtn.disabled = true;
                stopBtn.disabled = false;
            } else {
                statusElem.innerHTML = '<i class="fas fa-circle"></i> INACTIVE';
                statusElem.className = 'status-inactive';
                startBtn.disabled = false;
                stopBtn.disabled = true;
            }
        }
        
        // Periodically update stats
        function updateStats() {
            fetch('/api/recognition/status')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('faces-detected').textContent = data.stats.faces_detected;
                    document.getElementById('employees-recognized').textContent = data.stats.employees_recognized;
                    document.getElementById('unknown-faces').textContent = data.stats.unknown_faces;
                    
                    if (data.stats.last_recognition) {
                        // Format the timestamp if needed
                        document.getElementById('last-recognition').textContent = 'Recent';
                    }
                    
                    updateUI(data.active);
                });
        }
        
        // Update stats every 5 seconds
        setInterval(updateStats, 5000);
        updateStats(); // Initial update
    </script>
    {% endblock %}
</body>
</html>""")

with open('templates/debug.html', 'w') as f:
    f.write(""""<!-- Update the debug.html template -->
<!DOCTYPE html>
<html>
<head>
    <title>Debug - Recognition System</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/socket.io-client@4.0/dist/socket.io.min.js"></script>
</head>
<body>
    <div class="container mt-4">
        <h2>Recognition System Debug</h2>
        
        <div class="card mt-3">
            <div class="card-header d-flex justify-content-between align-items-center">
                <h5>System Status</h5>
                <button id="refresh-btn" class="btn btn-sm btn-outline-primary">Refresh Status</button>
            </div>
            <div class="card-body">
                <div class="row">
                    <div class="col-md-6">
                        <p><strong>AI Available:</strong> {{ debug_info.ai_available }}</p>
                        <p><strong>Model Loaded:</strong> 
                            <span id="model-status" class="badge {% if debug_info.model_loaded %}bg-success{% else %}bg-warning{% endif %}">
                                {% if debug_info.model_loaded %}Yes{% else %}No{% endif %}
                            </span>
                        </p>
                        <p><strong>Recognition Status:</strong> 
                            <span id="recognition-status" class="badge {% if recognition_active %}bg-success{% else %}bg-danger{% endif %}">
                                {% if recognition_active %}ACTIVE{% else %}INACTIVE{% endif %}
                            </span>
                        </p>
                    </div>
                    <div class="col-md-6">
                        <p><strong>Total Employees:</strong> {{ debug_info.total_employees }}</p>
                        <p><strong>Employees with Embeddings:</strong> {{ debug_info.employees_with_embeddings }}</p>
                        <p><strong>Training Status:</strong> 
                            {% if debug_info.employees_with_embeddings >= 2 %}
                                <span class="badge bg-success">Ready to train</span>
                            {% else %}
                                <span class="badge bg-warning">Need at least 2 employees with embeddings</span>
                            {% endif %}
                        </p>
                    </div>
                </div>
            </div>
        </div>

        <div class="card mt-3">
            <div class="card-header d-flex justify-content-between align-items-center">
                <h5>Model Control</h5>
                <button id="train-btn" class="btn btn-sm btn-warning">Train Model Now</button>
            </div>
            <div class="card-body">
                <div id="training-status" class="alert alert-info" style="display: none;">
                    <i class="fas fa-spinner fa-spin"></i> Training in progress...
                </div>
                <div id="training-result"></div>
            </div>
        </div>

        <div class="card mt-3">
            <div class="card-header">
                <h5>Employee Embedding Status</h5>
            </div>
            <div class="card-body">
                <table class="table table-striped">
                    <thead>
                        <tr>
                            <th>Employee ID</th>
                            <th>Name</th>
                            <th>Embedding Count</th>
                            <th>Status</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for emp in debug_info.employee_details %}
                        <tr>
                            <td>{{ emp.employee_id }}</td>
                            <td>{{ emp.name }}</td>
                            <td>{{ emp.embedding_count }}</td>
                            <td>
                                {% if emp.embedding_count > 0 %}
                                    <span class="badge bg-success">Registered</span>
                                {% else %}
                                    <span class="badge bg-warning">No Face Data</span>
                                {% endif %}
                            </td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </div>
        </div>

        <div class="mt-3">
            <a href="/admin/recognition" class="btn btn-primary">Back to Recognition Control</a>
            <a href="/admin/cctv_config" class="btn btn-info">CCTV Configuration</a>
        </div>
    </div>

    <script>
        const socket = io();
        
        // Handle model training
        document.getElementById('train-btn').addEventListener('click', function() {
            const trainingStatus = document.getElementById('training-status');
            const trainingResult = document.getElementById('training-result');
            
            trainingStatus.style.display = 'block';
            trainingResult.innerHTML = '';
            
            fetch('/admin/model/train', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            })
            .then(response => response.json())
            .then(data => {
                trainingStatus.style.display = 'none';
                
                if (data.success) {
                    trainingResult.innerHTML = `
                        <div class="alert alert-success">
                            <i class="fas fa-check-circle"></i> ${data.message}
                        </div>
                    `;
                    // Update model status
                    document.getElementById('model-status').textContent = 'Yes';
                    document.getElementById('model-status').className = 'badge bg-success';
                } else {
                    trainingResult.innerHTML = `
                        <div class="alert alert-danger">
                            <i class="fas fa-exclamation-circle"></i> ${data.message}
                        </div>
                    `;
                }
            })
            .catch(error => {
                trainingStatus.style.display = 'none';
                trainingResult.innerHTML = `
                    <div class="alert alert-danger">
                        <i class="fas fa-exclamation-circle"></i> Error: ${error}
                    </div>
                `;
            });
        });
        
        // Refresh status
        document.getElementById('refresh-btn').addEventListener('click', function() {
            location.reload();
        });
        
        // Socket event for model updates
        socket.on('model_update', function(data) {
            if (data.status === 'trained') {
                const trainingResult = document.getElementById('training-result');
                trainingResult.innerHTML = `
                    <div class="alert alert-success">
                        <i class="fas fa-check-circle"></i> ${data.message}
                    </div>
                `;
                document.getElementById('model-status').textContent = 'Yes';
                document.getElementById('model-status').className = 'badge bg-success';
            }
        });
        
        // Socket event for recognition status updates
        socket.on('recognition_status', function(data) {
            const statusElement = document.getElementById('recognition-status');
            statusElement.textContent = data.active ? 'ACTIVE' : 'INACTIVE';
            statusElement.className = data.active ? 'badge bg-success' : 'badge bg-danger';
        });
    </script>
</body>
</html>""")

with open('templates/cctv_config.html', 'w') as f:
    f.write("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CCTV Configuration | Skyhighes Technologies</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        .config-card {
            background: white;
            border-radius: 15px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            padding: 25px;
            margin-bottom: 30px;
        }
        .camera-card {
            background: #f8f9fa;
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 15px;
        }
        .camera-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }
        .camera-status {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            display: inline-block;
            margin-right: 5px;
        }
        .status-active {
            background-color: #28a745;
        }
        .status-inactive {
            background-color: #6c757d;
        }
        .status-maintenance {
            background-color: #ffc107;
        }
        .test-btn {
            margin-left: 10px;
        }
    </style>
</head>
<body>
    <div class="main-content">
        <div class="container-fluid">
            <div class="d-flex justify-content-between align-items-center mb-4">
                <h2>CCTV Configuration</h2>
            </div>
            
            {% with messages = get_flashed_messages(with_categories=true) %}
                {% if messages %}
                    {% for category, message in messages %}
                        <div class="alert alert-{{ category }} alert-dismissible fade show" role="alert">
                            {{ message }}
                            <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
                        </div>
                    {% endfor %}
                {% endif %}
            {% endwith %}
            
            <div class="config-card">
                <h4 class="mb-4">Add/Edit Camera</h4>
                <form method="POST">
                    <div class="row">
                        <div class="col-md-6">
                            <div class="mb-3">
                                <label for="camera_id" class="form-label">Camera ID</label>
                                <input type="text" class="form-control" id="camera_id" name="camera_id" required>
                                <div class="form-text">Unique identifier for the camera</div>
                            </div>
                            <div class="mb-3">
                                <label for="name" class="form-label">Camera Name</label>
                                <input type="text" class="form-control" id="name" name="name" required>
                            </div>
                            <div class="mb-3">
                                <label for="location" class="form-label">Location</label>
                                <input type="text" class="form-control" id="location" name="location" required>
                            </div>
                        </div>
                        <div class="col-md-6">
                            <div class="mb-3">
                                <label for="rtsp_url" class="form-label">RTSP URL</label>
                                <input type="text" class="form-control" id="rtsp_url" name="rtsp_url" required>
                                <div class="form-text">Example: rtsp://username:password@ip_address:port/stream</div>
                            </div>
                            <div class="mb-3">
                                <label for="status" class="form-label">Status</label>
                                <select class="form-select" id="status" name="status">
                                    <option value="active">Active</option>
                                    <option value="inactive">Inactive</option>
                                    <option value="maintenance">Maintenance</option>
                                </select>
                            </div>
                            <div class="mb-3">
                                <button type="submit" class="btn btn-primary">
                                    <i class="fas fa-save me-2"></i> Save Configuration
                                </button>
                                <button type="button" class="btn btn-outline-secondary test-btn" id="test-connection">
                                    <i class="fas fa-plug me-2"></i> Test Connection
                                </button>
                            </div>
                        </div>
                    </div>
                </form>
            </div>
            
            <div class="config-card">
                <h4 class="mb-4">Configured Cameras</h4>
                {% if cameras %}
                    {% for camera_id, camera in cameras.items() %}
                    <div class="camera-card">
                        <div class="camera-header">
                            <h5 class="mb-0">{{ camera.name }}</h5>
                            <span>
                                <span class="camera-status {% if camera.status == 'active' %}status-active{% elif camera.status == 'inactive' %}status-inactive{% else %}status-maintenance{% endif %}"></span>
                                {{ camera.status|title }}
                            </span>
                        </div>
                        <p class="mb-2"><strong>Location:</strong> {{ camera.location }}</p>
                        <p class="mb-2 text-truncate"><strong>RTSP URL:</strong> {{ camera.rtsp_url }}</p>
                        <div class="mt-2">
                            <button class="btn btn-sm btn-outline-primary edit-camera" data-camera-id="{{ camera_id }}" data-camera-name="{{ camera.name }}" data-camera-location="{{ camera.location }}" data-camera-rtsp="{{ camera.rtsp_url }}" data-camera-status="{{ camera.status }}">
                                <i class="fas fa-edit me-1"></i> Edit
                            </button>
                        </div>
                    </div>
                    {% endfor %}
                {% else %}
                    <div class="alert alert-info">
                        <i class="fas fa-info-circle me-2"></i> No cameras configured yet.
                    </div>
                {% endif %}
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        // Test RTSP connection
        document.getElementById('test-connection').addEventListener('click', function() {
            const rtspUrl = document.getElementById('rtsp_url').value;
            if (!rtspUrl) {
                alert('Please enter an RTSP URL first');
                return;
            }
            
            this.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i> Testing...';
            this.disabled = true;
            
            // In a real implementation, you would make an API call to test the connection
            // For now, we'll simulate a test with a timeout
            setTimeout(() => {
                alert('Connection test completed. This is a simulation - actual implementation would connect to the RTSP stream.');
                this.innerHTML = '<i class="fas fa-plug me-2"></i> Test Connection';
                this.disabled = false;
            }, 2000);
        });
        
        // Edit camera functionality
        document.querySelectorAll('.edit-camera').forEach(button => {
            button.addEventListener('click', function() {
                const cameraId = this.getAttribute('data-camera-id');
                const cameraName = this.getAttribute('data-camera-name');
                const cameraLocation = this.getAttribute('data-camera-location');
                const cameraRtsp = this.getAttribute('data-camera-rtsp');
                const cameraStatus = this.getAttribute('data-camera-status');
                
                document.getElementById('camera_id').value = cameraId;
                document.getElementById('name').value = cameraName;
                document.getElementById('location').value = cameraLocation;
                document.getElementById('rtsp_url').value = cameraRtsp;
                document.getElementById('status').value = cameraStatus;
                
                // Scroll to the form
                document.querySelector('.config-card').scrollIntoView({ behavior: 'smooth' });
            });
        });
    </script>
</body>
</html>""")

print("✅ All templates created successfully!")




# --- Post-startup model loading (injected) ---
try:
    try:
        ok = False
        try:
            ok = load_model()
        except Exception:
            ok = False
        if ok:
            MODEL_TRAINED = True
            logger.info("Saved ML model loaded on startup (model trained).")
        else:
            MODEL_TRAINED = False
            logger.info("No usable saved ML model found on startup.")
    except NameError:
        logger.warning("load_model() is not defined at this point; skipping model load.")
except Exception as e:
    logger.error(f"Post-startup model load error: {e}")

# --- end post-startup loader ---

if __name__ == '__main__':
    # Initialize system
    print("🚀 Starting Skyhighes Technologies Premium Attendance System...")
    
    # Initialize database

    # Load AI model
    if AI_AVAILABLE:
        load_model()
    
    # Start Flask apps
    print("✅ System initialized successfully!")
    print("🌐 Starting web server...")
    
    socketio.run(app, host='0.0.0.0', port=8080, debug=True)