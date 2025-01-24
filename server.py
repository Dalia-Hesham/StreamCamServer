import ssl
import cv2
import base64
import numpy as np
import sqlite3
from flask import Flask, render_template
from flask_socketio import SocketIO
from sklearn.metrics.pairwise import cosine_similarity

# Initialize Flask and Flask-SocketIO
app = Flask(__name__)
socketio = SocketIO(app, async_mode="threading")

# Database path
DATABASE_PATH = "face_embeddings.db"

# Initialize database
def initialize_database():
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS faces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            embedding BLOB NOT NULL
        )
    """)
    conn.commit()
    conn.close()
    print("Database initialized.")

# Save embedding to database
def save_embedding_to_database(name, embedding):
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO faces (name, embedding) VALUES (?, ?)", (name, embedding.tobytes()))
        conn.commit()
        print(f"Saved embedding for {name} to database.")
    except Exception as e:
        print(f"Error saving embedding to database: {e}")
    finally:
        conn.close()

# Load embeddings from database
def load_embeddings_from_database():
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT name, embedding FROM faces")
        data = cursor.fetchall()
        return [(row[0], np.frombuffer(row[1], dtype=np.float32)) for row in data]
    except Exception as e:
        print(f"Error loading embeddings from database: {e}")
        return []
    finally:
        conn.close()

# Feature extractor using HOG (placeholder, replace with advanced model if needed)
def extract_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hog = cv2.HOGDescriptor()
    return hog.compute(gray).flatten()

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('video_frame')
def handle_video_frame(data):
    print("Received video frame.")

    # Decode base64 image
    try:
        img_data = base64.b64decode(data.split(',')[1])
        np_img = np.frombuffer(img_data, dtype=np.uint8)
        frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"Error decoding image: {e}")
        return

    # Verify the frame is valid
    if frame is None:
        print("Error: Frame is None after decoding.")
        return

    print("Extracting features...")
    embedding = extract_features(frame)

    # Check if embedding is unique
    embeddings = load_embeddings_from_database()
    is_unique = all(
        cosine_similarity([embedding], [e])[0][0] < 0.8 for _, e in embeddings
    )

    if is_unique:
        name = f"Person_{len(embeddings) + 1}"  # Assign a unique name
        save_embedding_to_database(name, embedding)
        print(f"Saved unique face embedding for {name}.")
    else:
        print("Face already exists in the database.")

if __name__ == '__main__':
    # Initialize database
    initialize_database()

    # Run the Flask server
    socketio.run(app, host='0.0.0.0', port=5000)