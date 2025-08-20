import os
import json
import sys
import glob
import numpy as np
from flask import Flask, request, jsonify, send_from_directory, render_template
from werkzeug.utils import secure_filename
import librosa
import torch
from torch import nn, optim
from torchopenl3 import get_openl3_embedding  # Требует установки: pip install torchopenl3
import logging
from pathlib import Path

app = Flask(__name__)

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", 'audio_uploads')
EMBEDDINGS_DB = os.getenv("EMBEDDINGS_DB", 'embeddings_db.json')
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac', 'm4a'}

# Ограничение размера загружаемого файла (16 МБ)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = None

# =======================
# Вспомогательные функции
# =======================

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def compute_embedding(file_path):
    """Вычисление нейросетевого embedding с помощью OpenL3 на PyTorch."""
    try:
        y, sr = librosa.load(file_path, sr=None, mono=True)
        emb = get_openl3_embedding(y, sr, content_type="env", input_repr="mel128", embedding_size=512)
        return emb.tolist()  # Преобразование в список для JSON
    except Exception as e:
        raise Exception(f"Ошибка при вычислении эмбеддинга для {file_path}: {str(e)}")

def load_embeddings_db():
    try:
        if os.path.exists(EMBEDDINGS_DB):
            with open(EMBEDDINGS_DB, 'r') as f:
                return json.load(f)
        return {}
    except json.JSONDecodeError as e:
        raise Exception(f"Ошибка при чтении базы эмбеддингов: {str(e)}")
    except Exception as e:
        raise Exception(f"Ошибка при загрузке базы эмбеддингов: {str(e)}")

def save_embeddings_db(db):
    try:
        with open(EMBEDDINGS_DB, 'w') as f:
            json.dump(db, f, indent=2)
    except Exception as e:
        raise Exception(f"Ошибка при сохранении базы эмбеддингов: {str(e)}")

def find_similar_sounds(new_embedding, db, top_n=5):
    new_embedding = np.array(new_embedding)
    similarities = []
    for key, data in db.items():
        emb = np.array(data["embedding"])
        category = data.get("category", "unknown")  # Категория по умолчанию
        sim = np.dot(new_embedding, emb) / (np.linalg.norm(new_embedding) * np.linalg.norm(emb))  # Косинусное сходство
        similarities.append((key, sim, category))
    similarities.sort(key=lambda x: x[1], reverse=True)
    return [{"file": key, "similarity": sim, "category": category} for key, sim, category in similarities[:top_n]]

def train_model(db):
    global model
    categories = {'normal': 0, 'broken': 1, 'noise': 2}
    X = []
    y = []
    for key, data in db.items():
        cat = data.get("category", "unknown")
        if cat in categories:
            X.append(torch.tensor(data["embedding"], dtype=torch.float32))
            y.append(categories[cat])
    if len(X) < 3:  # Минимальное количество образцов для обучения
        logger.info("Not enough data to train model")
        return
    X = torch.stack(X)
    y = torch.tensor(y)
    model = nn.Linear(512, 3)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(50):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
    logger.info("Model trained")

def determine_status(similar_sounds, threshold=0.8):
    """Определение статуса на основе похожих звуков."""
    normal_count = sum(1 for s in similar_sounds if s["category"] == "normal" and s["similarity"] >= threshold)
    broken_count = sum(1 for s in similar_sounds if s["category"] == "broken" and s["similarity"] >= threshold)
    noise_count = sum(1 for s in similar_sounds if s["category"] == "noise" and s["similarity"] >= threshold)
    
    if normal_count > broken_count and normal_count > noise_count:
        return "Оборудование работает нормально"
    elif broken_count > normal_count and broken_count > noise_count:
        return "Обнаружена неполадка в оборудовании"
    elif noise_count > normal_count and noise_count > broken_count:
        return "Запись содержит посторонние шумы"
    else:
        return "Не удалось определить статус (низкое сходство или смешанные категории)"

def populate_embeddings_db(audio_folder, category_map=None):
    """Заполнение базы эмбеддингов из датасета с учетом структуры папок."""
    if category_map is None:
        category_map = {}
    
    db = load_embeddings_db()  # Загружаем существующую базу
    audio_folder = Path(audio_folder)  # Используем Path для кроссплатформенности
    
    if not audio_folder.exists():
        logger.error(f"Папка {audio_folder} не существует")
        raise FileNotFoundError(f"Папка {audio_folder} не найдена")

    patterns = [str(audio_folder / "**" / f"*.{ext}") for ext in ALLOWED_EXTENSIONS]
    for pattern in patterns:
        for audio_file in glob.glob(pattern, recursive=True):
            audio_path = Path(audio_file)
            filename = audio_path.name
            # Определяем категорию по имени подпапки
            category = audio_path.parent.name if audio_path.parent.name in ["normal", "broken", "noise"] else "unknown"
            
            # Если есть category_map, используем его в приоритете
            category = category_map.get(filename, category)
            
            # Простая классификация на основе имени файла, если категория не определена
            if category == "unknown":
                if "normal" in filename.lower():
                    category = "normal"
                elif "anomal" in filename.lower() or "broken" in filename.lower():
                    category = "broken"
                elif "noise" in filename.lower() or "rain" in filename.lower() or "gravel" in filename.lower():
                    category = "noise"
            
            # Вычисляем эмбеддинг
            emb = compute_embedding(audio_file)
            if emb is not None:
                db[filename] = {"embedding": emb, "category": category}
                logger.info(f"Добавлен эмбеддинг для {filename} (категория: {category})")
    
    try:
        with open(EMBEDDINGS_DB, 'w') as f:
            json.dump(db, f, indent=2)
        logger.info(f"База эмбеддингов сохранена в {EMBEDDINGS_DB}")
    except Exception as e:
        logger.error(f"Ошибка при сохранении базы: {str(e)}")
        raise

# =======================
# API
# =======================

@app.route('/', methods=['GET'])
def index():
    """Простой веб-интерфейс для загрузки файлов."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        if not allowed_file(file.filename):
            return jsonify({"error": "Invalid file type"}), 400

        filename = secure_filename(file.filename)
        input_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(input_path)

        # Вычисление эмбеддинга для цельного файла
        emb = compute_embedding(input_path)

        db = load_embeddings_db()

        global model
        if model is None:
            train_model(db)

        key = filename
        db[key] = {"embedding": emb, "category": "unknown"}

        # Сравнение с предыдущими
        sims = find_similar_sounds(emb, db)

        # Определение статуса
        if model is not None:
            new_emb = torch.tensor(emb, dtype=torch.float32)
            with torch.no_grad():
                output = model(new_emb.unsqueeze(0))
                pred = torch.argmax(output, dim=1).item()
            categories = ['normal', 'broken', 'noise']
            cat = categories[pred]
            if cat == 'normal':
                status = "Оборудование работает нормально"
            elif cat == 'broken':
                status = "Обнаружена неполадка в оборудовании"
            elif cat == 'noise':
                status = "Запись содержит посторонние шумы"
        else:
            status = determine_status(sims, threshold=0.8)

        save_embeddings_db(db)

        return jsonify({
            "message": "File uploaded, embedding calculated and compared.",
            "status": status,
            "similar_sounds": sims
        }), 200

    except Exception as e:
        logger.error(f"Server error: {str(e)}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/download/<path:filename>', methods=['GET'])
def download_file(filename):
    try:
        safe_path = os.path.abspath(os.path.join(UPLOAD_FOLDER, filename))
        if not safe_path.startswith(os.path.abspath(UPLOAD_FOLDER)):
            return jsonify({"error": "Invalid file path"}), 400
        return send_from_directory(UPLOAD_FOLDER, filename, as_attachment=True)
    except FileNotFoundError:
        return jsonify({"error": f"File {filename} not found"}), 404
    except Exception as e:
        logger.error(f"Server error: {str(e)}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500

# =======================
# Запуск
# =======================

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "populate":
        audio_folder = sys.argv[2] if len(sys.argv) > 2 else "audio_dataset"
        category_map = {
            "fan_id00_normal_00000000.wav": "normal",
            "fan_id00_abnormal_00000000.wav": "broken",
            "rain_noise.wav": "noise",
            "gravel_driving.wav": "noise",
            "engine_normal.wav": "normal"
        }
        populate_embeddings_db(audio_folder, category_map)
        db = load_embeddings_db()
        train_model(db)
    else:
        # Для разработки используйте debug=True, для продакшена - debug=False и запуск через gunicorn
        # Пример запуска в продакшен: gunicorn -w 4 -b 0.0.0.0:5000 MVPEquipment6:app
        app.run(debug=False, host='0.0.0.0', port=int(os.getenv('PORT', 5000)))