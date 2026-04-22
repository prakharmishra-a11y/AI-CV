"""One-time script: load known_faces/ photos, generate embeddings, store in DB."""

import os
import sys
import cv2

from config import KNOWN_FACES_DIR, MIN_DET_SCORE
import database
from recognizer import FaceRecognizer


def enroll_faces(faces_dir=KNOWN_FACES_DIR):
    """Scan known_faces/ folder and enroll each person.

    Expected structure:
      known_faces/
        rahul.jpg          -> person "rahul", 1 photo
        priya_1.jpg        -> person "priya", photo 1
        priya_2.jpg        -> person "priya", photo 2

    Naming: <person_name>.jpg or <person_name>_<N>.jpg
    Supported formats: .jpg, .jpeg, .png
    """
    database.create_tables()
    recognizer = FaceRecognizer()

    if not os.path.isdir(faces_dir):
        print(f"Error: Directory '{faces_dir}' not found.")
        sys.exit(1)

    valid_extensions = (".jpg", ".jpeg", ".png")
    files = sorted([
        f for f in os.listdir(faces_dir)
        if f.lower().endswith(valid_extensions)
    ])

    if not files:
        print(f"No image files found in '{faces_dir}'.")
        return

    enrolled = 0
    for filename in files:
        filepath = os.path.join(faces_dir, filename)
        # Extract person name: "rahul.jpg" -> "rahul", "priya_1.jpg" -> "priya"
        name_part = os.path.splitext(filename)[0]
        # Remove trailing _N suffix if present
        parts = name_part.rsplit("_", 1)
        if len(parts) == 2 and parts[1].isdigit():
            person_name = parts[0]
        else:
            person_name = name_part

        print(f"\nProcessing: {filename} -> person: {person_name}")

        img = cv2.imread(filepath)
        if img is None:
            print(f"  WARNING: Could not read image '{filepath}', skipping.")
            continue

        embedding, det_score = recognizer.get_embedding(img)
        if embedding is None:
            print(f"  WARNING: No face detected in '{filename}', skipping.")
            continue

        if det_score < MIN_DET_SCORE:
            print(f"  WARNING: Low detection score ({det_score:.3f}) for '{filename}'. "
                  "This may be a photo of a print or low-quality image.")

        person_id = database.get_or_create_person(person_name)
        database.insert_embedding(person_id, embedding, filepath)
        enrolled += 1
        print(f"  Enrolled: {person_name} (id={person_id}), "
              f"det_score={det_score:.3f}, embedding dim={len(embedding)}")

    print(f"\nDone. Enrolled {enrolled} face(s) from {len(files)} file(s).")


if __name__ == "__main__":
    enroll_faces()
