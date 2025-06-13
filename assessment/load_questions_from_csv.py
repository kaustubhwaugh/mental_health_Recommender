import csv
from .models import Question
from django.conf import settings
import os

def load_questions_from_csv():
    csv_path = os.path.join(settings.BASE_DIR, "QuestionBank.csv")
    with open(csv_path, newline='', encoding='utf-8-sig') as csvfile:
        reader = csv.DictReader(csvfile)
        count = 0
        for row in reader:
            qid = row['Question_Id'].strip()
            text = row['Question_Text'].strip()
            Question.objects.create(question_id=qid, text=text)
            count += 1
    print(f"✅ Loaded {count} questions from CSV")
