from .models import Question

def populate_question_ids():
    all_questions = Question.objects.all().order_by('id')
    for i, q in enumerate(all_questions, start=1):
        q.question_id = f"Q{i}"
        q.save()
    print(f"✅ Updated {len(all_questions)} questions with Q1, Q2,...")
    