
from django.db import models

class Question(models.Model):
    question_id = models.CharField(max_length=10, unique=True)
    text = models.TextField()

    def __str__(self):
        return f"{self.question_id}: {self.text}"


class Response(models.Model):
    question = models.ForeignKey(Question, on_delete=models.CASCADE)
    response = models.IntegerField(choices=[(0, 'Never'), (1, 'Sometimes'), (2, 'Often'), (3, 'Always')])

    def __str__(self):
        return f"{self.question} - {self.get_response_display()}"
from django.db import models
class ContactMessage(models.Model):
    name = models.CharField(max_length=255)
    email = models.EmailField()
    subject = models.CharField(max_length=255)
    message = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.subject



