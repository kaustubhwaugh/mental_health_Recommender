from django.contrib import admin
from .models import Question

class QuestionAdmin(admin.ModelAdmin):
    list_display = ('id', 'question_id', 'text')
    search_fields = ('question_id', 'text')

admin.site.register(Question, QuestionAdmin)


