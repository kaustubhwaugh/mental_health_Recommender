# mental_health_Recommender/assessment/urls.py
from django.urls import path
# Import all views explicitly from the views module
from .views import (
    ml_prediction_view, assessment_view, add_question, results,
    logout_request, about, contact, service, login1, register, assessment,
    home, customer_dashboard_view, eng_text, eng_voice, team_view,
    resources_view # Added resources_view import
)

urlpatterns = [
    path("predict/", ml_prediction_view, name="ml_prediction_view"),
    path('assessment_view/', assessment_view, name='assessment_view'),
    path('add_question/', add_question, name='add_question'),
    path('assessment/results/', results, name='results'),
    path("logout_request/", logout_request, name='logout_request'),
    path('about/', about, name='about'),
    path('contact/', contact, name='contact'),
    path('service/', service, name='service'),
    path("login1/", login1, name='login1'),
    path("register/", register, name="register"),
    path("assessment", assessment, name="assessment"),
    path("home/", home, name="home"),

    path('customer_dashboard_view/', customer_dashboard_view, name='customer_dashboard_view'),

    path("eng/text/", eng_text.as_view(), name="eng_text"),
    path("eng/voice/", eng_voice.as_view(), name="eng_voice"),

    path('team/', team_view, name='team'),
    path('resources/', resources_view, name='resources'), # New URL pattern for resources_view
]
