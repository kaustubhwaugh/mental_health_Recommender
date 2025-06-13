# mental_health_Recommender/mental_health/urls.py
from django.contrib import admin
from django.urls import path, include
from assessment import views  # Import views from your assessment app

urlpatterns = [
    path('admin/', admin.site.urls),
    # Directly map the root URL to the index view from the assessment app
    # This ensures '/' is handled and prevents recursion from assessment.urls's '' path
    path('', views.index, name='index'),

    # Include all other URLs from assessment.urls.
    # Since assessment.urls no longer has its own path('',...), this will correctly
    # handle paths like /predict/, /about/, etc., without any prefixes.
    path('', include('assessment.urls')),
]