# mental_health_Recommender/assessment/views.py
import joblib
import os
import pandas as pd
from django.conf import settings
from django.shortcuts import render, redirect
from django.urls import reverse
from django.http import HttpResponseRedirect
from django.contrib.auth.decorators import login_required, user_passes_test
from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth import login, logout, authenticate
from django.contrib import messages
from .models import Question, Response, ContactMessage
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from .forms import NewUserForm, ContactForm
from .predictor import make_predictions
import random
from django.core.mail import send_mail

# Import LoginRequiredMixin for class-based views
from django.contrib.auth.mixins import LoginRequiredMixin


BASE_DIR = settings.BASE_DIR

# --- Constants for Rule-Based Severe Override ---
SEVERE_OVERRIDE_HIGHEST_SCORE_THRESHOLD_PERCENTAGE = 0.75
CRITICAL_QUESTION_QID = "Q9"
CRITICAL_QUESTION_HIGHEST_SCORE = 3


# Global definition of video resources for reuse
VIDEO_RESOURCES_DATA = [
    {"title": "4-7-8 Calm Breathing Exercise", "description": "A simple breathing technique to calm your nervous system and reduce stress.", "link": "https://www.youtube.com/embed/LiUnFJ8P4gM"},
    {"title": "Anxiety Calming Exercise", "description": "Quick and effective exercises to soothe anxiety and bring a sense of peace.", "link": "https://www.youtube.com/embed/5zhnLG3GW-8"},
    {"title": "Relaxing Yoga For Mental Health", "description": "Gentle yoga flows designed to relax your mind and body, promoting mental clarity.", "link": "https://www.youtube.com/embed/COp7BR_Dvps"},
    {"title": "5-Minute Meditation", "description": "A brief guided meditation to help you find a moment of stillness in your day.", "link": "https://www.youtube.com/embed/inpok4MKVLM"},
    {"title": "How to Cope with Anxiety", "description": "Practical strategies and insights to help you manage and understand anxiety.", "link": "https://www.youtube.com/embed/WWloIAQpMcQ"},
    {"title": "Healing Meditation Music", "description": "Soothing music designed to aid relaxation, meditation, and emotional healing.", "link": "https://www.youtube.com/embed/vPvIxwh9N2w"},
    {"title": "Art Exercise for Stress Relief", "description": "Engage in creative expression as a therapeutic tool to alleviate stress and tension.", "link": "https://www.youtube.com/embed/nA5dGCeZO5k"},
    {"title": "Stress and Worry Release Dance Workout", "description": "A dynamic and fun way to release pent-up stress and emotional tension through movement.", "link": "https://www.youtube.com/embed/QH0vKexLMkI"},
]


def index(request):
    return render(request, 'index.html')


def home(request):
    return render(request, 'home.html')


def about(request):
    return render(request, 'about.html')


def contact(request):
    if request.method == 'POST':
        form = ContactForm(request.POST)
        if form.is_valid():
            # Save the message to the database
            ContactMessage.objects.create(
                name=form.cleaned_data['name'],
                email=form.cleaned_data['email'],
                subject=form.cleaned_data['subject'],
                message=form.cleaned_data['message']
            )

            # ✅ Send email to solaceapp25@gmail.com
            send_mail(
                subject=f"[Solace Feedback] {form.cleaned_data['subject']}",
                message=form.cleaned_data['message'],
                from_email=form.cleaned_data['email'],
                recipient_list=["solaceapp25@gmail.com"],
                fail_silently=False,
            )

            messages.success(request, 'Your message has been sent successfully!')
            return redirect('/contact')
        else:
            messages.error(request, 'There was an error sending your message. Please try again.')
    else:
        form = ContactForm()

    return render(request, 'contact.html', {'form': form})




def service(request):
    return render(request, 'service.html')


def team_view(request):
    """Renders the team.html page."""
    return render(request, 'team.html')


def assessment(request):
    return render(request, 'assessment.html')


def results(request):
    total_score = request.GET.get('total_score')
    severity = request.GET.get('severity')
    recommendation = request.GET.get('recommendation')

    return render(request, 'results.html', {
        'total_score': total_score,
        'severity': severity,
        'recommendation': recommendation
    })


def add_question(request):
    """Admin panel for adding mental health questions"""
    if request.method == "POST":
        question_text = request.POST.get("question_text")
        if question_text:
            Question.objects.create(text=question_text)
            return redirect('add_question')  # Refresh the page after adding
    questions = Question.objects.all()
    return render(request, 'add_questions.html', {'questions': questions})


# Helper function to generate structured recommendations based on severity
def get_detailed_recommendations(severity_results, user_input_indicates_severe_override=False):
    final_recommendations_data = {
        "general_recommendations": [],
        "condition_recommendations": {
            "Stress": [],
            "Anxiety": [],
            "Depression": []
        },
        "critical_recommendations": [],
        "media_resources": [] # This will now be populated
    }

    general_base_recommendations = [
        {"title": "Practice Mindfulness", "description": "Engage in mindfulness exercises to stay present and reduce overthinking. Try guided meditations for beginners.", "link": "https://www.mindful.org/meditation/mindfulness-getting-started/", "type": "Self-Care"},
        {"title": "Maintain a Healthy Lifestyle", "description": "Ensure balanced diet, regular exercise (even light walks), and sufficient sleep (7-9 hours). These are foundations for mental well-being.", "link": "https://www.mentalhealth.gov/basics/healthy-lifestyle", "type": "Self-Care"},
        {"title": "Connect with Loved Ones", "description": "Spend quality time with supportive friends and family members. Share your feelings if you feel comfortable.", "link": "#", "type": "Social Support"},
        {"title": "Journaling", "description": "Write down your thoughts and feelings to process emotions, identify triggers, and gain clarity. Even 5-10 minutes daily can help.", "link": "https://www.healthline.com/health/mental-health/benefits-of-journaling", "type": "Self-Care"},
    ]
    final_recommendations_data["general_recommendations"].extend(general_base_recommendations)

    # Populate media resources here from the global constant
    final_recommendations_data["media_resources"].extend(VIDEO_RESOURCES_DATA)


    stress_recs = {
        "Mild": [{"title": "Basic Stress Management Techniques", "description": "Learn and apply relaxation techniques like deep breathing exercises, progressive muscle relaxation, or guided imagery to manage everyday stress.", "link": "https://www.apa.org/topics/stress/tips", "type": "Self-Care"}, {"title": "Time Management Skills", "description": "Improve your organization and planning to reduce feelings of overwhelm. Prioritize tasks and learn to say no when necessary.", "link": "https://www.mindtools.com/pages/article/newHTE_00.htm", "type": "Self-Help"}],
        "Moderate": [{"title": "Cognitive Behavioral Therapy (CBT) for Stress", "description": "Consider online resources or initial therapy sessions focused on CBT techniques to reframe negative thought patterns and develop coping strategies for stress.", "link": "https://www.talkspace.com/online-therapy/conditions/stress/", "type": "Professional Help"}, {"title": "Mind-Body Practices", "description": "Explore yoga, tai chi, or meditation classes. These practices integrate physical postures, breathing exercises, and meditation to calm the mind.", "link": "#", "type": "Self-Care"}],
        "Severe": [{"title": "Professional Stress Counseling", "description": "Seek immediate consultation with a mental health professional specializing in severe stress and burnout. Intensive therapy may be beneficial.", "link": "https://www.betterhelp.com/", "type": "Urgent Professional Help"}, {"title": "Stress Management Programs", "description": "Enroll in structured stress management programs offered by clinics or community centers that provide comprehensive strategies for severe stress.", "link": "#", "type": "Community Support"}]
    }

    anxiety_recs = {
        "Mild": [{"title": "Anxiety Self-Help Apps & Resources", "description": "Utilize reputable mobile applications and online guides focused on managing mild anxiety symptoms through techniques like thought diaries and relaxation exercises.", "link": "https://www.calm.com/", "type": "Self-Help"}, {"title": "Understand Anxiety Triggers", "description": "Start identifying situations, thoughts, or feelings that trigger your anxiety. Awareness is the first step towards managing it.", "link": "https://www.anxietycanada.com/articles/identifying-triggers/", "type": "Self-Help"}],
        "Moderate": [{"title": "Online Therapy for Anxiety (e.g., CBT)", "description": "Online platforms connect you with licensed therapists experienced in treating moderate anxiety using evidence-based approaches like CBT.", "link": "https://www.talkspace.com/online-therapy/conditions/anxiety/", "type": "Professional Help"}, {"title": "Support Groups for Anxiety", "description": "Joining a local or online support group for anxiety can provide a sense of community and shared coping strategies.", "link": "https://www.nami.org/Support-Education/Support-Groups", "type": "Community Support"}],
        "Severe": [{"title": "Immediate Psychiatric Consultation for Severe Anxiety", "description": "For severe anxiety, consider consulting a psychiatrist for medication management alongside therapy. They can assess severe symptoms like panic attacks.", "link": "#", "type": "Urgent Professional Help"}, {"title": "Crisis Intervention Services", "description": "If severe anxiety leads to overwhelming distress or panic attacks affecting daily life, seek crisis intervention services.", "link": "https://www.samhsa.gov/find-help/national-helpline", "type": "Emergency Support"}]
    }

    depression_recs = {
        "Mild": [{"title": "Mood-Boosting Activities", "description": "Engage in activities that you previously enjoyed or new hobbies that can uplift your mood, suchs as listening to music, light exercise, or creative arts.", "link": "#", "type": "Self-Care"}, {"title": "Structure Your Day", "description": "Create a routine for your day, including meal times, activities, and sleep, to provide a sense of stability and purpose.", "link": "#", "type": "Self-Help"}],
        "Moderate": [{"title": "Cognitive Behavioral Therapy (CBT) for Depression", "description": "CBT helps identify and change negative thinking patterns associated with moderate depression. Online or in-person therapy can be effective.", "link": "https://www.betterhelp.com/online-therapy/depression/", "type": "Professional Help"}, {"title": "Light Therapy (Seasonal Affective Disorder)", "description": "If your depression is seasonal, consider light therapy boxes, after consulting with a healthcare professional.", "link": "#", "type": "Self-Help (with guidance)"}],
        "Severe": [{"title": "Emergency Mental Health Evaluation for Severe Depression", "description": "For severe depression, especially with thoughts of self-harm, immediate evaluation by a psychiatrist or mental health emergency service is critical.", "link": "https://www.nimhans.ac.in/nimhans-integrated-mental-health-helpline/", "type": "Urgent Professional Help"}, {"title": "Medication Assessment", "description": "A psychiatrist can assess if antidepressant medication is appropriate and safe for managing severe depressive symptoms.", "link": "#", "type": "Professional Help"}]
    }

    critical_emergency_recs = [
        {"title": "National Crisis Hotline", "description": "If you are in acute distress or having thoughts of harming yourself or others, please call a national crisis hotline immediately.", "link": "https://www.samhsa.gov/find-help/national-helpline", "type": "Emergency Support"},
        {"title": "Local Emergency Services", "description": "In an immediate life-threatening crisis, call your local emergency number or go to the nearest emergency room.", "link": "#", "type": "Emergency Support"},
        {"title": "NIMHANS Integrated Mental Health Helpline (India)", "description": "For immediate mental health support within India, NIMHANS offers a dedicated helpline.", "link": "tel:080-46110007", "type": "Emergency Support"}
    ]

    has_severe_case_flag = user_input_indicates_severe_override
    if not has_severe_case_flag:
        for level in severity_results.values():
            if level == "Severe":
                has_severe_case_flag = True
                break

    display_severity_results = severity_results.copy()

    if has_severe_case_flag:
        for cond in display_severity_results:
            display_severity_results[cond] = "Severe"
        final_recommendations_data["critical_recommendations"].extend(critical_emergency_recs)
        final_recommendations_data["critical_recommendations"] = list(
            {v['title']: v for v in final_recommendations_data["critical_recommendations"]}.values())

    for condition, level in display_severity_results.items():
        if level == "Severe":
            if condition == "Stress" and "Severe" in stress_recs:
                final_recommendations_data["condition_recommendations"]["Stress"].extend(stress_recs["Severe"])
            elif condition == "Anxiety" and "Severe" in anxiety_recs:
                final_recommendations_data["condition_recommendations"]["Anxiety"].extend(anxiety_recs["Severe"])
            elif condition == "Depression" and "Severe" in depression_recs:
                final_recommendations_data["condition_recommendations"]["Depression"].extend(depression_recs["Severe"])

            if condition == "Stress":
                final_recommendations_data["condition_recommendations"]["Stress"].extend(stress_recs.get("Moderate", []))
                final_recommendations_data["condition_recommendations"]["Stress"].extend(stress_recs.get("Mild", []))
            if condition == "Anxiety":
                final_recommendations_data["condition_recommendations"]["Anxiety"].extend(anxiety_recs.get("Moderate", []))
                final_recommendations_data["condition_recommendations"]["Anxiety"].extend(anxiety_recs.get("Mild", []))
            if condition == "Depression":
                final_recommendations_data["condition_recommendations"]["Depression"].extend(depression_recs.get("Moderate", []))
                final_recommendations_data["condition_recommendations"]["Depression"].extend(
                    depression_recs.get("Mild", []))

        elif level == "Moderate":
            if condition == "Stress" and "Moderate" in stress_recs:
                final_recommendations_data["condition_recommendations"]["Stress"].extend(stress_recs["Moderate"])
            elif condition == "Anxiety" and "Moderate" in anxiety_recs:
                final_recommendations_data["condition_recommendations"]["Anxiety"].extend(anxiety_recs["Moderate"])
            elif condition == "Depression" and "Moderate" in depression_recs:
                final_recommendations_data["condition_recommendations"]["Depression"].extend(
                    depression_recs["Moderate"])

            if condition == "Stress":
                final_recommendations_data["condition_recommendations"]["Stress"].extend(stress_recs.get("Mild", []))
            if condition == "Anxiety":
                final_recommendations_data["condition_recommendations"]["Anxiety"].extend(anxiety_recs.get("Mild", []))
            if condition == "Depression":
                final_recommendations_data["condition_recommendations"]["Depression"].extend(
                    depression_recs.get("Mild", []))

        elif level == "Mild":
            if condition == "Stress" and "Mild" in stress_recs:
                final_recommendations_data["condition_recommendations"]["Stress"].extend(stress_recs["Mild"])
            elif condition == "Anxiety" and "Mild" in anxiety_recs:
                final_recommendations_data["condition_recommendations"]["Anxiety"].extend(anxiety_recs["Mild"])
            elif condition == "Depression" and "Mild" in depression_recs:
                final_recommendations_data["condition_recommendations"]["Depression"].extend(depression_recs["Mild"])

    for condition in final_recommendations_data["condition_recommendations"]:
        final_recommendations_data["condition_recommendations"][condition] = list(
            {v['title']: v for v in final_recommendations_data["condition_recommendations"][condition]}.values())

    return final_recommendations_data, has_severe_case_flag, display_severity_results


@login_required(login_url='/login1/')
def assessment_view(request):
    """Displays questions from DB and calculates results"""
    print(f"DEBUG: Entering assessment_view. Request Method: {request.method}")

    if 'question_ids' not in request.session:
        print("DEBUG: 'question_ids' not in session. Selecting new questions.")
        question_ids_all = list(Question.objects.values_list('id', flat=True))
        print(f"DEBUG: All question IDs from DB: {question_ids_all}")

        if not question_ids_all:
            messages.error(request, 'No questions found in the database. Please add questions first.')
            print("ERROR: No questions found in the database.")
            return render(request, 'assessment.html', {'questions': []}) # Render empty form or message

        selected_questions = random.sample(question_ids_all, min(20, len(question_ids_all)))
        request.session['question_ids'] = selected_questions
        print(f"DEBUG: Selected questions for session: {selected_questions}")
    else:
        selected_questions = request.session['question_ids']
        print(f"DEBUG: 'question_ids' found in session: {selected_questions}")


    questions_for_template = list(
        Question.objects.filter(id__in=selected_questions).values('id', 'question_id', 'text'))
    print(f"DEBUG: Questions fetched for template: {questions_for_template}")


    if request.method == 'POST':
        print("DEBUG: Processing POST request for assessment.")
        data = {}
        highest_score_count = 0
        total_answered_questions = 0
        user_input_indicates_severe_override = False

        for q in questions_for_template:
            ans_str = request.POST.get(q['question_id'])
            if ans_str is None:
                messages.error(request, 'Please answer all questions.')
                print(f"ERROR: Question {q['question_id']} not answered.")
                return render(request, 'assessment.html',
                              {'questions': questions_for_template, 'error': 'Please answer all questions.'})

            ans_int = int(ans_str)
            data[q['question_id']] = ans_int

            total_answered_questions += 1
            if ans_int == CRITICAL_QUESTION_HIGHEST_SCORE:
                highest_score_count += 1

            if q['question_id'] == CRITICAL_QUESTION_QID and ans_int == CRITICAL_QUESTION_HIGHEST_SCORE:
                user_input_indicates_severe_override = True
                print(f"DEBUG: Critical question {CRITICAL_QUESTION_QID} answered severely. Override active.")

        if total_answered_questions > 0 and \
                (highest_score_count / total_answered_questions) >= SEVERE_OVERRIDE_HIGHEST_SCORE_THRESHOLD_PERCENTAGE:
            user_input_indicates_severe_override = True
            print(
                f"DEBUG: High percentage of severe answers ({highest_score_count}/{total_answered_questions}). Override active.")

        question_csv_path = os.path.join(BASE_DIR, "QuestionBank.csv")
        all_question_ids_from_csv = pd.read_csv(question_csv_path)["Question_Id"].tolist()
        all_question_ids_from_csv = sorted(all_question_ids_from_csv, key=lambda x: int(x[1:]))

        feature_vector_for_model = [data.get(qid, -1) for qid in all_question_ids_from_csv]

        df_for_prediction = pd.DataFrame([feature_vector_for_model], columns=all_question_ids_from_csv)

        severity_results = make_predictions(df_for_prediction)

        print(f"DEBUG: ml_prediction_view - Predicted Severity Results: {severity_results}")

        structured_recommendations, has_severe_case_flag, display_severity_results = \
            get_detailed_recommendations(severity_results, user_input_indicates_severe_override)

        if 'question_ids' in request.session:
            del request.session['question_ids']
        print("DEBUG: Redirecting to results.html after assessment POST.")
        return render(request, 'results.html', {
            'severity_results': display_severity_results,
            'recommendations': structured_recommendations,
            'has_severe_case': has_severe_case_flag
        })

    print("DEBUG: Rendering assessment.html for GET request.")
    return render(request, 'assessment.html', {'questions': questions_for_template})


#####################################Chatbot############################################
from googletrans import Translator
from django.views.generic import TemplateView
from .chat import get_response, bot_name
import speech_recognition as sr
from django.contrib.auth.mixins import LoginRequiredMixin


class eng_text(LoginRequiredMixin, TemplateView):
    template_name = "customer/eng_text.html"
    login_url = '/login1/'

    def get(self, request):
        if 'chat_history_text' not in request.session:
            request.session['chat_history_text'] = []
        context = {
            "bot": "Hi, how can I assist you?",
            "chat_history": request.session['chat_history_text']
        }
        return render(request, self.template_name, context)

    def post(self, request):
        if request.method == 'POST':
            user_input = request.POST.get('input', '')
            bot_response = get_response(user_input)
            if bot_response is None:
                bot_response = "I'm sorry, I don't understand your question."

            chat_history = request.session.get('chat_history_text', [])
            chat_history.append({"user": user_input, "bot": bot_response})
            request.session['chat_history_text'] = chat_history

            context = {
                "bot": bot_response,
                "chat_history": chat_history
            }
            return render(request, self.template_name, context)


class eng_voice(LoginRequiredMixin, TemplateView):
    template_name = "customer/eng_voice.html"
    login_url = '/login1/'

    def get(self, request):
        context = {}
        if 'chat_history_voice' not in request.session:
            request.session['chat_history_voice'] = []
        context['chat_history'] = request.session['chat_history_voice']
        context['bot'] = "Hi, I'm ready to listen. Please speak in English."
        return render(request, self.template_name, context)

    def post(self, request):
        if request.method == 'POST':
            r = sr.Recognizer()
            print("Please talk (English Voice)")
            user_speech_text = ""
            try:
                with sr.Microphone() as source:
                    audio_data = r.record(source, duration=5)
                    print("Recognizing (English Voice)...")
                    user_speech_text = r.recognize_google(audio_data, language="en-US")
                    print("Recognized Speech (English Voice):" + user_speech_text)
            except sr.UnknownValueError:
                user_speech_text = "Sorry, I could not understand the audio."
                print("Could not understand audio (English Voice)")
            except sr.RequestError as e:
                user_speech_text = f"Could not request results from Google Speech Recognition service; {e}"
                print(f"Could could not request results from Google Speech Recognition service; {e}")

            bot_response = get_response(user_speech_text)
            if bot_response is None:
                bot_response = "I'm sorry, I don't have a response for that."

            chat_history_voice = request.session.get('chat_history_voice', [])
            chat_history_voice.append({"user": user_speech_text, "bot": bot_response})
            request.session['chat_history_voice'] = chat_history_voice

            context = {"user": user_speech_text, "bot": bot_response, "chat_history": chat_history_voice}
            return render(request, self.template_name, context)


def resources_view(request): # New view for resources page
    """Renders the resources.html page with dynamic video data."""
    # This view now uses the globally defined VIDEO_RESOURCES_DATA
    return render(request, 'resources.html', {'video_resources': VIDEO_RESOURCES_DATA})


@login_required(login_url='login1')
def customer_dashboard_view(request):
    if request.method == 'POST':
        user_input = request.POST.get('input', '')
        context1 = {"user": user_input, "bot": get_response(user_input)}
    return render(request, 'customer/customer_dashboard.html')


def register(request):
    if request.method == "POST":
        form = NewUserForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            messages.success(request, "Registration successful.")
            return redirect('login1')
        messages.error(request, "Unsuccessful registration. Invalid information.")
    form = NewUserForm()
    return render(request=request, template_name="register.html", context={"register_form": form})


def login1(request):
    if request.method == "POST":
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            user = authenticate(username=username, password=password)
            if user is not None:
                login(request, user)
                messages.info(request, f"You are now logged in as {username}.")
                return redirect('home')
            else:
                messages.error(request, "Invalid username or password.")
        else:
            messages.error(request, "Invalid username or password.")

    form = AuthenticationForm()
    return render(request=request, template_name="login.html", context={"login_form": form})


def logout_request(request):
    logout(request)
    messages.info(request, "You have successfully logged out.")
    return redirect("login1")


def ml_prediction_view(request):
    if request.method == "POST":
        question_csv_path = os.path.join(BASE_DIR, "QuestionBank.csv")
        all_question_ids_from_csv = pd.read_csv(question_csv_path)["Question_Id"].tolist()
        all_question_ids_from_csv = sorted(all_question_ids_from_csv, key=lambda x: int(x[1:]))

        data = request.POST
        highest_score_count = 0
        total_answered_questions = 0
        user_input_indicates_severe_override = False

        feature_vector = []
        for qid in all_question_ids_from_csv:
            ans_str = data.get(qid)
            if ans_str is not None:
                ans_int = int(ans_str)
                feature_vector.append(ans_int)

                total_answered_questions += 1
                if ans_int == CRITICAL_QUESTION_HIGHEST_SCORE:
                    highest_score_count += 1

                if qid == CRITICAL_QUESTION_QID and ans_int == CRITICAL_QUESTION_HIGHEST_SCORE:
                    user_input_indicates_severe_override = True
                    print(
                        f"DEBUG: Critical question {CRITICAL_QUESTION_QID} answered severely. Override active in ml_prediction_view.")
            else:
                feature_vector.append(-1)

        if total_answered_questions > 0 and \
                (highest_score_count / total_answered_questions) >= SEVERE_OVERRIDE_HIGHEST_SCORE_THRESHOLD_PERCENTAGE:
            user_input_indicates_severe_override = True
            print(
                f"DEBUG: High percentage of severe answers ({highest_score_count}/{total_answered_questions}). Override active in ml_prediction_view.")

        X = pd.DataFrame([feature_vector], columns=all_question_ids_from_csv)

        severity_results = make_predictions(X)

        print(f"DEBUG: ml_prediction_view - Predicted Severity Results: {severity_results}")

        structured_recommendations, has_severe_case_flag, display_severity_results = \
            get_detailed_recommendations(severity_results, user_input_indicates_severe_override)

        return render(request, "results.html", {
            "severity_results": display_severity_results,
            "recommendations": structured_recommendations,
            'has_severe_case': has_severe_case_flag
        })

    return render(request, "results.html", {'questions': questions_for_template})
