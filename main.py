import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import difflib
import streamlit as st
import streamlit_authenticator as stauth
import json
import numpy as np
import spacy # Import spacy

# --- File Paths ---
USER_DATA_FILE = 'users.json'
DATASET_FILE = 'dataset.csv'
DESC_DATA_FILE = 'symptom_Description.csv'
PRECAUTION_DATA_FILE = 'symptom_precaution.csv'
SEVERITY_DATA_FILE = 'Symptom-severity.csv'

# --- TRANSLATIONS DICTIONARY ---
# Add ALL static text from your UI here for each language.
# If a key is missing for a language, it will fall back to English.
TRANSLATIONS = {
    "English": {
        # Authentication
        "Incorrect username or password": "Incorrect username or password",
        "Please log in or register to access the application.": "Please log in or register to access the application.",
        "Login": "Login",
        "Register": "Register",
        "Login to your account": "Login to your account",
        "Create a new account": "Create a new account",
        "User registered successfully! Please go to the 'Login' tab to sign in.": "User registered successfully! Please go to the 'Login' tab to sign in.",
        "An error occurred during registration: ": "An error occurred during registration: ",
        "Logout": "Logout",
        "Welcome ": "Welcome ",

        # General UI
        "Disease Prediction System": "Disease Prediction System",
        "Please provide your information and symptoms to get a potential disease prediction and nearby hospital details.": "Please provide your information and symptoms to get a potential disease prediction and nearby hospital information.",
        "Disclaimer: This system is for informational purposes only and not a substitute for professional medical advice. Always consult a qualified healthcare professional for any health concerns.": "Disclaimer: This system is for informational purposes only and not a substitute for professional medical advice. Always consult a qualified healthcare professional for any health concerns.",
        "Select language:": "Select language:",

        # Input Form - Your Information
        "Your Personal Information": "Your Personal Information",
        "What is your full name?": "What is your full name?",
        "Please enter your full name as it appears on your records.": "Please enter your full name as it appears on your records.",
        "What is your age in years?": "What is your age in years?",
        "Please enter your age. This helps us refine the prediction.": "Please enter your age. This helps us refine the prediction.",
        "What is your gender?": "What is your gender?",
        "Please select your biological gender.": "Please select your biological gender.",
        "How many days have you been experiencing these symptoms?": "How many days have you been experiencing these symptoms?",
        "This helps understand the duration of your current condition.": "This helps understand the duration of your current condition.",

        # Input Form - Symptoms & Conditions
        "Your Symptoms and Medical History": "Your Symptoms and Medical History",
        "Please list your current symptoms (comma-separated):": "Please list your current symptoms (comma-separated):",
        "e.g., headache, nausea, fever, cough": "e.g., headache, nausea, fever, cough",
        "Describe all symptoms you are currently experiencing. Use common and clear terms, separated by commas.": "Describe all symptoms you are currently experiencing. Use common and clear terms, separated by commas.",
        "Do you have any existing chronic medical conditions?": "Do you have any existing chronic medical conditions?",
        "Toggle this if you have a long-term disease like diabetes or hypertension.": "Toggle this if you have a long-term disease like diabetes or hypertension.",
        "Please specify the name of your chronic condition:": "Please specify the name of your chronic condition:",
        "e.g., diabetes, hypertension, asthma": "e.g., diabetes, hypertension, asthma",
        "Providing the name helps us understand your full medical context.": "Providing the name helps us understand your full medical context.",

        # Input Form - Location
        "Locate Nearby Hospitals": "Locate Nearby Hospitals",
        "In which city or area are you located?": "In which city or area are you located?",
        "e.g., Bengaluru, New Delhi, Mumbai": "e.g., Bengaluru, New Delhi, Mumbai",
        "This helps us find hospitals close to your current location.": "This helps us find hospitals close to your current location.",

        # Action Button
        "Get My Health Prediction": "Get My Health Prediction",

        # Prediction Results
        "No valid symptoms entered. Please enter at least one recognizable symptom.": "No valid symptoms entered. Please enter at least one recognizable symptom.",
        "interpreted as": "interpreted as",
        "Your Health Prediction Results": "Your Health Prediction Results",
        "Predicted Disease:": "Predicted Disease:",
        "Description": "Description",
        "No description available.": "No description available.",
        "Severity Level": "Severity Level",
        "Symptom Score:": "Symptom Score:",
        "Urgent Medical Attention Advised!": "Urgent Medical Attention Advised!",
        "Your symptoms indicate a SEVERE condition. Please consult a doctor immediately for urgent care.": "Your symptoms indicate a SEVERE condition. Please consult a doctor immediately for urgent care.",
        "Medical Consultation Recommended!": "Medical Consultation Recommended!",
        "Your symptoms are MODERATE. It's advisable to monitor your condition and consult a doctor if they persist or worsen.": "Your symptoms are MODERATE. It's advisable to monitor your condition and consult a doctor if they persist or worsen.",
        "Symptoms are MILD.": "Symptoms are MILD.",
        "Your symptoms appear to be MILD. Please follow general precautions and stay vigilant. Consult a doctor if symptoms change or worsen.": "Your symptoms appear to be MILD. Please follow general precautions and stay vigilant. Consult a doctor if symptoms change or worsen.",
        "Precautions and Recommendations:": "Precautions and Recommendations:",
        "No specific precautions listed for this condition.": "No specific precautions listed for this condition.",
        "No specific precautions found for your chronic condition: ": "No specific precautions found for your chronic condition: ",
        "Unrecognized symptoms (could not match to our database): ": "Unrecognized symptoms (could not match to our database): ",
        "These symptoms were not used in the prediction as they could not be recognized.": "These symptoms were not used in the prediction as they could not be recognized.",
        "Please double-check the spelling of unrecognized symptoms or try more common terms.": "Please double-check the spelling of unrecognized symptoms or try more common terms.",
        "Nearby Hospitals in ": "Nearby Hospitals in ",
        "Click here to view hospitals near ": "Click here to view hospitals near ",
        "on Google Maps": "on Google Maps",
        "Please enter your city/area to receive a useful link for nearby hospitals.": "Please enter your city/area to receive a useful link for nearby hospitals."
    },
    "Hindi": {
        # Authentication
        "Incorrect username or password": "गलत उपयोगकर्ता नाम या पासवर्ड",
        "Please log in or register to access the application.": "एप्लिकेशन तक पहुंचने के लिए कृपया लॉगिन या रजिस्टर करें।",
        "Login": "लॉग इन करें",
        "Register": "रजिस्टर करें",
        "Login to your account": "अपने खाते में लॉग इन करें",
        "Create a new account": "नया खाता बनाएं",
        "User registered successfully! Please go to the 'Login' tab to sign in.": "उपयोगकर्ता सफलतापूर्वक पंजीकृत हो गया! कृपया साइन इन करने के लिए 'लॉगिन' टैब पर जाएं।",
        "An error occurred during registration: ": "पंजीकरण के दौरान एक त्रुटि हुई: ",
        "Logout": "लॉग आउट",
        "Welcome ": "आपका स्वागत है ",

        # General UI
        "Disease Prediction System": "रोग भविष्यवाणी प्रणाली",
        "Please provide your information and symptoms to get a potential disease prediction and nearby hospital details.": "संभावित रोग भविष्यवाणी और पास के अस्पताल के विवरण प्राप्त करने के लिए कृपया अपनी जानकारी और लक्षण प्रदान करें।",
        "Disclaimer: This system is for informational purposes only and not a substitute for professional medical advice. Always consult a qualified healthcare professional for any health concerns.": "अस्वीकरण: यह प्रणाली केवल सूचनात्मक उद्देश्यों के लिए है और पेशेवर चिकित्सा सलाह का विकल्प नहीं है। किसी भी स्वास्थ्य संबंधी चिंता के लिए हमेशा योग्य स्वास्थ्य सेवा पेशेवर से परामर्श करें।",
        "Select language:": "भाषा चुनें:",

        # Input Form - Your Information
        "Your Personal Information": "आपकी व्यक्तिगत जानकारी",
        "What is your full name?": "आपका पूरा नाम क्या है?",
        "Please enter your full name as it appears on your records.": "कृपया अपना पूरा नाम दर्ज करें जैसा कि यह आपके रिकॉर्ड पर दिखाई देता है।",
        "What is your age in years?": "आपकी आयु वर्षों में कितनी है?",
        "Please enter your age. This helps us refine the prediction.": "कृपया अपनी आयु दर्ज करें। यह हमें भविष्यवाणी को परिष्कृत करने में मदद करता है।",
        "What is your gender?": "आपका लिंग क्या है?",
        "Please select your biological gender.": "कृपया अपना जैविक लिंग चुनें।",
        "How many days have you been experiencing these symptoms?": "आप इन लक्षणों का अनुभव कितने दिनों से कर रहे हैं?",
        "This helps understand the duration of your current condition.": "यह आपकी वर्तमान स्थिति की अवधि को समझने में मदद करता है।",

        # Input Form - Symptoms & Conditions
        "Your Symptoms and Medical History": "आपके लक्षण और चिकित्सा इतिहास",
        "Please list your current symptoms (comma-separated):": "कृपया अपने वर्तमान लक्षणों को सूचीबद्ध करें (कॉमा से अलग करके):",
        "e.g., headache, nausea, fever, cough": "उदाहरण: सिरदर्द, मतली, बुखार, खांसी",
        "Describe all symptoms you are currently experiencing. Use common and clear terms, separated by commas.": "आप वर्तमान में जिन सभी लक्षणों का अनुभव कर रहे हैं उनका वर्णन करें। सामान्य और स्पष्ट शब्दों का उपयोग करें, कॉमा से अलग करके।",
        "Do you have any existing chronic medical conditions?": "क्या आपको कोई पुरानी चिकित्सा स्थिति है?",
        "Toggle this if you have a long-term disease like diabetes or hypertension.": "यदि आपको मधुमेह या उच्च रक्तचाप जैसी दीर्घकालिक बीमारी है तो इसे टॉगल करें।",
        "Please specify the name of your chronic condition:": "कृपया अपनी पुरानी बीमारी का नाम बताएं:",
        "e.g., diabetes, hypertension, asthma": "उदाहरण: मधुमेह, उच्च रक्तचाप, अस्थमा",
        "Providing the name helps us understand your full medical context.": "नाम प्रदान करने से हमें आपकी पूरी चिकित्सा संदर्भ को समझने में मदद मिलती है।",

        # Input Form - Location
        "Locate Nearby Hospitals": "आस-पास के अस्पताल खोजें",
        "In which city or area are you located?": "आप किस शहर या क्षेत्र में स्थित हैं?",
        "e.g., Bengaluru, New Delhi, Mumbai": "उदाहरण: बेंगलुरु, नई दिल्ली, मुंबई",
        "This helps us find hospitals close to your current location.": "यह हमें आपके वर्तमान स्थान के पास अस्पताल खोजने में मदद करता है।",

        # Action Button
        "Get My Health Prediction": "मेरा स्वास्थ्य पूर्वानुमान प्राप्त करें",

        # Prediction Results
        "No valid symptoms entered. Please enter at least one recognizable symptom.": "कोई वैध लक्षण दर्ज नहीं किया गया। कृपया कम से कम एक पहचानने योग्य लक्षण दर्ज करें।",
        "interpreted as": "के रूप में व्याख्या की गई",
        "Your Health Prediction Results": "आपके स्वास्थ्य भविष्यवाणी के परिणाम",
        "Predicted Disease:": "अनुमानित रोग:",
        "Description": "विवरण",
        "No description available.": "कोई विवरण उपलब्ध नहीं है।",
        "Severity Level": "गंभीरता स्तर",
        "Symptom Score:": "लक्षण स्कोर:",
        "Urgent Medical Attention Advised!": "तत्काल चिकित्सा ध्यान देने की सलाह दी जाती है!",
        "Your symptoms indicate a SEVERE condition. Please consult a doctor immediately for urgent care.": "आपके लक्षण एक गंभीर स्थिति का संकेत देते हैं। कृपया तत्काल देखभाल के लिए तुरंत डॉक्टर से सलाह लें।",
        "Medical Consultation Recommended!": "चिकित्सा परामर्श की सिफारिश की जाती है!",
        "Your symptoms are MODERATE. It's advisable to monitor your condition and consult a doctor if they persist or worsen.": "आपके लक्षण मध्यम हैं। अपनी स्थिति पर नज़र रखना और यदि वे बने रहते हैं या बिगड़ते हैं तो डॉक्टर से सलाह लेना उचित है।",
        "Symptoms are MILD.": "लक्षण हल्के हैं।",
        "Your symptoms appear to be MILD. Please follow general precautions and stay vigilant. Consult a doctor if symptoms change or worsen.": "आपके लक्षण हल्के प्रतीत होते हैं। कृपया सामान्य सावधानियों का पालन करें और सतर्क रहें। यदि लक्षण बदलते या बिगड़ते हैं तो डॉक्टर से सलाह लें।",
        "Precautions and Recommendations:": "सावधानियां और सिफारिशें:",
        "No specific precautions listed for this condition.": "इस स्थिति के लिए कोई विशेष सावधानी सूचीबद्ध नहीं है।",
        "No specific precautions found for your chronic condition: ": "आपकी पुरानी स्थिति के लिए कोई विशेष सावधानी नहीं मिली: ",
        "Unrecognized symptoms (could not match to our database): ": "अमान्य लक्षण (हमारे डेटाबेस से मेल नहीं खा सके): ",
        "These symptoms were not used in the prediction as they could not be recognized.": "इन लक्षणों का उपयोग भविष्यवाणी में नहीं किया गया था क्योंकि उन्हें पहचाना नहीं जा सका।",
        "Please double-check the spelling of unrecognized symptoms or try more common terms.": "कृपया अमान्य लक्षणों की वर्तनी दोबारा जांचें या अधिक सामान्य शब्दों का प्रयास करें।",
        "Nearby Hospitals in ": "आस-पास के अस्पताल ",
        "Click here to view hospitals near ": "पास के अस्पतालों को देखने के लिए यहां क्लिक करें ",
        "on Google Maps": "Google Maps पर",
        "Please enter your city/area to receive a useful link for nearby hospitals.": "पास के अस्पतालों के लिए एक उपयोगी लिंक प्राप्त करने के लिए कृपया अपना शहर/क्षेत्र दर्ज करें।"
    },
    "Kannada": {
        # Authentication
        "Incorrect username or password": "ತಪ್ಪಾದ ಬಳಕೆದಾರ ಹೆಸರು ಅಥವಾ ಪಾಸ್ವರ್ಡ್",
        "Please log in or register to access the application.": "ಅಪ್ಲಿಕೇಶನ್ ಪ್ರವೇಶಿಸಲು ದಯವಿಟ್ಟು ಲಾಗಿನ್ ಮಾಡಿ ಅಥವಾ ನೋಂದಾಯಿಸಿ.",
        "Login": "ಲಾಗಿನ್",
        "Register": "ನೋಂದಣಿ",
        "Login to your account": "ನಿಮ್ಮ ಖಾತೆಗೆ ಲಾಗಿನ್ ಮಾಡಿ",
        "Create a new account": "ಹೊಸ ಖಾತೆ ರಚಿಸಿ",
        "User registered successfully! Please go to the 'Login' tab to sign in.": "ಬಳಕೆದಾರ ಯಶಸ್ವಿಯಾಗಿ ನೋಂದಾಯಿಸಲಾಗಿದೆ! ದಯವಿಟ್ಟು ಸೈನ್ ಇನ್ ಮಾಡಲು 'ಲಾಗಿನ್' ಟ್ಯಾಬ್‌ಗೆ ಹೋಗಿ.",
        "An error occurred during registration: ": "ನೋಂದಣಿ ಸಮಯದಲ್ಲಿ ದೋಷ ಸಂಭವಿಸಿದೆ: ",
        "Logout": "ಲಾಗ್‌ಔಟ್",
        "Welcome ": "ಸುಸ್ವಾಗತ ",

        # General UI
        "Disease Prediction System": "ರೋಗ ಭವಿಷ್ಯ ವ್ಯವಸ್ಥೆ",
        "Please provide your information and symptoms to get a potential disease prediction and nearby hospital details.": "ಸಂಭಾವ್ಯ ರೋಗ ಭವಿಷ್ಯ ಮತ್ತು ಹತ್ತಿರದ ಆಸ್ಪತ್ರೆ ವಿವರಗಳನ್ನು ಪಡೆಯಲು ದಯವಿಟ್ಟು ನಿಮ್ಮ ಮಾಹಿತಿ ಮತ್ತು ರೋಗಲಕ್ಷಣಗಳನ್ನು ಒದಗಿಸಿ.",
        "Disclaimer: This system is for informational purposes only and not a substitute for professional medical advice. Always consult a qualified healthcare professional for any health concerns.": "ಹಕ್ಕುತ್ಯಾಗ: ಈ ವ್ಯವಸ್ಥೆಯು ಮಾಹಿತಿ ಉದ್ದೇಶಗಳಿಗಾಗಿ ಮಾತ್ರ ಮತ್ತು ವೃತ್ತಿಪರ ವೈದ್ಯಕೀಯ ಸಲಹೆಗೆ ಬದಲಿಯಾಗಿಲ್ಲ. ಯಾವುದೇ ಆರೋಗ್ಯ ಕಾಳಜಿಗಳಿಗಾಗಿ ಯಾವಾಗಲೂ ಅರ್ಹ ಆರೋಗ್ಯ ವೃತ್ತಿಪರರನ್ನು ಸಂಪರ್ಕಿಸಿ.",
        "Select language:": "ಭಾಷೆಯನ್ನು ಆಯ್ಕೆಮಾಡಿ:",

        # Input Form - Your Information
        "Your Personal Information": "ನಿಮ್ಮ ವೈಯಕ್ತಿಕ ಮಾಹಿತಿ",
        "What is your full name?": "ನಿಮ್ಮ ಪೂರ್ಣ ಹೆಸರೇನು?",
        "Please enter your full name as it appears on your records.": "ದಯವಿಟ್ಟು ನಿಮ್ಮ ದಾಖಲೆಗಳಲ್ಲಿ ಇರುವಂತೆ ನಿಮ್ಮ ಪೂರ್ಣ ಹೆಸರನ್ನು ನಮೂದಿಸಿ.",
        "What is your age in years?": "ವರ್ಷಗಳಲ್ಲಿ ನಿಮ್ಮ ವಯಸ್ಸು ಎಷ್ಟು?",
        "Please enter your age. This helps us refine the prediction.": "ದಯವಿಟ್ಟು ನಿಮ್ಮ ವಯಸ್ಸನ್ನು ನಮೂದಿಸಿ. ಇದು ಭವಿಷ್ಯವನ್ನು ಉತ್ತಮಗೊಳಿಸಲು ನಮಗೆ ಸಹಾಯ ಮಾಡುತ್ತದೆ.",
        "What is your gender?": "ನಿಮ್ಮ ಲಿಂಗ ಯಾವುದು?",
        "Please select your biological gender.": "ದಯವಿಟ್ಟು ನಿಮ್ಮ ಜೈವಿಕ ಲಿಂಗವನ್ನು ಆಯ್ಕೆಮಾಡಿ.",
        "How many days have you been experiencing these symptoms?": "ಈ ರೋಗಲಕ್ಷಣಗಳನ್ನು ನೀವು ಎಷ್ಟು ದಿನಗಳಿಂದ ಅನುಭವಿಸುತ್ತಿದ್ದೀರಿ?",
        "This helps understand the duration of your current condition.": "ಇದು ನಿಮ್ಮ ಪ್ರಸ್ತುತ ಸ್ಥಿತಿಯ ಅವಧಿಯನ್ನು ಅರ್ಥಮಾಡಿಕೊಳ್ಳಲು ಸಹಾಯ ಮಾಡುತ್ತದೆ.",

        # Input Form - Symptoms & Conditions
        "Your Symptoms and Medical History": "ನಿಮ್ಮ ರೋಗಲಕ್ಷಣಗಳು ಮತ್ತು ವೈದ್ಯಕೀಯ ಇತಿಹಾಸ",
        "Please list your current symptoms (comma-separated):": "ದಯವಿಟ್ಟು ನಿಮ್ಮ ಪ್ರಸ್ತುತ ರೋಗಲಕ್ಷಣಗಳನ್ನು ಪಟ್ಟಿ ಮಾಡಿ (ಅಲ್ಪವಿರಾಮದಿಂದ ಬೇರ್ಪಡಿಸಿ):",
        "e.g., headache, nausea, fever, cough": "ಉದಾ: ತಲೆನೋವು, ವಾಕರಿಕೆ, ಜ್ವರ, ಕೆಮ್ಮು",
        "Describe all symptoms you are currently experiencing. Use common and clear terms, separated by commas.": "ನೀವು ಪ್ರಸ್ತುತ ಅನುಭವಿಸುತ್ತಿರುವ ಎಲ್ಲಾ ರೋಗಲಕ್ಷಣಗಳನ್ನು ವಿವರಿಸಿ. ಸಾಮಾನ್ಯ ಮತ್ತು ಸ್ಪಷ್ಟ ಪದಗಳನ್ನು ಬಳಸಿ, ಅಲ್ಪವಿರಾಮದಿಂದ ಬೇರ್ಪಡಿಸಿ.",
        "Do you have any existing chronic medical conditions?": "ನಿಮಗೆ ಯಾವುದೇ ಅಸ್ತಿತ್ವದಲ್ಲಿರುವ ದೀರ್ಘಕಾಲದ ವೈದ್ಯಕೀಯ ಪರಿಸ್ಥಿತಿಗಳಿವೆಯೇ?",
        "Toggle this if you have a long-term disease like diabetes or hypertension.": "ಮಧುಮೇಹ ಅಥವಾ ಅಧಿಕ ರಕ್ತದೊತ್ತಡದಂತಹ ದೀರ್ಘಕಾಲದ ಕಾಯಿಲೆ ಇದ್ದರೆ ಇದನ್ನು ಟಾಗಲ್ ಮಾಡಿ.",
        "Please specify the name of your chronic condition:": "ದಯವಿಟ್ಟು ನಿಮ್ಮ ದೀರ್ಘಕಾಲದ ಸ್ಥಿತಿಯ ಹೆಸರನ್ನು ನಿರ್ದಿಷ್ಟಪಡಿಸಿ:",
        "e.g., diabetes, hypertension, asthma": "ಉದಾ: ಮಧುಮೇಹ, ಅಧಿಕ ರಕ್ತದೊತ್ತಡ, ಅಸ್ತಮಾ",
        "Providing the name helps us understand your full medical context.": "ಹೆಸರನ್ನು ನೀಡುವುದರಿಂದ ನಿಮ್ಮ ಸಂಪೂರ್ಣ ವೈದ್ಯಕೀಯ ಸನ್ನಿವೇಶವನ್ನು ಅರ್ಥಮಾಡಿಕೊಳ್ಳಲು ನಮಗೆ ಸಹಾಯ ಮಾಡುತ್ತದೆ。",

        # Input Form - Location
        "Locate Nearby Hospitals": "ಹತ್ತಿರದ ಆಸ್ಪತ್ರೆಗಳನ್ನು ಪತ್ತೆ ಮಾಡಿ",
        "In which city or area are you located?": "ನೀವು ಯಾವ ನಗರ ಅಥವಾ ಪ್ರದೇಶದಲ್ಲಿ ನೆಲೆಸಿದ್ದೀರಿ?",
        "e.g., Bengaluru, New Delhi, Mumbai": "ಉದಾ: ಬೆಂಗಳೂರು, ನವದೆಹಲಿ, ಮುಂಬೈ",
        "This helps us find hospitals close to your current location.": "ಇದು ನಿಮ್ಮ ಪ್ರಸ್ತುತ ಸ್ಥಳಕ್ಕೆ ಹತ್ತಿರವಿರುವ ಆಸ್ಪತ್ರೆಗಳನ್ನು ಹುಡುಕಲು ನಮಗೆ ಸಹಾಯ ಮಾಡುತ್ತದೆ。",

        # Action Button
        "Get My Health Prediction": "ನನ್ನ ಆರೋಗ್ಯ ಭವಿಷ್ಯವನ್ನು ಪಡೆಯಿರಿ",

        # Prediction Results
        "No valid symptoms entered. Please enter at least one recognizable symptom.": "ಯಾವುದೇ ಮಾನ್ಯವಾದ ರೋಗಲಕ್ಷಣಗಳನ್ನು ನಮೂದಿಸಲಾಗಿಲ್ಲ. ದಯವಿಟ್ಟು ಕನಿಷ್ಠ ಒಂದು ಗುರುತಿಸಬಹುದಾದ ರೋಗಲಕ್ಷಣವನ್ನು ನಮೂದಿಸಿ.",
        "interpreted as": "ಎಂದು ಅರ್ಥೈಸಲಾಗಿದೆ",
        "Your Health Prediction Results": "ನಿಮ್ಮ ಆರೋಗ್ಯ ಭವಿಷ್ಯದ ಫಲಿತಾಂಶಗಳು",
        "Predicted Disease:": "ಊಹಿಸಿದ ರೋಗ:",
        "Description": "ವಿವರಣೆ",
        "No description available.": "ಯಾವುದೇ ವಿವರಣೆ ಲಭ್ಯವಿಲ್ಲ.",
        "Severity Level": "ತೀವ್ರತೆಯ ಮಟ್ಟ",
        "Symptom Score:": "ರೋಗಲಕ್ಷಣದ ಅಂಕ:",
        "Urgent Medical Attention Advised!": "ತಕ್ಷಣದ ವೈದ್ಯಕೀಯ ಗಮನವನ್ನು ಸಲಹೆ ಮಾಡಲಾಗಿದೆ!",
        "Your symptoms indicate a SEVERE condition. Please consult a doctor immediately for urgent care.": "ನಿಮ್ಮ ರೋಗಲಕ್ಷಣಗಳು ತೀವ್ರ ಸ್ಥಿತಿಯನ್ನು ಸೂಚಿಸುತ್ತವೆ. ತುರ್ತು ಚಿಕಿತ್ಸೆಗಾಗಿ ದಯವಿಟ್ಟು ತಕ್ಷಣ ವೈದ್ಯರನ್ನು ಸಂಪರ್ಕಿಸಿ.",
        "Medical Consultation Recommended!": "ವೈದ್ಯಕೀಯ ಸಮಾಲೋಚನೆ ಶಿಫಾರಸು ಮಾಡಲಾಗಿದೆ!",
        "Your symptoms are MODERATE. It's advisable to monitor your condition and consult a doctor if they persist or worsen.": "ನಿಮ್ಮ ರೋಗಲಕ್ಷಣಗಳು ಮಧ್ಯಮವಾಗಿವೆ. ನಿಮ್ಮ ಸ್ಥಿತಿಯನ್ನು ಮೇಲ್ವಿಚಾರಣೆ ಮಾಡುವುದು ಮತ್ತು ಅವುಗಳು ಮುಂದುವರಿದರೆ ಅಥವಾ ಹದಗೆಟ್ಟರೆ ವೈದ್ಯರನ್ನು ಸಂಪರ್ಕಿಸುವುದು ಸೂಕ್ತ。",
        "Symptoms are MILD.": "ರೋಗಲಕ್ಷಣಗಳು ಸೌಮ್ಯವಾಗಿವೆ。",
        "Your symptoms appear to be MILD. Please follow general precautions and stay vigilant. Consult a doctor if symptoms change or worsen.": "ನಿಮ್ಮ ರೋಗಲಕ್ಷಣಗಳು ಸೌಮ್ಯವಾಗಿ ಕಾಣಿಸುತ್ತವೆ. ದಯವಿಟ್ಟು ಸಾಮಾನ್ಯ ಮುನ್ನೆಚ್ಚರಿಕೆಗಳನ್ನು ಅನುಸರಿಸಿ ಮತ್ತು ಜಾಗರೂಕರಾಗಿರಿ. ರೋಗಲಕ್ಷಣಗಳು ಬದಲಾದರೆ ಅಥವಾ ಹದಗೆಟ್ಟರೆ ವೈದ್ಯರನ್ನು ಸಂಪರ್ಕಿಸಿ。",
        "Precautions and Recommendations:": "ಮುನ್ನೆಚ್ಚರಿಕೆಗಳು ಮತ್ತು ಶಿಫಾರಸುಗಳು:",
        "No specific precautions listed for this condition.": "ಈ ಸ್ಥಿತಿಗೆ ಯಾವುದೇ ನಿರ್ದಿಷ್ಟ ಮುನ್ನೆಚ್ಚರಿಕೆಗಳನ್ನು ಪಟ್ಟಿ ಮಾಡಲಾಗಿಲ್ಲ。",
        "No specific precautions found for your chronic condition: ": "ನಿಮ್ಮ ದೀರ್ಘಕಾಲದ ಸ್ಥಿತಿಗೆ ಯಾವುದೇ ನಿರ್ದಿಷ್ಟ ಮುನ್ನೆಚ್ಚರಿಕೆಗಳು ಕಂಡುಬಂದಿಲ್ಲ: ",
        "Unrecognized symptoms (could not match to our database): ": "ಗುರುತಿಸದ ರೋಗಲಕ್ಷಣಗಳು (ನಮ್ಮ ಡೇಟಾಬೇಸ್‌ಗೆ ಹೊಂದಿಕೆಯಾಗಲಿಲ್ಲ): ",
        "These symptoms were not used in the prediction as they could not be recognized.": "ಈ ರೋಗಲಕ್ಷಣಗಳನ್ನು ಗುರುತಿಸಲಾಗದ ಕಾರಣ ಭವಿಷ್ಯದಲ್ಲಿ ಬಳಸಲಾಗಿಲ್ಲ。",
        "Please double-check the spelling of unrecognized symptoms or try more common terms.": "ದಯವಿಟ್ಟು ಗುರುತಿಸದ ರೋಗಲಕ್ಷಣಗಳ ಕಾಗುಣಿತವನ್ನು ಎರಡು ಬಾರಿ ಪರಿಶೀಲಿಸಿ ಅಥವಾ ಹೆಚ್ಚು ಸಾಮಾನ್ಯ ಪದಗಳನ್ನು ಪ್ರಯತ್ನಿಸಿ。",
        "Nearby Hospitals in ": "ಹತ್ತಿರದ ಆಸ್ಪತ್ರೆಗಳು ",
        "Click here to view hospitals near ": "ಹತ್ತಿರದ ಆಸ್ಪತ್ರೆಗಳನ್ನು ವೀಕ್ಷಿಸಲು ಇಲ್ಲಿ ಕ್ಲಿಕ್ ಮಾಡಿ ",
        "on Google Maps": "Google Maps ನಲ್ಲಿ",
        "Please enter your city/area to receive a useful link for nearby hospitals.": "ಹತ್ತಿರದ ಆಸ್ಪತ್ರೆಗಳಿಗೆ ಉಪಯುಕ್ತ ಲಿಂಕ್ ಪಡೆಯಲು ದಯವಿಟ್ಟು ನಿಮ್ಮ ನಗರ/ಪ್ರದೇಶವನ್ನು ನಮೂದಿಸಿ।"
    }
}


# ---------- Helper Functions for User Data Persistence ----------
def load_user_data():
    """Loads user data from users.json."""
    try:
        with open(USER_DATA_FILE, 'r') as f:
            data = json.load(f)
            if "usernames" not in data:
                data = {"usernames": {}}
            return data
    except (FileNotFoundError, json.JSONDecodeError):
        return {"usernames": {}} # Return empty structure if file not found or corrupted

def save_user_data(data):
    """Saves user data to users.json."""
    with open(USER_DATA_FILE, 'w') as f:
        json.dump(data, f, indent=4)

# ---------- Page Configuration ----------
st.set_page_config(page_title="Disease Prediction System", layout="centered")

# --- Language Selection and Translation Function ---
# Initialize session state for language if not already set
if 'selected_language' not in st.session_state:
    st.session_state.selected_language = "English"

def get_translated_text(key):
    """Returns the translated text for a given English key."""
    return TRANSLATIONS.get(st.session_state.selected_language, {}).get(key, key)

# ---------- Authentication Setup ----------
user_data = load_user_data()
credentials = user_data

authenticator = stauth.Authenticate(
    credentials,
    "disease_predictor_cookie",
    "YOUR_STRONG_RANDOM_SECRET_KEY_FOR_SECURITY", # !!! CHANGE THIS TO A LONG, RANDOM STRING !!!
    cookie_expiry_days=0
)

# --- Global state for authentication ---
authentication_status = st.session_state.get('authentication_status')
name = st.session_state.get('name')
username = st.session_state.get('username')

# --- SpaCy Model Loading (Cached for performance) ---
@st.cache_resource
def load_spacy_model():
    """Loads the SpaCy English model, cached for efficiency."""
    try:
        nlp_model = spacy.load("en_core_web_sm")
    except OSError:
        st.error("SpaCy English model 'en_core_web_sm' not found. "
                 "Please run 'python -m spacy download en_core_web_sm' in your terminal "
                 "and restart the app.")
        st.stop() # Stop the app if model is not found
    return nlp_model

nlp = load_spacy_model()


# --- I. Initial User Interaction (Login/Register) ---
if authentication_status is False or authentication_status is None:
    if authentication_status is False:
        st.error(get_translated_text("Incorrect username or password"))
    elif authentication_status is None:
        st.warning(get_translated_text("Please log in or register to access the application."))

    tab1, tab2 = st.tabs([get_translated_text("Login"), get_translated_text("Register")])

    with tab1:
        st.subheader(get_translated_text("Login to your account"))
        authenticator.login(location="main")

    with tab2:
        st.subheader(get_translated_text("Create a new account"))
        try:
            email_of_registered_user, name_of_registered_user, username_of_registered_user = authenticator.register_user()
            if email_of_registered_user:
                st.success(get_translated_text("User registered successfully! Please go to the 'Login' tab to sign in."))
                st.balloons()
                save_user_data(credentials)
        except Exception as e:
            st.error(get_translated_text("An error occurred during registration: ") + str(e))

# --- II. Welcome & Onboarding (After Successful Login) ---
elif authentication_status:
    with st.sidebar:
        st.success(get_translated_text("Welcome ") + f"{name} 👋")
        authenticator.logout(get_translated_text("Logout"))

    st.title(get_translated_text("Disease Prediction System"))
    st.markdown(get_translated_text("Please provide your information and symptoms to get a potential disease prediction and nearby hospital details."))

    st.warning(get_translated_text("Disclaimer: This system is for informational purposes only and not a substitute for professional medical advice. Always consult a qualified healthcare professional for any health concerns."))

    st.session_state.selected_language = st.selectbox(
        get_translated_text("Select language:"),
        ["English", "Hindi", "Kannada"]
    )

    # ---------- Load Dataset (with error handling) ----------
    try:
        data = pd.read_csv(DATASET_FILE)
        desc_data = pd.read_csv(DESC_DATA_FILE)
        precaution_data = pd.read_csv(PRECAUTION_DATA_FILE)
        severity_data = pd.read_csv(SEVERITY_DATA_FILE)
    except FileNotFoundError as e:
        st.error(f"Error: Required data file not found ({e}). Please ensure all CSV files are in the same directory as 'hello.py'.")
        st.stop()

    # --- Data Preprocessing (Cached for performance) ---
    @st.cache_data
    def preprocess_data(data, severity_data):
        symptom_cols = [col for col in data.columns if col.startswith('Symptom')]
        for col in symptom_cols:
            data[col] = data[col].astype(str)
        data.fillna('None', inplace=True)

        unique_symptoms_raw = pd.unique(data[symptom_cols].values.ravel())
        unique_symptoms = sorted([
            s for s in unique_symptoms_raw
            if isinstance(s, str) and s not in ['None', 'nan']
        ])
        normalized_symptoms_dict = {s.replace("_", " ").lower(): s for s in unique_symptoms}

        if 'Symptom' in severity_data.columns and 'weight' in severity_data.columns:
            severity_data['Symptom'] = severity_data['Symptom'].str.strip().replace("_", " ")
            symptom_severity_dict = dict(zip(severity_data['Symptom'], severity_data['weight']))
        else:
            st.error("Error: 'Symptom-severity.csv' must contain 'Symptom' and 'weight' columns.")
            st.stop()

        # Create a base DataFrame for encoding: This will be our X for training
        
        # Binary symptom presence features
        symptom_df = pd.DataFrame(0, index=data.index, columns=unique_symptoms)
        for i, row in data.iterrows():
            for sym_col in symptom_cols:
                symptom = row[sym_col]
                if symptom in symptom_df.columns:
                    symptom_df.loc[i, symptom] = 1

        # Dummy data for non-symptom features to build the ColumnTransformer
        # These values will be replaced by actual user input during prediction
        dummy_age = [30] * len(data)
        dummy_days = [5] * len(data)
        dummy_gender = ['male'] * len(data) # Default gender
        dummy_chronic = [0] * len(data) # Default no chronic disease

        # Combine all features for the preprocessor pipeline
        temp_df_for_preprocessor = pd.DataFrame({
            'age': dummy_age,
            'days_with_symptoms': dummy_days,
            'gender': dummy_gender,
            'has_chronic_disease': dummy_chronic
        })
        
        # Add symptom features to the temporary DataFrame
        temp_df_for_preprocessor = pd.concat([temp_df_for_preprocessor, symptom_df], axis=1)

        # Define preprocessing steps
        numerical_features = ['age', 'days_with_symptoms']
        categorical_features = ['gender']
        binary_features = ['has_chronic_disease'] 
        passthrough_features = unique_symptoms

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', 'passthrough', numerical_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
                ('bin', 'passthrough', binary_features),
                ('sym', 'passthrough', passthrough_features)
            ])
        
        preprocessor.fit(temp_df_for_preprocessor)

        def get_feature_names(column_transformer):
            output_features = []
            for name, estimator, features in column_transformer.transformers_:
                if name == 'cat': # OneHotEncoder
                    output_features.extend(estimator.get_feature_names_out(features))
                elif estimator == 'passthrough':
                    output_features.extend(features)
            return output_features

        model_features_list = get_feature_names(preprocessor)
        
        X_train_symptoms = symptom_df 
        
        X_train_non_symptoms = pd.DataFrame({
            'age': [30] * len(data), 
            'days_with_symptoms': [5] * len(data), 
            'gender': ['male'] * len(data), 
            'has_chronic_disease': [0] * len(data) 
        }, index=data.index)

        X = preprocessor.transform(pd.concat([X_train_non_symptoms, X_train_symptoms], axis=1))
        
        y = data['Disease']
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        disease_labels = dict(zip(le.transform(le.classes_), le.classes_))

        return X, y_encoded, disease_labels, unique_symptoms, normalized_symptoms_dict, symptom_severity_dict, le, symptom_cols, preprocessor, model_features_list


    X, y_encoded, disease_labels, unique_symptoms, normalized_symptoms_dict, symptom_severity_dict, le, symptom_cols, preprocessor, model_features_list = preprocess_data(data, severity_data)

    # --- Model Training (Cached for performance) ---
    @st.cache_resource
    def train_model(X_data, y_data):
        model = RandomForestClassifier(random_state=42)
        model.fit(X_data, y_data)
        return model

    model = train_model(X, y_encoded)

    # --- Prediction Functions ---
    def predict_disease(input_symptoms_list, age, gender, days_with_symptoms, has_chronic_disease_flag):
        
        # Create a DataFrame for the non-symptom features first
        non_symptom_df = pd.DataFrame([{
            'age': age,
            'days_with_symptoms': days_with_symptoms,
            'gender': gender,
            'has_chronic_disease': has_chronic_disease_flag
        }])

        # Create a DataFrame for symptom features (binary 0/1)
        symptom_df_for_prediction = pd.DataFrame(0, index=[0], columns=unique_symptoms)
        for sym in input_symptoms_list:
            if sym in symptom_df_for_prediction.columns:
                symptom_df_for_prediction.loc[0, sym] = 1
        
        # Concatenate them for the preprocessor input
        input_df_for_preprocessor = pd.concat([non_symptom_df, symptom_df_for_prediction], axis=1)

        # Transform the single input sample using the fitted preprocessor
        transformed_input = preprocessor.transform(input_df_for_preprocessor)
        
        prediction = model.predict(transformed_input)[0]
        return disease_labels[prediction]


    def get_info(disease):
        desc = desc_data[desc_data['Disease'] == disease]['Description'].values
        desc = desc[0] if len(desc) > 0 else get_translated_text("No description available.")
        precautions = precaution_data[precaution_data['Disease'] == disease].iloc[:, 1:].values
        precautions = list(precautions[0][~pd.isna(precautions[0])]) if len(precautions) > 0 else [get_translated_text("No specific precautions listed for this condition.")]
        return desc, precautions

    def severity_score(symptoms_list):
        return sum([symptom_severity_dict.get(sym.replace("_", " "), 1) for sym in symptoms_list]) 

    # --- NEW NLP Function for Symptom Extraction ---
    def extract_and_normalize_symptoms_with_nlp(text, normalized_symptoms_dict, nlp_model):
        doc = nlp_model(text.lower())
        extracted_symptoms = set() # Use a set to avoid duplicates
        unrecognized_tokens = set() # To track what wasn't matched

        # Process tokens and look for matches
        for token in doc:
            potential_symptom = token.text.strip()
            if not potential_symptom:
                continue

            # Try direct match
            if potential_symptom in normalized_symptoms_dict:
                extracted_symptoms.add(normalized_symptoms_dict[potential_symptom])
                continue

            # Try fuzzy match if direct match fails
            close_matches = difflib.get_close_matches(potential_symptom, normalized_symptoms_dict.keys(), n=1, cutoff=0.7)
            if close_matches:
                extracted_symptoms.add(normalized_symptoms_dict[close_matches[0]])
            else:
                # Add to unrecognized only if it's not a common stop word or punctuation (basic filter)
                if not token.is_stop and not token.is_punct and not token.is_space:
                    unrecognized_tokens.add(potential_symptom)
        
        # Basic attempt at multi-word symptom detection from noun chunks
        # This is a simple approach and can be greatly expanded for medical NLP
        for chunk in doc.noun_chunks:
            chunk_text = chunk.text.lower().strip()
            if not chunk_text:
                continue
            if chunk_text in normalized_symptoms_dict:
                extracted_symptoms.add(normalized_symptoms_dict[chunk_text])
                # Remove individual tokens from unrecognized if they are part of a recognized chunk
                for token in chunk:
                    if token.text.lower() in unrecognized_tokens:
                        unrecognized_tokens.remove(token.text.lower())
            else:
                close_matches = difflib.get_close_matches(chunk_text, normalized_symptoms_dict.keys(), n=1, cutoff=0.7)
                if close_matches:
                    extracted_symptoms.add(normalized_symptoms_dict[close_matches[0]])
                    for token in chunk:
                        if token.text.lower() in unrecognized_tokens:
                            unrecognized_tokens.remove(token.text.lower())

        return list(extracted_symptoms), list(unrecognized_tokens)


    # --- III. Core Input Sections (using st.form) ---
    st.markdown(get_translated_text("Please provide your information and symptoms to get a potential disease prediction and nearby hospital details."))

    with st.form("user_input_form"):
        # Section 1: Your Personal Information
        with st.container(border=True):
            st.subheader(get_translated_text("Your Personal Information"))
            
            col_name, col_age, col_gender = st.columns(3)
            with col_name:
                name_input = st.text_input(
                    get_translated_text("What is your full name?"),
                    value=name if name else "",
                    placeholder="John Doe",
                    help=get_translated_text("Please enter your full name as it appears on your records.")
                )
            with col_age:
                age = st.number_input(
                    get_translated_text("What is your age in years?"),
                    min_value=1,
                    max_value=120,
                    value=30,
                    help=get_translated_text("Please enter your age. This helps us refine the prediction.")
                )
            with col_gender:
                gender = st.selectbox(
                    get_translated_text("What is your gender?"),
                    ["male", "female", "other"],
                    help=get_translated_text("Please select your biological gender.")
                )
            
            days_with_symptoms = st.number_input(
                get_translated_text("How many days have you been experiencing these symptoms?"),
                min_value=0, 
                max_value=60,
                value=3,
                help=get_translated_text("This helps understand the duration of your current condition.")
            )

        # Section 2: Locate Nearby Hospitals
        st.markdown("---")
        with st.container(border=True):
            st.subheader(get_translated_text("Locate Nearby Hospitals"))
            location = st.text_input(
                get_translated_text("In which city or area are you located?"),
                placeholder=get_translated_text("e.g., Bengaluru, New Delhi, Mumbai"),
                help=get_translated_text("This helps us find hospitals close to your current location.")
            )

        # Section 3: Your Symptoms and Medical History
        st.markdown("---")
        with st.container(border=True):
            st.subheader(get_translated_text("Your Symptoms and Medical History"))

            # Chronic conditions first
            chronic_disease_toggle = st.toggle(
                get_translated_text("Do you have any existing chronic medical conditions?"),
                help=get_translated_text("Toggle this if you have a long-term disease like diabetes or hypertension.")
            )
            chronic_disease_name = ""
            if chronic_disease_toggle:
                chronic_disease_name = st.text_input(
                    get_translated_text("Please specify the name of your chronic condition:"),
                    placeholder=get_translated_text("e.g., diabetes, hypertension, asthma"),
                    help=get_translated_text("Providing the name helps us understand your full medical context.")
                )
            has_chronic_disease_flag = 1 if chronic_disease_toggle else 0

            # Current symptoms (NLP-enabled input)
            input_symptom_text = st.text_area(
                get_translated_text("Please list your current symptoms (comma-separated):"),
                placeholder=get_translated_text("e.g., headache, nausea, fever, cough"),
                help=get_translated_text("Describe all symptoms you are currently experiencing. Use common and clear terms, separated by commas.")
            )
            
        # Action Button
        st.markdown("---")
        submitted = st.form_submit_button(get_translated_text("Get My Health Prediction"))

    # --- IV. Post-Submission Feedback & Results ---
    if submitted:
        # Process symptoms using the new NLP function
        final_symptoms, unmatched_symptoms_after_submit = extract_and_normalize_symptoms_with_nlp(
            input_symptom_text, normalized_symptoms_dict, nlp
        )
        
        if not final_symptoms: # Check if any symptoms were recognized by NLP
            st.error(get_translated_text("No valid symptoms entered. Please enter at least one recognizable symptom."))
        else:
            # --- PREDICT DISEASE with all features ---
            disease = predict_disease(final_symptoms, age, gender, days_with_symptoms, has_chronic_disease_flag)
            desc, precautions = get_info(disease)

            score = severity_score(final_symptoms)
            severity = "mild" if score < 10 else "moderate" if score < 20 else "severe"

            st.markdown("---")
            st.subheader(get_translated_text("Your Health Prediction Results"))
            st.success(f"🦠 {get_translated_text('Predicted Disease:')} **{disease}**")
            st.markdown(f"📃 **{get_translated_text('Description')}**: {desc}")
            st.markdown(f"📊 **{get_translated_text('Severity Level')}**: `{severity.upper()}` ({get_translated_text('Symptom Score:')} {score})")

            if severity == 'severe':
                st.error(f"🔴 **{get_translated_text('Urgent Medical Attention Advised!')}** {get_translated_text('Your symptoms indicate a SEVERE condition. Please consult a doctor immediately for urgent care.')}")
            elif severity == 'moderate':
                translated_rec = get_translated_text('Medical Consultation Recommended!')
                translated_mod_symptoms = get_translated_text('Your symptoms are MODERATE. It\'s advisable to monitor your condition and consult a doctor if they persist or worsen.')
                st.warning(f"🟠 **{translated_rec}** {translated_mod_symptoms}") 
            else:
                st.info(f"🟢 **{get_translated_text('Symptoms are MILD.')}** {get_translated_text('Your symptoms appear to be MILD. Please follow general precautions and stay vigilant. Consult a doctor if symptoms change or worsen.')}")

            chronic_precautions = []
            if chronic_disease_name: 
                chronic_row = precaution_data[precaution_data['Disease'].str.lower() == chronic_disease_name.lower()]
                if not chronic_row.empty:
                    cp_values = chronic_row.iloc[:, 1:].values[0]
                    chronic_precautions = list(cp_values[~pd.isna(cp_values)])
                else:
                    chronic_precautions = [get_translated_text("No specific precautions found for your chronic condition: ") + chronic_disease_name + "."]

            all_precautions = list(dict.fromkeys(precautions + chronic_precautions))

            st.markdown("---")
            st.subheader(get_translated_text("Precautions and Recommendations:"))
            if all_precautions:
                for p in all_precautions:
                    st.markdown(f"- {p}")
            else:
                st.info(get_translated_text("No specific precautions listed for this condition."))

            if unmatched_symptoms_after_submit: # Display unrecognized tokens from NLP
                st.warning(get_translated_text("Unrecognized symptoms (could not match to our database): ") + ", ".join(unmatched_symptoms_after_submit) +
                           ". " + get_translated_text("These symptoms were not used in the prediction as they could not be recognized."))
                st.info(get_translated_text("Please double-check the spelling of unrecognized symptoms or try more common terms."))

            if location:
                st.markdown("---")
                st.subheader(get_translated_text("Nearby Hospitals in ") + f"{location}")
                # Google Maps URL for searching hospitals in a specific location
                # Note: The provided URL `https://www.google.com/maps/search/hospitals+in+?q=hospitals+near+` is incorrect.
                # A correct Google Maps search URL would be:
                Maps_search_url = f"https://www.google.com/maps/search/hospitals+in+{location.replace(' ', '+')}"
                st.markdown(
                    f"📍 [{get_translated_text('Click here to view hospitals near ')}{location} {get_translated_text('on Google Maps')}]({Maps_search_url})",
                    unsafe_allow_html=True
                )
            else:
                st.info(get_translated_text("Please enter your city/area to receive a useful link for nearby hospitals."))


