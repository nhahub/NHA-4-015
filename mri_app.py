import streamlit as st
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import Adam
from PIL import Image
import numpy as np
import sqlite3
import io
import os
import pandas as pd
import base64

st.set_page_config(page_title="NeuroScope", layout="wide")

# Define paths
CLASS_NAMES = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]
DATABASE_PATH = "database.db"
FEEDBACK_PATH = "feedback_data.csv"



# Initialize database connection
def init_database():
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()

    # Create the user_accounts table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_accounts (
            phone_number TEXT PRIMARY KEY,
            first_name TEXT NOT NULL,
            last_name TEXT NOT NULL,
            password TEXT NOT NULL,
            email TEXT NOT NULL,
            date_of_birth TEXT NOT NULL,
            gender TEXT NOT NULL,
            address TEXT NOT NULL,
            insurance_number TEXT,
            chronic_illnesses TEXT,
            chronic_medications TEXT,
            weight REAL,
            height REAL,
            user_type TEXT CHECK(user_type IN ('doctor', 'patient')) NOT NULL,
            specialization TEXT,
            college TEXT,
            medical_degrees TEXT,
            about_doctor TEXT
        )
    ''')

    # Create the patient_data table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS patient_data (
            user_id TEXT PRIMARY KEY,
            additional_notes TEXT,
            predicted_class TEXT,
            mri_image BLOB,
            FOREIGN KEY(user_id) REFERENCES user_accounts(phone_number)
        )
    ''')

    # Create the appointments table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS appointments (
            appointment_id INTEGER PRIMARY KEY AUTOINCREMENT,
            doctor_phone TEXT NOT NULL,
            patient_phone TEXT NOT NULL,
            appointment_date TEXT NOT NULL,
            appointment_time TEXT NOT NULL,
            status TEXT CHECK(status IN ('Pending', 'Approved', 'Rejected')) DEFAULT 'Pending',
            FOREIGN KEY(doctor_phone) REFERENCES user_accounts(phone_number),
            FOREIGN KEY(patient_phone) REFERENCES user_accounts(phone_number)
        )
    ''')

    conn.commit()
    conn.close()

# Initialize the database
init_database()


# Load the trained model
@st.cache_resource
def load_single_model():
    return load_model("saved_model/inceptionresnetv2_mri_final.keras")

model = load_single_model()


import requests

def ask_chatbot_via_rapidapi(message, specialization="neurosurgery", language="en"):
    url = "https://ai-doctor-api-ai-medical-chatbot-healthcare-ai-assistant.p.rapidapi.com/chat?noqueue=1"

    payload = {
        "message": message,
        "specialization": specialization,
        "language": language
    }

    headers = {
        "content-type": "application/json",
        "x-rapidapi-host": "ai-doctor-api-ai-medical-chatbot-healthcare-ai-assistant.p.rapidapi.com",
        "x-rapidapi-key": "c3223f39fdmsh5729bd17b448e8fp144716jsnd86d352b7f9e"
    }

    response = requests.post(url, json=payload, headers=headers)

    try:
        data = response.json()
        if "result" in data and "response" in data["result"]:
            main_message = data["result"]["response"].get("message", "No main message found.")
            follow_up = data["result"]["response"].get("followUp", [])
            return f"{main_message}\n\n**Suggested follow-up questions:**\n" + "\n".join(f"- {q}" for q in follow_up)
        else:
            return f"⚠️ Unexpected structure:\n{data}"
    except Exception as e:
        return f"❌ Error decoding response: {e}\nRaw response:\n{response.text}"



def chatbot_page():
    st.title("🧠 AI Health Assistant")
    st.write("Ask me anything about your health or brain tumors. If I'm unsure, I'll let you know.")

    user_input = st.text_input("Your Question")

    if st.button("Ask") and user_input:
        reply = ask_chatbot_via_rapidapi(user_input)
        st.markdown(f"**Assistant:** {reply}")



# Database interaction functions
def add_user(phone_number, first_name, last_name, password, email, date_of_birth, gender, address, insurance_number=None, chronic_illnesses=None, chronic_medications=None, weight=None, height=None, user_type="patient", specialization=None, college=None, medical_degrees=None, about_doctor=None):
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO user_accounts (phone_number, first_name, last_name, password, email, date_of_birth, gender, address, insurance_number, chronic_illnesses, chronic_medications, weight, height, user_type, specialization, college, medical_degrees, about_doctor)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (phone_number, first_name, last_name, password, email, date_of_birth, gender, address, insurance_number, chronic_illnesses, chronic_medications, weight, height, user_type, specialization, college, medical_degrees, about_doctor))
    conn.commit()
    conn.close()

def authenticate(phone_number, password):
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT * FROM user_accounts WHERE phone_number = ? AND password = ?
    ''', (phone_number, password))
    user = cursor.fetchone()
    conn.close()
    return user

# Initialize session state
def initialize_session_state():
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "user_type" not in st.session_state:
        st.session_state.user_type = None
    if "user_id" not in st.session_state:
        st.session_state.user_id = None
    if "first_name" not in st.session_state:
        st.session_state.first_name = None
    if "last_name" not in st.session_state:
        st.session_state.last_name = None
    if "email" not in st.session_state:
        st.session_state.email = None
    if "date_of_birth" not in st.session_state:
        st.session_state.date_of_birth = None
    if "gender" not in st.session_state:
        st.session_state.gender = None
    if "address" not in st.session_state:
        st.session_state.address = None
    if "insurance_number" not in st.session_state:
        st.session_state.insurance_number = None
    if "chronic_illnesses" not in st.session_state:
        st.session_state.chronic_illnesses = None
    if "chronic_medications" not in st.session_state:
        st.session_state.chronic_medications = None
    if "weight" not in st.session_state:
        st.session_state.weight = None
    if "height" not in st.session_state:
        st.session_state.height = None
    if "specialization" not in st.session_state:
        st.session_state.specialization = None
    if "college" not in st.session_state:
        st.session_state.college = None
    if "medical_degrees" not in st.session_state:
        st.session_state.medical_degrees = None
    if "about_doctor" not in st.session_state:
        st.session_state.about_doctor = None

initialize_session_state()

def load_feedback_data(feedback_path):
    if os.path.exists(feedback_path):
        feedback_df = pd.read_csv(feedback_path)

        if not feedback_df.empty:
            images = []
            labels = []

            for _, row in feedback_df.iterrows():
                try:
                    # Decode the base64 image
                    image_data = base64.b64decode(row['image'])
                    image = Image.open(io.BytesIO(image_data)).resize((224, 224))
                    images.append(np.array(image) / 255.0)  # Normalize pixel values
                    labels.append(CLASS_NAMES.index(row['corrected_class']))
                except Exception as e:
                    st.warning(f"Skipping invalid image: {e}")

            return np.array(images), np.array(labels)
    return None, None


def retrain_model_with_feedback(original_model, feedback_images, feedback_labels):
    global model

    if feedback_images is not None and feedback_labels is not None:
        # Rebuild the model using Functional API to avoid shape mismatches
        base_model = load_model(MODEL_PATH)
        base_model.trainable = True  # Fine-tune the entire model

        base_model.compile(optimizer=Adam(learning_rate=0.0001), 
                           loss="sparse_categorical_crossentropy", 
                           metrics=["accuracy"])

        # Retrain the model using only the feedback data
        base_model.fit(feedback_images, feedback_labels, epochs=5, batch_size=16)

        # Save the updated model
        base_model.save(MODEL_PATH)
        model = base_model  # Update the global model
        st.success("Model retrained successfully and saved!")
    else:
        st.warning("No feedback data available for retraining.")



# Adding Edit Profile Feature
def edit_profile():
    st.header("Edit Profile")
    current_phone_number = st.session_state.user_id  # Store the current phone number
    phone_number = st.text_input("Phone Number", value=current_phone_number)  # Editable phone number field
    first_name = st.text_input("First Name", value=st.session_state.first_name)
    last_name = st.text_input("Last Name", value=st.session_state.last_name)
    email = st.text_input("Email", value=st.session_state.email)
    # Handle date_of_birth properly
    date_of_birth = st.date_input("Date of Birth", min_value=pd.Timestamp('1900-01-01'), value=pd.to_datetime(st.session_state.date_of_birth))
    gender = st.selectbox("Gender", ["Male", "Female", "Other"], index=["Male", "Female", "Other"].index(st.session_state.gender))
    address = st.text_area("Address", value=st.session_state.address)

    if st.session_state.user_type == "doctor":
        specialization = st.text_input("Specialization", value=st.session_state.specialization)
        college = st.text_input("College Graduated From", value=st.session_state.college)
        medical_degrees = st.text_area("Medical Degrees", value=st.session_state.medical_degrees)
        about_doctor = st.text_area("About the Doctor", value=st.session_state.about_doctor)
    else:
        specialization = None
        college = None
        medical_degrees = None
        about_doctor = None

    if st.button("Save Changes"):
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        try:
            # Update user data in the database
            cursor.execute('''
                UPDATE user_accounts
                SET phone_number = ?, first_name = ?, last_name = ?, email = ?, date_of_birth = ?, gender = ?, address = ?, specialization = ?, college = ?, medical_degrees = ?, about_doctor = ?
                WHERE phone_number = ?
            ''', (phone_number, first_name, last_name, email, date_of_birth.strftime("%Y-%m-%d"), gender, address, specialization, college, medical_degrees, about_doctor, current_phone_number))

            conn.commit()

            # Update session state with new values
            st.session_state.user_id = phone_number  # Update session state with the new phone number
            st.session_state.first_name = first_name
            st.session_state.last_name = last_name
            st.session_state.email = email
            st.session_state.date_of_birth = date_of_birth
            st.session_state.gender = gender
            st.session_state.address = address

            if st.session_state.user_type == "doctor":
                st.session_state.specialization = specialization
                st.session_state.college = college
                st.session_state.medical_degrees = medical_degrees
                st.session_state.about_doctor = about_doctor

            st.success("Profile updated successfully!")
        except sqlite3.IntegrityError:
            st.error("The phone number already exists. Please choose a different phone number.")
        except Exception as e:
            st.error(f"An error occurred: {e}")
        finally:
            conn.close()

        # Debugging: Log updated session state
        st.write("Session state after profile update:", st.session_state)

# Login Page
def login_page():
    st.title("Login")
    phone_number = st.text_input("Phone Number")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        user = authenticate(phone_number, password)
        if user:
            st.session_state.logged_in = True
            st.session_state.user_type = user[13]
            st.session_state.user_id = user[0]
            st.session_state.first_name = user[1]
            st.session_state.last_name = user[2]
            st.session_state.email = user[4]
            st.session_state.date_of_birth = user[5]
            st.session_state.gender = user[6]
            st.session_state.address = user[7]
            st.session_state.insurance_number = user[8]
            st.session_state.chronic_illnesses = user[9]
            st.session_state.chronic_medications = user[10]
            st.session_state.weight = user[11]
            st.session_state.height = user[12]
            st.session_state.specialization = user[14]
            st.session_state.college = user[15]
            st.session_state.medical_degrees = user[16]
            st.session_state.about_doctor = user[17]
            st.experimental_rerun()
        else:
            st.error("Invalid phone number or password.")



# Signup Page
def signup_page():
    st.title("Signup")
    user_type = st.selectbox("User Type", ["patient", "doctor"])

    phone_number = st.text_input("Phone Number")
    first_name = st.text_input("First Name")
    last_name = st.text_input("Last Name")
    password = st.text_input("Password", type="password")
    email = st.text_input("Email")
    date_of_birth = st.date_input("Date of Birth", min_value=pd.Timestamp('1900-01-01'))
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    address = st.text_area("Address")

    if user_type == "patient":
        insurance_number = st.text_input("Insurance Number (optional)")
        chronic_illnesses = st.text_area("Chronic Illnesses (optional)")
        chronic_medications = st.text_area("Chronic Medications (optional)")
        weight = st.number_input("Weight (kg)", min_value=0.0, format="%.2f")
        height = st.number_input("Height (cm)", min_value=0.0, format="%.2f")

        if st.button("Signup"):
            add_user(phone_number, first_name, last_name, password, email, date_of_birth, gender, address, insurance_number, chronic_illnesses, chronic_medications, weight, height, user_type)
            st.success("Account created successfully! Please log in.")

    elif user_type == "doctor":
        specialization = st.text_input("Specialization")
        college = st.text_input("College Graduated From")
        medical_degrees = st.text_area("Medical Degrees")
        about_doctor = st.text_area("About the Doctor")

        if st.button("Signup"):
            add_user(phone_number, first_name, last_name, password, email, date_of_birth, gender, address, user_type=user_type, specialization=specialization, college=college, medical_degrees=medical_degrees, about_doctor=about_doctor)
            st.success("Account created successfully! Please log in.")
            
# Patient Dashboard Features
def patient_dashboard():
    st.sidebar.title("Patient's Dashboard")
    tabs = st.sidebar.radio(
        "Navigation", ["Profile", "Edit Profile", "Book Appointment", "Health Records", "Search Doctor", "Medications", "Feedback", "Logout", "Ask AI Assistant"], key="patient_tabs"
)

    if tabs == "Profile":
        st.header("My Profile")
        st.write(f"**Name:** {st.session_state.first_name} {st.session_state.last_name}")
        st.write(f"**Phone Number:** {st.session_state.user_id}")
        st.write(f"**Email:** {st.session_state.email}")
        st.write(f"**Date of Birth:** {st.session_state.date_of_birth}")
        st.write(f"**Gender:** {st.session_state.gender}")
        st.write(f"**Address:** {st.session_state.address}")
        st.write(f"**Insurance Number:** {st.session_state.insurance_number if st.session_state.insurance_number else 'N/A'}")
        st.write(f"**Chronic Illnesses:** {st.session_state.chronic_illnesses if st.session_state.chronic_illnesses else 'N/A'}")
        st.write(f"**Chronic Medications:** {st.session_state.chronic_medications if st.session_state.chronic_medications else 'N/A'}")
        st.write(f"**Weight:** {st.session_state.weight if st.session_state.weight else 'N/A'}")
        st.write(f"**Height:** {st.session_state.height if st.session_state.height else 'N/A'}")

    elif tabs == "Edit Profile":
        edit_profile()

    elif tabs == "Book Appointment":
        st.header("Book an Appointment")
    
        # Fetch the list of doctors from the database
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT phone_number, first_name, last_name, specialization
            FROM user_accounts
            WHERE user_type = 'doctor'
        ''')
        doctors = cursor.fetchall()
        conn.close()
    
        # Format doctors for dropdown
        doctor_options = [f"Dr. {doctor[1]} {doctor[2]} ({doctor[3]})" for doctor in doctors]
        doctor_phone_map = {f"Dr. {doctor[1]} {doctor[2]} ({doctor[3]})": doctor[0] for doctor in doctors}
    
        if doctor_options:
            selected_doctor = st.selectbox("Select a Doctor", options=doctor_options)
            appointment_date = st.date_input("Appointment Date")
            appointment_time = st.time_input("Appointment Time")
    
            if st.button("Request Appointment"):
                doctor_phone = doctor_phone_map[selected_doctor]
                patient_phone = st.session_state.user_id  # Logged-in patient
    
                conn = sqlite3.connect(DATABASE_PATH)
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO appointments (doctor_phone, patient_phone, appointment_date, appointment_time)
                    VALUES (?, ?, ?, ?)
                ''', (doctor_phone, patient_phone, str(appointment_date), str(appointment_time)))
                conn.commit()
                conn.close()
    
                st.success(f"Appointment request sent successfully to {selected_doctor} on {appointment_date} at {appointment_time}!")
        else:
            st.warning("No doctors are currently signed up.")


    elif tabs == "Health Records":
        st.header("My Health Records")
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT additional_notes, predicted_class, mri_image
            FROM patient_data
            WHERE user_id = ?
        ''', (st.session_state.user_id,))
        records = cursor.fetchall()
        conn.close()
    
        if records:
            for record in records:
                additional_notes, predicted_class, mri_image_blob = record
                st.write(f"**Notes:** {additional_notes}")
                st.write(f"**Diagnosis:** {predicted_class}")
                if mri_image_blob:
                    st.image(io.BytesIO(mri_image_blob), caption="MRI Image")
                st.write("---")
        else:
            st.warning("No health records found.")

    
    elif tabs == "Search Doctor":
        st.header("Search for Doctors")
        search_query = st.text_input("Search by Doctor Name or Specialization")
        if st.button("Search"):
            conn = sqlite3.connect(DATABASE_PATH)
            cursor = conn.cursor()
            cursor.execute('''
                SELECT first_name, last_name, phone_number, specialization, college, email, about_doctor
                FROM user_accounts
                WHERE user_type = 'doctor' AND (first_name LIKE ? OR last_name LIKE ? OR specialization LIKE ?)
            ''', (f"%{search_query}%", f"%{search_query}%", f"%{search_query}%"))
            doctors = cursor.fetchall()
            conn.close()

            if doctors:
                st.subheader("Search Results:")
                for doctor in doctors:
                    st.write(f"**Name:** Dr. {doctor[0]} {doctor[1]}")
                    st.write(f"**Specialization:** {doctor[3]}")
                    st.write(f"**College:** {doctor[4]}")
                    st.write(f"**Email:** {doctor[5]}")
                    st.write(f"**About:** {doctor[6]}")
                    st.write("---")
            else:
                st.warning("No doctors found matching your search criteria.")

    elif tabs == "Medications":
        st.header("Manage Medications")
        med_name = st.text_input("Medication Name")
        dosage = st.text_input("Dosage (e.g., 1 tablet)")
        frequency = st.text_input("Frequency (e.g., Once a day)")

        if st.button("Add Medication"):
            st.success(f"Medication {med_name} added successfully!")

        st.subheader("Current Medications")
        st.write("- Medication 1: 1 tablet, Once a day")
        st.write("- Medication 2: 2 tablets, Twice a day")

    
    elif tabs == "Classify MRI Image":
        st.header("Classify MRI Image")
        uploaded_file = st.file_uploader("Upload an MRI image", type=["jpg", "jpeg", "png"])
    
        if uploaded_file:
            try:
                # Display the uploaded image
                img = Image.open(uploaded_file).resize((299, 299))
                st.image(uploaded_file, caption="Uploaded MRI Image", use_column_width=True)
    
                if model is None:
                    st.error("Model is not loaded. Please check the model path or initialization.")
                else:
                    # Process the image
                    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
    
                    # Model prediction
                    prediction = model.predict(img_array)
                    predicted_class = CLASS_NAMES[np.argmax(prediction)]
                    st.write(f"**Predicted Class:** {predicted_class}")
    
                    # Allow doctor to correct the prediction
                    corrected_class = st.selectbox(
                        "Correct Prediction (if incorrect)",
                        options=CLASS_NAMES,
                        index=CLASS_NAMES.index(predicted_class),
                        key="corrected_prediction"
                    )
    
                    # Save feedback
                    if st.button("Submit Correction"):
                        # Encode the image to base64
                        buffered = io.BytesIO()
                        img.save(buffered, format="PNG")
                        img_str = base64.b64encode(buffered.getvalue()).decode()
    
                        # Save to CSV
                        feedback_data = {
                            "image": [img_str],
                            "predicted_class": [predicted_class],
                            "corrected_class": [corrected_class],
                        }
                        feedback_df = pd.DataFrame(feedback_data)
    
                        if os.path.exists(FEEDBACK_PATH):
                            feedback_df.to_csv(FEEDBACK_PATH, mode="a", header=False, index=False)
                        else:
                            feedback_df.to_csv(FEEDBACK_PATH, mode="w", header=True, index=False)
    
                        st.success("Correction submitted successfully!")
    
                    # Check feedback size for automatic retraining
                    feedback_images, feedback_labels = load_feedback_data(FEEDBACK_PATH)
                    if feedback_images is not None and len(feedback_images) >= 50:
                        st.write("Retraining model with feedback data...")
    
                        # Retrain the model using only feedback data
                        model = retrain_model_with_feedback(model, feedback_images, feedback_labels)
                    else:
                        feedback_count = len(feedback_images) if feedback_images is not None else 0
                        st.write(f"Feedback count: {feedback_count}. Retraining requires at least 50 corrections.")
            except Exception as e:
                st.error(f"Invalid image uploaded: {e}")




    elif tabs == "Ask AI Assistant":
        chatbot_page()





    elif tabs == "Logout":
        st.session_state.logged_in = False
        st.session_state.user_type = None
        st.session_state.user_id = None
        st.experimental_rerun()


# Doctor Dashboard Features
def doctor_dashboard():
    st.sidebar.title("Doctor's Dashboard")
    tabs = st.sidebar.radio(
       "Navigation", ["My Profile", "Edit Profile", "Patient Records", "Classify MRI Image", "Search Patient", "Manage Appointments", "Update Patient Records", "View Feedback", "Ask AI Assistant", "Logout"], key="doctor_tabs"
)

    if tabs == "My Profile":
        st.header("Doctor Profile")
        st.write(f"**Name:** {st.session_state.first_name} {st.session_state.last_name}")
        st.write(f"**Specialization:** {st.session_state.specialization}")
        st.write(f"**College Graduated From:** {st.session_state.college}")
        st.write(f"**Medical Degrees:** {st.session_state.medical_degrees}")
        st.write(f"**About:** {st.session_state.about_doctor}")

    elif tabs == "Edit Profile":
        edit_profile()

    elif tabs == "Patient Records":
        st.header("All Patient Records")
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT ua.first_name, ua.last_name, pd.additional_notes, pd.predicted_class, pd.mri_image
            FROM user_accounts ua
            JOIN patient_data pd ON ua.phone_number = pd.user_id
            WHERE ua.user_type = 'patient'
        ''')
        patients = cursor.fetchall()
        conn.close()
    
        if patients:
            for patient in patients:
                first_name, last_name, additional_notes, predicted_class, mri_image_blob = patient
                st.write(f"**Name:** {first_name} {last_name}")
                st.write(f"**Notes:** {additional_notes}")
                st.write(f"**Diagnosis:** {predicted_class}")
                if mri_image_blob:
                    st.image(io.BytesIO(mri_image_blob), caption="MRI Image")
                st.write("---")
        else:
            st.warning("No patient records found.")


    elif tabs == "Classify MRI Image":
        st.header("Classify MRI Image")
        uploaded_file = st.file_uploader("Upload an MRI image", type=["jpg", "jpeg", "png"])
        
        if uploaded_file:
            try:
                # Display the uploaded image
                img = Image.open(uploaded_file)
                st.image(img, caption="Uploaded MRI Image", use_column_width=True)
    
                # Save image to temporary file path
                temp_path = "temp_uploaded_image.jpg"
                img.save(temp_path)
    
                if model is None:
                    st.error("Model is not loaded. Please check the model path or initialization.")
                else:
                    # Load and preprocess image from file path
                    img = Image.open(uploaded_file).convert("RGB").resize((299, 299))  # Force RGB for InceptionResNetV2
                    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)  # Shape: (1, 299, 299, 3)

                
                    # Predict with InceptionResNetV2 model
                    prediction = model.predict(img_array)
                    predicted_class = CLASS_NAMES[np.argmax(prediction)]
                    st.write(f"**Predicted Class:** {predicted_class}")


                    # Ask if the prediction is correct
                    is_correct = st.radio(
                        "Is the prediction correct?",
                        options=["Yes", "No"],
                        index=0,  # Default to "Yes"
                        key="prediction_correctness"
                    )
    
                    # If the prediction is incorrect, allow the doctor to submit a correction
                    if is_correct == "No":
                        corrected_class = st.selectbox(
                            "Submit the corrected class:",
                            options=CLASS_NAMES,
                            key="corrected_class_selection"
                        )
                        
                        if st.button("Submit Correction"):
                            # Encode the image to base64
                            buffered = io.BytesIO()
                            img.save(buffered, format="PNG")
                            img_str = base64.b64encode(buffered.getvalue()).decode()
    
                            # Save feedback to CSV
                            feedback_data = {
                                "image": [img_str],
                                "predicted_class": [predicted_class],
                                "corrected_class": [corrected_class],
                            }
                            feedback_df = pd.DataFrame(feedback_data)
    
                            if os.path.exists(FEEDBACK_PATH):
                                feedback_df.to_csv(FEEDBACK_PATH, mode="a", header=False, index=False)
                            else:
                                feedback_df.to_csv(FEEDBACK_PATH, mode="w", header=True, index=False)
    
                            st.success("Correction submitted successfully!")
    
                    else:
                        st.success("Thank you for confirming the prediction!")
    
            except Exception as e:
                st.error(f"Invalid image uploaded: {e}")



    

    elif tabs == "Search Patient":
        st.header("Search Patient Records")
        search_query = st.text_input("Search by Patient Name or Phone Number")
        if st.button("Search"):
            conn = sqlite3.connect(DATABASE_PATH)
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM user_accounts WHERE first_name LIKE ? OR phone_number = ?
            ''', (f"%{search_query}%", search_query))
            patients = cursor.fetchall()
            conn.close()
            if patients:
                for patient in patients:
                    st.write(f"**Name:** {patient[1]} {patient[2]}")
                    st.write(f"**Phone Number:** {patient[0]}")
                    st.write(f"**Email:** {patient[4]}")
                    st.write(f"**Address:** {patient[7]}")
                    st.write(f"**Chronic Illnesses:** {patient[9]}")
                    st.write("---")
            else:
                st.warning("No patient found.")

    elif tabs == "Update Patient Records":
        st.header("Update Patient Health Records")
        patient_phone = st.text_input("Enter Patient's Phone Number")
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        cursor.execute('SELECT first_name, last_name FROM user_accounts WHERE phone_number = ?', (patient_phone,))
        patient = cursor.fetchone()
    
        if patient:
            st.write(f"Updating records for: {patient[0]} {patient[1]}")
            additional_notes = st.text_area("Additional Notes")
            diagnosis = st.text_area("Diagnosis")
            mri_image = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"])
    
            if st.button("Save Record"):
                mri_image_blob = None
                if mri_image:
                    mri_image_blob = mri_image.read()
    
                cursor.execute('''
                    INSERT INTO patient_data (user_id, additional_notes, predicted_class, mri_image)
                    VALUES (?, ?, ?, ?)
                ''', (patient_phone, additional_notes, diagnosis, mri_image_blob))
    
                conn.commit()
                st.success("Patient record updated successfully!")
            conn.close()
        else:
            st.warning("Patient not found.")


    elif tabs == "Manage Appointments":
        st.header("Manage Appointment Requests")
    
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT appointment_id, patient_phone, appointment_date, appointment_time, status
            FROM appointments
            WHERE doctor_phone = ?
        ''', (st.session_state.user_id,))
        appointments = cursor.fetchall()
        conn.close()
    
        if appointments:
            for appointment in appointments:
                appointment_id, patient_phone, appointment_date, appointment_time, status = appointment
                st.write(f"**Patient Phone:** {patient_phone}")
                st.write(f"**Date:** {appointment_date}")
                st.write(f"**Time:** {appointment_time}")
                st.write(f"**Status:** {status}")
                
                new_status = st.radio("Update Status", ["Pending", "Approved", "Rejected"], index=["Pending", "Approved", "Rejected"].index(status), key=f"status_{appointment_id}")
                
                if st.button(f"Update Appointment {appointment_id}", key=f"update_{appointment_id}"):
                    conn = sqlite3.connect(DATABASE_PATH)
                    cursor = conn.cursor()
                    cursor.execute('''
                        UPDATE appointments
                        SET status = ?
                        WHERE appointment_id = ?
                    ''', (new_status, appointment_id))
                    conn.commit()
                    conn.close()
                    st.success(f"Appointment {appointment_id} updated to {new_status}.")
        else:
            st.warning("No appointments found.")

    
    elif tabs == "View Feedback":
        st.header("Patient Feedback")
    
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT phone_number, first_name, last_name FROM user_accounts WHERE user_type = 'patient'
        ''')
        patients = cursor.fetchall()
        conn.close()
    
        if patients:
            for patient in patients:
                patient_phone = patient[0]
                st.subheader(f"Feedback from {patient[1]} {patient[2]}")
    
                # Read feedback from the CSV file
                if os.path.exists(FEEDBACK_PATH):
                    feedback_df = pd.read_csv(FEEDBACK_PATH)
                    
                    # Update this line to match the actual column name in the CSV
                    if 'phone_number' in feedback_df.columns:
                        feedback_for_patient = feedback_df[feedback_df['phone_number'] == patient_phone]
                    else:
                        st.error("The feedback file does not have a 'phone_number' column.")
                        continue
                    
                    if not feedback_for_patient.empty:
                        for _, row in feedback_for_patient.iterrows():
                            st.write(f"**Rating:** {row['rating']}/5")
                            st.write(f"**Feedback:** {row['feedback']}")
                            st.write("---")
                    else:
                        st.write("No feedback provided yet.")
                else:
                    st.warning("No feedback file found.")
        else:
            st.warning("No patients found.")


    
    elif tabs == "Ask AI Assistant":
        chatbot_page()




    elif tabs == "Logout":
        st.session_state.logged_in = False
        st.session_state.user_type = None
        st.session_state.user_id = None
        st.experimental_rerun()




# Main Function
def main():
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
        st.session_state.user_type = None
        st.session_state.user_id = None

    if not st.session_state.logged_in:
        menu = st.sidebar.selectbox("Menu", ["Login", "Signup"])
        if menu == "Login":
            login_page()
        elif menu == "Signup":
            signup_page()
    else:
        if st.session_state.user_type == "doctor":
            doctor_dashboard()
        elif st.session_state.user_type == "patient":
            patient_dashboard()

if __name__ == "__main__":
    main()