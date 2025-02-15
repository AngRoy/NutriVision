import os
import sqlite3
import datetime
import json
import time
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, BlipProcessor, BlipForConditionalGeneration
import plotly.express as px
import numpy as np

# Helper: Rerun Function
# This injects JavaScript to reload the page twice in quick succession.
def jsrerun():
    st.markdown("""
        <script>
        setTimeout(function(){
        window.location.reload();
        }, 100);
        </script>
        """, unsafe_allow_html=True)
    
query_params = st.query_params
if "user_id" in query_params and query_params["user_id"]:
    st.session_state.logged_in = True
    st.session_state.user_id = int(query_params["user_id"][0])
    st.session_state.username = query_params["username"][0] if "username" in query_params else ""

st.markdown(
    """
    <style>
    /* Starfield background */
    #starfield {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: url('https://i.imgur.com/9z4xUO7.png') repeat;
        animation: moveStars 120s linear infinite;
        z-index: -2;
        opacity: 0.3;
    }
    @keyframes moveStars {
        from { background-position: 0 0; }
        to { background-position: -10000px 5000px; }
    }
    /* Main app container */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #000000, #121212) !important;
    }
    /* Header styling */
    header {
        background-color: #1F1F1F !important;
    }
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #1F1F1F !important;
    }
    /* General text styling */
    body, .stMarkdown, .stMetric, .css-1aumxhk {
        color: #E8EAF6 !important;
    }
    /* Tabs styling */
    .stTabs > div[role="tablist"] {
        background-color: #263238 !important;
        border-bottom: 2px solid #000000 !important;
    }
    .stTabs div[role="tab"] {
        font-size: 18px;
        font-weight: bold;
        color: #E8EAF6 !important;
        padding: 10px !important;
        transition: background-color 0.3s ease !important;
    }
    .stTabs div[role="tab"]:hover {
        background-color: #37474F !important;
    }
    /* Button styling - using bright red */
    .stButton>button {
        background-color: #E53935 !important;
        color: #FFFFFF !important;
        font-weight: bold !important;
        border: none !important;
        padding: 10px 20px !important;
        border-radius: 5px !important;
        transition: background-color 0.3s ease !important;
    }
    .stButton>button:hover {
        background-color: #EF5350 !important;
    }
    /* Input styling */
    .stNumberInput input, .stTextInput input, .stSelectbox select {
        background-color: #263238 !important;
        color: #E8EAF6 !important;
        border: 1px solid #37474F !important;
    }
    /* File uploader styling */
    .stFileUploader {
        background-color: #263238 !important;
        color: #E8EAF6 !important;
        border: 1px solid #37474F !important;
    }
    </style>
    <div id="starfield"></div>
    """,
    unsafe_allow_html=True,
)

if "app_loaded" not in st.session_state:
    placeholder = st.empty()
    placeholder.markdown(
        """
        <div class="loader"></div>
        <style>
        .loader {
          border: 16px solid #444;
          border-top: 16px solid #E53935;
          border-radius: 50%;
          width: 120px;
          height: 120px;
          animation: spin 2s linear infinite;
          position: fixed;
          top: 50%;
          left: 50%;
          transform: translate(-50%, -50%);
          z-index: 9999;
        }
        @keyframes spin {
          0% { transform: translate(-50%, -50%) rotate(0deg); }
          100% { transform: translate(-50%, -50%) rotate(360deg); }
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    time.sleep(2)
    st.session_state.app_loaded = True
    placeholder.empty()

# -------------------------------
# DATABASE SETUP & FUNCTIONS
# -------------------------------
@st.cache_resource(show_spinner=False)
def init_db():
    conn = sqlite3.connect("nutrivision_app.db", check_same_thread=False)
    c = conn.cursor()
    c.execute("PRAGMA table_info(users)")
    user_columns = [col[1] for col in c.fetchall()]
    if not user_columns or "height" not in user_columns or "profile_pic" not in user_columns:
        c.execute("DROP TABLE IF EXISTS users")
    c.execute("PRAGMA table_info(meals)")
    meal_columns = [col[1] for col in c.fetchall()]
    if not meal_columns or "meal_image" not in meal_columns:
        c.execute("DROP TABLE IF EXISTS meals")
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password TEXT,
            height REAL,
            weight REAL,
            age INTEGER,
            gender TEXT,
            profile_pic TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS meals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            meal_time TIMESTAMP,
            source TEXT,
            caption TEXT,
            predicted TEXT,
            calories REAL,
            meal_image TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)
    conn.commit()
    return conn

conn = init_db()

def register_user(username, password, height, weight, age, gender, profile_pic_path):
    try:
        c = conn.cursor()
        c.execute("INSERT INTO users (username, password, height, weight, age, gender, profile_pic) VALUES (?, ?, ?, ?, ?, ?, ?)", 
                  (username, password, height, weight, age, gender, profile_pic_path))
        conn.commit()
        c.execute("SELECT id FROM users WHERE username=?", (username,))
        user_id = c.fetchone()[0]
        return True, "User registered successfully.", user_id
    except sqlite3.IntegrityError:
        return False, "Username already exists.", None

def login_user(username, password):
    c = conn.cursor()
    c.execute("SELECT id FROM users WHERE username=? AND password=?", (username, password))
    result = c.fetchone()
    return result[0] if result else None

def store_meal(user_id, source, caption, predicted, calories, meal_image_path):
    c = conn.cursor()
    c.execute("INSERT INTO meals (user_id, meal_time, source, caption, predicted, calories, meal_image) VALUES (?, ?, ?, ?, ?, ?, ?)",
              (user_id, datetime.datetime.now(), source, caption, json.dumps(predicted), calories, meal_image_path))
    conn.commit()

def get_meal_history(user_id):
    c = conn.cursor()
    c.execute("SELECT meal_time, source, caption, predicted, calories, meal_image FROM meals WHERE user_id=? ORDER BY meal_time DESC", (user_id,))
    return c.fetchall()

def get_daily_calories(user_id, date):
    c = conn.cursor()
    start = datetime.datetime.combine(date, datetime.time.min)
    end = datetime.datetime.combine(date, datetime.time.max)
    c.execute("SELECT SUM(calories) FROM meals WHERE user_id=? AND meal_time BETWEEN ? AND ?", (user_id, start, end))
    result = c.fetchone()[0]
    return result if result else 0

def get_all_daily_calories(user_id):
    c = conn.cursor()
    c.execute("SELECT DATE(meal_time), SUM(calories) FROM meals WHERE user_id=? GROUP BY DATE(meal_time) ORDER BY DATE(meal_time)", (user_id,))
    return c.fetchall()

# -------------------------------
# HELPER: SAVE UPLOADED FILE
# -------------------------------
def save_uploaded_file(uploaded_file, folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
    file_path = os.path.join(folder, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

# -------------------------------
# MODEL DEFINITION: NutriVisionNetMultiHead
# -------------------------------
class NutriVisionNetMultiHead(nn.Module):
    def __init__(self, food_nutrition_dim, fv_dim, fastfood_dim, device="cuda", fine_tune_clip_image=False):
        super(NutriVisionNetMultiHead, self).__init__()
        self.device = device
        
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        for param in self.clip_model.text_model.parameters():
            param.requires_grad = False
        if not fine_tune_clip_image:
            for param in self.clip_model.vision_model.parameters():
                param.requires_grad = False
        
        self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        for param in self.blip_model.parameters():
            param.requires_grad = False
        
        fusion_dim = 1024
        hidden_dim = 512
        self.food_nutrition_head = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, food_nutrition_dim)
        )
        self.fv_head = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, fv_dim)
        )
        self.fastfood_head = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, fastfood_dim)
        )
        
    def forward(self, image, source):
        blip_inputs = self.blip_processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output_ids = self.blip_model.generate(**blip_inputs, max_length=50, num_beams=5)
        caption = self.blip_processor.decode(output_ids[0], skip_special_tokens=True)
        
        clip_image_inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)
        image_embeds = self.clip_model.get_image_features(**clip_image_inputs)
        
        clip_text_inputs = self.clip_processor(text=[caption], return_tensors="pt", padding=True).to(self.device)
        text_embeds = self.clip_model.get_text_features(**clip_text_inputs)
        
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
        fused = torch.cat([image_embeds, text_embeds], dim=-1)
        
        if source == "food_nutrition":
            pred = self.food_nutrition_head(fused)
        elif source == "fv":
            pred = self.fv_head(fused)
        elif source == "fastfood":
            pred = self.fastfood_head(fused)
        else:
            raise ValueError(f"Unknown source: {source}")
        return pred, caption

@st.cache_resource(show_spinner=False)
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    food_nutrition_dim = 3
    fv_dim = 9
    fastfood_dim = 8
    model = NutriVisionNetMultiHead(food_nutrition_dim, fv_dim, fastfood_dim, device=device, fine_tune_clip_image=True)
    model.to(device)
    checkpoint_path = "nutrivision_multihand.pt"
    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
    model.eval()
    return model, device

def infer_meal(image, source, model, device):
    with torch.no_grad():
        pred, caption = model(image, source)
    pred_list = pred.squeeze(0).cpu().numpy().tolist()
    calories = pred_list[0] if pred_list else 0
    return pred_list, caption, calories

# -------------------------------
# SESSION STATE INITIALIZATION
# -------------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "username" not in st.session_state:
    st.session_state.username = ""
if "user_info" not in st.session_state:
    st.session_state.user_info = {}
if "preferred_diet" not in st.session_state:
    st.session_state.preferred_diet = "Not specified"

# -------------------------------
# USER AUTHENTICATION (Login/Registration)
# Only show authentication forms if not logged in.
# -------------------------------
def login_form():
    st.subheader("Login")
    username = st.text_input("Username", key="login_username")
    password = st.text_input("Password", type="password", key="login_password")
    if st.button("Login", key="login_button"):
        user_id = login_user(username, password)
        if user_id:
            st.session_state.logged_in = True
            st.session_state.user_id = user_id
            st.session_state.username = username
            c = conn.cursor()
            c.execute("SELECT height, weight, age, gender, profile_pic FROM users WHERE id=?", (user_id,))
            info = c.fetchone()
            st.session_state.user_info = {
                "height": info[0],
                "weight": info[1],
                "age": info[2],
                "gender": info[3],
                "profile_pic": info[4]
            }
            st.session_state.preferred_diet = "Not specified"
            st.success("Logged in successfully!")
            st.button("Continue")
        else:
            st.error("Invalid username or password.")

def registration_form():
    st.subheader("Register")
    reg_username = st.text_input("Username", key="reg_username")
    reg_password = st.text_input("Password", type="password", key="reg_password")
    reg_height = st.number_input("Height (cm) [Optional]", min_value=0.0, max_value=300.0, step=0.1, key="reg_height", value=0.0)
    reg_weight = st.number_input("Weight (kg) [Optional]", min_value=0.0, max_value=300.0, step=0.1, key="reg_weight", value=0.0)
    reg_age = st.number_input("Age [Optional]", min_value=0, max_value=120, step=1, key="reg_age", value=0)
    reg_gender = st.selectbox("Gender [Optional]", ["", "Male", "Female", "Other"], key="reg_gender")
    reg_preferred_diet = st.text_input("Preferred Diet [Optional]", key="reg_preferred_diet")
    reg_profile = st.file_uploader("Upload Profile Picture [Optional]", type=["jpg", "jpeg", "png"], key="reg_profile")
    if st.button("Register", key="reg_button"):
        profile_pic_path = ""
        if reg_profile is not None:
            profile_pic_path = save_uploaded_file(reg_profile, "profile_pics")
        if reg_username == "" or reg_password == "":
            st.error("Username and password are required!")
        else:
            success, msg, new_user_id = register_user(
                reg_username, reg_password,
                float(reg_height) if reg_height > 0 else None,
                float(reg_weight) if reg_weight > 0 else None,
                int(reg_age) if reg_age > 0 else None,
                reg_gender if reg_gender != "" else None,
                profile_pic_path
            )
            if success:
                st.success(msg)
                st.session_state.logged_in = True
                st.session_state.user_id = new_user_id
                st.session_state.username = reg_username
                st.session_state.user_info = {
                    "height": float(reg_height) if reg_height > 0 else None,
                    "weight": float(reg_weight) if reg_weight > 0 else None,
                    "age": int(reg_age) if reg_age > 0 else None,
                    "gender": reg_gender if reg_gender != "" else None,
                    "profile_pic": profile_pic_path
                }
                st.session_state.preferred_diet = reg_preferred_diet if reg_preferred_diet != "" else "Not specified"
                st.query_params = {"user_id": [str(new_user_id)], "username": [reg_username]}
                jsrerun()
                st.button("Continue")
            else:
                st.error(msg)

if not st.session_state.logged_in:
    auth_choice = st.radio("Select Option", ["Login", "Register"], horizontal=True)
    if auth_choice == "Login":
        login_form()
    else:
        registration_form()
    st.stop()

# -------------------------------
# NAVIGATION TABS (only shown after login)
# -------------------------------
tabs = st.tabs(["Home", "Upload Meal", "Meal History", "Account", "Logout"])

# -------------------------------
# TAB 0: Home Dashboard
# -------------------------------
with tabs[0]:
    st.header("Dashboard")
    st.write(f"Welcome, {st.session_state.username}!")
    today = datetime.date.today()
    daily_cal = get_daily_calories(st.session_state.user_id, today)
    st.metric("Today's Calorie Intake", f"{daily_cal:.2f} kcal")
    
    daily_data = get_all_daily_calories(st.session_state.user_id)
    if daily_data:
        try:
            import pandas as pd
            df = pd.DataFrame(daily_data, columns=["Date", "Calories"])
            fig = px.line(df, x="Date", y="Calories", title="Daily Calorie Intake", markers=True)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.write("Error generating graph:", e)
    else:
        st.write("No meal records to display.")

# -------------------------------
# TAB 1: Upload Meal
# -------------------------------
with tabs[1]:
    st.header("Upload a Meal")
    st.write("Upload an image of your meal to get its estimated nutritional details.")
    source_options = {"Food Nutrition": "food_nutrition", "Fruits & Vegetables": "fv", "Fast Food": "fastfood"}
    selected_source_label = st.selectbox("Select Nutrition Category", list(source_options.keys()))
    selected_source = source_options[selected_source_label]
    
    uploaded_meal = st.file_uploader("Choose a meal image (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"], key="meal_upload")
    if uploaded_meal is not None:
        try:
            meal_image = Image.open(uploaded_meal).convert("RGB")
            st.image(meal_image, caption="Uploaded Meal Image", use_container_width=True)
        except Exception as e:
            st.error(f"Error loading image: {e}")
        
        if st.button("Analyze Meal", key="analyze_meal"):
            with st.spinner("⏳ Running inference... Please wait..."):
                model, device = load_model()
                pred_list, caption, calories = infer_meal(meal_image, selected_source, model, device)
            st.success("Inference completed!")
            st.markdown("### Inference Results")
            st.write("**Generated Caption:**", caption)
            st.write("**Predicted Nutritional Values:**")
            if selected_source == "food_nutrition":
                cols = ["Caloric Value", "Fat", "Carbohydrates"]
            elif selected_source == "fv":
                cols = ["energy (kcal/kJ)", "water (g)", "protein (g)", "total fat (g)", "carbohydrates (g)", "fiber (g)", "sugars (g)", "calcium (mg)", "iron (mg)"]
            elif selected_source == "fastfood":
                cols = ["calories", "cal_fat", "total_fat", "sat_fat", "trans_fat", "cholesterol", "sodium", "total_carb"]
            result_dict = {col: [round(val, 2)] for col, val in zip(cols, pred_list)}
            st.table(result_dict)
            st.write("**Predicted Calories:**", f"{calories:.2f} kcal")
            meal_img_path = save_uploaded_file(uploaded_meal, "meal_images")
            store_meal(st.session_state.user_id, selected_source, caption, pred_list, calories, meal_img_path)
            st.success("Meal recorded successfully!")

# -------------------------------
# TAB 2: Meal History
# -------------------------------
with tabs[2]:
    st.header("Meal History")
    history = get_meal_history(st.session_state.user_id)
    if history:
        for meal in history:
            meal_time, source, caption, predicted, calories, meal_img = meal
            st.markdown(f"**Time:** {meal_time}")
            st.markdown(f"**Category:** {source}")
            if meal_img and os.path.exists(meal_img):
                st.image(meal_img, width=250)
            st.markdown(f"**Caption:** {caption}")
            st.markdown(f"**Predicted Calories:** {calories:.2f} kcal")
            try:
                pred_vals = json.loads(predicted)
                if source == "food_nutrition":
                    nutrient_list = ["Caloric Value", "Fat", "Carbohydrates"]
                elif source == "fv":
                    nutrient_list = ["energy (kcal/kJ)", "water (g)", "protein (g)", "total fat (g)", "carbohydrates (g)", "fiber (g)", "sugars (g)", "calcium (mg)", "iron (mg)"]
                elif source == "fastfood":
                    nutrient_list = ["calories", "cal_fat", "total_fat", "sat_fat", "trans_fat", "cholesterol", "sodium", "total_carb"]
                else:
                    nutrient_list = []
                st.markdown("**Nutritional Details:**")
                st.table({
                    "Nutrient": nutrient_list,
                    "Value": [round(val, 2) for val in pred_vals]
                })
            except Exception as e:
                st.write("Predicted:", predicted)
            st.markdown("---")
    else:
        st.write("No meals recorded yet.")
    
    daily_data = get_all_daily_calories(st.session_state.user_id)
    if daily_data:
        try:
            import pandas as pd
            df = pd.DataFrame(daily_data, columns=["Date", "Calories"])
            fig = px.bar(df, x="Date", y="Calories", title="Daily Calorie Intake")
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.write("Error generating graph:", e)

# -------------------------------
# TAB 3: Account
# -------------------------------
with tabs[3]:
    st.header("Account Information")
    st.markdown(f"**Username:** {st.session_state.username}")
    if st.session_state.user_info:
        height = st.session_state.user_info.get('height', None)
        weight = st.session_state.user_info.get('weight', None)
        age = st.session_state.user_info.get('age', None)
        gender = st.session_state.user_info.get('gender', None)
        st.markdown(f"**Height:** {height if height is not None else 'N/A'} cm")
        st.markdown(f"**Weight:** {weight if weight is not None else 'N/A'} kg")
        st.markdown(f"**Age:** {age if age is not None else 'N/A'}")
        st.markdown(f"**Gender:** {gender if gender is not None else 'N/A'}")
        if height and weight and float(height) > 0:
            bmi = float(weight) / ((float(height)/100)**2)
            st.markdown(f"**Body Mass Index (BMI):** {bmi:.2f}")
        else:
            st.markdown("**Body Mass Index (BMI):** N/A")
        st.markdown(f"**Preferred Diet:** {st.session_state.preferred_diet}")
        profile_pic = st.session_state.user_info.get('profile_pic', '')
        if profile_pic and os.path.exists(profile_pic):
            st.image(profile_pic, width=200)
        else:
            st.write("No profile picture.")
    else:
        st.write("No user info available.")
    
    st.markdown("---")
    st.subheader("Update Profile Information")
    new_height = st.number_input("Height (cm)", min_value=0.0, max_value=300.0, step=0.1, key="upd_height", value=float(st.session_state.user_info.get('height') or 0))
    new_weight = st.number_input("Weight (kg)", min_value=0.0, max_value=300.0, step=0.1, key="upd_weight", value=float(st.session_state.user_info.get('weight') or 0))
    new_age = st.number_input("Age", min_value=0, max_value=120, step=1, key="upd_age", value=int(st.session_state.user_info.get('age') or 0))
    new_gender = st.selectbox("Gender", ["", "Male", "Female", "Other"], key="upd_gender", index=0 if not st.session_state.user_info.get('gender') else ["", "Male", "Female", "Other"].index(st.session_state.user_info.get('gender')))
    new_preferred_diet = st.text_input("Preferred Diet", key="upd_preferred_diet", value=st.session_state.preferred_diet)
    new_profile = st.file_uploader("Update Profile Picture", type=["jpg", "jpeg", "png"], key="upd_profile")
    if st.button("Update Profile"):
        c = conn.cursor()
        c.execute("UPDATE users SET height=?, weight=?, age=?, gender=? WHERE id=?",
                  (new_height if new_height > 0 else None, new_weight if new_weight > 0 else None, new_age if new_age > 0 else None, new_gender if new_gender != "" else None, st.session_state.user_id))
        conn.commit()
        profile_pic_path = st.session_state.user_info.get('profile_pic', '')
        if new_profile is not None:
            profile_pic_path = save_uploaded_file(new_profile, "profile_pics")
            c.execute("UPDATE users SET profile_pic=? WHERE id=?", (profile_pic_path, st.session_state.user_id))
            conn.commit()
        st.session_state.user_info['height'] = new_height if new_height > 0 else None
        st.session_state.user_info['weight'] = new_weight if new_weight > 0 else None
        st.session_state.user_info['age'] = new_age if new_age > 0 else None
        st.session_state.user_info['gender'] = new_gender if new_gender != "" else None
        st.session_state.preferred_diet = new_preferred_diet if new_preferred_diet != "" else "Not specified"
        if profile_pic_path:
            st.session_state.user_info['profile_pic'] = profile_pic_path
        st.success("Profile updated successfully!")
        jsrerun()

# -------------------------------
# TAB 4: Logout
# -------------------------------
with tabs[4]:
    st.header("Logout")
    if st.button("Confirm Logout"):
        st.session_state.logged_in = False
        st.session_state.user_id = None
        st.session_state.username = ""
        st.session_state.user_info = {}
        st.session_state.preferred_diet = "Not specified"
        # Clear query parameters (if needed)
        st.query_params = {}
        jsrerun()
        st.button("Confirm")