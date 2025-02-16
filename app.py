import os
import time
import json
import datetime
import sqlite3
import base64
import io
import streamlit as st
import torch
import torch.nn as nn
import plotly.express as px
from PIL import Image

# Transformers: for BLIP
from transformers import BlipProcessor, BlipForConditionalGeneration

####################################
# PAGE CONFIG & THEME
####################################
st.set_page_config(page_title="NutriVision", layout="wide")

STARFIELD_CSS = """
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
/* Gradient background */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #000000, #121212) !important;
}
/* Header and Sidebar */
header, [data-testid="stSidebar"] {
    background-color: #1F1F1F !important;
}
/* Text styling */
body, .stMarkdown, .stMetric, .css-1aumxhk {
    color: #E8EAF6 !important;
}
/* Larger titles & text */
h1, .app-title {
    font-size: 2.5rem !important;
    text-align: center;
    color: #FFF !important;
    margin-top: 10px;
}
h2, .app-subtitle {
    font-size: 1.75rem !important;
    color: #E8EAF6 !important;
}
h3 {
    font-size: 1.4rem !important;
    color: #E8EAF6 !important;
}
/* Tabs styling - advanced look */
.stTabs > div[role="tablist"] {
    background-color: #263238 !important;
    border-bottom: 2px solid #000000 !important;
    display: flex;
    justify-content: center;
}
.stTabs div[role="tab"] {
    font-size: 18px;
    font-weight: bold;
    margin: 0 10px;
    color: #E8EAF6 !important;
    padding: 10px;
    transition: background-color 0.3s ease;
    border-radius: 8px 8px 0 0;
}
.stTabs div[role="tab"]:hover {
    background-color: #37474F !important;
    cursor: pointer;
}
/* Buttons */
.stButton>button {
    background-color: #E53935 !important;
    color: #FFF !important;
    font-weight: bold;
    border: none;
    padding: 10px 20px;
    border-radius: 5px;
    transition: background-color 0.3s ease;
}
.stButton>button:hover {
    background-color: #EF5350 !important;
}
/* Inputs */
.stNumberInput input, .stTextInput input, .stSelectbox select {
    background-color: #263238 !important;
    color: #E8EAF6 !important;
    border: 1px solid #37474F !important;
}
/* File uploader & camera input */
.stFileUploader, [data-testid="stCameraInput"] {
    background-color: #263238 !important;
    color: #E8EAF6 !important;
    border: 1px solid #37474F !important;
}
/* Smooth transitions for images */
.upload-image img {
    max-height: 50vh;
    width: auto;
    display: block;
    margin: 0 auto;
    transition: opacity 0.5s ease-in-out;
}
.history-image img {
    max-height: 20vh;
    width: auto;
    display: block;
    margin: 0 auto;
    transition: opacity 0.5s ease-in-out;
}
</style>
<div id="starfield"></div>
"""
st.markdown(STARFIELD_CSS, unsafe_allow_html=True)

####################################
# LOADING SCREEN ONCE
####################################
if "loaded_once" not in st.session_state:
    placeholder = st.empty()
    placeholder.markdown("""
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
    """, unsafe_allow_html=True)
    time.sleep(2)
    placeholder.empty()
    st.session_state.loaded_once = True

####################################
# DATABASE SETUP
####################################
@st.cache_resource
def init_db():
    conn = sqlite3.connect("nutrivision_app.db", check_same_thread=False)
    c = conn.cursor()

    c.execute("PRAGMA table_info(users)")
    user_cols= [col[1] for col in c.fetchall()]
    needed_user= ["id","username","password","height","weight","age","gender","profile_pic"]
    if sorted(user_cols)!= sorted(needed_user):
        c.execute("DROP TABLE IF EXISTS users")

    c.execute("PRAGMA table_info(meals)")
    meal_cols= [col[1] for col in c.fetchall()]
    needed_meals= ["id","user_id","meal_time","source","caption","predicted","calories","meal_image"]
    if sorted(meal_cols)!= sorted(needed_meals):
        c.execute("DROP TABLE IF EXISTS meals")

    c.execute("""
    CREATE TABLE IF NOT EXISTS users(
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
    CREATE TABLE IF NOT EXISTS meals(
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      user_id INTEGER,
      meal_time TIMESTAMP,
      source TEXT,
      caption TEXT,
      predicted TEXT,
      calories REAL,
      meal_image TEXT,
      FOREIGN KEY(user_id) REFERENCES users(id)
    )
    """)
    conn.commit()
    return conn

conn= init_db()

def register_user(username, pw, h, w, a, g, ppic):
    c= conn.cursor()
    try:
        c.execute("""
        INSERT INTO users(username,password,height,weight,age,gender,profile_pic)
        VALUES(?,?,?,?,?,?,?)
        """,(username,pw,h,w,a,g,ppic))
        conn.commit()
        c.execute("SELECT id FROM users WHERE username=?",(username,))
        row= c.fetchone()
        return True,"Registration successful!", row[0]
    except sqlite3.IntegrityError:
        return False,"Username already exists.",None

def login_user(username,pw):
    c= conn.cursor()
    c.execute("SELECT id FROM users WHERE username=? AND password=?",(username,pw))
    row= c.fetchone()
    return row[0] if row else None

def store_meal(uid, src, caption, items_list, cals, path):
    c= conn.cursor()
    c.execute("""
    INSERT INTO meals(user_id,meal_time,source,caption,predicted,calories,meal_image)
    VALUES(?,?,?,?,?,?,?)
    """,(uid,datetime.datetime.now(),src,caption,json.dumps(items_list),cals,path))
    conn.commit()

def get_meal_history(uid):
    c= conn.cursor()
    c.execute("""
    SELECT meal_time,source,caption,predicted,calories,meal_image
    FROM meals
    WHERE user_id=? ORDER BY meal_time DESC
    """,(uid,))
    return c.fetchall()

def get_daily_cals(uid,date):
    c= conn.cursor()
    start= datetime.datetime.combine(date, datetime.time.min)
    end= datetime.datetime.combine(date, datetime.time.max)
    c.execute("""
    SELECT SUM(calories)
    FROM meals
    WHERE user_id=? AND meal_time BETWEEN ? AND ?
    """,(uid,start,end))
    val= c.fetchone()[0]
    return val if val else 0

def get_all_daily_cals(uid):
    c= conn.cursor()
    c.execute("""
    SELECT DATE(meal_time), SUM(calories)
    FROM meals
    WHERE user_id=?
    GROUP BY DATE(meal_time)
    ORDER BY DATE(meal_time)
    """,(uid,))
    return c.fetchall()

def save_uploaded_file(upf,folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
    path= os.path.join(folder, upf.name)
    with open(path,"wb") as f:
        f.write(upf.getbuffer())
    return path

####################################
# LOAD NUTRITION FROM CSV (Food Nutrition, Fruits/Veg, Fastfood)
####################################
import pandas as pd

def safe_convert_num(x):
    try:
        return float(x)
    except:
        return None

@st.cache_resource
def load_nutrition_db(nutrition_dir, fruits_csv, vegetables_csv, fastfood_csv):
    # 1) FOOD: from "FINAL FOOD DATASET" => FOOD-DATA-GROUP*.csv
    # We'll read "Caloric Value" and store as a dictionary
    food_dict= {}
    for fn in os.listdir(nutrition_dir):
        if fn.startswith("FOOD-DATA-GROUP") and fn.endswith(".csv"):
            fp= os.path.join(nutrition_dir, fn)
            df= pd.read_csv(fp)
            # must have columns "food" or "item" + "Caloric Value"
            if "food" in df.columns:
                df["FoodName"]= df["food"].astype(str).str.lower().str.strip()
            elif "item" in df.columns:
                df["FoodName"]= df["item"].astype(str).str.lower().str.strip()
            else:
                continue
            if "Caloric Value" not in df.columns:
                continue
            grouped= df.groupby("FoodName")["Caloric Value"].mean().reset_index()
            for _,row in grouped.iterrows():
                nm= str(row["FoodName"]).strip()
                val= safe_convert_num(row["Caloric Value"])
                if val is not None:
                    food_dict[nm]= float(val)
    # 2) FRUITS & VEGETABLES => read "energy (kcal/kJ)"
    # fruits.csv + vegetables.csv
    fv_dict= {}
    fdf= pd.read_csv(fruits_csv)
    vdf= pd.read_csv(vegetables_csv)
    fdf["Source"]= "fv"
    vdf["Source"]= "fv"
    cdf= pd.concat([fdf, vdf], ignore_index=True)
    if "name" in cdf.columns and "energy (kcal/kJ)" in cdf.columns:
        cdf["FoodName"]= cdf["name"].astype(str).str.lower().str.strip()
        cdf["energy (kcal/kJ)"]= cdf["energy (kcal/kJ)"].apply(safe_convert_num)
        grouped2= cdf.groupby("FoodName")["energy (kcal/kJ)"].mean().reset_index()
        for _,row in grouped2.iterrows():
            nm= row["FoodName"]
            val= row["energy (kcal/kJ)"]
            if val is not None:
                fv_dict[nm]= float(val)
    # 3) FASTFOOD => fastfood.csv => "item" + "calories"
    fast_dict= {}
    fcsv= pd.read_csv(fastfood_csv)
    if "item" in fcsv.columns and "calories" in fcsv.columns:
        fcsv["FoodName"]= fcsv["item"].astype(str).str.lower().str.strip()
        grouped3= fcsv.groupby("FoodName")["calories"].mean().reset_index()
        for _,row in grouped3.iterrows():
            nm= row["FoodName"]
            val= safe_convert_num(row["calories"])
            if val is not None:
                fast_dict[nm]= float(val)

    # unify them
    # if name exists in multiple => we can choose e.g. average or priority
    # let's do 'food_dict' -> 'fv_dict' -> 'fast_dict' priority
    unified= {}
    for k,v in fast_dict.items():
        unified[k]= v
    for k,v in fv_dict.items():
        if k not in unified:
            unified[k]= v
    for k,v in food_dict.items():
        if k not in unified:
            unified[k]= v
    return unified  # name-> calorie

####################################
# BLIP CAPTIONER
####################################
class BlipCaptioner(nn.Module):
    def __init__(self, device="cuda"):
        super().__init__()
        self.device= device
        self.processor= BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model= BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model.eval().to(device)
        for p in self.model.parameters():
            p.requires_grad=False

    def forward(self, pil_image):
        inputs= self.processor(images=pil_image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out_ids= self.model.generate(**inputs, max_length=50, num_beams=5)
        caption= self.processor.decode(out_ids[0], skip_special_tokens=True)
        return caption

@st.cache_resource
def load_captioner():
    dev= "cuda" if torch.cuda.is_available() else "cpu"
    return BlipCaptioner(dev), dev

####################################
# SESSION STATE
####################################
if "logged_in" not in st.session_state:  st.session_state.logged_in= False
if "user_id" not in st.session_state:    st.session_state.user_id= None
if "username" not in st.session_state:   st.session_state.username= ""
if "user_info" not in st.session_state:  st.session_state.user_info= {}
if "preferred_diet" not in st.session_state: st.session_state.preferred_diet= "Not specified"

####################################
# LOAD NUTRITION DB
####################################
nutrition_dir= "FINAL FOOD DATASET"
fruits_path= "fruits.csv"
vegetables_path= "vegetables.csv"
fastfood_path= "fastfood.csv"

@st.cache_resource
def load_unified_cal_db():
    return load_nutrition_db(nutrition_dir, fruits_path, vegetables_path, fastfood_path)

unified_cal_db= load_unified_cal_db()

####################################
# LOGIN / REGISTER
####################################
def login_form():
    st.markdown("<h2 class='app-subtitle'>Login</h2>",unsafe_allow_html=True)
    u= st.text_input("Username", key="login_u")
    p= st.text_input("Password", type="password", key="login_p")
    if st.button("Log In"):
        uid= login_user(u,p)
        if uid:
            st.session_state.logged_in= True
            st.session_state.user_id= uid
            st.session_state.username= u
            c= conn.cursor()
            c.execute("SELECT height, weight, age, gender, profile_pic FROM users WHERE id=?",(uid,))
            row= c.fetchone()
            st.session_state.user_info= {
              "height":row[0],
              "weight":row[1],
              "age":row[2],
              "gender":row[3],
              "profile_pic":row[4]
            }
            st.session_state.preferred_diet= "Not specified"
            st.success("Logged in!")
        else:
            st.error("Invalid credentials.")

def register_form():
    st.markdown("<h2 class='app-subtitle'>Register</h2>",unsafe_allow_html=True)
    ru= st.text_input("Username", key="reg_u")
    rp= st.text_input("Password", type="password", key="reg_p")
    rh= st.number_input("Height (cm) [Optional]",0.0,300.0,step=0.1)
    rw= st.number_input("Weight (kg) [Optional]",0.0,300.0,step=0.1)
    ra= st.number_input("Age [Optional]",0,120,step=1)
    rg= st.selectbox("Gender [Optional]",["","Male","Female","Other"],key="reg_g")
    rpd= st.text_input("Preferred Diet [Optional]",key="reg_pd")
    rpic= st.file_uploader("Profile Picture [Optional]", type=["jpg","jpeg","png"], key="reg_pic")

    if st.button("Register"):
        pic_path= ""
        if rpic:
            pic_path= save_uploaded_file(rpic,"profile_pics")
        if ru=="" or rp=="":
            st.error("Username & Password required!")
        else:
            succ,msg, new_uid= register_user(
              ru, rp,
              rh if rh>0 else None,
              rw if rw>0 else None,
              ra if ra>0 else None,
              rg if rg else None,
              pic_path
            )
            if succ:
                st.success(msg)
                st.session_state.logged_in= True
                st.session_state.user_id= new_uid
                st.session_state.username= ru
                st.session_state.user_info= {
                  "height": rh if rh>0 else None,
                  "weight": rw if rw>0 else None,
                  "age": ra if ra>0 else None,
                  "gender": rg if rg else None,
                  "profile_pic": pic_path
                }
                st.session_state.preferred_diet= rpd if rpd else "Not specified"
                st.success("Registered & logged in!")
            else:
                st.error(msg)

if not st.session_state.logged_in:
    st.markdown("<h1 class='app-title'>NutriVision</h1>",unsafe_allow_html=True)
    st.write("Pick your action")
    choice= st.radio("Login or Register", ["Login","Register"], horizontal=True)
    if choice=="Login":
        login_form()
    else:
        register_form()
    st.stop()

####################################
# MAIN TABS
####################################
tabs= st.tabs(["Home","Upload Meal","Meal History","Account","Logout"])

####################################
# TAB 0: HOME
####################################
with tabs[0]:
    st.markdown("<h1 class='app-title'>Dashboard</h1>",unsafe_allow_html=True)
    st.write(f"Hello, **{st.session_state.username}**!")
    tdy= datetime.date.today()
    day_cal= get_daily_cals(st.session_state.user_id, tdy)
    st.metric("Today's Calorie Intake",f"{day_cal:.2f} kcal")

    if st.button("Refresh Dashboard"):
        pass

    all_data= get_all_daily_cals(st.session_state.user_id)
    if all_data:
        dd= pd.DataFrame(all_data, columns=["Date","Calories"])
        fig= px.line(dd, x="Date", y="Calories", title="Daily Calorie Intake", markers=True)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("No meals so far.")

####################################
# TAB 1: UPLOAD
####################################
def parse_caption_for_dbitems(caption, db_map):
    """
    We'll parse quantity + item from caption tokens.
    If item in db_map => use db_map[item], else skip
    Return list of (item, qty, calsEach, subTotal)
    """
    words= caption.lower().split()
    number_map= {
      "one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,"eight":8,
      "nine":9,"ten":10,"eleven":11,"twelve":12,"thirteen":13,"fourteen":14,
      "fifteen":15,"twenty":20,"thirty":30,"forty":40,"fifty":50
    }
    results=[]
    i=0
    while i< len(words):
        word= words[i].strip(",.!?")
        qty=1
        if word.isdigit():
            qty= int(word)
            i+=1
            if i< len(words):
                item= words[i].strip(",.!?")
                i+=1
            else:
                break
        elif word in number_map:
            qty= number_map[word]
            i+=1
            if i< len(words):
                item= words[i].strip(",.!?")
                i+=1
            else:
                break
        else:
            # treat as item => qty=1
            item= word
            i+=1
        # check if item in db_map
        if item in db_map:
            cals= db_map[item]
            sub= qty*cals
            results.append((item, qty, cals, sub))
        else:
            # skip
            pass
    return results

with tabs[1]:
    st.markdown("<h1 class='app-title'>Upload a Meal</h1>",unsafe_allow_html=True)
    st.write("We'll generate a BLIP caption, parse recognized foods from the 3 CSV databases. Items not in DB are skipped.")
    
    # Load BLIP
    captioner, dev= load_captioner()

    cat_map= {"Food Nutrition":"food_nutrition","Fruits & Vegetables":"fv","Fast Food":"fastfood"}
    s_cat= st.selectbox("Meal Category", list(cat_map.keys()))
    source= cat_map[s_cat]

    # let user pick file or camera
    st.write("Select image source:")
    pic_choice= st.radio("", ["Upload","Camera"], horizontal=True)
    meal_file= None
    if pic_choice=="Upload":
        meal_file= st.file_uploader("Pick an image", type=["jpg","jpeg","png"])
    else:
        meal_file= st.camera_input("Take a meal photo")

    if meal_file:
        try:
            meal_img= Image.open(meal_file).convert("RGB")
            buff= io.BytesIO()
            meal_img.save(buff,format="PNG")
            b64= base64.b64encode(buff.getvalue()).decode()
            st.markdown(f"<div class='upload-image'><img src='data:image/png;base64,{b64}'/></div>",unsafe_allow_html=True)
        except:
            st.error("Error reading image data.")

        if st.button("Analyze Meal"):
            with st.spinner("Generating caption with BLIP..."):
                caption= captioner(meal_img)
            st.success("Caption done!")
            st.write("**Caption**:", caption)

            # parse items => skip unknown
            items_list= parse_caption_for_dbitems(caption, unified_cal_db)
            total_cals= 0.0
            row_data=[]
            for (itm, qty, calsEach, subT) in items_list:
                total_cals+= subT
                row_data.append((itm, qty, calsEach, subT))
            if row_data:
                pdf= pd.DataFrame(row_data, columns=["Item","Qty","Cals/Item","Subtotal"])
                st.table(pdf)
                st.write("**Total Calories**:", total_cals)
            else:
                st.write("No recognized items from the CSVs => total = 0")
            
            # store in DB
            up_path= save_uploaded_file(meal_file,"meal_images")
            # We'll store item details as json
            item_json= []
            for r in row_data:
                item_json.append({"item":r[0],"qty":r[1],"calsEach":r[2],"subtotal":r[3]})
            store_meal(st.session_state.user_id, source, caption, item_json, total_cals, up_path)
            st.success("Meal recorded successfully!")

####################################
# TAB 2: MEAL HISTORY
####################################
with tabs[2]:
    st.markdown("<h1 class='app-title'>Meal History</h1>",unsafe_allow_html=True)
    if st.session_state.user_id is None:
        st.write("No user. Please login again.")
    else:
        hist= get_meal_history(st.session_state.user_id)
        if hist:
            for rec in hist:
                meal_time, meal_src, meal_cap, meal_pred, meal_cals, meal_im= rec
                st.write(f"**Time**: {meal_time}, **Category**: {meal_src}")
                if meal_im and os.path.exists(meal_im):
                    i= Image.open(meal_im)
                    bf= io.BytesIO()
                    i.save(bf, format="PNG")
                    bb= base64.b64encode(bf.getvalue()).decode()
                    st.markdown(f"<div class='history-image'><img src='data:image/png;base64,{bb}'/></div>",unsafe_allow_html=True)
                st.write(f"**Caption**: {meal_cap}")
                st.write(f"**Calories**: {meal_cals:.2f}")
                try:
                    arr= json.loads(meal_pred)
                    st.write("**Items**:", arr)
                except:
                    st.write("Raw preds:", meal_pred)
                st.markdown("---")
        else:
            st.write("No meals so far.")

        dd= get_all_daily_cals(st.session_state.user_id)
        if dd:
            import pandas as pd
            ddf= pd.DataFrame(dd, columns=["Date","Calories"])
            fig= px.bar(ddf, x="Date", y="Calories", title="Daily Calorie Intake")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("No daily data available.")

####################################
# TAB 3: ACCOUNT
####################################
with tabs[3]:
    st.markdown("<h1 class='app-title'>Account Info</h1>",unsafe_allow_html=True)
    st.write(f"**Username**: {st.session_state.username}")
    inf= st.session_state.user_info
    if inf:
        st.write(f"**Height**: {inf.get('height','N/A')} cm")
        st.write(f"**Weight**: {inf.get('weight','N/A')} kg")
        st.write(f"**Age**: {inf.get('age','N/A')}")
        st.write(f"**Gender**: {inf.get('gender','N/A')}")
        if inf.get('height') and inf.get('weight') and inf['height']>0:
            bmi= inf['weight']/((inf['height']/100)**2)
            st.write(f"**BMI**: {bmi:.2f}")
        st.write(f"**Preferred Diet**: {st.session_state.preferred_diet}")
        pic= inf.get("profile_pic","")
        if pic and os.path.exists(pic):
            with open(pic,"rb") as f:
                enc= base64.b64encode(f.read()).decode()
            ext= os.path.splitext(pic)[1].lower()
            mime= "image/png" if ext==".png" else "image/jpeg"
            st.markdown(
                f"<div style='text-align:left;'><img src='data:{mime};base64,{enc}' style='max-height:40vh;'/></div>",
                unsafe_allow_html=True
            )
        else:
            st.write("No profile picture.")
    else:
        st.write("No user info.")
    st.markdown("---")
    st.subheader("Update Profile")
    new_ht= st.number_input("Height (cm)",0.0,300.0, step=0.1, value=float(inf.get('height') or 0))
    new_wt= st.number_input("Weight (kg)",0.0,300.0, step=0.1, value=float(inf.get('weight') or 0))
    new_ag= st.number_input("Age",0,120, step=1, value=int(inf.get('age') or 0))
    new_gd= st.selectbox("Gender",["","Male","Female","Other"], index=0 if not inf.get('gender') else ["","Male","Female","Other"].index(inf['gender']))
    new_pd= st.text_input("Preferred Diet", st.session_state.preferred_diet)
    upf= st.file_uploader("Update Profile Pic", type=["jpg","jpeg","png"], key="upd_pic")
    if st.button("Save Profile"):
        c= conn.cursor()
        c.execute("""
        UPDATE users SET height=?, weight=?, age=?, gender=? WHERE id=?
        """,(new_ht if new_ht>0 else None,
             new_wt if new_wt>0 else None,
             new_ag if new_ag>0 else None,
             new_gd if new_gd else None,
             st.session_state.user_id))
        conn.commit()
        new_path= inf.get('profile_pic','')
        if upf:
            new_path= save_uploaded_file(upf,"profile_pics")
            c.execute("UPDATE users SET profile_pic=? WHERE id=?",(new_path, st.session_state.user_id))
            conn.commit()
        st.session_state.user_info['height']= new_ht if new_ht>0 else None
        st.session_state.user_info['weight']= new_wt if new_wt>0 else None
        st.session_state.user_info['age']= new_ag if new_ag>0 else None
        st.session_state.user_info['gender']= new_gd if new_gd else None
        st.session_state.preferred_diet= new_pd if new_pd else "Not specified"
        if new_path:
            st.session_state.user_info['profile_pic']= new_path
        st.success("Profile updated successfully!")

####################################
# TAB 4: LOGOUT
####################################
with tabs[4]:
    st.markdown("<h1 class='app-title'>Logout</h1>",unsafe_allow_html=True)
    if st.button("Confirm Logout"):
        st.session_state.logged_in= False
        st.session_state.user_id= None
        st.session_state.username= ""
        st.session_state.user_info= {}
        st.session_state.preferred_diet= "Not specified"
        # Clear query params if any
        st.query_params= {}
        st.success("You have been logged out.")
        st.info("Go to the Login/Register or refresh.")
        st.button("Ok")