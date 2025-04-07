import streamlit as st
import pymongo
import pandas as pd
from datetime import datetime
import uuid
import cv2
from PIL import Image
import numpy as np
import tempfile
import time
import json


from pymongo import MongoClient


MONGO_URI = st.secrets["MONGO_URI"]

if not MONGO_URI:
    st.error("MongoDB connection string not found! Check your Streamlit Secrets.")
    st.stop()

# MongoDB Connection
client = MongoClient(MONGO_URI)




# MongoDB Connection
client = MongoClient(MONGO_URI)

# # MongoDB Configuration

DB_NAME = "quality"
COLLECTION_NAME = "samples"

client = pymongo.MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]




def detect_qr_from_image(image):
    qr_detector = cv2.QRCodeDetector()
    img = np.array(image.convert('RGB'))
    img_cv = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    data, bbox, _ = qr_detector.detectAndDecode(img_cv)
    return data


st.subheader("Scan QR Code for RM Inward Details")

if "qr_scanned" not in st.session_state:
    st.session_state.qr_scanned = False
if "rm_details" not in st.session_state:
    st.session_state.rm_details = {}

scan_method = st.radio("Choose QR Input Method", ["📷 Webcam", "🖼️ Upload Image"])

def scan_qr_from_camera():
    image_data = st.camera_input("Take a photo of the QR code")
    if image_data is not None:
        image = Image.open(image_data)
        qr_data = detect_qr_from_image(image)
        return qr_data
    return None

# Inside your QR scanning section
if scan_method == "📷 Webcam" and not st.session_state.qr_scanned:
    scanned_qr = scan_qr_from_camera()
    if scanned_qr:
        try:
            st.session_state.rm_details = json.loads(scanned_qr)
            st.session_state.qr_scanned = True
            st.success("✅ Scan successful! RM details loaded.")
        except Exception as e:
            st.error(f"❌ Failed to parse QR: {e}")
    else:
        st.info("📷 Please take a photo of the QR code above to scan.")

elif scan_method == "🖼️ Upload Image" and not st.session_state.qr_scanned:
    uploaded_file = st.file_uploader("Upload QR Code Image", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        qr_data = detect_qr_from_image(image)
        if qr_data:
            try:
                st.session_state.rm_details = json.loads(qr_data)
                st.session_state.qr_scanned = True
                st.success("✅ Scan successful! RM details loaded.")
            except Exception as e:
                st.error(f"❌ Failed to parse QR: {e}")
        else:
            st.warning("❌ No QR Code detected in the image.")

# Show and allow editing of scanned RM details
rm_details = st.session_state.rm_details
rm_lot_id = st.text_input("RM LOT ID", value=rm_details.get("rm_lot_id", ""))

# rm_lot_id = st.text_input("RM LOT ID", value=rm_details.get("rm_lot_id", ""))
rm_date = st.date_input("Date", value=pd.to_datetime(rm_details.get("date", datetime.now().date())))
rm_quality = st.selectbox("RM Quality", ["High", "Medium", "Low"], index=["High", "Medium", "Low"].index(rm_details.get("quality", "High")))
# spec_size = st.text_input("Spec Size", value=rm_details.get("spec_size", ""))
# supplier = st.text_input("Supplier", value=rm_details.get("supplier", ""))   
spec_size = st.text_input("Spec Size", value=rm_details.get("spec_size", ""))
supplier = st.text_input("Supplier", value=rm_details.get("supplier", ""))
     
# Field Names
field_names = ["OK", "Melanin", "Colour", "Deformation", "Skin Wound", "Blood", "Blood Spot"]

# Function to create an empty grid
def create_grid(rows=20, cols=7):
    return [[False] * cols for _ in range(rows)]

# Function to compute cumulative stats
def compute_stats(grid_data):
    total_selected = sum(sum(row) for row in grid_data)
    per_field_counts = [sum(col) for col in zip(*grid_data)]
    return total_selected, per_field_counts

# Streamlit UI
st.title("RM Quality Check List")

# Initialize session state
grid_sets = st.session_state.get("grid_sets", [create_grid()])
show_modal = st.session_state.get("show_modal", False)

# Compute overall stats
total_bubbles = 0
field_wise_counts = [0] * 7
for grid in grid_sets:
    total, field_counts = compute_stats(grid)
    total_bubbles += total
    field_wise_counts = [sum(x) for x in zip(field_wise_counts, field_counts)]

# Sidebar - Cumulative stats
st.sidebar.subheader("Quality Stats")
st.sidebar.metric(label="Total Checked Parameters", value=total_bubbles)
for i, count in enumerate(field_wise_counts):
    st.sidebar.metric(label=f"{field_names[i]} Selected", value=count)

# Display grids in tabs with highlighted active tab
tabs = st.tabs([f"Grid Set {i+1}" for i in range(len(grid_sets))])

sample_counter = 1  # Serial sample counter

for grid_index, tab in enumerate(tabs):
    with tab:
        st.markdown(f"""
            <style>
            div[role='tablist'] > div:nth-child({grid_index+1}) {{ 
                background-color: #FFDD57 !important; 
            }}
            .stCheckbox > label > div:first-child {{
                width: 30px;
                height: 30px;
                border-radius: 50%;
                border: 2px solid #000;
            }}
            </style>
        """, unsafe_allow_html=True)

        st.subheader(f"Grid Set {grid_index + 1}")

        header_cols = st.columns(8)
        header_cols[0].write("**Sample**")
        for col_idx, name in enumerate(field_names):
            header_cols[col_idx + 1].write(f"**{name}**")

        for row_idx in range(20):
            cols = st.columns(8)  # Extra column for serial numbers
            cols[0].write(f"{sample_counter}")
            sample_counter += 1
            for col_idx in range(7):
                with cols[col_idx + 1]:
                    grid_sets[grid_index][row_idx][col_idx] = st.checkbox(" ", value=grid_sets[grid_index][row_idx][col_idx], key=f"g{grid_index}_r{row_idx}_c{col_idx}")

# Store updated grids in session
st.session_state["grid_sets"] = grid_sets

# Option to add more grid sets (up to 7 more)
if len(grid_sets) < 7:
    if st.button("Add More Samples"):
        grid_sets.append(create_grid())
        st.session_state["grid_sets"] = grid_sets
        st.rerun()

# Final submission with confirmation modal
if st.button("Submit Data"):
    st.session_state["show_modal"] = True
    st.rerun()

if st.session_state.get("show_modal", False):
    with st.expander("Confirm Submission", expanded=True):
        st.warning("Are you sure you want to submit the RM Quality Data?")
        if st.button("Yes, Submit"):
            timestamp = datetime.now().isoformat()
            sample_id = str(uuid.uuid4())
            marked_samples = []

            for grid_index, grid in enumerate(grid_sets):
                for row_idx, row in enumerate(grid):
                    marked_fields = [field_names[col_idx] for col_idx, checked in enumerate(row) if checked]
                    if marked_fields:
                        marked_samples.append({
                            "grid_set": grid_index + 1,
                            "sample_number": row_idx + 1,
                            "marked_fields": marked_fields
                        })

            data_to_store = {
                "sample_id": sample_id,
                "timestamp": timestamp,
                "marked_samples": marked_samples,
                "total_bubbles": total_bubbles,
                "field_wise_counts": field_wise_counts,
                # Additional RM details
                "rm_lot_id": rm_lot_id,
                "rm_date": rm_date.isoformat(),
                "rm_quality": rm_quality,
                "spec_size": spec_size,
                "supplier": supplier




            }
            collection.insert_one(data_to_store)
            st.success(f"RM Quality Data saved successfully! Sample ID: {sample_id}")
            st.session_state["grid_sets"] = [create_grid()]
            st.session_state["show_modal"] = False
            st.rerun()
        if st.button("Cancel"):
            st.session_state["show_modal"] = False
            st.rerun()

# Download Submitted Data
st.subheader("Download Submitted Samples Report")
if st.button("Generate Report"):
    all_data = list(collection.find({}, {"_id": 0}))
    if all_data:
        df = pd.DataFrame(all_data)
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="submitted_samples_report.csv",
            mime="text/csv"
        )
    else:
        st.error("No data available for download.")
