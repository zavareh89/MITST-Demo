import json
import os
import pickle
import traceback
import copy
from tqdm import tqdm

import numpy as np
import pandas as pd

from feature_generation_utilities import get_categories_mappings


import streamlit as st


# Define a reusable function for handling offsets
def handle_offset_input(unique_id, record_key):
    """
    Handles offset input validation and processing.

    Args:
        unique_id (int): The unique ID of the record.
        record_key (str): The key for the record in session state.

    Returns:
        int or None: The validated offset value, or None if invalid.
    """
    offset_box = st.text_input(
        f"Offset #{unique_id} (integer)",
        key=f"{record_key}_offset_{unique_id}",
    )
    try:
        offset_val = int(offset_box) if offset_box else None
    except ValueError:
        offset_val = None

    # if offset_box and (not offset_box.isdigit()):
    #     st.warning("Offset must be an integer.")

    return offset_val


@st.cache_data
def load_examples():
    with open("examples.json", "r") as f:
        return json.load(f)


EXAMPLES = load_examples()


# Load model and cache
@st.cache_resource
def load_model():
    from models import EHRTransformer

    model_path = os.path.join(".", "weights_epoch31_16_32_444_888_888_seq.pth")
    shared_dim = 32
    n_classes = 3
    tf_n_heads = (8, 8, 8)
    tf_dim_head = (8, 8, 8)
    tf_depths = (4, 4, 4)
    n_classes = 3

    model = EHRTransformer(
        shared_dim=shared_dim,
        tf_n_heads=tf_n_heads,
        tf_depths=tf_depths,
        tf_dim_head=tf_dim_head,
        n_classes=n_classes,
    )
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    return model


# %% normalization functions
def grouped_mean_std_scaler(x, groups, scaler):
    # null vlues are replaced with mean (i.e. 0)
    output = [(v - scaler[k][0]) / scaler[k][1] if k else 0 for k, v in zip(groups, x)]
    return np.array(output, dtype=np.float32).reshape(-1, 1)


def mean_std_scaler(x, scaler):
    if isinstance(x, int) or isinstance(x, float):
        return (x - scaler[0]) / scaler[1]
    x = (x.reshape(-1, 1) - scaler[0]) / scaler[1]
    return x


# 2. Inference function
def feautre_generator(session_state, categories_to_int, scaler):
    """
    Preprocesses session state and returns classification label.
    """

    # All (even missing) sources are available in session_state. Missing sources are empty lists or dictionaries.
    features = []
    sources, offsets = {}, {}
    # vital sign (excluding temperature)
    for k in (
        "HR",
        "RR",
        "SpO2",
        "ibp_mean",
        "ibp_diastolic",
        "ibp_systolic",
        "nibp_mean",
        "nibp_diastolic",
        "nibp_systolic",
    ):
        values = torch.zeros((1, 1), dtype=torch.float32)
        offsets[k] = torch.zeros(1, dtype=torch.int32)
        if session_state[k]["values"]:
            values = np.array(session_state[k]["values"], dtype=np.float32)
            values = mean_std_scaler(values, scaler[k])
            values = torch.tensor(values, dtype=torch.float32)
            offsets[k] = torch.tensor(session_state[k]["offsets"], dtype=torch.int32)
        sources[k] = {
            "categorical": None,
            "continuous": values,
        }

    # temperature, lab, infusion, and IO with 1 categorical and 1 numerical variable
    keys = ("Temp", "lab", "infusion", "IO")
    s_names = ("location", "labname", "drugname", "IOname")  # session_state keys
    s_values = ("value", "labresult", "dosage", "IOvalue")  # session_state values
    cat_mapping_keys = ("temp_location", "labname", "infusion_drug", "IO_cell_label")
    for k, s_name, s_value, cat_key in zip(keys, s_names, s_values, cat_mapping_keys):
        continuous = torch.zeros((1, 1), dtype=torch.float32)
        categorical = -1 * torch.ones((1, 1), dtype=torch.int32)
        offsets[k] = torch.zeros(1, dtype=torch.int32)
        L = len(session_state[k])
        if L > 0:
            continuous = [session_state[k][i][s_value] for i in range(L)]
            categorical = [session_state[k][i][s_name] for i in range(L)]
            offsets[k] = torch.tensor(
                [session_state[k][i]["offset"] for i in range(L)], dtype=torch.int32
            )
            continuous = grouped_mean_std_scaler(
                continuous, categorical, scaler[k]
            )  # scale values by s_names
            continuous = torch.tensor(continuous, dtype=torch.float32)
            categorical = [categories_to_int[cat_key][c] for c in categorical]
            categorical = torch.tensor(categorical, dtype=torch.int32).unsqueeze(1)
        sources[k] = {
            "categorical": categorical,
            "continuous": continuous,
        }

    # scores, past_history, treatment and addx (all with a single categorical variables)
    for k in ("GCS", "sedation", "past_history", "treatment", "addx"):
        categorical = -1 * torch.ones((1, 1), dtype=torch.int32)
        offsets[k] = torch.zeros(1, dtype=torch.int32)
        L = len(session_state[k])
        if L > 0:
            if k in ("GCS", "sedation"):
                categorical = [session_state[k][i]["score"] for i in range(L)]
                categorical = [categories_to_int[f"{k}_scores"][c] for c in categorical]
            else:
                categorical = [session_state[k][i][k] for i in range(L)]
                categorical = [categories_to_int[k][c] for c in categorical]
            categorical = torch.tensor(categorical, dtype=torch.int32).unsqueeze(1)
            offsets[k] = torch.tensor(
                [session_state[k][i]["offset"] for i in range(L)], dtype=torch.int32
            )
        sources[k] = {
            "categorical": categorical,
            "continuous": None,
        }

    # diagnosis (with two categorical variables)
    k = "diagnosis"
    categorical = -1 * torch.ones((1, 2), dtype=torch.int32)
    offsets[k] = torch.zeros(1, dtype=torch.int32)
    L = len(session_state[k])
    if L > 0:
        categorical = [
            [
                categories_to_int["diagnosis"][session_state[k][i]["diagnosis"]],
                categories_to_int["diagnosispriority"][
                    session_state[k][i]["diagnosispriority"]
                ],
            ]
            for i in range(L)
        ]
        categorical = torch.tensor(categorical, dtype=torch.int32)
        offsets[k] = torch.tensor(
            [session_state[k][i]["offset"] for i in range(L)], dtype=torch.int32
        )
    sources[k] = {
        "categorical": categorical,
        "continuous": None,
    }

    # medication
    k = "med"
    key = "Medications"
    continuous = torch.zeros((1, 1), dtype=torch.float32)
    categorical = -1 * torch.ones((1, 5), dtype=torch.int32)
    offsets[k] = torch.zeros(1, dtype=torch.int32)
    L = len(session_state[key])
    if L > 0:
        continuous = [session_state[key][i]["dosage"] for i in range(L)]
        categorical = [
            [
                categories_to_int["medication"][session_state[key][i]["drugname"]],
                categories_to_int["frequency"][session_state[key][i]["frequency"]],
                categories_to_int["route_admin"][session_state[key][i]["routeadmin"]],
                int(session_state[key][i]["prn"] == "True"),
                int(session_state[key][i]["drugivadmixture"] == "True"),
            ]
            for i in range(L)
        ]
        drugnames = [session_state[key][i]["drugname"] for i in range(L)]
        continuous = grouped_mean_std_scaler(continuous, drugnames, scaler["med"])
        continuous = torch.tensor(continuous, dtype=torch.float32)
        categorical = torch.tensor(categorical, dtype=torch.int32)
        offsets[k] = torch.tensor(
            [session_state[key][i]["offset"] for i in range(L)], dtype=torch.int32
        )

    sources[k] = {
        "categorical": categorical,
        "continuous": continuous,
    }

    # IO_total
    k = "IO_num_reg"
    values = torch.zeros((1, 4), dtype=torch.float32)
    offsets[k] = torch.zeros(1, dtype=torch.int32)
    L = len(session_state["IO_total"]["num_registrations"]["values"])
    if L > 0:
        values = np.zeros((L, 4), dtype=np.float32)
        for i, key in enumerate(("num_registrations", "intake", "output", "dialysis")):
            v = np.array(session_state["IO_total"][key]["values"], dtype=np.float32)
            values[:, i] = mean_std_scaler(v, scaler[key])
        values = torch.tensor(values, dtype=torch.float32)

        offsets[k] = torch.tensor(
            session_state["IO_total"][key]["offsets"], dtype=torch.int32
        )
    sources[k] = {
        "categorical": None,
        "continuous": values,
    }

    # unit_info
    k = "unit_info"
    categorical = -1 * torch.ones((1, 3), dtype=torch.int32)
    offsets[k] = torch.zeros(1, dtype=torch.int32)
    L = len(session_state[k])
    if L > 0:
        categorical = [
            [
                categories_to_int["unittype"][session_state[k][i]["unittype"]],
                categories_to_int["unitstaytype"][session_state[k][i]["unitstaytype"]],
                categories_to_int["admitsource"][
                    session_state[k][i]["unitadmitsource"]
                ],
            ]
            for i in range(L)
        ]
        categorical = torch.tensor(categorical, dtype=torch.int32)
        offsets[k] = torch.tensor(
            [session_state[k][i]["offset"] for i in range(L)], dtype=torch.int32
        )
    sources[k] = {
        "categorical": categorical,
        "continuous": None,
    }

    # static
    k = "static"
    offsets[k] = torch.zeros(1, dtype=torch.int32)
    static = session_state[k]
    continuous = np.zeros((1, 3), dtype=np.float32)
    categorical = np.zeros((1, 7), dtype=np.int32)

    for i, key in enumerate(("weight", "age", "height")):
        v = mean_std_scaler(static[key], scaler[key])
        continuous[0, i] = v
    for i, key in enumerate(
        ("gender", "ethnicity", "hospital_id", "num_beds", "region", "admitsource")
    ):
        categorical[0, i] = categories_to_int[key][static[key]]
    categorical[0, -1] = int(
        static["teachingstatus"] == "True"
    )  # convert boolean to int
    continuous = torch.tensor(continuous, dtype=torch.float32)
    categorical = torch.tensor(categorical, dtype=torch.int32)
    sources[k] = {
        "categorical": categorical,
        "continuous": continuous,
    }
    return sources, offsets


def classify_patient(session_state, categories_to_int, scaler):

    try:
        sources, offsets = feautre_generator(session_state, categories_to_int, scaler)
        # make it like a btach with batch size=1
        batch_size = 1
        seq_len_per_source = {}
        mask = {}  # True for masked values
        for k in sources.keys():
            seq_len_per_source[k] = torch.tensor([len(offsets[k])])
            mask[k] = torch.zeros(
                batch_size, seq_len_per_source[k].item(), dtype=torch.bool
            )

        model = load_model()
        with torch.no_grad():
            source_data = (sources, offsets)
            logits = model(source_data, mask, seq_len_per_source)
            predictions = logits.softmax(dim=-1).squeeze(0)  # Remove batch dimension
            predictions = predictions.numpy()
            if predictions[0] > 0.372:  # euglycemia threshold
                pred_class = 0
            elif predictions[1] > 0.311:  # hypoglycemia threshold
                pred_class = 1
            elif predictions[2] > 0.420:  # hyperglycemia threshold
                pred_class = 2
            else:
                pred_class = 0
        # Map prediction to class label
        label_map = {0: "euglycemia", 1: "hypoglycemia", 2: "hyperglycemia"}
        return label_map.get(pred_class, "Unknown"), predictions

    except Exception as e:
        st.error(f"Error in classification: {str(e)}")
        st.text("Detailed error:")
        st.text(traceback.format_exc())
        return "Error"


# load scaler and categorical mappings
scaler_path = os.path.join(".", "scaler.pkl")
if os.path.exists(scaler_path):
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

categories_to_int, int_to_categories = get_categories_mappings()

# sort categorical variables
gender = sorted(categories_to_int["gender"])
ethnicity = categories_to_int["ethnicity"]
num_beds = sorted(categories_to_int["num_beds"], key=lambda x: (x is None, x))
hospital_ids = sorted(categories_to_int["hospital_id"])
region = sorted(categories_to_int["region"], key=lambda x: (x is None, x))
admitsource = sorted(categories_to_int["admitsource"])
unittype = sorted(categories_to_int["unittype"])
unitstaytype = sorted(categories_to_int["unitstaytype"])
temp_location = sorted(categories_to_int["temp_location"])
GCS_scores = sorted(
    categories_to_int["GCS_scores"],
    key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else float("inf")),
)
sedation_scores = sorted(categories_to_int["sedation_scores"])
drugnames = sorted(categories_to_int["medication"])
freqs = sorted(categories_to_int["frequency"])
routeadmin = sorted(categories_to_int["route_admin"])
infusion_drugnames = sorted(categories_to_int["infusion_drug"])
labnames = sorted(categories_to_int["labname"])
IOnames = sorted(categories_to_int["IO_cell_label"])
past_history = sorted(categories_to_int["past_history"])
treatments = sorted(categories_to_int["treatment"])
diagnosis = sorted(categories_to_int["diagnosis"])
diagnosispriority = sorted(categories_to_int["diagnosispriority"])
addx = sorted(categories_to_int["addx"])

# compute limits for drug dosages, lab results and IO values
drug_dosage_limits = {}
valid_drugnames = []
for dname in drugnames:
    if dname in scaler["med"]:
        m, std = scaler["med"][dname]
        min_val = round(m - 10 * std, 3)
        # min_val = max(min_val, 0.0)  # Ensure minimum is non-negative
        max_val = round(m + 10 * std, 3)
        drug_dosage_limits[dname] = (min_val, max_val)
        valid_drugnames.append(dname)

drugnames = valid_drugnames

inf_dosage_limits = {}
for dname in infusion_drugnames:
    m, std = scaler["infusion"][dname]
    min_val = round(m - 10 * std, 3)
    # min_val = max(min_val, 0.0)  # Ensure minimum is non-negative
    max_val = round(m + 10 * std, 3)
    inf_dosage_limits[dname] = (min_val, max_val)

labresult_limits = {}
for lname in labnames:
    m, std = scaler["lab"][lname]
    min_val = round(m - 10 * std, 3)
    # min_val = max(min_val, 0.0)  # Ensure minimum is non-negative
    max_val = round(m + 10 * std, 3)
    labresult_limits[lname] = (min_val, max_val)

IO_limits = {}
for IOname in IOnames:
    m, std = scaler["IO"][IOname]
    min_val = round(m - 10 * std, 3)
    # min_val = max(min_val, 0.0)  # Ensure minimum is non-negative
    max_val = round(m + 10 * std, 3)
    IO_limits[IOname] = (min_val, max_val)


# Define numerical limits for IO_total values based on the database
IO_numerical = {}
IO_numerical["num_registrations"] = (0, 18)
IO_numerical["intake"] = (-700, 10000)
IO_numerical["output"] = (-5000, 3000)
IO_numerical["dialysis"] = (-40000, 25000)

# Example config for three source groups
SOURCE_GROUPS = {
    "Vital Signs": {
        "HR": (30, 200),
        "RR": (1, 60),
        "SpO2": (83, 110),
        "ibp_mean": (10, 150),
        "ibp_diastolic": (10, 110),
        "ibp_systolic": (20, 220),
        "nibp_mean": (10, 150),
        "nibp_diastolic": (10, 110),
        "nibp_systolic": (20, 220),
        "Temp": {"value": (33.0, 42.0), "location": temp_location},
    },
    "Medications": {
        "med": {
            "drugname": drugnames,
            "frequency": freqs,
            "routeadmin": routeadmin,
            "prn": ["True", "False"],
            "drugivadmixture": ["True", "False"],
        },
        "infusion": {"drugname": infusion_drugnames},
    },
    "Lab": {"labname": labnames},
    "Scores": {  # Level of Consciousness (LOC) Assessment
        "GCS": GCS_scores,
        "sedation": sedation_scores,
    },
    "IO": {"IO": IOnames, "IO_total": IO_numerical},
    "Past history and treatment": {
        "past_history": past_history,
        "treatment": treatments,
    },
    "Diagnosis": {
        "diagnosis": {"diagnosis": diagnosis, "diagnosispriority": diagnosispriority},
        "addx": addx,
    },
    "Demographics and hospital info": {
        "weight": (0, 200),  # kg
        "height": (0, 250),  # cm
        "age": (0, 90),  # years
        "gender": gender,
        "ethnicity": ethnicity,
        "num_beds": num_beds,
        "hospital_id": hospital_ids,
        "region": region,
        "hospital_admitsource": admitsource,
        "teachingstatus": ["True", "False"],
        "unittype": unittype,
        "unitstaytype": unitstaytype,
        "unitadmitsource": admitsource,
    },
}


# Session state setup
for vs in SOURCE_GROUPS["Vital Signs"].keys():
    if vs == "Temp":
        continue
    if vs not in st.session_state:
        st.session_state[vs] = {}
if "Temp" not in st.session_state:
    st.session_state["Temp"] = []
if "IO_total" not in st.session_state:
    st.session_state["IO_total"] = {}
if "Medications" not in st.session_state:
    st.session_state["Medications"] = []
if "infusion" not in st.session_state:
    st.session_state["infusion"] = []
if "lab" not in st.session_state:
    st.session_state["lab"] = []
if "GCS" not in st.session_state:
    st.session_state["GCS"] = []
if "sedation" not in st.session_state:
    st.session_state["sedation"] = []
if "IO" not in st.session_state:
    st.session_state["IO"] = []
if "past_history" not in st.session_state:
    st.session_state["past_history"] = []
if "treatment" not in st.session_state:
    st.session_state["treatment"] = []
if "diagnosis" not in st.session_state:
    st.session_state["diagnosis"] = []
if "addx" not in st.session_state:
    st.session_state["addx"] = []
if "unit_info" not in st.session_state:
    st.session_state["unit_info"] = []

# add counters
if "Temp_id_counter" not in st.session_state:
    st.session_state["Temp_id_counter"] = 0
if "Medications_id_counter" not in st.session_state:
    st.session_state["Medications_id_counter"] = 0
if "infusion_id_counter" not in st.session_state:
    st.session_state["infusion_id_counter"] = 0
if "lab_id_counter" not in st.session_state:
    st.session_state["lab_id_counter"] = 0
if "GCS_id_counter" not in st.session_state:
    st.session_state["GCS_id_counter"] = 0
if "sedation_id_counter" not in st.session_state:
    st.session_state["sedation_id_counter"] = 0
if "IO_id_counter" not in st.session_state:
    st.session_state["IO_id_counter"] = 0
if "past_history_id_counter" not in st.session_state:
    st.session_state["past_history_id_counter"] = 0
if "treatment_id_counter" not in st.session_state:
    st.session_state["treatment_id_counter"] = 0
if "diagnosis_id_counter" not in st.session_state:
    st.session_state["diagnosis_id_counter"] = 0
if "addx_id_counter" not in st.session_state:
    st.session_state["addx_id_counter"] = 0
if "unit_info_id_counter" not in st.session_state:
    st.session_state["unit_info_id_counter"] = 0


st.title("MITST Demo")
col_buttons, _ = st.columns([5, 1])
with col_buttons:
    st.markdown("**Examples:**", unsafe_allow_html=True)
    cols = st.columns(6)
    example_keys = list(EXAMPLES.keys())
    for i, label in enumerate(example_keys):
        if cols[i].button(label):
            st.session_state.clear()  # Optional: reset first
            st.session_state.update(copy.deepcopy(EXAMPLES[label]))
            st.session_state["example_loaded"] = label
            st.rerun()

st.markdown("---")
st.markdown(
    """
The MITST model has been pretrained and deployed on the server. When you press **"Run MITST Model"**, the system performs inference using the input data you have provided. This inference runs on **CPU only**.
"""
)
st.markdown("---")

tab_labels = list(SOURCE_GROUPS.keys())
tabs = st.tabs(tab_labels)

# %% --- Vital Signs Tab ---
with tabs[0]:
    st.header("Vital Signs")
    st.write(
        "For each vital sign except temperature, enter **comma-separated values** and corresponding **comma-separated offsets** (offsets must be integers)."
    )
    for vs, (minv, maxv) in SOURCE_GROUPS["Vital Signs"].items():
        if vs == "Temp":
            continue  # Skip Temp; handled separately below
        st.subheader(vs)
        val_str = st.text_input(
            f"{vs} values (min={minv}, max={maxv})", key=f"{vs}_values"
        )
        offset_str = st.text_input(f"{vs} offsets (integers)", key=f"{vs}_offsets")
        try:
            values = [float(x) for x in val_str.split(",") if x.strip() != ""]
        except:
            values = []
        try:
            offsets = [int(x) for x in offset_str.split(",") if x.strip() != ""]
        except:
            offsets = []
        # if offset_str and any(
        #     (not x.isdigit())
        #     for x in offset_str.split(",")
        #     if x.strip() != ""
        # ):
        #     st.warning("All offsets must be integers.")
        if len(values) != len(offsets):
            st.warning(
                f"Number of values ({len(values)}) and offsets ({len(offsets)}) do not match."
            )
        if values and offsets and len(values) == len(offsets):
            st.success(f"{len(values)} entries for {vs}.")
        st.session_state[vs] = {"values": values, "offsets": offsets}

    # --- Temperature Subsection ---
    st.subheader("Temperature Records")
    record_key = "Temp"

    if st.button("➕ Add Temperature Record", key=f"add_{record_key}"):
        new_id = st.session_state[f"{record_key}_id_counter"]
        st.session_state[record_key].append(
            {"id": new_id, "location": temp_location[0], "value": 36.5, "offset": 0}
        )
        st.session_state[f"{record_key}_id_counter"] += 1

    if not st.session_state[record_key]:
        st.info(
            "No temperature records found. Click 'Add Temperature Record' to create one."
        )

    for i, record in enumerate(st.session_state[record_key]):
        unique_id = record["id"]
        with st.container():
            st.markdown(f"**Temperature Record {unique_id}**")
            cols = st.columns([3, 1])
            with cols[0]:
                values = {}
                loc = st.selectbox(
                    f"Measurement Location #{unique_id}",
                    temp_location,
                    key=f"{record_key}_location_{unique_id}",
                )
                temp_min, temp_max = SOURCE_GROUPS["Vital Signs"]["Temp"]["value"]
                val = st.slider(
                    f"Temperature (°C) #{unique_id}",
                    min_value=temp_min,
                    max_value=temp_max,
                    # value=36.5,
                    key=f"{record_key}_value_{unique_id}",
                )
                values["location"] = loc
                values["value"] = val
                values["offset"] = handle_offset_input(unique_id, record_key)
                values["id"] = unique_id
                st.session_state[record_key][i] = values
            with cols[1]:
                st.write("")
                if st.button("❌ Remove", key=f"remove_{record_key}_{unique_id}"):
                    st.session_state[record_key] = [
                        r for r in st.session_state[record_key] if r["id"] != unique_id
                    ]
                    st.rerun()

# %% --- Medications Tab ---
with tabs[1]:
    st.header("Medications")
    med_config = SOURCE_GROUPS["Medications"]["med"]
    record_key = "Medications"

    # Add Medication Record Button
    if st.button("➕ Add Medication record", key=f"add_{record_key}"):
        new_id = st.session_state["Medications_id_counter"]
        default_record = {feat: options[0] for feat, options in med_config.items()}
        default_record["dosage"] = 0
        default_record["offset"] = 0
        default_record["id"] = new_id
        st.session_state[record_key].append(default_record)
        st.session_state["Medications_id_counter"] += 1

    # Display message if no records exist
    if not st.session_state[record_key]:
        st.info("No records for Medications. Click 'Add Medication record' to add one.")

    # Iterate through existing medication records
    for i, record in enumerate(st.session_state[record_key]):
        unique_id = record["id"]
        with st.container():
            st.markdown(f"**Medication Record {unique_id}**")
            cols = st.columns([3, 1])

            with cols[0]:
                values = {}
                # Categorical Features Dropdowns
                for cat_feat, options in med_config.items():
                    val = st.selectbox(
                        f"{cat_feat} #{unique_id}",
                        options,
                        key=f"{record_key}_{cat_feat}_{unique_id}",
                    )
                    values[cat_feat] = val

                # Dosage Slider
                drugname = values["drugname"]
                min_dose, max_dose = drug_dosage_limits.get(drugname, (0, 100))
                dosage = st.slider(
                    f"Dosage #{unique_id} (min={min_dose}, max={max_dose})",
                    min_value=min_dose,
                    max_value=max_dose,
                    # value=min_dose,
                    key=f"{drugname}_dosage_{unique_id}",
                )
                values["dosage"] = dosage
                values["offset"] = handle_offset_input(unique_id, record_key)
                values["id"] = unique_id
                # Save the updated record
                st.session_state[record_key][i] = values

            with cols[1]:
                st.write("")
                # Remove Record Button
                if st.button("❌ Remove", key=f"remove_{record_key}_{unique_id}"):
                    st.session_state[record_key] = [
                        r
                        for r in st.session_state[record_key]
                        if r.get("id") != unique_id
                    ]
                    st.rerun()

    # %% Infusion Medications
    st.header("Infusions")
    infusion_drugnames = SOURCE_GROUPS["Medications"]["infusion"]["drugname"]
    record_key = "infusion"

    # Add Infusion Record Button
    if st.button("➕ Add Infusion record", key=f"add_{record_key}"):
        new_id = st.session_state["infusion_id_counter"]
        default_record = {"drugname": infusion_drugnames[0]}
        default_record["dosage"] = 0
        default_record["offset"] = 0
        default_record["id"] = new_id
        st.session_state[record_key].append(default_record)
        st.session_state["infusion_id_counter"] += 1

    # Display message if no records exist
    if not st.session_state[record_key]:
        st.info("No records for Infusions. Click 'Infusion record' to add one.")

    # Display and update each record
    for i, record in enumerate(st.session_state[record_key]):
        unique_id = record["id"]
        with st.container():
            st.markdown(f"**Infusion Record {unique_id}**")
            cols = st.columns([3, 1])

            with cols[0]:
                values = {}
                drugname = st.selectbox(
                    f"Drug name #{unique_id}",
                    infusion_drugnames,
                    key=f"{record_key}_drugname_{unique_id}",
                )
                values["drugname"] = drugname
                min_val, max_val = inf_dosage_limits.get(drugname, (0.0, 100.0))

                dosage = st.slider(
                    f"Dosage #{unique_id} (range: {min_val:.1f} to {max_val:.1f})",
                    min_value=min_val,
                    max_value=max_val,
                    # value=min_val,
                    key=f"{drugname}_dosage_{unique_id}",
                )

                values["dosage"] = dosage
                values["offset"] = handle_offset_input(unique_id, record_key)
                values["id"] = unique_id
                st.session_state[record_key][i] = values

            with cols[1]:
                st.write("")
                if st.button("❌ Remove", key=f"remove_{record_key}_{unique_id}"):
                    st.session_state[record_key] = [
                        r
                        for r in st.session_state[record_key]
                        if r.get("id") != unique_id
                    ]
                    st.rerun()


# %% --- Lab Tab ---
with tabs[2]:
    st.header("Lab Results")
    labnames = SOURCE_GROUPS["Lab"]["labname"]
    record_key = "lab"

    # Add Lab Record Button
    if st.button("➕ Add Lab record", key=f"add_{record_key}"):
        new_id = st.session_state["lab_id_counter"]
        default_record = {"labname": labnames[0]}
        default_record["labresult"] = 0
        default_record["offset"] = 0
        default_record["id"] = new_id
        st.session_state[record_key].append(default_record)
        st.session_state["lab_id_counter"] += 1

    # Display message if no records exist
    if not st.session_state[record_key]:
        st.info("No records for Labs. Click 'Add Lab record' to add one.")

    # Display and update each record
    for i, record in enumerate(st.session_state[record_key]):
        unique_id = record["id"]
        with st.container():
            st.markdown(f"**Lab Record {unique_id}**")
            cols = st.columns([3, 1])

            with cols[0]:
                values = {}
                labname = st.selectbox(
                    f"Lab name #{unique_id}",
                    labnames,
                    key=f"{record_key}_labname_{unique_id}",
                )
                values["labname"] = labname
                min_val, max_val = labresult_limits.get(labname, (0.0, 100.0))

                value = st.slider(
                    f"{labname} Value #{unique_id} (range: {min_val:.1f} to {max_val:.1f})",
                    min_value=min_val,
                    max_value=max_val,
                    # value=min_val,
                    key=f"{labname}_value_{unique_id}",
                )

                values["labresult"] = value
                values["offset"] = handle_offset_input(unique_id, record_key)
                values["id"] = unique_id
                st.session_state[record_key][i] = values

            with cols[1]:
                st.write("")
                if st.button("❌ Remove", key=f"remove_{record_key}_{unique_id}"):
                    st.session_state[record_key] = [
                        r
                        for r in st.session_state[record_key]
                        if r.get("id") != unique_id
                    ]
                    st.rerun()

# %% --- Level of Consciousness (LOC) Assessment ---
with tabs[3]:
    # GCS
    GCS_scores = SOURCE_GROUPS["Scores"]["GCS"]
    record_key = "GCS"

    st.header("GCS Scores")
    if st.button("➕ Add GCS record", key="add_GCS"):
        new_id = st.session_state["GCS_id_counter"]
        default_record = {"id": new_id, "score": GCS_scores[0], "offset": 0}
        st.session_state[record_key].append(default_record)
        st.session_state["GCS_id_counter"] += 1

    if not st.session_state[record_key]:
        st.info("No records for GCS. Click 'Add GCS record' to add one.")

    for i, record in enumerate(st.session_state[record_key]):
        unique_id = record["id"]
        with st.container():
            st.markdown(f"**GCS Record {unique_id}**")
            cols = st.columns([3, 1])
            with cols[0]:
                values = {}
                score = st.selectbox(
                    f"GCS score #{unique_id}",
                    GCS_scores,
                    key=f"{record_key}_score_{unique_id}",
                )
                values["score"] = score
                values["offset"] = handle_offset_input(unique_id, record_key)
                values["id"] = unique_id
                st.session_state[record_key][i] = values
            with cols[1]:
                st.write("")
                if st.button("❌ Remove", key=f"remove_{record_key}_{unique_id}"):
                    st.session_state[record_key] = [
                        r
                        for r in st.session_state[record_key]
                        if r.get("id") != unique_id
                    ]
                    st.rerun()

    # sedation
    sedation_scores = SOURCE_GROUPS["Scores"]["sedation"]
    record_key = "sedation"

    st.header("Sedation Scores")
    if st.button("➕ Add Sedation record", key="add_sedation"):
        new_id = st.session_state["sedation_id_counter"]
        default_record = {"id": new_id, "score": sedation_scores[0], "offset": 0}
        st.session_state[record_key].append(default_record)
        st.session_state["sedation_id_counter"] += 1

    if not st.session_state[record_key]:
        st.info("No records for Sedation. Click 'Add Sedation record' to add one.")

    for i, record in enumerate(st.session_state[record_key]):
        unique_id = record["id"]
        with st.container():
            st.markdown(f"**Sedation Record {unique_id}**")
            cols = st.columns([3, 1])
            with cols[0]:
                values = {}
                score = st.selectbox(
                    f"Sedation score #{unique_id}",
                    sedation_scores,
                    key=f"{record_key}_score_{unique_id}",
                )
                values["score"] = score
                values["offset"] = handle_offset_input(unique_id, record_key)
                values["id"] = unique_id
                st.session_state[record_key][i] = values
            with cols[1]:
                st.write("")
                if st.button("❌ Remove", key=f"remove_{record_key}_{unique_id}"):
                    st.session_state[record_key] = [
                        r
                        for r in st.session_state[record_key]
                        if r.get("id") != unique_id
                    ]
                    st.rerun()

# --- IO Tab ---
with tabs[4]:
    # IO cell names and labels
    st.header("IO Records")
    IOnames = SOURCE_GROUPS["IO"]["IO"]
    record_key = "IO"

    # Add Lab Record Button
    if st.button("➕ Add IO record", key=f"add_{record_key}"):
        new_id = st.session_state["IO_id_counter"]
        default_record = {"IOname": IOnames[0]}
        default_record["IOvalue"] = 0
        default_record["offset"] = 0
        default_record["id"] = new_id
        st.session_state[record_key].append(default_record)
        st.session_state["IO_id_counter"] += 1

    # Display message if no records exist
    if not st.session_state[record_key]:
        st.info("No records for IO. Click 'Add IO record' to add one.")

    # Display and update each record
    for i, record in enumerate(st.session_state[record_key]):
        unique_id = record["id"]
        with st.container():
            st.markdown(f"**IO Record {unique_id}**")
            cols = st.columns([3, 1])

            with cols[0]:
                values = {}
                IOname = st.selectbox(
                    f"IO cell name #{unique_id}",
                    IOnames,
                    key=f"{record_key}_IOname_{unique_id}",
                )
                values["IOname"] = IOname
                min_val, max_val = IO_limits.get(IOname, (0.0, 100.0))

                value = st.slider(
                    f"{IOname} Value #{unique_id} (range: {min_val:.1f} to {max_val:.1f})",
                    min_value=min_val,
                    max_value=max_val,
                    # value=min_val,
                    key=f"{IOname}_value_{unique_id}",
                )

                values["IOvalue"] = value
                values["offset"] = handle_offset_input(unique_id, record_key)
                values["id"] = unique_id
                st.session_state[record_key][i] = values

            with cols[1]:
                st.write("")
                if st.button("❌ Remove", key=f"remove_{record_key}_{unique_id}"):
                    st.session_state[record_key] = [
                        r
                        for r in st.session_state[record_key]
                        if r.get("id") != unique_id
                    ]
                    st.rerun()

    # --- IO Total Records ---
    st.header("IO Total Records")
    st.write(
        "For each of the following variables, enter **comma-separated values** and corresponding **comma-separated offsets** (offsets must be integers)."
    )
    for var, (min_val, max_val) in IO_numerical.items():
        st.subheader(var)
        val_str = st.text_input(
            f"{var} values (min={minv}, max={maxv})", key=f"{var}_values"
        )
        offset_str = st.text_input(f"{var} offsets (integers)", key=f"{var}_offsets")
        try:
            values = [float(x) for x in val_str.split(",") if x.strip() != ""]
        except:
            values = []
        try:
            offsets = [int(x) for x in offset_str.split(",") if x.strip() != ""]
        except:
            offsets = []
        # if offset_str and any(
        #     (not x.isdigit() )
        #     for x in offset_str.split(",")
        #     if x.strip() != ""
        # ):
        #     st.warning("All offsets must be integers.")
        if len(values) != len(offsets):
            st.warning(
                f"Number of values ({len(values)}) and offsets ({len(offsets)}) do not match."
            )
        if values and offsets and len(values) == len(offsets):
            st.success(f"{len(values)} entries for {var}.")
        st.session_state["IO_total"][var] = {"values": values, "offsets": offsets}

# %% --- Past History and Treatment Tab ---
with tabs[5]:
    # Past History
    past_history_options = SOURCE_GROUPS["Past history and treatment"]["past_history"]
    record_key = "past_history"
    st.header("Past history records")

    if st.button("➕ Add Past History record", key="add_past_history"):
        new_id = st.session_state["past_history_id_counter"]
        st.session_state[record_key].append(
            {"id": new_id, "past_history": past_history_options[0], "offset": 0}
        )
        st.session_state["past_history_id_counter"] += 1

    if not st.session_state[record_key]:
        st.info(
            "No records for Past History. Click 'Add Past History record' to add one."
        )

    for i, record in enumerate(st.session_state[record_key]):
        unique_id = record["id"]
        with st.container():
            st.markdown(f"**Past History Record {unique_id}**")
            cols = st.columns([3, 1])
            with cols[0]:
                values = {}
                ph = st.selectbox(
                    f"Past history #{unique_id}",
                    past_history_options,
                    key=f"{record_key}_past_history_{unique_id}",
                )
                values["past_history"] = ph
                values["offset"] = handle_offset_input(unique_id, record_key)
                values["id"] = unique_id
                st.session_state[record_key][i] = values
            with cols[1]:
                st.write("")
                if st.button("❌ Remove", key=f"remove_{record_key}_{unique_id}"):
                    st.session_state[record_key] = [
                        r for r in st.session_state[record_key] if r["id"] != unique_id
                    ]
                    st.rerun()

    # %% Treatments
    treatment_options = SOURCE_GROUPS["Past history and treatment"]["treatment"]
    record_key = "treatment"
    st.header("Treatment records")

    if st.button("➕ Add Treatment record", key="add_treatment"):
        new_id = st.session_state["treatment_id_counter"]
        st.session_state[record_key].append(
            {"id": new_id, "treatment": treatment_options[0], "offset": 0}
        )
        st.session_state["treatment_id_counter"] += 1

    if not st.session_state[record_key]:
        st.info("No records for Treatments. Click 'Add Treatment record' to add one.")

    for i, record in enumerate(st.session_state[record_key]):
        unique_id = record["id"]
        with st.container():
            st.markdown(f"**Treatment Record {unique_id}**")
            cols = st.columns([3, 1])
            with cols[0]:
                values = {}
                treatment = st.selectbox(
                    f"treatment #{unique_id}",
                    treatment_options,
                    key=f"{record_key}_treatment_{unique_id}",
                )
                values["treatment"] = treatment
                values["offset"] = handle_offset_input(unique_id, record_key)
                values["id"] = unique_id
                st.session_state[record_key][i] = values
            with cols[1]:
                st.write("")
                if st.button("❌ Remove", key=f"remove_{record_key}_{unique_id}"):
                    st.session_state[record_key] = [
                        r for r in st.session_state[record_key] if r["id"] != unique_id
                    ]
                    st.rerun()

# %% --- Diagnosis Tab ---
with tabs[6]:
    # Diagnosis Section
    st.header("Diagnosis Records")
    diagnosis_config = SOURCE_GROUPS["Diagnosis"]["diagnosis"]
    record_key = "diagnosis"

    if st.button("➕ Add Diagnosis record", key="add_diagnosis"):
        new_id = st.session_state["diagnosis_id_counter"]
        default_record = {
            "id": new_id,
            "diagnosis": diagnosis_config["diagnosis"][0],
            "diagnosispriority": diagnosis_config["diagnosispriority"][0],
        }
        st.session_state[record_key].append(default_record)
        st.session_state["diagnosis_id_counter"] += 1

    if not st.session_state[record_key]:
        st.info("No records for Diagnosis. Click 'Add Diagnosis record' to add one.")

    for i, record in enumerate(st.session_state[record_key]):
        unique_id = record["id"]
        with st.container():
            st.markdown(f"**Diagnosis Record {unique_id}**")
            cols = st.columns([3, 1])
            with cols[0]:
                values = {}
                diag = st.selectbox(
                    f"Diagnosis #{unique_id}",
                    diagnosis_config["diagnosis"],
                    key=f"{record_key}_diagnosis_{unique_id}",
                )
                diag_priority = st.selectbox(
                    f"Priority #{unique_id}",
                    diagnosis_config["diagnosispriority"],
                    key=f"{record_key}_diagnosispriority_{unique_id}",
                )
                values["diagnosis"] = diag
                values["diagnosispriority"] = diag_priority
                values["id"] = unique_id
                values["offset"] = handle_offset_input(unique_id, record_key)
                st.session_state[record_key][i] = values
            with cols[1]:
                st.write("")
                if st.button("❌ Remove", key=f"remove_{record_key}_{unique_id}"):
                    st.session_state[record_key] = [
                        r for r in st.session_state[record_key] if r["id"] != unique_id
                    ]
                    st.rerun()

    # %% Admission Diagnosis (Addx) Section
    st.header("Admission Diagnosis (Addx) Records")
    addx_options = SOURCE_GROUPS["Diagnosis"]["addx"]
    record_key = "addx"

    if st.button("➕ Add Addx record", key="add_addx"):
        new_id = st.session_state["addx_id_counter"]
        st.session_state[record_key].append(
            {"id": new_id, "addx": addx_options[0], "offset": 0}
        )
        st.session_state["addx_id_counter"] += 1

    if not st.session_state[record_key]:
        st.info("No records for Addx. Click 'Add Addx record' to add one.")

    for i, record in enumerate(st.session_state[record_key]):
        unique_id = record["id"]
        with st.container():
            st.markdown(f"**Addx Record {unique_id}**")
            cols = st.columns([3, 1])
            with cols[0]:
                values = {}
                val = st.selectbox(
                    f"Addx #{unique_id}",
                    addx_options,
                    key=f"{record_key}_addx_{unique_id}",
                )

                values["addx"] = val
                values["offset"] = handle_offset_input(unique_id, record_key)
                values["id"] = unique_id
                st.session_state[record_key][i] = values
            with cols[1]:
                st.write("")
                if st.button("❌ Remove", key=f"remove_{record_key}_{unique_id}"):
                    st.session_state[record_key] = [
                        r for r in st.session_state[record_key] if r["id"] != unique_id
                    ]
                    st.rerun()

# %% --- Demographics and Hospital Info Tab ---
with tabs[7]:
    st.header("Demographics and Hospital Info")

    # Sliders for numerical demographic values
    weight = st.slider("Weight (kg)", min_value=0, max_value=200, key="weight")
    height = st.slider("Height (cm)", min_value=0, max_value=250, key="height")
    age = st.slider("Age (years)", min_value=0, max_value=90, key="age")

    # Dropdowns for categorical variables
    gender_val = st.selectbox("Gender", gender, key="gender")
    ethnicity_val = st.selectbox("Ethnicity", ethnicity, key="ethnicity")
    num_beds_val = st.selectbox("Hospital Num Beds", num_beds, key="num_beds")
    hospital_id_val = st.selectbox("Hospital ID", hospital_ids, key="hospital_id")
    region_val = st.selectbox("Region", region, key="region")
    admitsource_val = st.selectbox(
        "Hospital Admit Source", admitsource, key="admitsource"
    )
    teachingstatus_val = st.selectbox(
        "Teaching Status", ["True", "False"], key="teachingstatus"
    )

    # Store demographic values
    st.session_state["static"] = {
        "weight": weight,
        "height": height,
        "age": age,
        "gender": gender_val,
        "ethnicity": ethnicity_val,
        "num_beds": num_beds_val,
        "hospital_id": hospital_id_val,
        "region": region_val,
        "admitsource": admitsource_val,
        "teachingstatus": teachingstatus_val,
    }

    # --- Unit Info Subsection ---
    st.subheader("Unit Information Records")
    unit_config = {
        "unittype": unittype,
        "unitstaytype": unitstaytype,
        "unitadmitsource": admitsource,
    }
    record_key = "unit_info"

    if st.button("➕ Add Unit Info record", key=f"add_{record_key}"):
        new_id = st.session_state[f"{record_key}_id_counter"]
        default_record = {feat: options[0] for feat, options in unit_config.items()}
        default_record["offset"] = 0
        default_record["id"] = new_id
        st.session_state[record_key].append(default_record)
        st.session_state[f"{record_key}_id_counter"] += 1

    if not st.session_state[record_key]:
        st.info("No records for Unit Info. Click 'Add Unit Info record' to add one.")

    for i, record in enumerate(st.session_state[record_key]):
        unique_id = record["id"]
        with st.container():
            st.markdown(f"**Unit Info Record {unique_id}**")
            cols = st.columns([3, 1])
            with cols[0]:
                values = {}
                for feat, options in unit_config.items():
                    values[feat] = st.selectbox(
                        f"{feat} #{unique_id}",
                        options,
                        key=f"{record_key}_{feat}_{unique_id}",
                    )
                values["offset"] = handle_offset_input(unique_id, record_key)
                values["id"] = unique_id
                st.session_state[record_key][i] = values

            with cols[1]:
                st.write("")
                if st.button("❌ Remove", key=f"remove_{record_key}_{unique_id}"):
                    st.session_state[record_key] = [
                        r for r in st.session_state[record_key] if r["id"] != unique_id
                    ]
                    st.rerun()


st.markdown("---")
if st.button("Run MITST Model"):
    import torch

    # 3. UI section for model inference
    result, preds = classify_patient(st.session_state, categories_to_int, scaler)
    st.success(f"Predicted Glucose Status: **{result}**")
    st.write(f"Predictions: 1- {preds[0]}, 2-{preds[1]}, 3-{preds[2]}")
