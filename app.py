from pymongo import MongoClient
import voyageai
import csv
from typing import List, Dict, Any

# -------------------------------
# CONFIG — EDIT THESE
# -------------------------------
MONGO_URI = ""
VOYAGE_API_KEY = ""

DB_NAME = "Optum_Match"
PATIENT_COLL = "patient_survey_responses"
CLINICIAN_COLL = "clinician_survey_responses"

VECTOR_INDEX_NAME = "vector_index"      # index on clinician_survey_responses
MODEL_NAME = "voyage-3-large"
EXPECTED_DIM = 1024

OUTPUT_CSV = "vector_matches_results.csv"

# -------------------------------
# INIT CLIENTS
# -------------------------------
mongo_client = MongoClient(MONGO_URI)
db = mongo_client[DB_NAME]

vo = voyageai.Client(api_key=VOYAGE_API_KEY)

# -------------------------------
# EMBEDDING HELPERS
# -------------------------------
def flatten_survey_doc(doc: Dict[str, Any]) -> str:
    chunks = []
    for key, val in doc.items():
        if key in ["_id", "patientId", "clinicianId"]:
            continue
        if isinstance(val, list):
            chunks.extend([str(v) for v in val])
        else:
            chunks.append(str(val))
    return " ".join(chunks)


def embed_text(text: str) -> List[float]:
    resp = vo.embed([text], model=MODEL_NAME, input_type="document")
    emb = resp.embeddings[0]
    if EXPECTED_DIM and len(emb) != EXPECTED_DIM:
        raise ValueError(f"Embedding dim {len(emb)} != EXPECTED_DIM {EXPECTED_DIM}")
    return emb


def ensure_embeddings():
    print("🔄 Ensuring patient embeddings...")
    start = time.time()
    for doc in db[PATIENT_COLL].find():
        emb = doc.get("embedding")
        if not emb or (EXPECTED_DIM and len(emb) != EXPECTED_DIM):
            text = flatten_survey_doc(doc)
            vector = embed_text(text)
            db[PATIENT_COLL].update_one({"_id": doc["_id"]},
                                        {"$set": {"embedding": vector}})
            print(f"  ✓ Embedded patient {doc.get('patientId')}")
    print(f"✅ Patient embeddings ready. ({time.time() - start:.2f}s)\n")

    print("🔄 Ensuring clinician embeddings...")
    start = time.time()
    for doc in db[CLINICIAN_COLL].find():
        emb = doc.get("embedding")
        if not emb or (EXPECTED_DIM and len(emb) != EXPECTED_DIM):
            text = flatten_survey_doc(doc)
            vector = embed_text(text)
            db[CLINICIAN_COLL].update_one({"_id": doc["_id"]},
                                          {"$set": {"embedding": vector}})
            print(f"  ✓ Embedded clinician {doc.get('clinicianId')}")
    print(f"✅ Clinician embeddings ready. ({time.time() - start:.2f}s)\n")


# -------------------------------
# REASONS HELPERS
# -------------------------------
EXPERTISE_MAP = {
    "Q9": "Aging well",
    "Q10": "BIPOC health",
    "Q11": "Body positive care",
    "Q12": "Caregiver support",
    "Q13": "Caring for deaf and hard of hearing patients",
    "Q14": "End of life care",
    "Q15": "Gender affirming care",
    "Q16": "Immigrant or refugee health",
    "Q17": "LGBTQ+ health",
    "Q18": "Men's health care",
    "Q19": "Women's health care",
    "Q20": "Neurodiversity affirming care",
    "Q21": "Period positive care",
    "Q22": "Sex positive care",
    "Q23": "Substance use and addiction recovery",
    "Q24": "Trauma informed care",
    "Q25": "Veteran supportive care",
    "Q26": "Spiritual care",
}

def as_list(x):
    if isinstance(x, list):
        return x
    if x is None:
        return []
    return [x]

def generate_reasons(patient: Dict[str, Any], clinician: Dict[str, Any]) -> List[str]:
    reasons = []

    # Languages (Q5)
    p_langs = set(as_list(patient.get("Q5")))
    c_langs = set(as_list(clinician.get("Q5")))
    lang_overlap = p_langs & c_langs
    if lang_overlap:
        reasons.append("Shared language(s): " + ", ".join(sorted(lang_overlap)))

    # Care expertise (patient Q9 vs clinician Q9–Q26 'expert')
    patient_care_needs = set(as_list(patient.get("Q9")))
    clinician_expertise = set()
    for q_num, care_area in EXPERTISE_MAP.items():
        text = clinician.get(q_num)
        if isinstance(text, str) and "expert" in text.lower():
            clinician_expertise.add(care_area)
    care_overlap = patient_care_needs & clinician_expertise
    if care_overlap:
        reasons.append("Care expertise overlap: " + ", ".join(sorted(care_overlap)))

    # Communication style (Q2)
    if patient.get("Q2") == clinician.get("Q2"):
        reasons.append("Matched communication style (Q2)")

    # Military experience (Q7)
    if patient.get("Q7") == "Yes" and clinician.get("Q7") == "Yes":
        reasons.append("Shared military experience")

    return reasons


# -------------------------------
# VECTOR SEARCH
# -------------------------------
def get_vector_matches_for_patient(patient_doc: Dict[str, Any], k: int = 10):
    patient_vec = patient_doc.get("embedding")
    if not patient_vec:
        raise ValueError(f"Patient {patient_doc.get('patientId')} has no embedding")

    pipeline = [
        {
            "$vectorSearch": {
                "index": VECTOR_INDEX_NAME,
                "path": "embedding",
                "queryVector": patient_vec,
                "numCandidates": 50,
                "limit": k
            }
        },
        {
            "$project": {
                "_id": 0,
                "clinicianId": 1,
                "Q2": 1,
                "Q5": 1,
                "Q9": 1, "Q10": 1, "Q11": 1, "Q12": 1, "Q13": 1,
                "Q14": 1, "Q15": 1, "Q16": 1, "Q17": 1, "Q18": 1,
                "Q19": 1, "Q20": 1, "Q21": 1, "Q22": 1, "Q23": 1,
                "Q24": 1, "Q25": 1, "Q26": 1,
                "Q7": 1,
                "score": {"$meta": "vectorSearchScore"}
            }
        }
    ]

    return list(db[CLINICIAN_COLL].aggregate(pipeline))


# -------------------------------
# MAIN
# -------------------------------
def run_vector_matching():
    ensure_embeddings()

    patients = list(db[PATIENT_COLL].find().sort("patientId", 1))

    print("🔍 Running vector search for each patient...")
    rows = []
    printed_sample = False

    for patient in patients:
        pid = patient.get("patientId")
        matches = get_vector_matches_for_patient(patient, k=10)

        if not matches:
            print(f"  ⚠ No matches returned for patient {pid}")
            continue

        # print one raw sample so you can see the score field
        if not printed_sample:
            print("\n🧪 Sample match document:")
            print(matches[0])
            printed_sample = True

        for rank, clinician in enumerate(matches, start=1):
            cid = clinician.get("clinicianId")
            score = clinician.get("score")

            rows.append({
                "patient_id": pid,
                "clinician_id": cid,
                "rank": rank,
                "score": float(score) if score is not None else None,
                "reasons": " | ".join(generate_reasons(patient, clinician))
            })

        print(f"  ✓ Patient {pid}: collected {len(matches)} matches")

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["patient_id", "clinician_id", "rank", "score", "reasons"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n💾 Vector match results exported to: {OUTPUT_CSV}")
    print("🎯 Vector matching complete.")


if __name__ == "__main__":
    run_vector_matching()
