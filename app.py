import streamlit as st
import pandas as pd
import numpy as np
import json
import math
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==========================================
# 1. Page Configuration & Styling
# ==========================================
st.set_page_config(
    page_title="Bangkok Urban Heat & Green Space Priority Dashboard",
    page_icon="🌳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Dark Navy, Green, and Professional Design
st.markdown("""
<style>
    .main { background-color: #0d1b2a; color: #e0e1dd; }
    .stApp { background-color: #0b132b; color: #ffffff; }
    h1, h2, h3 { color: #52b788 !important; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
    .metric-card {
        background-color: #1c2541;
        border-left: 5px solid #52b788;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 15px;
    }
    .metric-title { font-size: 0.9rem; color: #a3b18a; margin-bottom: 5px; }
    .metric-value { font-size: 1.8rem; font-weight: bold; color: #ffffff; }
    .stAlert { background-color: #1c2541; border: 1px solid #3a506b; color: #e0e1dd; }
    .species-card {
        background-color: #1c2541;
        border: 1px solid #3a506b;
        border-radius: 10px;
        padding: 18px;
        margin-bottom: 20px;
    }
    .species-title { color: #52b788; font-size: 1.3rem; font-weight: bold; }
    .species-scientific { color: #a3b18a; font-style: italic; font-size: 1.0rem; margin-bottom: 10px; }
    .badge {
        background-color: #2d6a4f;
        color: #ffffff;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
        margin-right: 5px;
        margin-bottom: 5px;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. Data Loading & Preprocessing Functions
# ==========================================
@st.cache_data
def load_and_preprocess_data():
    grid_df = pd.read_csv("data/bangkok_250m_grid_data.csv")
    district_df = pd.read_csv("data/bangkok_district_summary.csv")
    
    # Extract Longitude and Latitude from .geo JSON column if present
    if '.geo' in grid_df.columns:
        def parse_coords(geo_str):
            try:
                data = json.loads(geo_str)
                coords = data['coordinates']
                return coords[0], coords[1] # lon, lat
            except:
                return np.nan, np.nan
        grid_df['longitude'], grid_df['latitude'] = zip(*grid_df['.geo'].apply(parse_coords))
    
    # Clean null values and extreme outliers
    grid_df = grid_df.dropna(subset=['LST', 'NDVI', 'population', 'latitude', 'longitude']).copy()
    grid_df = grid_df[(grid_df['LST'] >= 15) & (grid_df['LST'] <= 50)]
    grid_df = grid_df[(grid_df['NDVI'] >= -0.2) & (grid_df['NDVI'] <= 1.0)]
    grid_df = grid_df[grid_df['population'] >= 0]
    
    # Normalize factors between 0 and 1
    lst_min, lst_max = grid_df['LST'].min(), grid_df['LST'].max()
    ndvi_min, ndvi_max = grid_df['NDVI'].min(), grid_df['NDVI'].max()
    pop_min, pop_max = grid_df['population'].min(), grid_df['population'].max()
    
    grid_df['LST_norm'] = (grid_df['LST'] - lst_min) / (lst_max - lst_min + 1e-6)
    grid_df['NDVI_norm'] = (ndvi_max - grid_df['NDVI']) / (ndvi_max - ndvi_min + 1e-6) # Inverse: low vegetation = higher priority
    grid_df['Pop_norm'] = (grid_df['population'] - pop_min) / (pop_max - pop_min + 1e-6)
    
    # Priority Scoring (LST: 40%, NDVI: 35%, Population Exposure: 25%)
    grid_df['Priority_Score'] = (
        0.40 * grid_df['LST_norm'] +
        0.35 * grid_df['NDVI_norm'] +
        0.25 * grid_df['Pop_norm']
    ) * 100.0
    
    # Priority Categories
    quantiles = grid_df['Priority_Score'].quantile([0.25, 0.50, 0.75]).values
    def get_category(score):
        if score >= quantiles[2]: return 'Very High'
        elif score >= quantiles[1]: return 'High'
        elif score >= quantiles[0]: return 'Medium'
        else: return 'Low'
    
    grid_df['Priority_Category'] = grid_df['Priority_Score'].apply(get_category)
    
    # Clean district summaries
    district_df = district_df.dropna(subset=['LST', 'NDVI']).copy()
    if 'ADM2_NAME' in district_df.columns:
        district_df.rename(columns={'ADM2_NAME': 'District'}, inplace=True)
    elif 'system:index' in district_df.columns:
        district_df.rename(columns={'system:index': 'District'}, inplace=True)
        
    return grid_df, district_df

@st.cache_resource
def train_random_forest_model(grid_df):
    features = ['longitude', 'latitude', 'NDVI', 'population']
    X = grid_df[features]
    y = grid_df['LST']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    metrics = {
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "R2": r2_score(y_test, y_pred),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "importances": dict(zip(features, model.feature_importances_))
    }
    
    grid_df['LST_Predicted'] = model.predict(X)
    return model, metrics, grid_df

# Load datasets and model
try:
    grid_df, district_df = load_and_preprocess_data()
    rf_model, ml_metrics, grid_df = train_random_forest_model(grid_df)
except Exception as e:
    st.error(f"Error loading project datasets: {e}")
    st.stop()

# Static Tree Species Data (30 Candidate Species for Bangkok Urban Greening)
SPECIES_DATA = [
    {
        "thai_name": "มะค่าโมง",
        "scientific_name": "Afzelia xylocarpa",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/6/69/Afzelia_xylocarpa_leaves.jpg/640px-Afzelia_xylocarpa_leaves.jpg",
        "description": "Large native deciduous forest tree with a broad, dense rounded crown. Exceptional timber strength and broad canopy coverage.",
        "canopy_size": "Large (12 - 18 m crown diameter, ~110 - 250 m² canopy)",
        "environments": ["Park", "Open Public Space"],
        "benefits": "Dense shade canopy, high carbon storage capacity, atmospheric nitrogen fixation through root nodules.",
        "limitations": "Massive trunk buttresses and root expansion. Unsuitable for narrow sidewalks, road dividers, or near built infrastructure.",
        "source": "Bangkok Metropolitan Administration (BMA) & Royal Forest Department (RFD)"
    },
    {
        "thai_name": "ยางนา",
        "scientific_name": "Dipterocarpus alatus",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b5/Dipterocarpus_alatus_in_Thailand.jpg/640px-Dipterocarpus_alatus_in_Thailand.jpg",
        "description": "Iconic large emergent native tree of the Chao Phraya river basin. Tall straight trunk with broad spreading upper canopy branches.",
        "canopy_size": "Very Large (15 - 25 m crown diameter, ~150 - 350 m² canopy)",
        "environments": ["Park", "Open Public Space"],
        "benefits": "Superior airborne PM2.5 particulate interception, high windbreak potential, major carbon sequestration reservoir.",
        "limitations": "Extreme mature height (up to 40m). Branch drop hazard during monsoon storms if unpruned. Requires wide soil footprint.",
        "source": "BMA Parks & Gardening Division / Kasetsart University Forestry Research"
    },
    {
        "thai_name": "ประดู่บ้าน",
        "scientific_name": "Pterocarpus indicus",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c2/Pterocarpus_indicus_tree.jpg/640px-Pterocarpus_indicus_tree.jpg",
        "description": "Widely planted urban shade tree across Bangkok. Features umbrella-like spreading branches with vibrant golden seasonal yellow flowers.",
        "canopy_size": "Medium to Large (10 - 15 m crown diameter, ~80 - 180 m² canopy)",
        "environments": ["Roadside", "Park", "Open Public Space"],
        "benefits": "Excellent microclimate cooling efficiency, dense leaf canopy for solar radiation reduction, high tolerance to heat.",
        "limitations": "Shallow aggressive lateral roots can lift light sidewalk pavement if planted in restricted soil pits.",
        "source": "Bangkok Urban Greening Master Plan & Royal Forest Department"
    },
    {
        "thai_name": "ชมพูพันธุ์ทิพย์",
        "scientific_name": "Tabebuia rosea",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1d/Tabebuia_rosea_flowers.jpg/640px-Tabebuia_rosea_flowers.jpg",
        "description": "Popular ornamental medium-to-large shade tree known for striking pink floral blooms. Broad dome-shaped canopy.",
        "canopy_size": "Medium (8 - 14 m crown diameter, ~50 - 150 m² canopy)",
        "environments": ["Park", "Open Public Space", "Residential Area"],
        "benefits": "Rapid juvenile growth rate, high visual aesthetic value, effective neighborhood canopy shading.",
        "limitations": "Heavy seasonal leaf and flower litter. Softwood branches prone to wind snapping during severe squalls.",
        "source": "BMA Environment Department & Faculty of Forestry, Kasetsart University"
    },
    {
        "thai_name": "อินทนิลน้ำ",
        "scientific_name": "Lagerstroemia speciosa",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/9/90/Lagerstroemia_speciosa_flowers.jpg/640px-Lagerstroemia_speciosa_flowers.jpg",
        "description": "Medium-sized deciduous flowering shade tree with purple and pink blossoms. Highly adaptable to harsh urban streetscapes.",
        "canopy_size": "Medium (6 - 10 m crown diameter, ~30 - 80 m² canopy)",
        "environments": ["Roadside", "Narrow Planting Space", "Residential Area", "Park"],
        "benefits": "Non-invasive deep root system, high heat and drought tolerance, excellent fit for roadside medians and streetscapes.",
        "limitations": "Slower growth rate in severely compacted urban soils. Susceptible to powdery mildew in heavily shaded environments.",
        "source": "Bangkok Tree Planting 1-Million Trees Campaign / RFD Guidelines"
    },
    {
        "thai_name": "นนทรี",
        "scientific_name": "Peltophorum pterocarpum",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b7/Peltophorum_pterocarpum_tree.jpg/640px-Peltophorum_pterocarpum_tree.jpg",
        "description": "Medium to large spreading shade tree with umbrella-shaped canopy and fragrant yellow blossoms.",
        "canopy_size": "Medium to Large (10 - 16 m crown diameter, ~80 - 200 m² canopy)",
        "environments": ["Roadside", "Park", "Open Public Space", "Residential Area"],
        "benefits": "High shade density, nitrogen-fixing soil enrichment, excellent urban heat resilience.",
        "limitations": "Brittle wood prone to wind damage if unpruned; pod litter in dry season.",
        "source": "BMA Environment Department & Kasetsart University"
    },
    {
        "thai_name": "ราชพฤกษ์ (คูน)",
        "scientific_name": "Cassia fistula",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f7/Cassia_fistula_in_Kolkata_W_IMG_2448.jpg/640px-Cassia_fistula_in_Kolkata_W_IMG_2448.jpg",
        "description": "National tree of Thailand, medium-sized tree producing cascading bright yellow flower clusters.",
        "canopy_size": "Medium (6 - 10 m crown diameter, ~30 - 80 m² canopy)",
        "environments": ["Roadside", "Park", "Residential Area"],
        "benefits": "Strong drought tolerance, non-aggressive root system, iconic aesthetic appeal.",
        "limitations": "Deciduous during bloom season (temporary canopy density drop); long pod drop.",
        "source": "Royal Forest Department & BMA Greening Guidelines"
    },
    {
        "thai_name": "กัลปพฤกษ์",
        "scientific_name": "Cassia bakeriana",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/8/89/Cassia_bakeriana.jpg/640px-Cassia_bakeriana.jpg",
        "description": "Small-to-medium native flowering tree known as Northern Pink Cassia, with soft pink-white blossoms.",
        "canopy_size": "Small to Medium (5 - 9 m crown diameter, ~20 - 65 m² canopy)",
        "environments": ["Roadside", "Narrow Planting Space", "Residential Area", "Park"],
        "benefits": "Compact footprint suitable for urban streets, moderate shade, minimal root disruption.",
        "limitations": "Loses leaves briefly before blooming; requires well-drained soil.",
        "source": "Royal Forest Department Guidelines"
    },
    {
        "thai_name": "ชัยพฤกษ์",
        "scientific_name": "Cassia javanica",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a4/Cassia_javanica_flowers.jpg/640px-Cassia_javanica_flowers.jpg",
        "description": "Medium deciduous tree with spreading horizontal branches and pink floral displays.",
        "canopy_size": "Medium (8 - 12 m crown diameter, ~50 - 110 m² canopy)",
        "environments": ["Park", "Open Public Space", "Roadside"],
        "benefits": "Wide branch spread providing solid sidewalk shade; highly tolerant of heat.",
        "limitations": "Requires wide overhead space; branches extend low if not crown-lifted.",
        "source": "BMA Parks & Gardening Division"
    },
    {
        "thai_name": "พิกุล",
        "scientific_name": "Mimusops elengi",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/7/7b/Mimusops_elengi_tree.jpg/640px-Mimusops_elengi_tree.jpg",
        "description": "Dense evergreen medium tree with fragrant star-shaped white flowers and compact foliage.",
        "canopy_size": "Medium (6 - 10 m crown diameter, ~30 - 80 m² canopy)",
        "environments": ["Roadside", "Narrow Planting Space", "Park", "Residential Area"],
        "benefits": "Year-round dense shade (evergreen), deep non-invasive taproot, highly wind-resistant.",
        "limitations": "Slow growth rate; fallen small fruits can attract insects if uncleaned.",
        "source": "BMA Tree Planting Manual & RFD"
    },
    {
        "thai_name": "สารภี",
        "scientific_name": "Mammea siamensis",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d4/Mammea_siamensis.jpg/640px-Mammea_siamensis.jpg",
        "description": "Native evergreen shade tree with glossy dark green leaves and small fragrant cream flowers.",
        "canopy_size": "Small to Medium (5 - 8 m crown diameter, ~20 - 50 m² canopy)",
        "environments": ["Narrow Planting Space", "Residential Area", "Park"],
        "benefits": "Compact upright canopy, excellent shade density, deep non-disruptive roots.",
        "limitations": "Slow growth rate; sensitive to prolonged waterlogging.",
        "source": "Royal Forest Department"
    },
    {
        "thai_name": "แก้ว",
        "scientific_name": "Murraya paniculata",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b3/Murraya_paniculata_flowers.jpg/640px-Murraya_paniculata_flowers.jpg",
        "description": "Small compact evergreen tree or large shrub with intensely sweet fragrant white flowers.",
        "canopy_size": "Small (3 - 6 m crown diameter, ~7 - 28 m² canopy)",
        "environments": ["Narrow Planting Space", "Residential Area", "Roadside"],
        "benefits": "Ideal for tight urban pockets, sidewalk planters, and screening hedges.",
        "limitations": "Requires regular pruning to maintain tree form; low canopy shading area.",
        "source": "BMA Urban Greening Guidelines"
    },
    {
        "thai_name": "โมกบ้าน",
        "scientific_name": "Wrightia religiosa",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/2/23/Wrightia_religiosa_flowers.jpg/640px-Wrightia_religiosa_flowers.jpg",
        "description": "Small traditional Thai garden tree with drooping white fragrant flowers.",
        "canopy_size": "Small (2 - 5 m crown diameter, ~3 - 20 m² canopy)",
        "environments": ["Narrow Planting Space", "Residential Area", "Roadside"],
        "benefits": "Excellent for constrained urban corridors, air pollution buffer, low root impact.",
        "limitations": "Provides limited shade footprint compared to large canopy trees.",
        "source": "BMA Urban Planting Guide"
    },
    {
        "thai_name": "ลีลาวดี (ลั่นทม)",
        "scientific_name": "Plumeria obtusa",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/5/53/Plumeria_obtusa_white_flowers.jpg/640px-Plumeria_obtusa_white_flowers.jpg",
        "description": "Small ornamental succulent-branched tree with glossy leaves and fragrant flowers.",
        "canopy_size": "Small to Medium (4 - 7 m crown diameter, ~12 - 40 m² canopy)",
        "environments": ["Park", "Open Public Space", "Residential Area"],
        "benefits": "Extremely drought-tolerant, non-destructive roots, aesthetic landmark tree.",
        "limitations": "Soft branches snap easily; poisonous milky latex sap when cut.",
        "source": "BMA Parks Division"
    },
    {
        "thai_name": "จามจุรี (ก้ามปู)",
        "scientific_name": "Samanea saman",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/2/29/Samanea_saman_tree.jpg/640px-Samanea_saman_tree.jpg",
        "description": "Massive dome-shaped rain tree with sprawling horizontal branch canopy.",
        "canopy_size": "Very Large (20 - 35 m crown diameter, ~300 - 950 m² canopy)",
        "environments": ["Park", "Open Public Space"],
        "benefits": "Unrivaled cooling canopy area per tree, massive carbon sink, rapid growth.",
        "limitations": "Requires huge open space; aggressive surface roots destroy pavement and structures.",
        "source": "Chulalongkorn University Campus Greening / BMA Parks"
    },
    {
        "thai_name": "ตะแบกนา",
        "scientific_name": "Lagerstroemia floribunda",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e5/Lagerstroemia_floribunda.jpg/640px-Lagerstroemia_floribunda.jpg",
        "description": "Native medium tree with mottled smooth bark and dense pinkish-purple flower plumes.",
        "canopy_size": "Medium (7 - 11 m crown diameter, ~40 - 95 m² canopy)",
        "environments": ["Roadside", "Park", "Open Public Space", "Residential Area"],
        "benefits": "Strong adaptability to Bangkok urban soils, good canopy shade, deep root structure.",
        "limitations": "Deciduous period in dry season; moderate maintenance needed for shape.",
        "source": "Royal Forest Department"
    },
    {
        "thai_name": "เสลา",
        "scientific_name": "Lagerstroemia loudonii",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3d/Lagerstroemia_loudonii.jpg/640px-Lagerstroemia_loudonii.jpg",
        "description": "Medium native tree featuring dark fissured bark and striking purple floral displays.",
        "canopy_size": "Medium (6 - 10 m crown diameter, ~30 - 80 m² canopy)",
        "environments": ["Roadside", "Park", "Residential Area"],
        "benefits": "Highly flood and heat tolerant, compact crown footprint for urban streets.",
        "limitations": "Slower growth rate; leaf drop prior to flowering.",
        "source": "BMA Tree Planting Masterplan"
    },
    {
        "thai_name": "สุพรรณิการ์",
        "scientific_name": "Cochlospermum regium",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a2/Cochlospermum_regium.jpg/640px-Cochlospermum_regium.jpg",
        "description": "Small to medium deciduous tree with showy bright yellow blossom clusters.",
        "canopy_size": "Small to Medium (5 - 8 m crown diameter, ~20 - 50 m² canopy)",
        "environments": ["Roadside", "Narrow Planting Space", "Park"],
        "benefits": "Exceptional drought resistance, lightweight canopy suitable for narrow buffers.",
        "limitations": "Completely drops leaves during blooming season (no shade during bloom).",
        "source": "RFD Urban Forestry Guidelines"
    },
    {
        "thai_name": "เหลืองปรีดียาธร",
        "scientific_name": "Tabebuia aurea",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4b/Tabebuia_aurea_flower.jpg/640px-Tabebuia_aurea_flower.jpg",
        "description": "Small to medium silver trumpet tree with vibrant golden blossoms.",
        "canopy_size": "Small to Medium (4 - 8 m crown diameter, ~12 - 50 m² canopy)",
        "environments": ["Roadside", "Narrow Planting Space", "Residential Area"],
        "benefits": "Thrives in dry poor soils, small root profile suitable for sidewalk planters.",
        "limitations": "Brittle wood prone to wind damage; sparse foliage during flowering.",
        "source": "BMA Street Tree Selection Guide"
    },
    {
        "thai_name": "แคนา",
        "scientific_name": "Dolichandrone serrulata",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f6/Dolichandrone_serrulata.jpg/640px-Dolichandrone_serrulata.jpg",
        "description": "Medium native tree with trumpet-shaped white nocturnal flowers.",
        "canopy_size": "Medium (6 - 10 m crown diameter, ~30 - 80 m² canopy)",
        "environments": ["Roadside", "Park", "Open Public Space"],
        "benefits": "High survival rate in urban soils, rapid growth, effective shade provider.",
        "limitations": "Fallen trumpet flowers require daily sweeping on pedestrian paths.",
        "source": "Royal Forest Department & BMA"
    },
    {
        "thai_name": "ปีบ (กาซะลอง)",
        "scientific_name": "Millingtonia hortensis",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/44/Millingtonia_hortensis.jpg/640px-Millingtonia_hortensis.jpg",
        "description": "Tall slender tree with corky bark and fragrant white tubular flowers.",
        "canopy_size": "Medium (5 - 9 m crown diameter, ~20 - 65 m² canopy)",
        "environments": ["Park", "Open Public Space", "Roadside"],
        "benefits": "Narrow vertical growth profile, pleasant floral aroma, high air filter capacity.",
        "limitations": "Shallow root suckers can sprout aggressively; weak wood in high winds.",
        "source": "BMA Greening Guidelines"
    },
    {
        "thai_name": "บุนนาค",
        "scientific_name": "Mesua ferrea",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e4/Mesua_ferrea_flower.jpg/640px-Mesua_ferrea_flower.jpg",
        "description": "Slow-growing evergreen tree with dense conical canopy and fragrant white flowers.",
        "canopy_size": "Medium (6 - 10 m crown diameter, ~30 - 80 m² canopy)",
        "environments": ["Park", "Residential Area", "Narrow Planting Space"],
        "benefits": "Year-round dense canopy shade, highly storm-resistant, deep non-invasive taproot.",
        "limitations": "Very slow growth rate; high initial sapling cost.",
        "source": "Royal Forest Department"
    },
    {
        "thai_name": "กันเกรา",
        "scientific_name": "Cyrtophyllum fragrans",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/6/60/Cyrtophyllum_fragrans.jpg/640px-Cyrtophyllum_fragrans.jpg",
        "description": "Auspicious native evergreen tree with small yellowish fragrant flowers.",
        "canopy_size": "Medium (6 - 10 m crown diameter, ~30 - 80 m² canopy)",
        "environments": ["Park", "Open Public Space", "Residential Area"],
        "benefits": "Strong windbreak, highly resistant to pests, dense evergreen foliage.",
        "limitations": "Slow growth rate; requires good drainage to avoid root rot.",
        "source": "Royal Forest Department"
    },
    {
        "thai_name": "กระทิง (สารภีทะเล)",
        "scientific_name": "Calophyllum inophyllum",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/30/Calophyllum_inophyllum_tree.jpg/640px-Calophyllum_inophyllum_tree.jpg",
        "description": "Evergreen tree with thick leathery leaves, native to coastal and lowland areas.",
        "canopy_size": "Medium to Large (8 - 14 m crown diameter, ~50 - 150 m² canopy)",
        "environments": ["Park", "Open Public Space", "Roadside"],
        "benefits": "Extremely hardy against pollution, salt, heat, and flooding; dense shading.",
        "limitations": "Large round hard fruits create dropping hazards along roadways.",
        "source": "BMA Environment Department"
    },
    {
        "thai_name": "มะฮอกกานีใบใหญ่",
        "scientific_name": "Swietenia macrophylla",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e0/Swietenia_macrophylla_tree.jpg/640px-Swietenia_macrophylla_tree.jpg",
        "description": "Large fast-growing shade tree with broad compound leaves and robust trunk.",
        "canopy_size": "Large (12 - 18 m crown diameter, ~110 - 250 m² canopy)",
        "environments": ["Park", "Open Public Space", "Roadside"],
        "benefits": "Exceptional shade canopy coverage, rapid growth rate, high carbon uptake.",
        "limitations": "Strong surface root expansion; requires wide sidewalk offsets.",
        "source": "BMA Parks & Gardening Division"
    },
    {
        "thai_name": "นนทรีน้ำ (กระพี้จั่น)",
        "scientific_name": "Millettia brandisiana",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/0/07/Millettia_brandisiana.jpg/640px-Millettia_brandisiana.jpg",
        "description": "Medium native deciduous tree with purple-blue pea-like flowers.",
        "canopy_size": "Medium (6 - 10 m crown diameter, ~30 - 80 m² canopy)",
        "environments": ["Roadside", "Park", "Residential Area"],
        "benefits": "Fixes atmospheric nitrogen, highly drought resistant, attractive bloom.",
        "limitations": "Deciduous drop during peak dry season.",
        "source": "RFD Native Species Catalog"
    },
    {
        "thai_name": "จันผา",
        "scientific_name": "Dracaena cochinchinensis",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/5/50/Dracaena_cochinchinensis.jpg/640px-Dracaena_cochinchinensis.jpg",
        "description": "Distinctive architectural palm-like tree with sword-shaped leaves.",
        "canopy_size": "Small (2 - 4 m crown diameter, ~3 - 12 m² canopy)",
        "environments": ["Narrow Planting Space", "Residential Area", "Open Public Space"],
        "benefits": "Minimal soil volume requirement, highly drought resistant, striking visual design.",
        "limitations": "Low shading area; purely architectural/micro-buffer utility.",
        "source": "BMA Landscape Guidelines"
    },
    {
        "thai_name": "หว้า",
        "scientific_name": "Syzygium cumini",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a6/Syzygium_cumini_tree.jpg/640px-Syzygium_cumini_tree.jpg",
        "description": "Large dense evergreen tree producing small edible dark purple berries.",
        "canopy_size": "Large (10 - 16 m crown diameter, ~80 - 200 m² canopy)",
        "environments": ["Park", "Open Public Space"],
        "benefits": "Dense cooling canopy, high urban bird biodiversity support, flood tolerance.",
        "limitations": "Fallen dark berries stain pavement and vehicles in fruiting season.",
        "source": "BMA Parks & RFD"
    },
    {
        "thai_name": "มะดัน",
        "scientific_name": "Garcinia schomburgkiana",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4e/Garcinia_schomburgkiana.jpg/640px-Garcinia_schomburgkiana.jpg",
        "description": "Small to medium native evergreen tree adapted to lowlands and riverbanks.",
        "canopy_size": "Small to Medium (4 - 7 m crown diameter, ~12 - 40 m² canopy)",
        "environments": ["Narrow Planting Space", "Park", "Residential Area"],
        "benefits": "Tolerates high groundwater and urban moisture, compact evergreen shade.",
        "limitations": "Moderate growth rate; requires moist soil conditions.",
        "source": "Royal Forest Department"
    },
    {
        "thai_name": "จิกน้ำ",
        "scientific_name": "Barringtonia acutangula",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/2/28/Barringtonia_acutangula_flowers.jpg/640px-Barringtonia_acutangula_flowers.jpg",
        "description": "Medium wetland tree with long drooping red flower racemes.",
        "canopy_size": "Medium (6 - 10 m crown diameter, ~30 - 80 m² canopy)",
        "environments": ["Park", "Open Public Space", "Roadside"],
        "benefits": "Thrives in canal banks (khlongs) and flood-prone Bangkok zones; excellent shade.",
        "limitations": "Requires adequate water; drops long flower tassels daily.",
        "source": "BMA Canal Greening & RFD"
    }
]

# ==========================================
# 3. Sidebar Navigation
# ==========================================
st.sidebar.title("🌳 Navigation")
page = st.sidebar.radio(
    "Select Page:",
    [
        "Homepage & Executive Overview", 
        "Interactive Priority Map", 
        "District Comparison", 
        "Model Performance & Machine Learning",
        "Tree Planting Estimation & Species Guide" # NEW PAGE
    ]
)

st.sidebar.markdown("---")
st.sidebar.subheader("📌 Project Details")
st.sidebar.info("""
**Bangkok Urban Heat & Green Space Priority Dashboard**
- **Spatial Resolution:** 250m × 250m Grid (~25,000 cells)
- **Satellite Data:** Landsat 8/9 (2020–2025)
- **Target Users:** Urban Planners & BMA Policy Researchers
""")

# ==========================================
# PAGE 1: Homepage & Executive Overview
# ==========================================
if page == "Homepage & Executive Overview":
    st.title("Bangkok Urban Heat & Green Space Priority Dashboard")
    st.markdown("##### *An Academic Decision-Support System for Targeted Urban Greening Interventions*")
    st.markdown("---")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Mean Land Surface Temp (LST)</div>
            <div class="metric-value">{grid_df['LST'].mean():.1f} °C</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Mean Vegetation (NDVI)</div>
            <div class="metric-value">{grid_df['NDVI'].mean():.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Analytical Grid Cells</div>
            <div class="metric-value">{len(grid_df):,}</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        very_high_count = len(grid_df[grid_df['Priority_Category'] == 'Very High'])
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Very High Priority Zones</div>
            <div class="metric-value">{very_high_count:,}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 📋 Executive Summary & Key Findings")
    st.write("""
    Urban heat island effects present significant environmental challenges across Bangkok. This academic prototype integrates multi-year Landsat thermal satellite observations, NDVI vegetation indexes, and WorldPop population density models to answer key spatial planning questions:
    
    - **Thermal Distribution:** Satellite-derived Land Surface Temperature (LST) reveals pronounced heat concentration in dense, paved urban centers.
    - **Vegetation Deficits:** Low vegetation coverage coincides directly with commercial and residential corridors lacking public tree canopy.
    - **Priority Targeting:** Weighted priority scoring highlights areas where high surface temperature, low vegetation, and high population density intersect.
    """)

    st.warning("**Scientific & Planning Disclaimer:** High-priority grid cells represent zones for further field investigation. The system does not assess underground utilities, land ownership, tree species selection, or guaranteed microclimate cooling effects.")

# ==========================================
# PAGE 2: Interactive Priority Map
# ==========================================
elif page == "Interactive Priority Map":
    st.title("🗺️ Interactive Urban Heat & Green Space Priority Map")
    st.markdown("Explore heat exposure, vegetation deficits, and prioritized greening intervention zones.")
    
    col_filter1, col_filter2 = st.columns(2)
    with col_filter1:
        category_filter = st.multiselect(
            "Filter Priority Categories:",
            options=['Very High', 'High', 'Medium', 'Low'],
            default=['Very High', 'High']
        )
    with col_filter2:
        max_display = st.slider("Max Map Markers Displayed:", min_value=100, max_value=2000, value=500, step=100)
        
    filtered_df = grid_df[grid_df['Priority_Category'].isin(category_filter)].copy()
    if len(filtered_df) > max_display:
        filtered_df = filtered_df.sample(n=max_display, random_state=42)
        
    m = folium.Map(location=[13.7563, 100.5018], zoom_start=11, tiles="OpenStreetMap")
    
    color_map = {
        'Very High': 'red',
        'High': 'orange',
        'Medium': 'yellow',
        'Low': 'green'
    }
    
    for idx, row in filtered_df.iterrows():
        popup_text = f"""
        <b>Priority Level:</b> {row['Priority_Category']}<br>
        <b>Priority Score:</b> {row['Priority_Score']:.1f}/100<br>
        <b>LST:</b> {row['LST']:.1f} °C<br>
        <b>NDVI:</b> {row['NDVI']:.2f}<br>
        <b>Population Density:</b> {row['population']:.0f} persons/grid
        """
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=4,
            color=color_map.get(row['Priority_Category'], 'blue'),
            fill=True,
            fill_opacity=0.7,
            popup=folium.Popup(popup_text, max_width=250)
        ).add_to(m)
        
    st_folium(m, width=1200, height=600)

# ==========================================
# PAGE 3: District Comparison
# ==========================================
elif page == "District Comparison":
    st.title("📊 Bangkok District Comparisons")
    st.markdown("Side-by-side district summary analysis derived from multi-year spatial aggregations.")
    
    fig_lst = px.bar(
        district_df.sort_values(by='LST', ascending=False).head(20),
        x='District', y='LST',
        title="Top 20 Districts by Average Land Surface Temperature (°C)",
        color='LST', color_continuous_scale='Reds'
    )
    fig_lst.update_layout(template="plotly_dark")
    st.plotly_chart(fig_lst, use_container_width=True)
    
    fig_ndvi = px.bar(
        district_df.sort_values(by='NDVI', ascending=True).head(20),
        x='District', y='NDVI',
        title="Top 20 Districts with Lowest Average NDVI Vegetation Index",
        color='NDVI', color_continuous_scale='Greens_r'
    )
    fig_ndvi.update_layout(template="plotly_dark")
    st.plotly_chart(fig_ndvi, use_container_width=True)

# ==========================================
# PAGE 4: Model Performance & Machine Learning
# ==========================================
elif page == "Model Performance & Machine Learning":
    st.title("🤖 Machine Learning Model Performance")
    st.markdown("##### *Random Forest Regression for Land Surface Temperature Estimation*")
    st.markdown("---")
    
    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.metric("Mean Absolute Error (MAE)", f"{ml_metrics['MAE']:.3f} °C")
    with col_m2:
        st.metric("Root Mean Squared Error (RMSE)", f"{ml_metrics['RMSE']:.3f} °C")
    with col_m3:
        st.metric("R² Variance Score", f"{ml_metrics['R2']:.3f}")
        
    st.markdown("### 📊 Feature Importance Breakdown")
    importance_df = pd.DataFrame({
        'Feature': list(ml_metrics['importances'].keys()),
        'Importance': list(ml_metrics['importances'].values())
    }).sort_values(by='Importance', ascending=True)
    
    fig_imp = px.bar(importance_df, x='Importance', y='Feature', orientation='h', title="Random Forest Feature Importance")
    fig_imp.update_layout(template="plotly_dark")
    st.plotly_chart(fig_imp, use_container_width=True)
    
    st.markdown("### ⚠️ Methodological & Spatial Limitations")
    st.info("""
    - **Random Split Leakage:** Standard 80/20 train-test splits on spatial datasets can overestimate model performance due to spatial autocorrelation between adjacent 250m grid cells.
    - **Surface vs Air Temperature:** Satellite thermal sensors measure skin/surface temperature (LST), which fluctuates more dramatically than ambient air temperature.
    """)

# ==========================================
# PAGE 5: Tree Planting Estimation & Species Guide (NEW FEATURE)
# ==========================================
elif page == "Tree Planting Estimation & Species Guide":
    st.title("🌱 Tree Planting Estimation & Species Guide")
    st.markdown("##### *Decision-Support Planning Tool for Target Greening Requirements & Candidate Tree Selection*")
    st.markdown("---")
    
    st.markdown("""
    This feature builds directly upon the **Heat Priority Map**. By selecting specific priority tiers or administrative districts, 
    urban planners can estimate tree canopy expansion requirements under transparent, scenario-based assumptions.
    """)
    
    # ------------------------------------------
    # 1. Target Selection Controls
    # ------------------------------------------
    st.subheader("1. Select Heat Priority Target Areas")
    
    col_preset, col_dist = st.columns([1, 1])
    with col_preset:
        preset_choice = st.radio(
            "Quick Priority Tier Presets:",
            options=["Very High Only", "High Only", "Very High + High", "All Priority Tiers", "Custom Selection"],
            index=2,
            horizontal=True
        )
        
        if preset_choice == "Very High Only":
            selected_tiers = ["Very High"]
        elif preset_choice == "High Only":
            selected_tiers = ["High"]
        elif preset_choice == "Very High + High":
            selected_tiers = ["Very High", "High"]
        elif preset_choice == "All Priority Tiers":
            selected_tiers = ["Very High", "High", "Medium", "Low"]
        else:
            selected_tiers = st.multiselect(
                "Custom Priority Tier Selection:",
                options=["Very High", "High", "Medium", "Low"],
                default=["Very High", "High"]
            )
            
    with col_dist:
        all_districts = sorted(district_df['District'].unique().tolist()) if 'District' in district_df.columns else []
        select_all_districts = st.checkbox("Select All Bangkok Districts", value=True)
        if select_all_districts or not all_districts:
            selected_districts = all_districts
        else:
            selected_districts = st.multiselect(
                "Filter Specific District(s):",
                options=all_districts,
                default=all_districts[:5]
            )

    # Read-only Filtering of Existing Heat Priority Output Grid
    selected_grid = grid_df[grid_df['Priority_Category'].isin(selected_tiers)].copy()
    if all_districts and 'District' in selected_grid.columns and selected_districts:
        selected_grid = selected_grid[selected_grid['District'].isin(selected_districts)]
        
    num_selected_cells = len(selected_grid)
    
    # ------------------------------------------
    # 2. Interactive Calculator Assumptions
    # ------------------------------------------
    st.markdown("---")
    st.subheader("2. Interactive Scenario Assumptions")
    st.caption("Adjust the sliders below to explore different tree planting density and canopy scenarios.")
    
    col_param1, col_param2, col_param3, col_param4 = st.columns(4)
    with col_param1:
        plantable_pct = st.slider("Assumed Plantable Land (%)", min_value=1.0, max_value=50.0, value=15.0, step=1.0, help="Percentage of grid area available for tree planting (excluding existing buildings, roads, water bodies).")
    with col_param2:
        target_canopy_pct = st.slider("Target Canopy Coverage (%)", min_value=5.0, max_value=50.0, value=30.0, step=1.0, help="Desired percentage of plantable land covered by mature tree canopy.")
    with col_param3:
        tree_canopy_m2 = st.number_input("Mature Canopy Area per Tree (m²)", min_value=5.0, max_value=200.0, value=25.0, step=5.0, help="Assumed ground shadow area of a single mature tree crown.")
    with col_param4:
        tree_spacing_m = st.number_input("Tree Spacing Distance (m)", min_value=2.0, max_value=15.0, value=5.0, step=0.5, help="Assumed grid distance between adjacent planted trees.")
        
    # ------------------------------------------
    # 3. Transparent Calculation Engine
    # ------------------------------------------
    # Step 1: Area Calculation (Each 250m x 250m cell = 62,500 m²)
    CELL_AREA_M2 = 250.0 * 250.0
    total_selected_area_m2 = num_selected_cells * CELL_AREA_M2
    total_selected_area_km2 = total_selected_area_m2 / 1_000_000.0
    
    # Step 2: Plantable Area
    plantable_area_m2 = total_selected_area_m2 * (plantable_pct / 100.0)
    plantable_area_ha = plantable_area_m2 / 10_000.0
    
    # Step 3: Target Canopy Area
    target_canopy_area_m2 = plantable_area_m2 * (target_canopy_pct / 100.0)
    target_canopy_area_ha = target_canopy_area_m2 / 10_000.0
    
    # Step 4: Estimated Trees Needed
    if tree_canopy_m2 > 0:
        trees_needed = math.ceil(target_canopy_area_m2 / tree_canopy_m2)
    else:
        trees_needed = 0
        
    # Step 5: Planting Capacity
    spacing_area = tree_spacing_m ** 2
    if spacing_area > 0:
        planting_capacity = math.floor(plantable_area_m2 / spacing_area)
    else:
        planting_capacity = 0
        
    # Step 6: Feasibility Check
    is_feasible = trees_needed <= planting_capacity
    
    # ------------------------------------------
    # 4. Display Results & Feasibility Alert
    # ------------------------------------------
    st.markdown("---")
    st.subheader("3. Estimation Results & Capacity Feasibility")
    
    if num_selected_cells == 0:
        st.warning("⚠️ No grid cells match your selected filters. Please expand your priority tier or district selections.")
    else:
        if is_feasible:
            st.success(f"✅ **Target Scenario Feasible:** The estimated tree target ({trees_needed:,} trees) is within the maximum physical planting capacity ({planting_capacity:,} trees) under your selected spacing ({tree_spacing_m}m).")
        else:
            st.error(f"⚠️ **Target Scenario Infeasible Warning:** The estimated tree target ({trees_needed:,} trees) exceeds the physical planting capacity ({planting_capacity:,} trees) at {tree_spacing_m}m spacing. Increase tree spacing, reduce target canopy coverage, or adjust plantable land percentage.")
            
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        with m_col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Selected Grid Cells</div>
                <div class="metric-value">{num_selected_cells:,}</div>
                <small>{total_selected_area_km2:.2f} km² Total Area</small>
            </div>
            """, unsafe_allow_html=True)
        with m_col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Estimated Plantable Area</div>
                <div class="metric-value">{plantable_area_ha:.1f} ha</div>
                <small>{plantable_pct:.0f}% of Selected Area</small>
            </div>
            """, unsafe_allow_html=True)
        with m_col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Target Trees Needed</div>
                <div class="metric-value">{trees_needed:,}</div>
                <small>{target_canopy_area_ha:.1f} ha Canopy Target</small>
            </div>
            """, unsafe_allow_html=True)
        with m_col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Max Planting Capacity</div>
                <div class="metric-value">{planting_capacity:,}</div>
                <small>At {tree_spacing_m}m Spacing</small>
            </div>
            """, unsafe_allow_html=True)

        # Interactive Map Display of Selected Priority Cells
        st.markdown("#### 🗺️ Selected Priority Target Map")
        map_sample = selected_grid.sample(n=min(500, len(selected_grid)), random_state=42) if len(selected_grid) > 500 else selected_grid
        
        m_est = folium.Map(location=[13.7563, 100.5018], zoom_start=11, tiles="OpenStreetMap")
        color_map = {'Very High': 'red', 'High': 'orange', 'Medium': 'yellow', 'Low': 'green'}
        
        for idx, row in map_sample.iterrows():
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=4,
                color=color_map.get(row['Priority_Category'], 'blue'),
                fill=True,
                fill_opacity=0.7,
                popup=f"Priority: {row['Priority_Category']}<br>LST: {row['LST']:.1f}°C"
            ).add_to(m_est)
            
        st_folium(m_est, width=1200, height=450)

    # ------------------------------------------
    # 5. Tree Species Recommendation Guide
    # ------------------------------------------
    st.markdown("---")
    st.subheader("4. Candidate Tree Species Guide & Suitability Matrix")
    st.markdown("Filter recommended urban tree species based on target municipal planting environments:")
    
    env_filter = st.selectbox(
        "Select Target Planting Environment:",
        options=["All Environments", "Roadside", "Park", "Open Public Space", "Narrow Planting Space", "Residential Area"],
        index=0
    )
    
    # Filter species list based on environment selection
    if env_filter == "All Environments":
        filtered_species = SPECIES_DATA
    else:
        filtered_species = [s for s in SPECIES_DATA if env_filter in s['environments']]
        
    st.caption(f"Showing {len(filtered_species)} of {len(SPECIES_DATA)} candidate tree species suitable for **{env_filter}**.")
    
    # Display Species Cards
    for s in filtered_species:
        st.markdown(f"""
        <div class="species-card">
            <div class="species-title">{s['thai_name']}</div>
            <div class="species-scientific">{s['scientific_name']}</div>
            <div>{" ".join([f'<span class="badge">{e}</span>' for e in s['environments']])}</div>
            <hr style="border-color: #3a506b; margin: 10px 0;">
            <p><b>Description:</b> {s['description']}</p>
            <p><b>Mature Canopy Size:</b> {s['canopy_size']}</p>
            <p><b>Key Environmental Benefits:</b> {s['benefits']}</p>
            <p style="color: #e63946;"><b>Important Limitations:</b> {s['limitations']}</p>
            <p><small><b>Reference / Source:</b> {s['source']}</small></p>
        </div>
        """, unsafe_allow_html=True)

    # ------------------------------------------
    # 6. Important Scientific & Planning Disclaimers
    # ------------------------------------------
    st.markdown("---")
    st.subheader("⚠️ Important Scientific & Planning Limitations")
    st.info("""
    1. **Planning Estimates Only:** The tree numbers generated by this tool are educational planning scenarios based on user-defined inputs. They do not represent exact municipal tree planting directives.
    2. **Assumed Available Land:** The model assumes a uniform plantable land percentage. It does not inspect real-time property boundaries, land ownership, pavement coverage, or underground utilities.
    3. **Non-Guarantee of Survival:** Calculated targets do not account for sapling mortality rates, soil compaction, water availability, or maintenance schedules.
    4. **Air vs. Surface Temperature:** Adding trees reduces Land Surface Temperature (LST) through direct shading, but microclimate ambient air cooling depends on local wind patterns, building geometries, and humidity.
    5. **Site-Specific Arboricultural Requirements:** Species suitability guidance provides general ecological rules of thumb and must be verified by a certified arborist prior to field planting.
    """)
